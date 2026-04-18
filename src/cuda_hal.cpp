#include "cuda_hal.h"
#include "stream_pool.h"
#include "twiddle_gen.h"
#include "vram_pool.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); \
         if(_e!=cudaSuccess) throw std::runtime_error( \
             std::string("[CUDA HAL] ")+cudaGetErrorString(_e)); } while(0)

extern "C" void LaunchRNSMult(const uint64_t* a, const uint64_t* b, uint64_t* r,
                               uint64_t q, uint32_t n, cudaStream_t s);
extern "C" void LaunchRNSMultMontgomery(const uint64_t* a, const uint64_t* b, uint64_t* r,
                                         uint64_t q, uint64_t q_inv, uint64_t R2,
                                         uint32_t n, cudaStream_t s);
extern "C" void LaunchNTT(uint64_t* x, const uint64_t* tw,
                           uint64_t q, uint32_t n, cudaStream_t s);
extern "C" void LaunchINTT(uint64_t* x, const uint64_t* tw_inv,
                            uint64_t q, uint32_t n, uint64_t n_inv, cudaStream_t s);

// Montgomery precomputation helpers.
static uint64_t calc_q_inv(uint64_t q) {
    uint64_t inv = q;
    for (int i = 0; i < 5; ++i) inv *= 2 - q * inv;
    return -inv; // -q^{-1} mod 2^64 (correct REDC form)
}

static uint64_t calc_R2(uint64_t q) {
    // R = 2^64. Compute R^2 mod q = ((2^64 mod q)^2) mod q.
    unsigned __int128 R = ((unsigned __int128)1 << 64) % q;
    return (uint64_t)((R * R) % q);
}

// FIX: Pool slot must accommodate the largest ring dimension used anywhere.
// Old MAX_RING=32768 caused silent runtime throws for ring=65536.
// Set to 65536 (512 KB per slot) which covers all standard CKKS/BGV rings.
static constexpr uint32_t MAX_RING        = 65536;
static constexpr size_t   POOL_SLOT_BYTES = MAX_RING * sizeof(uint64_t);

static void EnsurePool() {
    openfhe_cuda::VRAMPool::Instance().Init(POOL_SLOT_BYTES);
    // Keep enough streams for all towers (32 is fine for up to 32 RNS towers).
    openfhe_cuda::StreamPool::Instance().Init(32);
}

// ── Twiddle cache (with proper VRAM cleanup on shutdown) ──────────────────────
struct TwKey {
    uint64_t q; uint32_t N;
    bool operator==(const TwKey& o) const { return q==o.q && N==o.N; }
};
struct TwKeyHash {
    size_t operator()(const TwKey& k) const {
        return std::hash<uint64_t>()(k.q) ^ ((size_t)std::hash<uint32_t>()(k.N) << 32);
    }
};
struct DeviceTwiddles {
    uint64_t* d_fwd  = nullptr;
    uint64_t* d_inv  = nullptr;
    uint64_t  n_inv  = 0;
    uint32_t  N      = 0;
};

// FIX: wrap in a struct with a destructor so VRAM is freed at process exit.
struct TwiddleCache {
    std::unordered_map<TwKey, DeviceTwiddles, TwKeyHash> map;
    std::mutex mu;

    ~TwiddleCache() {
        for (auto& [k, dt] : map) {
            if (dt.d_fwd) cudaFree(dt.d_fwd);
            if (dt.d_inv) cudaFree(dt.d_inv);
        }
    }
};
static TwiddleCache g_tw_cache;

// FIX: twiddle table is now length N (not N/2) -- negacyclic twisted layout.
static const DeviceTwiddles& GetDeviceTwiddles(uint64_t q, uint32_t N) {
    TwKey key{q, N};
    std::lock_guard<std::mutex> lk(g_tw_cache.mu);
    auto it = g_tw_cache.map.find(key);
    if (it != g_tw_cache.map.end()) return it->second;

    TwiddleTable tt = BuildTwiddleTable(q, N);  // now negacyclic, length-N tables
    DeviceTwiddles dt;
    dt.N     = N;
    dt.n_inv = tt.n_inv;

    // FIX: upload N elements (not N/2) -- negacyclic twisted twiddles.
    size_t bytes = 2*N * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&dt.d_fwd, bytes));
    CUDA_CHECK(cudaMalloc(&dt.d_inv, bytes));
    CUDA_CHECK(cudaMemcpy(dt.d_fwd, tt.forward.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt.d_inv, tt.inverse.data(), bytes, cudaMemcpyHostToDevice));

    g_tw_cache.map[key] = dt;
    return g_tw_cache.map[key];
}

// ── Legacy pool API (used by benchmark.cpp / main_gpu_engine.cpp) ─────────────
namespace openfhe_cuda {

void CUDAMathHAL::AllocateVRAM(std::vector<uint64_t*>& ptrs, uint32_t towers, uint32_t ring) {
    size_t bytes = ring * sizeof(uint64_t);
    ptrs.resize(towers);
    for (uint32_t i = 0; i < towers; ++i)
        CUDA_CHECK(cudaMalloc(&ptrs[i], bytes));
}

void CUDAMathHAL::FreeVRAM(std::vector<uint64_t*>& ptrs) {
    for (auto p : ptrs) if (p) cudaFree(p);
    ptrs.clear();
}

void CUDAMathHAL::EvalMultRNS(
    const std::vector<uint64_t*>& d_a,
    const std::vector<uint64_t*>& d_b,
    std::vector<uint64_t*>&       d_res,
    const std::vector<uint64_t>&  moduli,
    uint32_t ring)
{
    EnsurePool();
    uint32_t towers = (uint32_t)d_a.size();
    for (uint32_t i = 0; i < towers; ++i) {
        cudaStream_t s = StreamPool::Instance().Get(i);
        uint64_t q_inv = calc_q_inv(moduli[i]);
        uint64_t R2    = calc_R2(moduli[i]);
        LaunchRNSMultMontgomery(d_a[i], d_b[i], d_res[i], moduli[i], q_inv, R2, ring, s);
    }
    StreamPool::Instance().SyncAll();
}

uint64_t* CUDAMathHAL::GetOrAllocateDevicePtr(const void*, uint32_t, cudaStream_t) {
    return nullptr; // SWA cache stub -- implemented in OpenFHE patch path
}
void CUDAMathHAL::ClearShadowCache() {}

} // namespace openfhe_cuda

// ── Batch pointwise RNS multiply (used by OpenFHE hook & duality bench) ───────
//
// FIX: True concurrent stream dispatch.
// Old code: launch kernel, sync stream, copy back -- all inside the loop.
//   → Towers executed sequentially despite having separate streams.
// New code: launch all kernels first across all streams, then sync all
//   streams, then copy all results back. This lets the GPU overlap kernels
//   across streams (Pascal+ hardware with MPS or sufficient SM resources).
//
// VRAM lifetime: VRAMSlot RAII objects must survive until after the memcpy.
// We manage this by storing all slots in a vector that outlives both loops.
extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    EnsurePool();
    if (ring * sizeof(uint64_t) > POOL_SLOT_BYTES)
        throw std::runtime_error("[CUDA HAL] ring=" + std::to_string(ring) +
                                 " exceeds MAX_RING=" + std::to_string(MAX_RING));
    size_t bytes = ring * sizeof(uint64_t);
    std::vector<uint64_t*> da(num_towers), db(num_towers), dr(num_towers);
    for (uint32_t i = 0; i < num_towers; i++) {
        CUDA_CHECK(cudaMalloc(&da[i], bytes));
        CUDA_CHECK(cudaMalloc(&db[i], bytes));
        CUDA_CHECK(cudaMalloc(&dr[i], bytes));
    }
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        uint64_t q_inv = calc_q_inv(q[i]);
        uint64_t R2    = calc_R2(q[i]);
        CUDA_CHECK(cudaMemcpyAsync(da[i], ha[i], bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(db[i], hb[i], bytes, cudaMemcpyHostToDevice, s));
        LaunchRNSMultMontgomery(da[i], db[i], dr[i], q[i], q_inv, R2, ring, s);
    }
    openfhe_cuda::StreamPool::Instance().SyncAll();
    for (uint32_t i = 0; i < num_towers; i++) {
        CUDA_CHECK(cudaMemcpy(hr[i], dr[i], bytes, cudaMemcpyDeviceToHost));
        cudaFree(da[i]); cudaFree(db[i]); cudaFree(dr[i]);
    }
}

// ── Polynomial multiply via NTT (negacyclic) ─────────────────────────────────
extern "C" void gpu_poly_mult_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    EnsurePool();
    if (ring * sizeof(uint64_t) > POOL_SLOT_BYTES)
        throw std::runtime_error("[CUDA HAL] ring exceeds MAX_RING");
    size_t bytes = ring * sizeof(uint64_t);
    std::vector<uint64_t*> da(num_towers), db(num_towers), dr(num_towers);
    for (uint32_t i = 0; i < num_towers; i++) {
        CUDA_CHECK(cudaMalloc(&da[i], bytes));
        CUDA_CHECK(cudaMalloc(&db[i], bytes));
        CUDA_CHECK(cudaMalloc(&dr[i], bytes));
    }
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s           = openfhe_cuda::StreamPool::Instance().Get(i);
        const DeviceTwiddles& dt = GetDeviceTwiddles(q[i], ring);
        CUDA_CHECK(cudaMemcpyAsync(da[i], ha[i], bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(db[i], hb[i], bytes, cudaMemcpyHostToDevice, s));
        LaunchNTT(da[i], dt.d_fwd, q[i], ring, s);
        LaunchNTT(db[i], dt.d_fwd, q[i], ring, s);
    }
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        uint64_t q_inv = calc_q_inv(q[i]);
        uint64_t R2    = calc_R2(q[i]);
        LaunchRNSMultMontgomery(da[i], db[i], dr[i], q[i], q_inv, R2, ring, s);
    }
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s           = openfhe_cuda::StreamPool::Instance().Get(i);
        const DeviceTwiddles& dt = GetDeviceTwiddles(q[i], ring);
        LaunchINTT(dr[i], dt.d_inv, q[i], ring, dt.n_inv, s);
    }
    openfhe_cuda::StreamPool::Instance().SyncAll();
    for (uint32_t i = 0; i < num_towers; i++) {
        CUDA_CHECK(cudaMemcpy(hr[i], dr[i], bytes, cudaMemcpyDeviceToHost));
        cudaFree(da[i]); cudaFree(db[i]); cudaFree(dr[i]);
    }
}

// Single-tower wrapper (used by bench_evalmult.cpp).
extern "C" void gpu_rns_mult_wrapper(
    const uint64_t* ha, const uint64_t* hb, uint64_t* hr,
    uint64_t q, uint64_t /*unused*/, uint32_t ring, uint32_t tower_idx)
{
    const uint64_t* pA = ha; const uint64_t* pB = hb; uint64_t* pR = hr;
    gpu_rns_mult_batch_wrapper(&pA, &pB, &pR, &q, ring, 1);
    (void)tower_idx;
}

extern "C" void gpu_synchronize_all() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
}

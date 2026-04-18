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
extern "C" void LaunchNTT(uint64_t* x, const uint64_t* tw,
                           uint64_t q, uint32_t n, cudaStream_t s);
extern "C" void LaunchINTT(uint64_t* x, const uint64_t* tw_inv,
                            uint64_t q, uint32_t n, uint64_t n_inv, cudaStream_t s);

// ── Pool configuration ────────────────────────────────────────────────────────
// 3 device buffers per OMP thread (a, b, r), 8 threads = 24 slots minimum.
// We use 48 to allow headroom for batch calls that pipeline multiple towers.
// Slot size = largest ring we expect: 32768 coefficients * 8 bytes = 256 KB.
static constexpr uint32_t POOL_SLOTS     = 48;
static constexpr uint32_t MAX_RING       = 32768;
static constexpr size_t   POOL_SLOT_BYTES = MAX_RING * sizeof(uint64_t);

static void EnsurePool() {
    openfhe_cuda::VRAMPool::Instance().Init(POOL_SLOTS, POOL_SLOT_BYTES);
    openfhe_cuda::StreamPool::Instance().Init(32);
}

// ── Twiddle cache ─────────────────────────────────────────────────────────────
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

static std::unordered_map<TwKey, DeviceTwiddles, TwKeyHash> g_tw_cache;
static std::mutex g_tw_mu;

static const DeviceTwiddles& GetDeviceTwiddles(uint64_t q, uint32_t N) {
    TwKey key{q, N};
    std::lock_guard<std::mutex> lk(g_tw_mu);
    auto it = g_tw_cache.find(key);
    if (it != g_tw_cache.end()) return it->second;
    TwiddleTable tt = BuildTwiddleTable(q, N);
    DeviceTwiddles dt;
    dt.N     = N;
    dt.n_inv = tt.n_inv;
    size_t bytes = (N / 2) * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&dt.d_fwd, bytes));
    CUDA_CHECK(cudaMalloc(&dt.d_inv, bytes));
    CUDA_CHECK(cudaMemcpy(dt.d_fwd, tt.forward.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt.d_inv, tt.inverse.data(), bytes, cudaMemcpyHostToDevice));
    g_tw_cache[key] = dt;
    return g_tw_cache[key];
}

// ── Pointwise RNS multiply — pool-backed, zero dynamic allocation ─────────────
extern "C" void gpu_rns_mult_wrapper(
    const uint64_t* ha, const uint64_t* hb, uint64_t* hr,
    uint64_t q, uint64_t /*unused*/, uint32_t ring, uint32_t tower_idx)
{
    EnsurePool();
    if (ring * sizeof(uint64_t) > POOL_SLOT_BYTES)
        throw std::runtime_error("[CUDA HAL] ring too large for pool slot");

    size_t bytes = ring * sizeof(uint64_t);
    cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(tower_idx);

    openfhe_cuda::VRAMSlot da, db, dr;
    CUDA_CHECK(cudaMemcpyAsync(da.ptr, ha, bytes, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(db.ptr, hb, bytes, cudaMemcpyHostToDevice, s));
    LaunchRNSMult(da.ptr, db.ptr, dr.ptr, q, ring, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(hr, dr.ptr, bytes, cudaMemcpyDeviceToHost));
    // VRAMSlot destructors return da, db, dr to pool here
}

extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    EnsurePool();
    if (ring * sizeof(uint64_t) > POOL_SLOT_BYTES)
        throw std::runtime_error("[CUDA HAL] ring too large for pool slot");

    size_t bytes = ring * sizeof(uint64_t);

    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        openfhe_cuda::VRAMSlot da, db, dr;
        CUDA_CHECK(cudaMemcpyAsync(da.ptr, ha[i], bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(db.ptr, hb[i], bytes, cudaMemcpyHostToDevice, s));
        LaunchRNSMult(da.ptr, db.ptr, dr.ptr, q[i], ring, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(hr[i], dr.ptr, bytes, cudaMemcpyDeviceToHost));
    }
}

// ── Polynomial convolution: NTT -> pointwise mult -> INTT ─────────────────────
extern "C" void gpu_poly_mult_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    EnsurePool();
    if (ring * sizeof(uint64_t) > POOL_SLOT_BYTES)
        throw std::runtime_error("[CUDA HAL] ring too large for pool slot");

    size_t bytes = ring * sizeof(uint64_t);

    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        const DeviceTwiddles& dt = GetDeviceTwiddles(q[i], ring);

        openfhe_cuda::VRAMSlot da, db, dr;
        CUDA_CHECK(cudaMemcpyAsync(da.ptr, ha[i], bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(db.ptr, hb[i], bytes, cudaMemcpyHostToDevice, s));

        LaunchNTT(da.ptr, dt.d_fwd, q[i], ring, s);
        LaunchNTT(db.ptr, dt.d_fwd, q[i], ring, s);
        LaunchRNSMult(da.ptr, db.ptr, dr.ptr, q[i], ring, s);
        LaunchINTT(dr.ptr, dt.d_inv, q[i], ring, dt.n_inv, s);

        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(hr[i], dr.ptr, bytes, cudaMemcpyDeviceToHost));
    }
}

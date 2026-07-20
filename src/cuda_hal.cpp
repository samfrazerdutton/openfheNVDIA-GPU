#include "cuda_hal.h"
#include "stream_pool.h"
#include "twiddle_gen.h"
#include "shadow_registry.h"
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { cudaError_t _e = (call); \
         if (_e != cudaSuccess) \
             throw std::runtime_error( \
                 std::string("[CUDA HAL] " #call ": ") + cudaGetErrorString(_e)); \
    } while (0)

extern "C" void LaunchRNSMultMontgomery(const uint64_t* a, const uint64_t* b, uint64_t* r,
    uint64_t q, uint64_t q_inv, uint64_t R2, uint32_t n, cudaStream_t s);
extern "C" void LaunchNTT(uint64_t* x, const uint64_t* tw, uint64_t q, uint64_t q_inv,
    uint32_t n, cudaStream_t s);
extern "C" void LaunchINTT(uint64_t* x, const uint64_t* tw_inv, uint64_t q, uint64_t q_inv,
    uint32_t n, uint64_t n_inv, cudaStream_t s);

static uint64_t calc_q_inv(uint64_t q) {
    uint64_t x = q;
    for (int i = 0; i < 5; ++i) x *= 2 - q * x;
    return -x;
}
static uint64_t calc_R2(uint64_t q) {
    unsigned __int128 R = ((unsigned __int128)1 << 64) % q;
    return (uint64_t)((R * R) % q);
}

struct TwKey { uint64_t q; uint32_t N;
    bool operator==(const TwKey& o) const { return q==o.q && N==o.N; } };
struct TwKeyHash { size_t operator()(const TwKey& k) const {
    return std::hash<uint64_t>()(k.q) ^ ((size_t)std::hash<uint32_t>()(k.N) << 32); } };
struct DeviceTwiddles { uint64_t* d_fwd=nullptr; uint64_t* d_inv=nullptr;
    uint64_t n_inv=0; uint32_t N=0; };

static std::unordered_map<TwKey, DeviceTwiddles, TwKeyHash> g_tw_map;
static std::mutex g_tw_mu;

static const DeviceTwiddles& GetDeviceTwiddles(uint64_t q, uint32_t N) {
    TwKey key{q, N};
    std::lock_guard<std::mutex> lk(g_tw_mu);
    auto it = g_tw_map.find(key);
    if (it != g_tw_map.end()) return it->second;
    TwiddleTable tt = BuildTwiddleTable(q, N);
    DeviceTwiddles dt; dt.N=N; dt.n_inv=tt.n_inv;
    size_t bytes = 2*N*sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&dt.d_fwd, bytes));
    CUDA_CHECK(cudaMalloc(&dt.d_inv, bytes));
    CUDA_CHECK(cudaMemcpy(dt.d_fwd, tt.forward.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt.d_inv, tt.inverse.data(), bytes, cudaMemcpyHostToDevice));
    g_tw_map[key] = dt;
    return g_tw_map[key];
}

namespace openfhe_cuda {
void CUDAMathHAL::InitStreams(uint32_t n)  { StreamPool::Instance().Init(n); }
void CUDAMathHAL::DestroyStreams()         {}
void CUDAMathHAL::Synchronize()           { cudaDeviceSynchronize(); }
void CUDAMathHAL::AllocateVRAM(std::vector<uint64_t*>& ptrs, uint32_t towers, uint32_t ring) {
    size_t bytes = (size_t)ring * sizeof(uint64_t);
    ptrs.resize(towers);
    for (uint32_t i = 0; i < towers; ++i) CUDA_CHECK(cudaMalloc(&ptrs[i], bytes));
}
void CUDAMathHAL::FreeVRAM(std::vector<uint64_t*>& ptrs) {
    for (auto p : ptrs) if (p) cudaFree(p); ptrs.clear();
}
void CUDAMathHAL::EvalMultRNS(
    const std::vector<uint64_t*>& d_a, const std::vector<uint64_t*>& d_b,
    std::vector<uint64_t*>& d_res, const std::vector<uint64_t>& moduli, uint32_t ring)
{
    StreamPool::Instance().Init(32);
    uint32_t towers = (uint32_t)d_a.size();
    for (uint32_t i = 0; i < towers; ++i) {
        cudaStream_t s = StreamPool::Instance().Get(i);
        LaunchRNSMultMontgomery(d_a[i], d_b[i], d_res[i],
            moduli[i], calc_q_inv(moduli[i]), calc_R2(moduli[i]), ring, s);
    }
    StreamPool::Instance().SyncAll();
}
} // namespace openfhe_cuda

extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    openfhe_cuda::StreamPool::Instance().Init(32);
    size_t bytes = (size_t)ring * sizeof(uint64_t);
    auto& reg = ShadowRegistry::Instance();

    // d_out is now cached the same way da/db already were: keyed by the real,
    // unique output host pointer via ShadowRegistry. ShadowRegistry already
    // proved safe under 8 concurrent OMP threads for da/db -- reusing that
    // exact mechanism (rather than a new pool keyed only by tower index)
    // avoids the per-call cudaMalloc/cudaFree without introducing a new
    // buffer-aliasing race across concurrent callers.
    std::vector<uint64_t*> d_out(num_towers, nullptr);

    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        uint64_t* da = reg.GetDevicePtr(ha[i], bytes);
        uint64_t* db = reg.GetDevicePtr(hb[i], bytes);
        d_out[i]      = reg.GetDevicePtr(hr[i], bytes);
        // cudaMemcpyDefault: required for cudaMallocManaged buffers.
        // cudaMemcpyHostToDevice is illegal when dst is a managed pointer.
        if (!da || !db || !d_out[i]) {
            throw std::runtime_error("[CUDA HAL] Null pointer from cudaMallocManaged. VRAM exhausted.");
        }
        CUDA_CHECK(cudaMemcpyAsync(da, ha[i], bytes, cudaMemcpyDefault, s));
        CUDA_CHECK(cudaMemcpyAsync(db, hb[i], bytes, cudaMemcpyDefault, s));
        LaunchRNSMultMontgomery(da, db, d_out[i],
            q[i], calc_q_inv(q[i]), calc_R2(q[i]), ring, s);
    }
    openfhe_cuda::StreamPool::Instance().SyncAll();

    for (uint32_t i = 0; i < num_towers; i++)
        CUDA_CHECK(cudaMemcpy(hr[i], d_out[i], bytes, cudaMemcpyDeviceToHost));
    // No cudaFree -- ShadowRegistry retains ownership and reuses this
    // allocation the next time the same host pointer is seen.
}

extern "C" void gpu_poly_mult_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    openfhe_cuda::StreamPool::Instance().Init(32);
    size_t bytes = (size_t)ring * sizeof(uint64_t);
    auto& reg = ShadowRegistry::Instance();

    // Dynamically sized -- no hardcoded tower cap (previously a fixed
    // 64-entry static array with no bounds check: silent memory corruption
    // past 64 towers).
    static std::vector<uint64_t> scratch_a;
    static std::vector<uint64_t> scratch_b;
    if (scratch_a.size() < num_towers) scratch_a.resize(num_towers);
    if (scratch_b.size() < num_towers) scratch_b.resize(num_towers);

    std::vector<uint64_t*> d_out(num_towers, nullptr);

    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        const DeviceTwiddles& dt = GetDeviceTwiddles(q[i], ring);
        uint64_t q_inv = calc_q_inv(q[i]);
        uint64_t* da = reg.GetDevicePtr(&scratch_a[i], bytes);
        uint64_t* db = reg.GetDevicePtr(&scratch_b[i], bytes);
        d_out[i]      = reg.GetDevicePtr(hr[i], bytes);
        if (!da || !db || !d_out[i]) {
            throw std::runtime_error("[CUDA HAL] Null pointer from cudaMallocManaged. VRAM exhausted.");
        }
        CUDA_CHECK(cudaMemcpyAsync(da, ha[i], bytes, cudaMemcpyDefault, s));
        CUDA_CHECK(cudaMemcpyAsync(db, hb[i], bytes, cudaMemcpyDefault, s));
        LaunchNTT(da, dt.d_fwd, q[i], q_inv, ring, s);
        LaunchNTT(db, dt.d_fwd, q[i], q_inv, ring, s);
        LaunchRNSMultMontgomery(da, db, d_out[i], q[i], q_inv, calc_R2(q[i]), ring, s);
        LaunchINTT(d_out[i], dt.d_inv, q[i], q_inv, ring, dt.n_inv, s);
    }
    openfhe_cuda::StreamPool::Instance().SyncAll();

    for (uint32_t i = 0; i < num_towers; i++)
        CUDA_CHECK(cudaMemcpy(hr[i], d_out[i], bytes, cudaMemcpyDeviceToHost));
    // No cudaFree -- ShadowRegistry retains ownership and reuses this
    // allocation the next time the same host pointer is seen.
}

extern "C" void gpu_rns_mult_wrapper(
    const uint64_t* ha, const uint64_t* hb, uint64_t* hr,
    uint64_t q, uint64_t, uint32_t ring, uint32_t)
{
    const uint64_t* pA=ha; const uint64_t* pB=hb; uint64_t* pR=hr;
    gpu_rns_mult_batch_wrapper(&pA, &pB, &pR, &q, ring, 1);
}

extern "C" void gpu_synchronize_all() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
}

extern "C" void gpu_clear_vram_cache() {
    ShadowRegistry::Instance().Clear();
}

extern "C" void gpu_prepare_for_decrypt() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
}

thread_local bool gpu_evalmult_enabled = true;
extern "C" void gpu_disable_for_decrypt() { gpu_evalmult_enabled = false; }
extern "C" void gpu_enable_evalmult()     { gpu_evalmult_enabled = true;  }

extern "C" void gpu_sync_all_to_host() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
    cudaDeviceSynchronize();
}

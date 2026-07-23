#include "cuda_hal.h"
#include "stream_pool.h"
#include "twiddle_gen.h"
#include "shadow_registry.h"
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <cuda_runtime.h>
#include <cstring>
#include <atomic>
#include <cstdio>
#include <cstdlib>

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

// Initialize CUDA context + stream pool at library load, so the ~300ms
// first-touch cost (WSL) never lands inside a timed FHE operation.
__attribute__((constructor)) static void gpu_hal_warmup() {
    const char* e = std::getenv("OPENFHE_GPU");
    if (e && e[0] == '0') return;
    cudaFree(nullptr);  // force context creation
    openfhe_cuda::StreamPool::Instance().Init(32);
}

extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    static const bool log_calls = [] {
        const char* e = std::getenv("OPENFHE_GPU_LOG");
        return e && e[0] == '1';
    }();
    static std::atomic<uint64_t> call_no{0};
    if (log_calls)
        printf("[GPU_LOG] call=%llu towers=%u ring=%u\n",
                (unsigned long long)++call_no, num_towers, ring);
    openfhe_cuda::StreamPool::Instance().Init(32);
    size_t bytes = (size_t)ring * sizeof(uint64_t);
    auto& reg = ShadowRegistry::Instance();

    // Pinned staging path (Fix 1): host -> HAL-owned pinned buffer (memcpy)
    // -> device (true async DMA); results come back the same way. The HAL
    // owns the pinned memory (cudaMallocHost, reused forever), so OpenFHE
    // freeing its own buffers can never dangle a registration. If pinned
    // allocation fails we fall back to pageable copies (slower, correct).
    std::vector<uint64_t*> d_out(num_towers, nullptr);
    auto& pool = PinnedStagingPool::Instance();
    std::vector<void*> st_a(num_towers, nullptr), st_b(num_towers, nullptr),
                       st_r(num_towers, nullptr);
    bool staged = true;
    for (uint32_t i = 0; i < num_towers; i++) {
        st_a[i] = pool.Acquire(bytes);
        st_b[i] = pool.Acquire(bytes);
        st_r[i] = pool.Acquire(bytes);
        if (!st_a[i] || !st_b[i] || !st_r[i]) staged = false;
    }

    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        uint64_t* da = reg.GetDevicePtr(ha[i], bytes);
        uint64_t* db = reg.GetDevicePtr(hb[i], bytes);
        d_out[i]      = reg.GetDevicePtr(hr[i], bytes);
        if (staged) {
            std::memcpy(st_a[i], ha[i], bytes);
            std::memcpy(st_b[i], hb[i], bytes);
            CUDA_CHECK(cudaMemcpyAsync(da, st_a[i], bytes, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(db, st_b[i], bytes, cudaMemcpyHostToDevice, s));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(da, ha[i], bytes, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(db, hb[i], bytes, cudaMemcpyHostToDevice, s));
        }
        LaunchRNSMultMontgomery(da, db, d_out[i],
            q[i], calc_q_inv(q[i]), calc_R2(q[i]), ring, s);
        if (staged)
            CUDA_CHECK(cudaMemcpyAsync(st_r[i], d_out[i], bytes, cudaMemcpyDeviceToHost, s));
        else
            CUDA_CHECK(cudaMemcpyAsync(hr[i], d_out[i], bytes, cudaMemcpyDeviceToHost, s));
    }
    openfhe_cuda::StreamPool::Instance().SyncAll();

    for (uint32_t i = 0; i < num_towers; i++) {
        if (staged) std::memcpy(hr[i], st_r[i], bytes);
        pool.Release(st_a[i]);
        pool.Release(st_b[i]);
        pool.Release(st_r[i]);
    }
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

// ---- Runtime mode control -------------------------------------------------
// The GPU/KS gates were read once into function-local statics, which made
// in-process A/B impossible: any benchmark had to fork a new process per mode,
// so CPU and GPU samples were never drawn at the same thermal moment. These
// accessors keep the env vars as defaults but allow a harness to switch modes
// between reps, which is what paired-difference timing requires.
static int g_gpu_mode = -1;   // -1 uninitialized, 0 off, 1 on
static int g_ks_mode  = -1;

extern "C" int gpu_hal_enabled() {
    if (g_gpu_mode < 0) {
        const char* e = std::getenv("OPENFHE_GPU");
        g_gpu_mode = (e && e[0] == '0') ? 0 : 1;
    }
    return g_gpu_mode;
}

extern "C" int gpu_ks_enabled() {
    if (g_ks_mode < 0) {
        const char* e = std::getenv("OPENFHE_GPU_KS");
        g_ks_mode = (e && e[0] == '1') ? 1 : 0;
    }
    return g_ks_mode;
}

extern "C" void gpu_set_mode(int gpu, int ks) {
    g_gpu_mode = gpu;
    g_ks_mode  = ks;
}

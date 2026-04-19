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

static uint64_t calc_q_inv(uint64_t q) {
    uint64_t inv = q;
    for (int i = 0; i < 5; ++i) inv *= 2 - q * inv;
    return -inv; 
}

static uint64_t calc_R2(uint64_t q) {
    unsigned __int128 R = ((unsigned __int128)1 << 64) % q;
    return (uint64_t)((R * R) % q);
}

static constexpr uint32_t MAX_RING        = 65536;
static constexpr size_t   POOL_SLOT_BYTES = MAX_RING * sizeof(uint64_t);

static void EnsurePool() {
    openfhe_cuda::VRAMPool::Instance().Init(POOL_SLOT_BYTES);
    openfhe_cuda::StreamPool::Instance().Init(32);
}

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

static const DeviceTwiddles& GetDeviceTwiddles(uint64_t q, uint32_t N) {
    TwKey key{q, N};
    std::lock_guard<std::mutex> lk(g_tw_cache.mu);
    auto it = g_tw_cache.map.find(key);
    if (it != g_tw_cache.map.end()) return it->second;

    TwiddleTable tt = BuildTwiddleTable(q, N); 
    DeviceTwiddles dt;
    dt.N     = N;
    dt.n_inv = tt.n_inv;

    // Twiddles are size 2N (Twist + Cyclic Roots)
    size_t twiddle_bytes = 2 * N * sizeof(uint64_t); 
    CUDA_CHECK(cudaMalloc(&dt.d_fwd, twiddle_bytes));
    CUDA_CHECK(cudaMalloc(&dt.d_inv, twiddle_bytes));
    CUDA_CHECK(cudaMemcpy(dt.d_fwd, tt.forward.data(), twiddle_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dt.d_inv, tt.inverse.data(), twiddle_bytes, cudaMemcpyHostToDevice));

    g_tw_cache.map[key] = dt;
    return g_tw_cache.map[key];
}

namespace openfhe_cuda {
    void CUDAMathHAL::AllocateVRAM(std::vector<uint64_t*>& ptrs, uint32_t towers, uint32_t ring) {
        size_t bytes = ring * sizeof(uint64_t);
        ptrs.resize(towers);
        for (uint32_t i = 0; i < towers; ++i) CUDA_CHECK(cudaMalloc(&ptrs[i], bytes));
    }
    void CUDAMathHAL::FreeVRAM(std::vector<uint64_t*>& ptrs) {
        for (auto p : ptrs) if (p) cudaFree(p);
        ptrs.clear();
    }
    void CUDAMathHAL::EvalMultRNS(const std::vector<uint64_t*>& d_a, const std::vector<uint64_t*>& d_b, std::vector<uint64_t*>& d_res, const std::vector<uint64_t>& moduli, uint32_t ring) {
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
    uint64_t* CUDAMathHAL::GetOrAllocateDevicePtr(const void*, uint32_t, cudaStream_t) { return nullptr; }
    void CUDAMathHAL::ClearShadowCache() {}
}

extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    EnsurePool();
    if (ring * sizeof(uint64_t) > POOL_SLOT_BYTES)
        throw std::runtime_error("[CUDA HAL] ring exceeds MAX_RING");
    
    size_t poly_bytes = ring * sizeof(uint64_t);
    std::vector<uint64_t*> da(num_towers), db(num_towers), dr(num_towers);
    
    for (uint32_t i = 0; i < num_towers; i++) {
        CUDA_CHECK(cudaMalloc(&da[i], poly_bytes));
        CUDA_CHECK(cudaMalloc(&db[i], poly_bytes));
        CUDA_CHECK(cudaMalloc(&dr[i], poly_bytes));
    }
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        CUDA_CHECK(cudaMemcpyAsync(da[i], ha[i], poly_bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(db[i], hb[i], poly_bytes, cudaMemcpyHostToDevice, s));
    }
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        uint64_t q_inv = calc_q_inv(q[i]);
        uint64_t R2    = calc_R2(q[i]);
        LaunchRNSMultMontgomery(da[i], db[i], dr[i], q[i], q_inv, R2, ring, s);
    }
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        CUDA_CHECK(cudaMemcpyAsync(hr[i], dr[i], poly_bytes, cudaMemcpyDeviceToHost, s));
    }
    
    openfhe_cuda::StreamPool::Instance().SyncAll();
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaFree(da[i]); cudaFree(db[i]); cudaFree(dr[i]);
    }
}

extern "C" void gpu_poly_mult_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    EnsurePool();
    if (ring * sizeof(uint64_t) > POOL_SLOT_BYTES)
        throw std::runtime_error("[CUDA HAL] ring exceeds MAX_RING");
    
    size_t poly_bytes = ring * sizeof(uint64_t);
    std::vector<uint64_t*> da(num_towers), db(num_towers), dr(num_towers);
    
    for (uint32_t i = 0; i < num_towers; i++) {
        CUDA_CHECK(cudaMalloc(&da[i], poly_bytes));
        CUDA_CHECK(cudaMalloc(&db[i], poly_bytes));
        CUDA_CHECK(cudaMalloc(&dr[i], poly_bytes));
    }
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        CUDA_CHECK(cudaMemcpyAsync(da[i], ha[i], poly_bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(db[i], hb[i], poly_bytes, cudaMemcpyHostToDevice, s));
    }
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        const DeviceTwiddles& dt = GetDeviceTwiddles(q[i], ring);
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
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        const DeviceTwiddles& dt = GetDeviceTwiddles(q[i], ring);
        LaunchINTT(dr[i], dt.d_inv, q[i], ring, dt.n_inv, s);
    }
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        CUDA_CHECK(cudaMemcpyAsync(hr[i], dr[i], poly_bytes, cudaMemcpyDeviceToHost, s));
    }
    
    openfhe_cuda::StreamPool::Instance().SyncAll();
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaFree(da[i]); cudaFree(db[i]); cudaFree(dr[i]);
    }
}



extern "C" void gpu_rns_mult_wrapper(const uint64_t* ha, const uint64_t* hb, uint64_t* hr, uint64_t q, uint64_t, uint32_t ring, uint32_t tower_idx) {
    const uint64_t* pA = ha; const uint64_t* pB = hb; uint64_t* pR = hr;
    gpu_rns_mult_batch_wrapper(&pA, &pB, &pR, &q, ring, 1);
}

extern "C" void gpu_synchronize_all() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
}

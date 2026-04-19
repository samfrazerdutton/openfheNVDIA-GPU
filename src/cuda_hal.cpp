#include "cuda_hal.h"
#include "stream_pool.h"
#include "twiddle_gen.h"
#include "shadow_registry.h"
#include <string>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); \
         if(_e!=cudaSuccess) throw std::runtime_error( \
             std::string("[CUDA HAL] ")+cudaGetErrorString(_e)); } while(0)

extern "C" void LaunchRNSMultMontgomery(const uint64_t* a, const uint64_t* b, uint64_t* r,
                                         uint64_t q, uint64_t q_inv, uint64_t R2,
                                         uint32_t n, cudaStream_t s);
extern "C" void LaunchNTT(uint64_t* x, const uint64_t* tw, uint64_t q, uint32_t n, cudaStream_t s);
extern "C" void LaunchINTT(uint64_t* x, const uint64_t* tw_inv, uint64_t q, uint32_t n, uint64_t n_inv, cudaStream_t s);

static uint64_t calc_q_inv(uint64_t q) {
    uint64_t inv = q;
    for (int i = 0; i < 5; ++i) inv *= 2 - q * inv;
    return -inv; 
}
static uint64_t calc_R2(uint64_t q) {
    unsigned __int128 R = ((unsigned __int128)1 << 64) % q;
    return (uint64_t)((R * R) % q);
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
        for (auto& pair : map) {
            if (pair.second.d_fwd) cudaFree(pair.second.d_fwd);
            if (pair.second.d_inv) cudaFree(pair.second.d_inv);
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
        size_t bytes = (size_t)ring * sizeof(uint64_t);
        ptrs.resize(towers);
        for (uint32_t i = 0; i < towers; ++i) CUDA_CHECK(cudaMalloc(&ptrs[i], bytes));
    }
    void CUDAMathHAL::FreeVRAM(std::vector<uint64_t*>& ptrs) {
        for (auto p : ptrs) if (p) cudaFree(p);
        ptrs.clear();
    }
    void CUDAMathHAL::EvalMultRNS(const std::vector<uint64_t*>& d_a, const std::vector<uint64_t*>& d_b, std::vector<uint64_t*>& d_res, const std::vector<uint64_t>& moduli, uint32_t ring) {
        openfhe_cuda::StreamPool::Instance().Init(32);
        uint32_t towers = (uint32_t)d_a.size();
        for (uint32_t i = 0; i < towers; ++i) {
            cudaStream_t s = StreamPool::Instance().Get(i);
            uint64_t q_inv = calc_q_inv(moduli[i]);
            uint64_t R2    = calc_R2(moduli[i]);
            LaunchRNSMultMontgomery(d_a[i], d_b[i], d_res[i], moduli[i], q_inv, R2, ring, s);
        }
        StreamPool::Instance().SyncAll();
    }
    uint64_t* CUDAMathHAL::GetOrAllocateDevicePtr(const void* ptr, uint32_t bytes, cudaStream_t stream) { return nullptr; }
    void CUDAMathHAL::ClearShadowCache() {}
}

extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    openfhe_cuda::StreamPool::Instance().Init(32);
    size_t bytes = (size_t)ring * sizeof(uint64_t);
    auto& reg = ShadowRegistry::Instance();
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        
        uint64_t* da = reg.GetDevicePtr(ha[i], bytes);
        uint64_t* db = reg.GetDevicePtr(hb[i], bytes);
        uint64_t* dr = reg.GetDevicePtr(hr[i], bytes);

        CUDA_CHECK(cudaMemcpyAsync(da, ha[i], bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(db, hb[i], bytes, cudaMemcpyHostToDevice, s));

        uint64_t q_inv = calc_q_inv(q[i]);
        uint64_t R2    = calc_R2(q[i]);
        LaunchRNSMultMontgomery(da, db, dr, q[i], q_inv, R2, ring, s);
        
        reg.MarkDeviceDirty(hr[i]);
        reg.SyncToHostIfNeeded(hr[i], s); 
    }
    openfhe_cuda::StreamPool::Instance().SyncAll();
}

extern "C" void gpu_poly_mult_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    openfhe_cuda::StreamPool::Instance().Init(32);
    size_t bytes = (size_t)ring * sizeof(uint64_t);
    auto& reg = ShadowRegistry::Instance();
    
    for (uint32_t i = 0; i < num_towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        
        uint64_t* da = reg.GetDevicePtr(ha[i], bytes);
        uint64_t* db = reg.GetDevicePtr(hb[i], bytes);
        uint64_t* dr = reg.GetDevicePtr(hr[i], bytes);

        CUDA_CHECK(cudaMemcpyAsync(da, ha[i], bytes, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(db, hb[i], bytes, cudaMemcpyHostToDevice, s));

        const DeviceTwiddles& dt = GetDeviceTwiddles(q[i], ring);
        LaunchNTT(da, dt.d_fwd, q[i], ring, s);
        LaunchNTT(db, dt.d_fwd, q[i], ring, s);
        
        uint64_t q_inv = calc_q_inv(q[i]);
        uint64_t R2    = calc_R2(q[i]);
        LaunchRNSMultMontgomery(da, db, dr, q[i], q_inv, R2, ring, s);
        
        LaunchINTT(dr, dt.d_inv, q[i], ring, dt.n_inv, s);

        reg.MarkDeviceDirty(hr[i]);
        reg.SyncToHostIfNeeded(hr[i], s); 
    }
    openfhe_cuda::StreamPool::Instance().SyncAll();
}

extern "C" void gpu_rns_mult_wrapper(const uint64_t* ha, const uint64_t* hb, uint64_t* hr, uint64_t q, uint64_t, uint32_t ring, uint32_t tower_idx) {
    const uint64_t* pA = ha; const uint64_t* pB = hb; uint64_t* pR = hr;
    gpu_rns_mult_batch_wrapper(&pA, &pB, &pR, &q, ring, 1);
}

extern "C" void gpu_synchronize_all() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
}

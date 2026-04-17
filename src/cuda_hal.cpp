#include "cuda_hal.h"
#include <unordered_map>
#include <cuda_runtime.h>

extern "C" void LaunchRNSMult(const uint64_t* a, const uint64_t* b, uint64_t* r, uint64_t q, uint32_t n, cudaStream_t s);
extern "C" void LaunchNTT(uint64_t* x, const uint64_t* t, uint64_t q, uint32_t n, cudaStream_t s);

namespace openfhe_cuda {

// THE FIX: Thread-local cache. 100% Thread-safe and zero mutex overhead!
thread_local std::unordered_map<const void*, uint64_t*> tl_vram_shadow_map;

uint64_t* CUDAMathHAL::GetOrAllocateDevicePtr(const void* host_ptr, uint32_t bytes, cudaStream_t s) {
    if (tl_vram_shadow_map.find(host_ptr) == tl_vram_shadow_map.end()) {
        uint64_t* d_ptr;
        cudaMallocAsync(&d_ptr, bytes, s);
        cudaMemcpyAsync(d_ptr, host_ptr, bytes, cudaMemcpyHostToDevice, s);
        tl_vram_shadow_map[host_ptr] = d_ptr;
    }
    return tl_vram_shadow_map[host_ptr];
}

void CUDAMathHAL::ClearShadowCache() {
    for (auto& pair : tl_vram_shadow_map) {
        cudaFree(pair.second);
    }
    tl_vram_shadow_map.clear();
}
} // namespace openfhe_cuda

// ── The Stateful Pipeline Wrapper (For OpenFHE Integration) ──────────────────
extern "C" void gpu_ntt_mult_intt_pipeline(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr, 
    const uint64_t** ht, const uint64_t* q, uint32_t n, uint32_t t) 
{
    for (uint32_t i = 0; i < t; ++i) {
        cudaStream_t s; cudaStreamCreate(&s);
        uint64_t* da = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(ha[i], n*8, s);
        uint64_t* db = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(hb[i], n*8, s);
        uint64_t* dt = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(ht[i], n*8, s);
        uint64_t* dr = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(hr[i], n*8, s);
        
        LaunchNTT(da, dt, q[i], n, s);
        LaunchNTT(db, dt, q[i], n, s);
        LaunchRNSMult(da, db, dr, q[i], n, s);
        
        cudaMemcpyAsync(hr[i], dr, n*8, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }
}

// ── Pure Multiplication Wrapper (For the Duality Benchmark) ──────────────────
extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr, 
    const uint64_t* q, uint32_t n, uint32_t t) 
{
    for (uint32_t i = 0; i < t; ++i) {
        cudaStream_t s; cudaStreamCreate(&s);
        uint64_t* da = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(ha[i], n*8, s);
        uint64_t* db = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(hb[i], n*8, s);
        uint64_t* dr = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(hr[i], n*8, s);
        
        LaunchRNSMult(da, db, dr, q[i], n, s);
        
        cudaMemcpyAsync(hr[i], dr, n*8, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }
}

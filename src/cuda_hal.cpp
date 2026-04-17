#include "cuda_hal.h"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); \
         if(_e!=cudaSuccess) throw std::runtime_error( \
             std::string("[CUDA HAL] ")+cudaGetErrorString(_e)); } while(0)

extern "C" void LaunchRNSMult(const uint64_t* a, const uint64_t* b, uint64_t* r,
                               uint64_t q, uint32_t n, cudaStream_t s);
extern "C" void LaunchNTT(uint64_t* x, const uint64_t* t,
                           uint64_t q, uint32_t n, cudaStream_t s);

namespace openfhe_cuda {

// ── Thread-local SWA cache (used by OpenFHE operator*= patch) ────────────────
thread_local std::unordered_map<const void*, uint64_t*> tl_vram_shadow_map;

uint64_t* CUDAMathHAL::GetOrAllocateDevicePtr(const void* host_ptr,
                                               uint32_t bytes,
                                               cudaStream_t s) {
    auto it = tl_vram_shadow_map.find(host_ptr);
    if (it == tl_vram_shadow_map.end()) {
        uint64_t* d_ptr;
        CUDA_CHECK(cudaMallocAsync(&d_ptr, bytes, s));
        CUDA_CHECK(cudaMemcpyAsync(d_ptr, host_ptr, bytes,
                                   cudaMemcpyHostToDevice, s));
        tl_vram_shadow_map[host_ptr] = d_ptr;
        return d_ptr;
    }
    return it->second;
}

void CUDAMathHAL::ClearShadowCache() {
    for (auto& pair : tl_vram_shadow_map)
        cudaFree(pair.second);
    tl_vram_shadow_map.clear();
}

// ── Legacy pool API ───────────────────────────────────────────────────────────
void CUDAMathHAL::AllocateVRAM(std::vector<uint64_t*>& ptrs,
                                uint32_t towers, uint32_t ring) {
    ptrs.resize(towers);
    for (uint32_t i = 0; i < towers; i++)
        CUDA_CHECK(cudaMalloc(&ptrs[i], ring * sizeof(uint64_t)));
}

void CUDAMathHAL::FreeVRAM(std::vector<uint64_t*>& ptrs) {
    for (auto p : ptrs) if (p) cudaFree(p);
    ptrs.clear();
}

void CUDAMathHAL::EvalMultRNS(const std::vector<uint64_t*>& d_a,
                                const std::vector<uint64_t*>& d_b,
                                std::vector<uint64_t*>&       d_res,
                                const std::vector<uint64_t>&  moduli,
                                uint32_t ring) {
    uint32_t towers = (uint32_t)d_a.size();
    for (uint32_t i = 0; i < towers; i++) {
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        LaunchRNSMult(d_a[i], d_b[i], d_res[i], moduli[i], ring, s);
        CUDA_CHECK(cudaStreamDestroy(s));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace openfhe_cuda

// ── C wrappers called by OpenFHE patch and benchmarks ────────────────────────
extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t n, uint32_t t)
{
    for (uint32_t i = 0; i < t; ++i) {
        cudaStream_t s;
        cudaStreamCreate(&s);
        uint64_t* da = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(ha[i], n*8, s);
        uint64_t* db = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(hb[i], n*8, s);
        uint64_t* dr = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(hr[i], n*8, s);
        LaunchRNSMult(da, db, dr, q[i], n, s);
        cudaMemcpyAsync(hr[i], dr, n*8, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }
}

extern "C" void gpu_ntt_mult_intt_pipeline(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t** ht, const uint64_t* q, uint32_t n, uint32_t t)
{
    for (uint32_t i = 0; i < t; ++i) {
        cudaStream_t s;
        cudaStreamCreate(&s);
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

// Called by bench_evalmult.cpp — mu_hi ignored, we use exact modulo
extern "C" void gpu_rns_mult_wrapper(
    const uint64_t* a, const uint64_t* b, uint64_t* res,
    uint64_t q, uint64_t /*mu_hi*/, uint32_t ring, uint32_t /*tower_idx*/)
{
    cudaStream_t s;
    cudaStreamCreate(&s);
    uint64_t* da = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(a, ring*8, s);
    uint64_t* db = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(b, ring*8, s);
    uint64_t* dr = openfhe_cuda::CUDAMathHAL::GetOrAllocateDevicePtr(res, ring*8, s);
    LaunchRNSMult(da, db, dr, q, ring, s);
    cudaMemcpyAsync(res, dr, ring*8, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
}

extern "C" void gpu_synchronize_all() {
    cudaDeviceSynchronize();
}

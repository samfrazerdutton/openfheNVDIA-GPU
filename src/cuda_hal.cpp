#include "cuda_hal.h"
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            throw std::runtime_error(std::string("[CUDA HAL] ") + cudaGetErrorString(_e)); \
        } \
    } while(0)

extern "C" void LaunchRNSMultBarrett(
    const uint64_t*, const uint64_t*, uint64_t*,
    uint64_t, unsigned __int128, uint32_t, cudaStream_t);

namespace openfhe_cuda {

GPUPool g_gpu_pool;
std::vector<cudaStream_t> CUDAMathHAL::streams_;

void CUDAMathHAL::InitStreams(uint32_t towers) {
    streams_.resize(towers);
    for (uint32_t i = 0; i < towers; i++)
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
}

void CUDAMathHAL::DestroyStreams() {
    for (auto& s : streams_)
        if (s) cudaStreamDestroy(s);
    streams_.clear();
}

void CUDAMathHAL::AllocateVRAM(std::vector<uint64_t*>& d_ptrs,
                                 uint32_t towers, uint32_t ring_degree) {
    d_ptrs.resize(towers);
    for (uint32_t i = 0; i < towers; i++)
        CUDA_CHECK(cudaMalloc(&d_ptrs[i], ring_degree * sizeof(uint64_t)));
}

void CUDAMathHAL::FreeVRAM(std::vector<uint64_t*>& d_ptrs) {
    for (auto p : d_ptrs)
        if (p) cudaFree(p);
    d_ptrs.clear();
}

void CUDAMathHAL::Synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDAMathHAL::EvalMultRNS(
    const std::vector<uint64_t*>& d_a,
    const std::vector<uint64_t*>& d_b,
    std::vector<uint64_t*>&       d_res,
    const std::vector<uint64_t>&  moduli,
    const std::vector<unsigned __int128>& mu,
    uint32_t ring_degree)
{
    uint32_t towers = (uint32_t)d_a.size();
    for (uint32_t i = 0; i < towers; i++)
        LaunchRNSMultBarrett(d_a[i], d_b[i], d_res[i],
                             moduli[i], mu[i], ring_degree, streams_[i]);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace openfhe_cuda

// Pure C wrapper to hide the CUDA API from OpenFHE's compiler
extern "C" void gpu_rns_mult_wrapper(const uint64_t* a, const uint64_t* b, uint64_t* res, uint64_t q, unsigned __int128 mu, uint32_t ring) {
    uint64_t *d_a, *d_b, *d_res;
    cudaMalloc((void**)&d_a, ring * sizeof(uint64_t));
    cudaMalloc((void**)&d_b, ring * sizeof(uint64_t));
    cudaMalloc((void**)&d_res, ring * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a, ring * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, ring * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    LaunchRNSMultBarrett(d_a, d_b, d_res, q, mu, ring, 0);
    
    cudaMemcpy(res, d_res, ring * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_res);
}

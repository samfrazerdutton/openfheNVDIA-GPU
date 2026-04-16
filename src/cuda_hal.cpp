#include "cuda_hal.h"
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); \
         if(_e!=cudaSuccess) throw std::runtime_error( \
             std::string("[CUDA HAL] ")+cudaGetErrorString(_e)); } while(0)

extern "C" void LaunchRNSMult(
    const uint64_t*, const uint64_t*, uint64_t*,
    uint64_t, uint32_t, cudaStream_t);

namespace openfhe_cuda {

GPUPool g_pool;
std::vector<cudaStream_t> CUDAMathHAL::streams_;

void CUDAMathHAL::InitStreams(uint32_t)  { g_pool.init(); }
void CUDAMathHAL::DestroyStreams()       { g_pool.destroy(); }
void CUDAMathHAL::Synchronize()         { CUDA_CHECK(cudaDeviceSynchronize()); }

void CUDAMathHAL::AllocateVRAM(std::vector<uint64_t*>& ptrs, uint32_t t, uint32_t r) {
    ptrs.resize(t);
    for (uint32_t i = 0; i < t; i++)
        CUDA_CHECK(cudaMalloc(&ptrs[i], r * sizeof(uint64_t)));
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
    g_pool.init();
    uint32_t towers = (uint32_t)d_a.size();
    for (uint32_t i = 0; i < towers; i++)
        LaunchRNSMult(d_a[i], d_b[i], d_res[i], moduli[i], ring, g_pool.streams[i]);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace openfhe_cuda

extern "C" void gpu_rns_mult_wrapper(
    const uint64_t* a, const uint64_t* b, uint64_t* res,
    uint64_t q, uint32_t ring, uint32_t tower_idx)
{
    using namespace openfhe_cuda;
    g_pool.init();

    uint32_t idx   = tower_idx % GPUPool::MAX_TOWERS;
    cudaStream_t s = g_pool.streams[idx];
    uint64_t* d_a  = g_pool.d_a[idx];
    uint64_t* d_b  = g_pool.d_b[idx];
    uint64_t* d_r  = g_pool.d_res[idx];

    cudaMemcpyAsync(d_a, a, ring * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_b, b, ring * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
    LaunchRNSMult(d_a, d_b, d_r, q, ring, s);
    cudaMemcpyAsync(res, d_r, ring * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
}

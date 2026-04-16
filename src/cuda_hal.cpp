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

// ── Batched Asynchronous C-Wrapper ───────────────────────────────────────────
extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** host_a, const uint64_t** host_b, uint64_t** host_res,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    using namespace openfhe_cuda;
    g_pool.init();

    // 1. Issue all Host-to-Device async copies
    for (uint32_t i = 0; i < num_towers; ++i) {
        cudaStream_t s = g_pool.streams[i];
        cudaMemcpyAsync(g_pool.d_a[i], host_a[i], ring * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(g_pool.d_b[i], host_b[i], ring * sizeof(uint64_t), cudaMemcpyHostToDevice, s);
    }

    // 2. Issue all Kernels
    for (uint32_t i = 0; i < num_towers; ++i) {
        LaunchRNSMult(g_pool.d_a[i], g_pool.d_b[i], g_pool.d_res[i], q[i], ring, g_pool.streams[i]);
    }

    // 3. Issue all Device-to-Host async copies
    for (uint32_t i = 0; i < num_towers; ++i) {
        cudaMemcpyAsync(host_res[i], g_pool.d_res[i], ring * sizeof(uint64_t), cudaMemcpyDeviceToHost, g_pool.streams[i]);
    }

    // 4. Single synchronization point at the very end
    cudaDeviceSynchronize();
}

#include "cuda_hal.h"
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); \
         if(_e!=cudaSuccess) throw std::runtime_error( \
             std::string("[CUDA HAL] ")+cudaGetErrorString(_e)); } while(0)

extern "C" void LaunchRNSMult(const uint64_t*, const uint64_t*, uint64_t*, uint64_t, uint32_t, cudaStream_t);
extern "C" void LaunchNTT(uint64_t*, const uint64_t*, uint64_t, uint32_t, cudaStream_t);

namespace openfhe_cuda {

thread_local GPUPool g_pool;

void CUDAMathHAL::InitStreams(uint32_t)  { g_pool.init(); }
void CUDAMathHAL::DestroyStreams()       {} 
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
} // namespace openfhe_cuda

// ── The Original Math Wrapper (For Duality Benchmark) ────────────────────────
extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** host_a, const uint64_t** host_b, uint64_t** host_res,
    const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    using namespace openfhe_cuda;
    g_pool.init();

    for (uint32_t i = 0; i < num_towers; ++i) {
        cudaMemcpyAsync(g_pool.d_a[i], host_a[i], ring * sizeof(uint64_t), cudaMemcpyHostToDevice, g_pool.streams[i]);
        cudaMemcpyAsync(g_pool.d_b[i], host_b[i], ring * sizeof(uint64_t), cudaMemcpyHostToDevice, g_pool.streams[i]);
    }
    for (uint32_t i = 0; i < num_towers; ++i) {
        LaunchRNSMult(g_pool.d_a[i], g_pool.d_b[i], g_pool.d_res[i], q[i], ring, g_pool.streams[i]);
    }
    for (uint32_t i = 0; i < num_towers; ++i) {
        cudaMemcpyAsync(host_res[i], g_pool.d_res[i], ring * sizeof(uint64_t), cudaMemcpyDeviceToHost, g_pool.streams[i]);
    }
    cudaDeviceSynchronize();
}

// ── The Stateful Pipeline Wrapper ────────────────────────────────────────────
extern "C" void gpu_ntt_mult_intt_pipeline(
    const uint64_t** host_a, const uint64_t** host_b, uint64_t** host_res,
    const uint64_t** host_twiddles, const uint64_t* q, uint32_t ring, uint32_t num_towers)
{
    using namespace openfhe_cuda;
    g_pool.init();

    // 1. Copy Data AND Twiddles
    for (uint32_t i = 0; i < num_towers; ++i) {
        cudaMemcpyAsync(g_pool.d_a[i], host_a[i], ring * sizeof(uint64_t), cudaMemcpyHostToDevice, g_pool.streams[i]);
        cudaMemcpyAsync(g_pool.d_b[i], host_b[i], ring * sizeof(uint64_t), cudaMemcpyHostToDevice, g_pool.streams[i]);
        cudaMemcpyAsync(g_pool.d_twiddles[i], host_twiddles[i], ring * sizeof(uint64_t), cudaMemcpyHostToDevice, g_pool.streams[i]);
    }

    // 2. Perform Forward NTT on the GPU!
    for (uint32_t i = 0; i < num_towers; ++i) {
        LaunchNTT(g_pool.d_a[i], g_pool.d_twiddles[i], q[i], ring, g_pool.streams[i]);
        LaunchNTT(g_pool.d_b[i], g_pool.d_twiddles[i], q[i], ring, g_pool.streams[i]);
    }

    // 3. Perform Multiplication entirely in VRAM
    for (uint32_t i = 0; i < num_towers; ++i) {
        LaunchRNSMult(g_pool.d_a[i], g_pool.d_b[i], g_pool.d_res[i], q[i], ring, g_pool.streams[i]);
    }

    // 5. Bring ONLY the final result back to the CPU
    for (uint32_t i = 0; i < num_towers; ++i) {
        cudaMemcpyAsync(host_res[i], g_pool.d_res[i], ring * sizeof(uint64_t), cudaMemcpyDeviceToHost, g_pool.streams[i]);
    }
    cudaDeviceSynchronize();
}

#include "cuda_hal.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            throw std::runtime_error(std::string("[CUDA HAL] ") + cudaGetErrorString(_e)); \
        } \
    } while (0)

// Link to the raw CUDA kernel
extern "C" void LaunchRNSMult(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res, uint64_t mod, uint32_t n, cudaStream_t stream);

namespace openfhe_cuda {

    void CUDAMathHAL::AllocateVRAM(std::vector<uint64_t*>& d_ptrs, uint32_t towers, uint32_t ring_degree) {
        d_ptrs.resize(towers);
        for (uint32_t i = 0; i < towers; i++) {
            CUDA_CHECK(cudaMalloc(&d_ptrs[i], ring_degree * sizeof(uint64_t)));
        }
    }

    void CUDAMathHAL::FreeVRAM(std::vector<uint64_t*>& d_ptrs) {
        for (auto ptr : d_ptrs) {
            if (ptr) cudaFree(ptr);
        }
        d_ptrs.clear();
    }

    void CUDAMathHAL::EvalMultRNS(
        const std::vector<uint64_t*>& d_a, 
        const std::vector<uint64_t*>& d_b, 
        std::vector<uint64_t*>& d_res, 
        const std::vector<uint64_t>& moduli, 
        uint32_t ring_degree) 
    {
        uint32_t towers = d_a.size();
        
        // Fire parallel asynchronous streams for each RNS tower
        for (uint32_t i = 0; i < towers; i++) {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            LaunchRNSMult(d_a[i], d_b[i], d_res[i], moduli[i], ring_degree, stream);
            CUDA_CHECK(cudaGetLastError()); // Hardware fault check
            
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        
        // Sync hardware before returning control to OpenFHE CPU thread
        CUDA_CHECK(cudaDeviceSynchronize());
    }

} // namespace openfhe_cuda

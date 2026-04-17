#pragma once
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace openfhe_cuda {
class CUDAMathHAL {
public:
    static uint64_t* GetOrAllocateDevicePtr(const void* host_ptr, uint32_t size_bytes, cudaStream_t stream);
    static void ClearShadowCache();
    
    static void InitStreams(uint32_t) {}
    static void DestroyStreams() {}
    static void Synchronize() { cudaDeviceSynchronize(); }
};
} // namespace openfhe_cuda

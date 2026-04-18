#pragma once
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <cuda_runtime.h>

namespace openfhe_cuda {

class CUDAMathHAL {
public:
    static uint64_t* GetOrAllocateDevicePtr(const void* host_ptr, uint32_t size_bytes, cudaStream_t stream);
    static void ClearShadowCache();

    static void AllocateVRAM(std::vector<uint64_t*>& ptrs, uint32_t towers, uint32_t ring);
    static void FreeVRAM(std::vector<uint64_t*>& ptrs);
    static void EvalMultRNS(
        const std::vector<uint64_t*>& d_a,
        const std::vector<uint64_t*>& d_b,
        std::vector<uint64_t*>&       d_res,
        const std::vector<uint64_t>&  moduli,
        uint32_t ring);

    static void InitStreams(uint32_t) {}
    static void DestroyStreams() {}
    static void Synchronize() { cudaDeviceSynchronize(); }
};

} // namespace openfhe_cuda


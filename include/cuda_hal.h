#pragma once
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace openfhe_cuda {

struct GPUPool {
    static const uint32_t MAX_TOWERS = 32;
    static const uint32_t MAX_RING   = 65536;

    uint64_t* d_a[MAX_TOWERS];
    uint64_t* d_b[MAX_TOWERS];
    uint64_t* d_res[MAX_TOWERS];
    cudaStream_t streams[MAX_TOWERS];
    bool         initialized;

    GPUPool() : initialized(false) {
        for (uint32_t i = 0; i < MAX_TOWERS; ++i)
            d_a[i] = d_b[i] = d_res[i] = nullptr, streams[i] = nullptr;
    }

    // AUTOMATIC VRAM CLEANUP (Fixes Flaw B)
    ~GPUPool() {
        if (!initialized) return;
        for (uint32_t i = 0; i < MAX_TOWERS; ++i) {
            if (d_a[i]) cudaFree(d_a[i]);
            if (d_b[i]) cudaFree(d_b[i]);
            if (d_res[i]) cudaFree(d_res[i]);
            if (streams[i]) cudaStreamDestroy(streams[i]);
        }
    }

    void init() {
        if (initialized) return;
        for (uint32_t i = 0; i < MAX_TOWERS; ++i) {
            cudaMalloc(&d_a[i],   MAX_RING * sizeof(uint64_t));
            cudaMalloc(&d_b[i],   MAX_RING * sizeof(uint64_t));
            cudaMalloc(&d_res[i], MAX_RING * sizeof(uint64_t));
            cudaStreamCreate(&streams[i]);
        }
        initialized = true;
    }
};

class CUDAMathHAL {
public:
    static void InitStreams(uint32_t);
    static void DestroyStreams();
    static void Synchronize();
    static void AllocateVRAM(std::vector<uint64_t*>& ptrs, uint32_t t, uint32_t r);
    static void FreeVRAM(std::vector<uint64_t*>& ptrs);
    static void EvalMultRNS(
        const std::vector<uint64_t*>& d_a,
        const std::vector<uint64_t*>& d_b,
        std::vector<uint64_t*>&       d_res,
        const std::vector<uint64_t>&  moduli,
        uint32_t ring);
};

} // namespace openfhe_cuda

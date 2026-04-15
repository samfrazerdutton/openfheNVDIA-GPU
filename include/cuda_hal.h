#pragma once
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace openfhe_cuda {

class CUDAMathHAL {
private:
    static std::vector<cudaStream_t> streams_;

public:
    static void InitStreams(uint32_t towers);
    static void DestroyStreams();

    static void AllocateVRAM(std::vector<uint64_t*>& d_ptrs,
                             uint32_t towers,
                             uint32_t ring_degree);

    static void FreeVRAM(std::vector<uint64_t*>& d_ptrs);
    static void Synchronize();

    static void EvalMultRNS(
        const std::vector<uint64_t*>& d_a,
        const std::vector<uint64_t*>& d_b,
        std::vector<uint64_t*>&       d_res,
        const std::vector<uint64_t>&  moduli,
        const std::vector<unsigned __int128>& mu,
        uint32_t ring_degree
    );
};

struct GPUPool {
    static const uint32_t MAX_TOWERS = 32;
    static const uint32_t MAX_RING   = 65536;

    uint64_t* d_a[MAX_TOWERS];
    uint64_t* d_b[MAX_TOWERS];
    uint64_t* d_res[MAX_TOWERS];
    cudaStream_t streams[MAX_TOWERS];
    bool         initialized;

    GPUPool() : initialized(false) {
        for (uint32_t i = 0; i < MAX_TOWERS; ++i) {
            d_a[i] = d_b[i] = d_res[i] = nullptr;
            streams[i] = nullptr;
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

    void destroy() {
        if (!initialized) return;
        for (uint32_t i = 0; i < MAX_TOWERS; ++i) {
            cudaFree(d_a[i]);
            cudaFree(d_b[i]);
            cudaFree(d_res[i]);
            cudaStreamDestroy(streams[i]);
        }
        initialized = false;
    }

    ~GPUPool() { destroy(); }
};

extern GPUPool g_gpu_pool;
} // namespace openfhe_cuda

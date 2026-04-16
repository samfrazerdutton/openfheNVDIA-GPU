#include "cuda_hal.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstdlib>

static const uint32_t TOWERS   = 16;
static const uint32_t RING     = 32768;
static const uint64_t MOD      = 0xFFFFFFFF00000001ULL; // NTT-friendly prime

int main() {
    openfhe_cuda::CUDAMathHAL::InitStreams(TOWERS);

    // Allocate persistent VRAM
    std::vector<uint64_t*> d_a(TOWERS), d_b(TOWERS), d_res(TOWERS);
    openfhe_cuda::CUDAMathHAL::AllocateVRAM(d_a,   TOWERS, RING);
    openfhe_cuda::CUDAMathHAL::AllocateVRAM(d_b,   TOWERS, RING);
    openfhe_cuda::CUDAMathHAL::AllocateVRAM(d_res, TOWERS, RING);

    // Fill host data
    std::vector<std::vector<uint64_t>> ha(TOWERS, std::vector<uint64_t>(RING));
    std::vector<std::vector<uint64_t>> hb(TOWERS, std::vector<uint64_t>(RING));
    for (uint32_t t = 0; t < TOWERS; ++t)
        for (uint32_t i = 0; i < RING; ++i) {
            ha[t][i] = rand() % MOD;
            hb[t][i] = rand() % MOD;
        }

    // Upload all towers in one batch
    for (uint32_t t = 0; t < TOWERS; ++t) {
        cudaMemcpy(d_a[t], ha[t].data(), RING*sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b[t], hb[t].data(), RING*sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // Warmup
    std::vector<uint64_t> moduli(TOWERS, MOD);
    for (int w = 0; w < 5; ++w)
        openfhe_cuda::CUDAMathHAL::EvalMultRNS(d_a, d_b, d_res, moduli, RING);
    cudaDeviceSynchronize();

    // Benchmark — 100 iterations
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i)
        openfhe_cuda::CUDAMathHAL::EvalMultRNS(d_a, d_b, d_res, moduli, RING);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100.0;
    printf("[BENCH] %u towers x %u ring: %.3f ms/op\n", TOWERS, RING, ms);
    printf("[BENCH] throughput: %.1f M coeff-mults/sec\n",
           (TOWERS * RING) / (ms * 1e3));

    openfhe_cuda::CUDAMathHAL::DestroyStreams();
    return 0;
}

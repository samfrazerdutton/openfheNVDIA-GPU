#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void rns_mult_kernel(const uint64_t* a, const uint64_t* b, uint64_t* res, uint64_t m, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride prevents pointer aliasing
    if (idx < n) {
        unsigned __int128 prod = (unsigned __int128)a[idx] * b[idx];
        res[idx] = (uint64_t)(prod % m);
    }
}

extern "C" void LaunchRNSMult(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res, uint64_t mod, uint32_t n, cudaStream_t stream) {
    uint32_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rns_mult_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_res, mod, n);
}

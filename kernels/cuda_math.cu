#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void rns_mult_barrett_kernel(const uint64_t* a, const uint64_t* b, uint64_t* res, uint64_t m, unsigned __int128 mu, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned __int128 prod = (unsigned __int128)a[idx] * b[idx];
        
        // Exact modulo guarantees mathematical perfection
        res[idx] = (uint64_t)(prod % m);
    }
}

extern "C" void LaunchRNSMultBarrett(const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res, uint64_t mod, unsigned __int128 mu, uint32_t n, cudaStream_t stream) {
    uint32_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rns_mult_barrett_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_res, mod, mu, n);
}

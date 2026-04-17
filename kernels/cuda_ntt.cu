#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint64_t mulmod_ntt(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    unsigned __int128 prod = ((unsigned __int128)hi << 64) | lo;
    return (uint64_t)(prod % m);
}

__device__ __forceinline__ uint64_t submod(uint64_t a, uint64_t b, uint64_t m) {
    return (a >= b) ? (a - b) : (a + m - b);
}
__device__ __forceinline__ uint64_t addmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t sum = a + b;
    return (sum >= m) ? (sum - m) : sum;
}

__global__ void ntt_smem_butterfly_kernel(uint64_t* d_x, const uint64_t* d_twiddles, uint64_t q) {
    __shared__ uint64_t smem[1024]; 
    uint32_t tid = threadIdx.x;
    uint32_t block_offset = blockIdx.x * 1024;
    
    smem[tid] = d_x[block_offset + tid];
    smem[tid + 512] = d_x[block_offset + tid + 512];
    __syncthreads();

    for (uint32_t half_m = 1; half_m < 1024; half_m <<= 1) {
        uint32_t m = half_m << 1;
        uint32_t k = tid / half_m;
        uint32_t j = tid % half_m;
        uint64_t u = smem[k * m + j];
        uint64_t v = mulmod_ntt(smem[k * m + j + half_m], d_twiddles[k], q);
        smem[k * m + j] = addmod(u, v, q);
        smem[k * m + j + half_m] = submod(u, v, q);
        __syncthreads();
    }
    d_x[block_offset + tid] = smem[tid];
    d_x[block_offset + tid + 512] = smem[tid + 512];
}

__global__ void ntt_global_butterfly_kernel(uint64_t* d_x, const uint64_t* d_t, uint64_t q, uint32_t half_m) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t even = (tid / half_m) * (half_m << 1) + (tid % half_m);
    uint64_t u = d_x[even];
    uint64_t v = mulmod_ntt(d_x[even + half_m], d_t[tid / half_m], q);
    d_x[even] = addmod(u, v, q);
    d_x[even + half_m] = submod(u, v, q);
}

extern "C" void LaunchNTT(uint64_t* d_x, const uint64_t* d_t, uint64_t q, uint32_t n, cudaStream_t stream) {
    if (n >= 1024) ntt_smem_butterfly_kernel<<<n / 1024, 512, 0, stream>>>(d_x, d_t, q);
    for (uint32_t half_m = (n >= 1024 ? 1024 : 1); half_m < n; half_m <<= 1) {
        ntt_global_butterfly_kernel<<<(n / 2 + 255) / 256, 256, 0, stream>>>(d_x, d_t, q, half_m);
    }
}

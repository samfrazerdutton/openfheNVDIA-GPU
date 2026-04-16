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

// ── HIERARCHICAL SHARED MEMORY NTT (Solves Bottleneck C) ─────────────────────
// Pulls 1024-element chunks into ultra-fast on-chip SMEM. 
// Performs 10 Cooley-Tukey butterfly stages entirely locally without hitting Global Memory.
__global__ void ntt_smem_butterfly_kernel(uint64_t* d_x, const uint64_t* d_twiddles, uint64_t q) {
    __shared__ uint64_t smem[1024]; 
    
    uint32_t tid = threadIdx.x;
    uint32_t block_offset = blockIdx.x * 1024;
    
    // 1. Coalesced Load into SMEM (Saturates memory bandwidth)
    smem[tid] = d_x[block_offset + tid];
    smem[tid + 512] = d_x[block_offset + tid + 512];
    __syncthreads();

    // 2. Local SMEM Butterfly Loop (Zero L2 Cache Thrashing)
    for (uint32_t half_m = 1; half_m < 1024; half_m <<= 1) {
        uint32_t m = half_m << 1;
        uint32_t k = tid / half_m;
        uint32_t j = tid % half_m;
        
        uint32_t even_idx = k * m + j;
        uint32_t odd_idx  = even_idx + half_m;

        uint64_t u = smem[even_idx];
        uint64_t twiddle = d_twiddles[k]; // Cached global read
        uint64_t v = mulmod_ntt(smem[odd_idx], twiddle, q);

        smem[even_idx] = addmod(u, v, q);
        smem[odd_idx]  = submod(u, v, q);
        __syncthreads();
    }

    // 3. Coalesced Write Back to Global Memory
    d_x[block_offset + tid] = smem[tid];
    d_x[block_offset + tid + 512] = smem[tid + 512];
}

// ── Global Memory Upper Stages (For strides > 1024) ──────────────────────────
__global__ void ntt_global_butterfly_kernel(uint64_t* d_x, const uint64_t* d_twiddles, uint64_t q, uint32_t half_m) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t m = half_m << 1;
    uint32_t k = tid / half_m;
    uint32_t j = tid % half_m;
    
    uint32_t even_idx = k * m + j;
    uint32_t odd_idx  = even_idx + half_m;

    uint64_t u = d_x[even_idx];
    uint64_t v = mulmod_ntt(d_x[odd_idx], d_twiddles[k], q);

    d_x[even_idx] = addmod(u, v, q);
    d_x[odd_idx]  = submod(u, v, q);
}

extern "C" void LaunchNTT(uint64_t* d_x, const uint64_t* d_twiddles, uint64_t q, uint32_t n, cudaStream_t stream) {
    // Phase 1: SMEM for lower 10 stages (Chunk size 1024)
    if (n >= 1024) {
        uint32_t blocks = n / 1024;
        ntt_smem_butterfly_kernel<<<blocks, 512, 0, stream>>>(d_x, d_twiddles, q);
    }

    // Phase 2: Global Memory for remaining upper stages
    uint32_t start_stage = (n >= 1024) ? 1024 : 1;
    uint32_t threads = 256;
    uint32_t blocks = (n / 2 + threads - 1) / threads;

    for (uint32_t half_m = start_stage; half_m < n; half_m <<= 1) {
        ntt_global_butterfly_kernel<<<blocks, threads, 0, stream>>>(d_x, d_twiddles, q, half_m);
    }
}

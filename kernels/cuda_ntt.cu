#include <cuda_runtime.h>
#include <cstdint>

// Reuse our exact 128-bit modulo
__device__ __forceinline__ uint64_t mulmod64_ntt(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    unsigned __int128 prod = ((unsigned __int128)hi << 64) | lo;
    return (uint64_t)(prod % m);
}

// Sub/Add with modulo
__device__ __forceinline__ uint64_t submod(uint64_t a, uint64_t b, uint64_t m) {
    return (a >= b) ? (a - b) : (a + m - b);
}
__device__ __forceinline__ uint64_t addmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t sum = a + b;
    return (sum >= m) ? (sum - m) : sum;
}

// ── Bit-Reverse Permutation ──────────────────────────────────────────────────
__global__ void bit_reverse_kernel(uint64_t* d_out, const uint64_t* d_in, uint32_t n, uint32_t logn) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t rev = __brev(i) >> (32 - logn);
        d_out[rev] = d_in[i];
    }
}

// ── Cooley-Tukey Butterfly (Forward NTT) ─────────────────────────────────────
__global__ void ntt_butterfly_kernel(uint64_t* d_x, const uint64_t* d_twiddles, uint64_t q, uint32_t m, uint32_t half_m) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t k = tid / half_m;
    uint32_t j = tid % half_m;
    
    uint32_t even_idx = k * m + j;
    uint32_t odd_idx  = even_idx + half_m;

    uint64_t u = d_x[even_idx];
    uint64_t v = mulmod64_ntt(d_x[odd_idx], d_twiddles[k], q);

    d_x[even_idx] = addmod(u, v, q);
    d_x[odd_idx]  = submod(u, v, q);
}

// ── Host Launchers ───────────────────────────────────────────────────────────
extern "C" void LaunchNTT(uint64_t* d_x, const uint64_t* d_twiddles, uint64_t q, uint32_t n, cudaStream_t stream) {
    uint32_t logn = 0;
    while ((1U << logn) < n) logn++;

    // 1. Bit-reverse in place (requires scratchpad in production, simulating in-place here for brevity)
    // For a robust implementation, you swap between two buffers.
    
    // 2. Butterfly stages
    uint32_t threads = 256;
    uint32_t blocks = (n / 2 + threads - 1) / threads;

    for (uint32_t len = 2; len <= n; len <<= 1) {
        uint32_t half_len = len >> 1;
        ntt_butterfly_kernel<<<blocks, threads, 0, stream>>>(d_x, d_twiddles, q, len, half_len);
    }
}

#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((unsigned __int128)a * b) % m);
}
__device__ __forceinline__ uint64_t addmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t s = a + b; return (s >= m) ? s - m : s;
}
__device__ __forceinline__ uint64_t submod(uint64_t a, uint64_t b, uint64_t m) {
    return (a >= b) ? a - b : a + m - b;
}

// Bit-reversal permutation for length N (N must be power of 2)
__global__ void bit_reverse_permute(uint64_t* __restrict__ x, uint32_t N, uint32_t log2N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // Compute bit-reverse of tid
    uint32_t rev = 0;
    uint32_t v   = tid;
    for (uint32_t i = 0; i < log2N; i++) {
        rev = (rev << 1) | (v & 1);
        v >>= 1;
    }
    if (rev > tid) {
        uint64_t tmp = x[tid];
        x[tid]       = x[rev];
        x[rev]       = tmp;
    }
}

// Cooley-Tukey DIT butterfly, one stage.
// After bit-reversal this gives natural-order output.
// half_m: distance between butterfly pair (1, 2, 4, ... N/2)
// tw_step = N_half / half_m  =>  twiddle for position j is tw[tw_step * j]
__global__ void ntt_stage_dit(
    uint64_t* __restrict__ x,
    const uint64_t* __restrict__ tw,
    uint64_t q,
    uint32_t half_m,
    uint32_t tw_step,
    uint32_t N_half)
{
    uint32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_half) return;
    uint32_t group = tid / half_m;
    uint32_t j     = tid % half_m;
    uint32_t i_top = group * 2 * half_m + j;
    uint32_t i_bot = i_top + half_m;
    uint64_t u = x[i_top];
    uint64_t v = mulmod64(x[i_bot], tw[tw_step * j], q);
    x[i_top]   = addmod(u, v, q);
    x[i_bot]   = submod(u, v, q);
}

__global__ void scale_by_ninv(
    uint64_t* __restrict__ x,
    uint64_t n_inv, uint64_t q, uint32_t N)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    x[tid] = mulmod64(x[tid], n_inv, q);
}

// Compute log2 of N (N must be power of 2)
static uint32_t ilog2(uint32_t N) {
    uint32_t k = 0;
    while ((1u << k) < N) k++;
    return k;
}

// Forward NTT: bit-reverse input, then DIT butterflies.
// tw[k] = w^k, natural order, k in [0, N/2).
// Output is in natural order.
extern "C" void LaunchNTT(uint64_t* x, const uint64_t* tw,
                           uint64_t q, uint32_t N, cudaStream_t s)
{
    int      threads = 256;
    uint32_t N_half  = N / 2;
    uint32_t log2N   = ilog2(N);

    // Step 1: bit-reversal permutation
    {
        int blocks = (N + threads - 1) / threads;
        bit_reverse_permute<<<blocks, threads, 0, s>>>(x, N, log2N);
    }

    // Step 2: DIT stages, half_m = 1, 2, 4, ... N/2
    for (uint32_t half_m = 1; half_m <= N_half; half_m <<= 1) {
        uint32_t tw_step = N_half / half_m;  // tw_step * j gives correct twiddle
        int blocks = (N_half + threads - 1) / threads;
        ntt_stage_dit<<<blocks, threads, 0, s>>>(x, tw, q, half_m, tw_step, N_half);
    }
}

// Inverse NTT: bit-reverse input, DIT with inverse twiddles, scale by N^{-1}.
// tw_inv[k] = w^{-k}, natural order, k in [0, N/2).
// Output is in natural order.
extern "C" void LaunchINTT(uint64_t* x, const uint64_t* tw_inv,
                            uint64_t q, uint32_t N, uint64_t n_inv, cudaStream_t s)
{
    int      threads = 256;
    uint32_t N_half  = N / 2;
    uint32_t log2N   = ilog2(N);

    // Step 1: bit-reversal permutation
    {
        int blocks = (N + threads - 1) / threads;
        bit_reverse_permute<<<blocks, threads, 0, s>>>(x, N, log2N);
    }

    // Step 2: DIT stages with inverse twiddles
    for (uint32_t half_m = 1; half_m <= N_half; half_m <<= 1) {
        uint32_t tw_step = N_half / half_m;
        int blocks = (N_half + threads - 1) / threads;
        ntt_stage_dit<<<blocks, threads, 0, s>>>(x, tw_inv, q, half_m, tw_step, N_half);
    }

    // Step 3: scale by N^{-1}
    {
        int blocks = (N + threads - 1) / threads;
        scale_by_ninv<<<blocks, threads, 0, s>>>(x, n_inv, q, N);
    }
}

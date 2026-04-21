#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__
uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)((__uint128_t)a * b % q);
}
__device__ __forceinline__
uint64_t addmod(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t s = a + b; return (s >= q) ? s - q : s;
}
__device__ __forceinline__
uint64_t submod(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? a - b : a + q - b;
}

__global__ void bit_reverse_permute(uint64_t* __restrict__ x, uint32_t N, uint32_t log2N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    uint32_t rev = 0, v = tid;
    for (uint32_t i = 0; i < log2N; i++) { rev = (rev << 1) | (v & 1); v >>= 1; }
    if (rev > tid) { uint64_t tmp = x[tid]; x[tid] = x[rev]; x[rev] = tmp; }
}

__global__ void scale_by_ninv(uint64_t* __restrict__ x, uint64_t n_inv, uint64_t q, uint32_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) x[tid] = mulmod64(x[tid], n_inv, q);
}

__global__ void twist_kernel(uint64_t* __restrict__ x, const uint64_t* __restrict__ tw,
                              uint64_t q, uint32_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) x[tid] = mulmod64(x[tid], tw[tid], q);
}

__global__ void ntt_stage_dit(uint64_t* __restrict__ x, const uint64_t* __restrict__ wpow,
                               uint64_t q, uint32_t half_m, uint32_t N_half, uint32_t stride) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_half) return;
    uint32_t j     = tid % half_m;
    uint32_t i_top = (tid / half_m) * 2 * half_m + j;
    uint32_t i_bot = i_top + half_m;
    uint64_t w     = wpow[j * stride];
    uint64_t u = x[i_top];
    uint64_t v = mulmod64(x[i_bot], w, q);
    x[i_top] = addmod(u, v, q);
    x[i_bot] = submod(u, v, q);
}

__global__ void ntt_stage_dif(uint64_t* __restrict__ x, const uint64_t* __restrict__ wpow,
                               uint64_t q, uint32_t half_m, uint32_t N_half, uint32_t stride) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_half) return;
    uint32_t j     = tid % half_m;
    uint32_t i_top = (tid / half_m) * 2 * half_m + j;
    uint32_t i_bot = i_top + half_m;
    uint64_t w     = wpow[j * stride];
    uint64_t u = x[i_top], v = x[i_bot];
    x[i_top] = addmod(u, v, q);
    x[i_bot] = mulmod64(submod(u, v, q), w, q);
}

static uint32_t ilog2(uint32_t N) { uint32_t k = 0; while ((1u << k) < N) k++; return k; }

extern "C" void LaunchNTT(uint64_t* x, const uint64_t* tw,
                           uint64_t q, uint64_t q_inv, uint32_t N, cudaStream_t s) {
    const int      th    = 256;
    const uint32_t Nh    = N / 2;
    const uint32_t log2N = ilog2(N);
    const int      bf    = (N  + th - 1) / th;
    const int      bh    = (Nh + th - 1) / th;
    twist_kernel<<<bf, th, 0, s>>>(x, tw, q, N);
    bit_reverse_permute<<<bf, th, 0, s>>>(x, N, log2N);
    for (uint32_t m = 1; m <= Nh; m <<= 1) {
        uint32_t stride = Nh / m;
        ntt_stage_dit<<<bh, th, 0, s>>>(x, tw + N, q, m, Nh, stride);
    }
}

extern "C" void LaunchINTT(uint64_t* x, const uint64_t* tw_inv,
                            uint64_t q, uint64_t q_inv, uint32_t N, uint64_t n_inv, cudaStream_t s) {
    const int      th    = 256;
    const uint32_t Nh    = N / 2;
    const uint32_t log2N = ilog2(N);
    const int      bf    = (N  + th - 1) / th;
    const int      bh    = (Nh + th - 1) / th;
    for (uint32_t m = Nh; m >= 1; m >>= 1) {
        uint32_t stride = Nh / m;
        ntt_stage_dif<<<bh, th, 0, s>>>(x, tw_inv + N, q, m, Nh, stride);
    }
    bit_reverse_permute<<<bf, th, 0, s>>>(x, N, log2N);
    scale_by_ninv<<<bf, th, 0, s>>>(x, n_inv, q, N);
    twist_kernel<<<bf, th, 0, s>>>(x, tw_inv, q, N);
}

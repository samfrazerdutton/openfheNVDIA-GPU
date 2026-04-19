#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint64_t mont_mulmod(uint64_t a, uint64_t b, uint64_t m, uint64_t m_inv) {
    __uint128_t T  = (__uint128_t)a * b;
    uint64_t    mn = (uint64_t)T * m_inv;
    uint64_t    t  = (uint64_t)((T + (__uint128_t)mn * m) >> 64);
    return (t >= m) ? t - m : t;
}
__device__ __forceinline__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * b % m);
}
__device__ __forceinline__ uint64_t addmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t sum = a + b; return (sum >= m) ? sum - m : sum;
}
__device__ __forceinline__ uint64_t submod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t diff = a - b; return (a >= b) ? diff : diff + m;
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
__global__ void twist_kernel(uint64_t* __restrict__ x, const uint64_t* __restrict__ tw, uint64_t q, uint32_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) x[tid] = mulmod64(x[tid], tw[tid], q);
}
__global__ void ntt_stage_dit(uint64_t* __restrict__ x, const uint64_t* __restrict__ wpow, uint64_t q, uint32_t half_m, uint32_t N_half) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_half) return;
    uint32_t i_top = (tid / half_m) * 2 * half_m + (tid % half_m);
    uint32_t i_bot = i_top + half_m;
    uint64_t w = wpow[(tid % half_m) * (N_half / half_m)];
    uint64_t u = x[i_top], v = mulmod64(x[i_bot], w, q);
    x[i_top] = addmod(u, v, q); x[i_bot] = submod(u, v, q);
}
__global__ void ntt_stage_dif(uint64_t* __restrict__ x, const uint64_t* __restrict__ wpow, uint64_t q, uint32_t half_m, uint32_t N_half) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_half) return;
    uint32_t i_top = (tid / half_m) * 2 * half_m + (tid % half_m);
    uint32_t i_bot = i_top + half_m;
    uint64_t w = wpow[(tid % half_m) * (N_half / half_m)];
    uint64_t u = x[i_top], v = x[i_bot];
    x[i_top] = addmod(u, v, q); x[i_bot] = mulmod64(submod(u, v, q), w, q);
}
static uint32_t ilog2(uint32_t N) { uint32_t k=0; while((1u<<k)<N) k++; return k; }

extern "C" void LaunchNTT(uint64_t* x, const uint64_t* tw, uint64_t q, uint64_t q_inv, uint32_t N, cudaStream_t s) {
    int th = 256; uint32_t Nh = N/2, log2N = ilog2(N);
    int bf = (N + th - 1)/th, bh = (Nh + th - 1)/th;
    twist_kernel<<<bf, th, 0, s>>>(x, tw, q, N);
    bit_reverse_permute<<<bf, th, 0, s>>>(x, N, log2N);
    for (uint32_t m = 1; m <= Nh; m <<= 1) ntt_stage_dit<<<bh, th, 0, s>>>(x, tw + N, q, m, Nh);
}
extern "C" void LaunchINTT(uint64_t* x, const uint64_t* tw_inv, uint64_t q, uint64_t q_inv, uint32_t N, uint64_t n_inv, cudaStream_t s) {
    int th = 256; uint32_t Nh = N/2, log2N = ilog2(N);
    int bf = (N + th - 1)/th, bh = (Nh + th - 1)/th;
    for (uint32_t m = Nh; m >= 1; m >>= 1) ntt_stage_dif<<<bh, th, 0, s>>>(x, tw_inv + N, q, m, Nh);
    bit_reverse_permute<<<bf, th, 0, s>>>(x, N, log2N);
    scale_by_ninv<<<bf, th, 0, s>>>(x, n_inv, q, N);
    twist_kernel<<<bf, th, 0, s>>>(x, tw_inv, q, N);
}

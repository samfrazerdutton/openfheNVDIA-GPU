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

__global__ void bit_reverse_permute(uint64_t* __restrict__ x, uint32_t N, uint32_t log2N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    uint32_t rev = 0, v = tid;
    for (uint32_t i = 0; i < log2N; i++) { rev = (rev << 1) | (v & 1); v >>= 1; }
    if (rev > tid) { uint64_t tmp = x[tid]; x[tid] = x[rev]; x[rev] = tmp; }
}

__global__ void scale_by_ninv(uint64_t* __restrict__ x, uint64_t n_inv, uint64_t q, uint32_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    x[tid] = mulmod64(x[tid], n_inv, q);
}

__global__ void twist_kernel(uint64_t* __restrict__ x, const uint64_t* __restrict__ tw, uint64_t q, uint32_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    x[tid] = mulmod64(x[tid], tw[tid], q); 
}

__global__ void ntt_stage_dit(uint64_t* __restrict__ x, const uint64_t* __restrict__ wpow, uint64_t q, uint32_t half_m, uint32_t N_half) {
    uint32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_half) return;
    uint32_t group = tid / half_m;
    uint32_t j     = tid % half_m;
    uint32_t i_top = group * 2 * half_m + j;
    uint32_t i_bot = i_top + half_m;

    uint32_t step  = N_half / half_m;
    uint64_t w     = wpow[j * step];

    uint64_t u = x[i_top];
    uint64_t v = mulmod64(x[i_bot], w, q);
    x[i_top] = addmod(u, v, q);
    x[i_bot] = submod(u, v, q);
}

__global__ void ntt_stage_dif(uint64_t* __restrict__ x, const uint64_t* __restrict__ wpow, uint64_t q, uint32_t half_m, uint32_t N_half) {
    uint32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_half) return;
    uint32_t group = tid / half_m;
    uint32_t j     = tid % half_m;
    uint32_t i_top = group * 2 * half_m + j;
    uint32_t i_bot = i_top + half_m;

    uint32_t step  = N_half / half_m;
    uint64_t w     = wpow[j * step];

    uint64_t u = x[i_top];
    uint64_t v = x[i_bot];
    x[i_top] = addmod(u, v, q);
    x[i_bot] = mulmod64(submod(u, v, q), w, q);
}

static uint32_t ilog2(uint32_t N) { uint32_t k=0; while((1u<<k)<N) k++; return k; }

extern "C" void LaunchNTT(uint64_t* x, const uint64_t* tw, uint64_t q, uint32_t N, cudaStream_t s) {
    int threads = 256;
    uint32_t N_half = N / 2;
    uint32_t log2N  = ilog2(N);
    int bf = (N + threads - 1) / threads;
    int bh = (N_half + threads - 1) / threads;

    twist_kernel<<<bf, threads, 0, s>>>(x, tw, q, N);                   
    bit_reverse_permute<<<bf, threads, 0, s>>>(x, N, log2N);
    const uint64_t* wpow = tw + N;                                     
    for (uint32_t half_m = 1; half_m <= N_half; half_m <<= 1) {
        ntt_stage_dit<<<bh, threads, 0, s>>>(x, wpow, q, half_m, N_half);
    }
}

extern "C" void LaunchINTT(uint64_t* x, const uint64_t* tw_inv, uint64_t q, uint32_t N, uint64_t n_inv, cudaStream_t s) {
    int threads = 256;
    uint32_t N_half = N / 2;
    uint32_t log2N  = ilog2(N);
    int bf = (N + threads - 1) / threads;
    int bh = (N_half + threads - 1) / threads;

    const uint64_t* wpow_inv = tw_inv + N;
    for (uint32_t half_m = N_half; half_m >= 1; half_m >>= 1) {
        ntt_stage_dif<<<bh, threads, 0, s>>>(x, wpow_inv, q, half_m, N_half);
    }
    bit_reverse_permute<<<bf, threads, 0, s>>>(x, N, log2N);
    scale_by_ninv<<<bf, threads, 0, s>>>(x, n_inv, q, N);
    twist_kernel<<<bf, threads, 0, s>>>(x, tw_inv, q, N);             
}

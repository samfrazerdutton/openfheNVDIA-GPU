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

// tw[0..N-1] = psi^0..psi^(N-1)  (pre-twist powers)
// tw[N..2N-1] = w^0..w^(N-1) where w=psi^2  (cyclic NTT powers, w^j for j=0..N-1)
__global__ void pre_twist(uint64_t* __restrict__ x, const uint64_t* __restrict__ tw,
                           uint64_t q, uint32_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    x[tid] = mulmod64(x[tid], tw[tid], q);  // x[k] *= psi^k
}

__global__ void post_untwist(uint64_t* __restrict__ x, const uint64_t* __restrict__ tw,
                              uint64_t q, uint32_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    x[tid] = mulmod64(x[tid], tw[tid], q);  // x[k] *= psi_inv^k
}

__global__ void ntt_stage_dit(uint64_t* __restrict__ x,
                               const uint64_t* __restrict__ wpow,
                               uint64_t q, uint32_t half_m, uint32_t N_half,
                               uint32_t stride) {
    uint32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_half) return;
    uint32_t group = tid / half_m;
    uint32_t j     = tid % half_m;
    uint32_t i_top = group * 2 * half_m + j;
    uint32_t i_bot = i_top + half_m;
    uint64_t w  = wpow[j * stride];   // w^(j * N/(2*half_m)) = w^(stride*j)
    uint64_t u  = x[i_top];
    uint64_t v  = mulmod64(x[i_bot], w, q);
    x[i_top]    = addmod(u, v, q);
    x[i_bot]    = submod(u, v, q);
}

__global__ void scale_by_ninv(uint64_t* __restrict__ x, uint64_t n_inv, uint64_t q, uint32_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    x[tid] = mulmod64(x[tid], n_inv, q);
}

static uint32_t ilog2(uint32_t N) { uint32_t k=0; while((1u<<k)<N) k++; return k; }

// Twiddle table layout (set by new BuildTwiddleTable):
//   forward[0..N-1]   = psi^0, psi^1, ..., psi^(N-1)       (pre-twist)
//   forward[N..2N-1]  = w^0,   w^1,   ..., w^(N-1)          (cyclic NTT, w=psi^2)
//   inverse[0..N-1]   = psi_inv^0, ..., psi_inv^(N-1)       (post-untwist)
//   inverse[N..2N-1]  = w_inv^0, ..., w_inv^(N-1)           (cyclic INTT)

extern "C" void LaunchNTT(uint64_t* x, const uint64_t* tw,
                           uint64_t q, uint32_t N, cudaStream_t s)
{
    int threads = 256;
    uint32_t log2N  = ilog2(N);
    uint32_t N_half = N / 2;
    int bf = (N + threads-1)/threads;
    int bh = (N_half + threads-1)/threads;

    pre_twist<<<bf, threads, 0, s>>>(x, tw, q, N);                   // x[k] *= psi^k
    bit_reverse_permute<<<bf, threads, 0, s>>>(x, N, log2N);
    const uint64_t* wpow = tw + N;                                     // w-power section
    for (uint32_t half_m = 1; half_m <= N_half; half_m <<= 1) {
        uint32_t stride = N_half / half_m;                             // = N/(2*half_m)
        ntt_stage_dit<<<bh, threads, 0, s>>>(x, wpow, q, half_m, N_half, stride);
    }
}

extern "C" void LaunchINTT(uint64_t* x, const uint64_t* tw_inv,
                            uint64_t q, uint32_t N, uint64_t n_inv, cudaStream_t s)
{
    int threads = 256;
    uint32_t log2N  = ilog2(N);
    uint32_t N_half = N / 2;
    int bf = (N + threads-1)/threads;
    int bh = (N_half + threads-1)/threads;

    bit_reverse_permute<<<bf, threads, 0, s>>>(x, N, log2N);
    const uint64_t* wpow_inv = tw_inv + N;
    for (uint32_t half_m = 1; half_m <= N_half; half_m <<= 1) {
        uint32_t stride = N_half / half_m;
        ntt_stage_dit<<<bh, threads, 0, s>>>(x, wpow_inv, q, half_m, N_half, stride);
    }
    scale_by_ninv<<<bf, threads, 0, s>>>(x, n_inv, q, N);
    post_untwist<<<bf, threads, 0, s>>>(x, tw_inv, q, N);             // x[k] *= psi_inv^k
}

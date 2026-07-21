// cuda_keyswitch.cu — GPU RNS inner-product kernels for hybrid key-switching.
#include <cuda_runtime.h>
#include <cstdint>

// ── Original per-tower kernels (kept for reference / fallback) ───────────
__global__ void rns_mac_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t*       __restrict__ out,
    uint64_t q, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned __int128 prod = (unsigned __int128)a[idx] * b[idx];
    uint64_t r = (uint64_t)(prod % q);
    uint64_t s = out[idx] + r;
    out[idx]   = (s >= q) ? s - q : s;
}

extern "C" void LaunchRNSMac(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_out,
    uint64_t q, uint32_t n, cudaStream_t s)
{
    uint32_t blocks = (n + 255) / 256;
    rns_mac_kernel<<<blocks, 256, 0, s>>>(d_a, d_b, d_out, q, n);
}

__device__ __forceinline__ uint64_t ks_mont_reduce(unsigned __int128 T,
                                                   uint64_t q, uint64_t q_inv)
{
    uint64_t m = (uint64_t)T * q_inv;
    unsigned __int128 mq = (unsigned __int128)m * q;
    uint64_t t = (uint64_t)((T + mq) >> 64);
    return (t >= q) ? t - q : t;
}

__global__ void rns_mac_mont_kernel(
    const uint64_t* __restrict__ c,
    const uint64_t* __restrict__ bR,
    uint64_t*       __restrict__ out,
    uint64_t q, uint64_t q_inv, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned __int128 T = (unsigned __int128)c[idx] * bR[idx];
    uint64_t r = ks_mont_reduce(T, q, q_inv);
    uint64_t s = out[idx] + r;
    out[idx]   = (s >= q) ? s - q : s;
}

extern "C" void LaunchRNSMacMont(
    const uint64_t* d_c, const uint64_t* d_bR, uint64_t* d_out,
    uint64_t q, uint64_t q_inv, uint32_t n, cudaStream_t s)
{
    uint32_t blocks = (n + 255) / 256;
    rns_mac_mont_kernel<<<blocks, 256, 0, s>>>(d_c, d_bR, d_out, q, q_inv, n);
}

// ── Fused multi-tower Montgomery MAC ─────────────────────────────────────
//
// One launch handles EVERY tower of one digit. Previously keyswitch issued
// 2 kernels per (digit, tower) = 36 launches at depth-5 params; at ~10 us
// of launch overhead each (WSL) that dominated the ~4 us of real work per
// kernel. This collapses it to 2 launches per digit.
//
// Thread t covers global element (tower = t / n, coeff = t % n) with a
// grid-stride loop, so any grid size is valid. Per-tower parameters come
// from device arrays built once per call by the host.
//
// Keys must be Montgomery-pre-scaled (kR = k * 2^64 mod q) — done at
// upload time by KeyResidencyCache. Accumulators stay in the plain domain.
__global__ void rns_mac_mont_fused_kernel(
    const uint64_t* const* __restrict__ c,      // [towers] digit tower ptrs
    const uint64_t* const* __restrict__ kR,     // [towers] R-scaled key ptrs
    uint64_t* const*       __restrict__ acc,    // [towers] accumulator ptrs
    const uint64_t* __restrict__ q,             // [towers]
    const uint64_t* __restrict__ q_inv,         // [towers]
    uint32_t n, uint32_t towers)
{
    const uint64_t total = (uint64_t)n * towers;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         g < total; g += stride) {
        const uint32_t t = (uint32_t)(g / n);
        const uint32_t i = (uint32_t)(g - (uint64_t)t * n);
        const uint64_t qt = q[t];
        unsigned __int128 T = (unsigned __int128)c[t][i] * kR[t][i];
        uint64_t r = ks_mont_reduce(T, qt, q_inv[t]);
        uint64_t s = acc[t][i] + r;
        acc[t][i]  = (s >= qt) ? s - qt : s;
    }
}

extern "C" void LaunchRNSMacMontFused(
    const uint64_t* const* d_c, const uint64_t* const* d_kR,
    uint64_t* const* d_acc, const uint64_t* d_q, const uint64_t* d_qinv,
    uint32_t n, uint32_t towers, cudaStream_t s)
{
    const uint64_t total = (uint64_t)n * towers;
    const uint32_t threads = 256;
    uint64_t want = (total + threads - 1) / threads;
    // Cap the grid; the stride loop covers the remainder.
    const uint32_t blocks = (uint32_t)(want > 65535 ? 65535 : (want ? want : 1));
    rns_mac_mont_fused_kernel<<<blocks, threads, 0, s>>>(
        d_c, d_kR, d_acc, d_q, d_qinv, n, towers);
}

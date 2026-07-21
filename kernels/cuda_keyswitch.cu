// cuda_keyswitch.cu — GPU RNS inner-product kernels for hybrid key-switching.
#include <cuda_runtime.h>
#include <cstdint>

// Original kernel (plain 128-bit modulo). Kept for reference/fallback.
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

// ── Fix 3: Montgomery MAC ────────────────────────────────────────────────
// Requires the SECOND operand pre-scaled: bR[i] = b[i] * 2^64 mod q
// (done once at key-residency upload). Then:
//   mont_reduce(c * bR) = c * b * R * R^{-1} = c * b  (mod q)
// so the result is in the plain domain with no 128-bit division.
// q_inv = -q^{-1} mod 2^64.
__device__ __forceinline__ uint64_t ks_mont_reduce(unsigned __int128 T,
                                                   uint64_t q, uint64_t q_inv)
{
    uint64_t m = (uint64_t)T * q_inv;
    unsigned __int128 mq = (unsigned __int128)m * q;
    uint64_t t = (uint64_t)((T + mq) >> 64);
    return (t >= q) ? t - q : t;
}

__global__ void rns_mac_mont_kernel(
    const uint64_t* __restrict__ c,     // plain domain
    const uint64_t* __restrict__ bR,    // pre-scaled by R mod q
    uint64_t*       __restrict__ out,   // plain-domain accumulator
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

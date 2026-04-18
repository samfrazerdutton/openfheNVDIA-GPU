#include <cuda_runtime.h>
#include <cstdint>

// Montgomery reduction. R = 2^64.
// Given T < q*R, returns T * R^{-1} mod q.
__device__ __forceinline__ uint64_t mont_reduce(__uint128_t T, uint64_t q, uint64_t q_inv) {
    uint64_t m = (uint64_t)T * q_inv;
    __uint128_t mq = (__uint128_t)m * q;
    uint64_t t = (uint64_t)((T + mq) >> 64);
    return (t >= q) ? t - q : t;
}

__global__ void rns_mult_montgomery_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ r,
    uint64_t q, uint64_t q_inv, uint64_t R2, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    __uint128_t T  = (__uint128_t)a[idx] * b[idx];
    uint64_t res1  = mont_reduce(T, q, q_inv);
    __uint128_t T2 = (__uint128_t)res1 * R2;
    r[idx]         = mont_reduce(T2, q, q_inv);
}

__global__ void rns_mult_exact_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ r,
    uint64_t q, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        r[idx] = (uint64_t)(((unsigned __int128)a[idx] * b[idx]) % q);
}

extern "C" void LaunchRNSMult(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res,
    uint64_t q, uint32_t n, cudaStream_t stream)
{
    uint32_t blocks = (n + 255) / 256;
    rns_mult_exact_kernel<<<blocks, 256, 0, stream>>>(d_a, d_b, d_res, q, n);
}

extern "C" void LaunchRNSMultMontgomery(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res,
    uint64_t q, uint64_t q_inv, uint64_t R2, uint32_t n, cudaStream_t stream)
{
    uint32_t blocks = (n + 255) / 256;
    rns_mult_montgomery_kernel<<<blocks, 256, 0, stream>>>(d_a, d_b, d_res, q, q_inv, R2, n);
}

extern "C" void LaunchRNSMultBarrett(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res,
    uint64_t mod, uint64_t mu_hi, uint32_t n, cudaStream_t stream)
{
    // mu_hi ignored -- exact fallback is correct and avoids approximation error.
    uint32_t blocks = (n + 255) / 256;
    rns_mult_exact_kernel<<<blocks, 256, 0, stream>>>(d_a, d_b, d_res, mod, n);
}


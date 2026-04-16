#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__
uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    unsigned __int128 prod = ((unsigned __int128)hi << 64) | lo;
    return (uint64_t)(prod % m);
}

__global__ void rns_mult_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ res,
    uint64_t m, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        res[idx] = mulmod64(a[idx], b[idx], m);
}

extern "C" void LaunchRNSMult(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res,
    uint64_t mod, uint32_t n, cudaStream_t stream)
{
    uint32_t blocks = (n + 255) / 256;
    rns_mult_kernel<<<blocks, 256, 0, stream>>>(d_a, d_b, d_res, mod, n);
}

// Alias expected by patched dcrtpoly.h
extern "C" void LaunchRNSMultBarrett(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_res,
    uint64_t mod, uint64_t /*mu_hi*/, uint32_t n, cudaStream_t stream)
{
    // mu_hi ignored — we use exact modulo, not Barrett
    LaunchRNSMult(d_a, d_b, d_res, mod, n, stream);
}

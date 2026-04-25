// cuda_keyswitch.cu — GPU RNS inner-product kernel for hybrid key-switching.
#include <cuda_runtime.h>
#include <cstdint>

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

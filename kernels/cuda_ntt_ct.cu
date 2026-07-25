// cuda_ntt_ct.cu — Cooley-Tukey NTT matching OpenFHE's Shoup butterfly
// bit-for-bit. Consumes OpenFHE's own bit-reversed root table and its Shoup
// precon table (floor(omega * 2^64 / q)); no twiddle generation here. Input is
// standard order, output is bit-reversed order, in [0, q) — identical layout to
// what DCRTPolyImpl::operator* expects, which is the whole point: a residency
// design can only stay correct if the on-device transform is not merely
// congruent to OpenFHE's but exactly equal.

#include <cuda_runtime.h>
#include <cstdint>

// Shoup modular multiply: x * omega mod q, given precon = floor(omega*2^64/q).
// Matches NativeInteger::ModMulFastConstEq: one high-multiply, one subtract-back,
// one conditional reduction. Result in [0, q).
__device__ __forceinline__
uint64_t shoup_mulmod(uint64_t x, uint64_t omega, uint64_t precon, uint64_t q) {
    uint64_t hi = __umul64hi(x, precon);      // floor(x * omega / q), approx
    uint64_t r  = x * omega - hi * q;         // in [0, 2q)
    return (r >= q) ? r - q : r;              // in [0, q)
}

// One CT stage. Grid-strided over the n/2 butterflies of this stage.
// m       = number of blocks this stage (1,2,4,...)
// t       = half-block width (n, n/2, ...)  -- distance between butterfly pair
// logt    = log2(t) -- so j1 = i << logt
// roots   = OpenFHE bit-reversed root table (device)  -- omega = roots[i+m]
// precon  = matching Shoup precon table (device)
__global__ void ct_stage(uint64_t* __restrict__ x,
                         const uint64_t* __restrict__ roots,
                         const uint64_t* __restrict__ precon,
                         uint32_t n, uint32_t m, uint32_t t, uint32_t logt,
                         uint64_t q) {
    const uint32_t nbf = n >> 1;              // butterflies this stage
    for (uint32_t bf = blockIdx.x * blockDim.x + threadIdx.x;
         bf < nbf; bf += gridDim.x * blockDim.x) {
        // Decompose the flat butterfly index into (i, offset-within-block).
        const uint32_t i   = bf >> logt;      // which block -> which omega
        const uint32_t off = bf & (t - 1);    // position within the block
        const uint32_t j1  = (i << (logt + 1)) + off;
        const uint32_t j2  = j1 + t;

        const uint64_t omega  = roots[i + m];
        const uint64_t pomega = precon[i + m];

        const uint64_t of = shoup_mulmod(x[j2], omega, pomega, q);
        const uint64_t lo = x[j1];

        uint64_t hi = lo + of;
        if (hi >= q) hi -= q;
        uint64_t lv = lo;
        if (lv < of) lv += q;
        lv -= of;

        x[j1] = hi;
        x[j2] = lv;
    }
}

static uint32_t ilog2_u(uint32_t v) { uint32_t k = 0; while ((1u << k) < v) k++; return k; }

// Runs the full forward transform in place on device buffer d_x (length n).
// roots/precon are device pointers to OpenFHE's tables for this modulus.
extern "C" void LaunchNTT_CT(uint64_t* d_x,
                            const uint64_t* d_roots, const uint64_t* d_precon,
                            uint32_t n, uint64_t q, cudaStream_t s) {
    const int th  = 256;
    const int nbf = (int)(n >> 1);
    const int bl  = (nbf + th - 1) / th;
    for (uint32_t m = 1, t = n >> 1; m < n; m <<= 1, t >>= 1) {
        uint32_t logt = ilog2_u(t);
        ct_stage<<<bl, th, 0, s>>>(d_x, d_roots, d_precon, n, m, t, logt, q);
    }
}

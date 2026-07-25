// test_ntt_vs_openfhe.cpp — bit-exact validation of the GPU CT-NTT against
// OpenFHE's own ForwardTransformToBitReverseInPlace. No tolerance: every
// coefficient must match exactly, because the residency architecture this
// unblocks depends on the GPU transform being substitutable for OpenFHE's,
// not merely numerically close. A 1e-12 "pass" is what let a wrong residency
// result through before; this asserts equality.

#include "openfhe.h"
#include "math/hal/intnat/transformnat.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;

extern "C" void LaunchNTT_CT(uint64_t*, const uint64_t*, const uint64_t*,
                             uint32_t, uint64_t, cudaStream_t);

#define CK(x) do { cudaError_t e=(x); if(e){ \
    std::cerr<<"CUDA "<<cudaGetErrorString(e)<<" @"<<__LINE__<<"\n"; return 2;} } while(0)

int main() {
    using NI  = NativeInteger;
    using NV  = NativeVector;

    // A negacyclic-NTT-friendly prime and ring dim OpenFHE will accept.
    const uint32_t n = 4096;
    const uint32_t cycloOrder = 2 * n;
    NI q = FirstPrime<NI>(40, cycloOrder);
    NI root = RootOfUnity<NI>(cycloOrder, q);

    // Force OpenFHE to build its bit-reversed root + Shoup precon tables.
    intnat::ChineseRemainderTransformFTTNat<NV> crt;
    NV in(n, q);
    std::mt19937_64 rng(20260724);
    for (uint32_t i = 0; i < n; ++i) in[i] = NI(rng() % q.ConvertToInt());

    NV ref = in;                          // OpenFHE will transform this copy
    crt.ForwardTransformToBitReverseInPlace(root, cycloOrder, &ref);

    // Pull OpenFHE's tables straight out of the static maps, keyed by modulus.
    // Reconstruct OpenFHE's bit-reversed root table publicly.
    auto bitrev = [](uint32_t v, uint32_t bits){ uint32_t r=0;
        for(uint32_t k=0;k<bits;k++){ r=(r<<1)|(v&1); v>>=1; } return r; };
    uint32_t logn = 0; while ((1u<<logn) < n) ++logn;
    std::vector<uint64_t> h_roots(n), h_precon(n);
    NI omega_i(1);
    std::vector<NI> powers(n);
    powers[0] = NI(1);
    for (uint32_t i = 1; i < n; ++i) powers[i] = powers[i-1].ModMul(root, q);
    for (uint32_t i = 0; i < n; ++i) {
        NI r_i = powers[bitrev(i, logn)];
        h_roots[i] = r_i.ConvertToInt();
        // Shoup precon = floor(r_i * 2^64 / q)
        __uint128_t num = ((__uint128_t)r_i.ConvertToInt() << 64);
        h_precon[i] = (uint64_t)(num / (__uint128_t)q.ConvertToInt());
    }

    std::vector<uint64_t> h_in(n);
    for (uint32_t i = 0; i < n; ++i) h_in[i] = in[i].ConvertToInt();

    uint64_t *d_x=nullptr, *d_r=nullptr, *d_p=nullptr;
    const size_t B = (size_t)n * sizeof(uint64_t);
    CK(cudaMalloc(&d_x, B)); CK(cudaMalloc(&d_r, B)); CK(cudaMalloc(&d_p, B));
    CK(cudaMemcpy(d_x, h_in.data(),     B, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_r, h_roots.data(),  B, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_p, h_precon.data(), B, cudaMemcpyHostToDevice));

    LaunchNTT_CT(d_x, d_r, d_p, n, q.ConvertToInt(), 0);
    CK(cudaDeviceSynchronize());

    std::vector<uint64_t> h_out(n);
    CK(cudaMemcpy(h_out.data(), d_x, B, cudaMemcpyDeviceToHost));

    uint32_t mism = 0; int first = -1;
    for (uint32_t i = 0; i < n; ++i) {
        if (h_out[i] != ref[i].ConvertToInt()) {
            if (first < 0) first = (int)i;
            ++mism;
        }
    }

    cudaFree(d_x); cudaFree(d_r); cudaFree(d_p);

    if (mism == 0) {
        std::cout << "[PASS] GPU CT-NTT bit-exact vs OpenFHE  (n=" << n
                  << ", q=" << q.ConvertToInt() << ", " << n << "/" << n << " match)\n";
        return 0;
    }
    std::cout << "[FAIL] " << mism << "/" << n << " mismatches, first at index "
              << first << ": gpu=" << h_out[first]
              << " ref=" << ref[first].ConvertToInt() << "\n";
    return 1;
}

// test_ntt_towers.cpp — bit-exact NTT validation across a REAL ciphertext's
// tower moduli, not one synthetic prime. This is the bridge from "kernel is
// correct in isolation" to "kernel is correct on the primes the pipeline
// actually uses". CKKS tower primes vary in bit-width; the Shoup intermediate
// x*omega - hi*q must stay correct for each. A pass here is the precondition
// for feeding the kernel real DCRTPoly data.

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

static int validate_modulus(uint32_t n, NativeInteger q, NativeInteger root) {
    using NI = NativeInteger;
    using NV = intnat::NativeVectorT<NI>;
    const uint32_t cycloOrder = 2 * n;

    intnat::ChineseRemainderTransformFTTNat<NV> crt;
    NV in(n, q);
    std::mt19937_64 rng(20260724);
    for (uint32_t i = 0; i < n; ++i) in[i] = NI(rng() % q.ConvertToInt());

    NV ref = in;
    crt.ForwardTransformToBitReverseInPlace(root, cycloOrder, &ref);

    auto bitrev = [](uint32_t v, uint32_t bits){ uint32_t r=0;
        for(uint32_t k=0;k<bits;k++){ r=(r<<1)|(v&1); v>>=1; } return r; };
    uint32_t logn = 0; while ((1u<<logn) < n) ++logn;

    std::vector<uint64_t> h_in(n), h_roots(n), h_precon(n);
    std::vector<NI> powers(n);
    powers[0] = NI(1);
    for (uint32_t i = 1; i < n; ++i) powers[i] = powers[i-1].ModMul(root, q);
    for (uint32_t i = 0; i < n; ++i) {
        NI r_i = powers[bitrev(i, logn)];
        h_roots[i]  = r_i.ConvertToInt();
        __uint128_t num = ((__uint128_t)r_i.ConvertToInt() << 64);
        h_precon[i] = (uint64_t)(num / (__uint128_t)q.ConvertToInt());
        h_in[i] = in[i].ConvertToInt();
    }

    uint64_t *d_x=nullptr,*d_r=nullptr,*d_p=nullptr;
    const size_t B = (size_t)n*sizeof(uint64_t);
    cudaMalloc(&d_x,B); cudaMalloc(&d_r,B); cudaMalloc(&d_p,B);
    cudaMemcpy(d_x,h_in.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,h_roots.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_p,h_precon.data(),B,cudaMemcpyHostToDevice);
    LaunchNTT_CT(d_x,d_r,d_p,n,q.ConvertToInt(),0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> h_out(n);
    cudaMemcpy(h_out.data(),d_x,B,cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_p);

    uint32_t mism=0; int first=-1;
    for (uint32_t i=0;i<n;++i)
        if (h_out[i]!=ref[i].ConvertToInt()) { if(first<0) first=(int)i; ++mism; }

    const uint32_t bits = q.GetMSB();
    if (mism==0) {
        std::cout << "  [ok]  q=" << q.ConvertToInt() << " (" << bits << "-bit)  "
                  << n << "/" << n << "\n";
        return 0;
    }
    std::cout << "  [BAD] q=" << q.ConvertToInt() << " (" << bits << "-bit)  "
              << mism << " mismatches, first@" << first
              << " gpu=" << h_out[first] << " ref=" << ref[first].ConvertToInt() << "\n";
    return 1;
}

int main() {
    using NI = NativeInteger;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(10);
    params.SetScalingModSize(50);
    params.SetRingDim(32768);
    params.SetBatchSize(16384);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(LEVELEDSHE);

    // Pull the actual RNS tower moduli for this context.
    auto ep = cc->GetCryptoParameters()->GetElementParams();
    const uint32_t n = ep->GetRingDimension();
    const uint32_t cyclo = 2 * n;

    std::cout << "=== GPU NTT vs OpenFHE across real CKKS tower moduli ===\n"
              << "ring=" << n << " towers=" << ep->GetParams().size() << "\n";

    int fails = 0, k = 0;
    for (const auto& p : ep->GetParams()) {
        NI q = p->GetModulus();
        NI root = RootOfUnity<NI>(cyclo, q);
        std::cout << " tower " << k++ << ":";
        fails += validate_modulus(n, q, root);
    }

    if (!fails) { std::cout << "[PASS] all towers bit-exact\n"; return 0; }
    std::cout << "[FAIL] " << fails << " tower(s) mismatched\n"; return 1;
}

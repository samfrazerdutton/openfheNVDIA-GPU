#include "openfhe.h"
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace lbcrypto;
using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
    const int NREPS  = 100;
    const int WARMUP = 5;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(10);
    params.SetScalingModSize(50);
    params.SetRingDim(32768);
    params.SetBatchSize(16384);
    params.SetScalingTechnique(FLEXIBLEAUTO);
    params.SetSecurityLevel(HEStd_128_classic);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    std::vector<double> x(16384, 0.1);
    std::vector<double> y(16384, 0.2);
    Plaintext ptx1 = cc->MakeCKKSPackedPlaintext(x);
    Plaintext ptx2 = cc->MakeCKKSPackedPlaintext(y);

    auto ctx1 = cc->Encrypt(keyPair.publicKey, ptx1);
    auto ctx2 = cc->Encrypt(keyPair.publicKey, ptx2);

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        auto tmp = cc->EvalMult(ctx1, ctx2);
    }

    auto t_start = Clock::now();
    auto ctx_res = ctx1;
    for (int i = 0; i < NREPS; i++) {
        ctx_res = cc->EvalMult(ctx1, ctx2);
    }
    auto t_end = Clock::now();

    Plaintext ptx_res;
    cc->Decrypt(keyPair.secretKey, ctx_res, &ptx_res);
    ptx_res->SetLength(16384);

    double expected = 0.02;
    double got = ptx_res->GetRealPackedValue()[0];
    double err = std::abs(expected - got);

    std::cout << "Ring dim : 32768\n";
    std::cout << "Towers   : " << ctx1->GetElements()[0].GetNumOfElements() << "\n";
    std::cout << "Reps     : " << NREPS << "\n\n";

    double total_ms = elapsed_ms(t_start, t_end);
    double mean_ms = total_ms / NREPS;

    std::cout << "EvalMult over " << NREPS << " reps:\n";
    std::cout << "  mean   " << mean_ms << " ms\n\n";

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Correctness: expected=" << expected << " got=" << got << " err=" << err << "\n";
    
    if (err < 1e-4) {
        std::cout << "[PASS]\n";
        return 0;
    } else {
        std::cout << "[FAIL] Error too large\n";
        return 1;
    }
}

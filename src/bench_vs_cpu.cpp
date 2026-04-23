#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace lbcrypto;
using clk = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    bool is_gpu = (argc > 1 && std::string(argv[1]) == "--gpu");
    
    std::cout << "======================================================\n";
    std::cout << (is_gpu ? "[*] GPU (RTX 2060) Benchmark" : "[*] CPU (Native OpenMP) Benchmark") << "\n";
    std::cout << "======================================================\n";

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(5);
    params.SetScalingModSize(50);
    params.SetBatchSize(16384);
    params.SetRingDim(32768);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);

    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    std::vector<double> x(16384, 0.5), y(16384, 0.5);
    auto ctx = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(x));
    auto cty = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(y));

    // Warmup (PCIe upload + EVK upload for GPU cache)
    Ciphertext<DCRTPoly> result = cc->EvalMult(ctx, cty);

    int reps = 25;
    double total_ms = 0.0;

    for (int i = 0; i < reps; i++) {
        auto t0 = clk::now();
        result = cc->EvalMult(ctx, cty);
        total_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    }

    std::cout << "Ring dim : 32768 | Towers : 11 | Reps : " << reps << "\n";
    std::cout << "Mean Latency: " << std::fixed << std::setprecision(2) << (total_ms / reps) << " ms\n\n";

    return 0;
}

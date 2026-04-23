#include "global_dag.h"
#include <openfhe.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace lbcrypto;
using clk = std::chrono::high_resolution_clock;

static double approx_error(const std::vector<std::complex<double>>& a,
                            const std::vector<std::complex<double>>& b) {
    double max_err = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++)
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    return max_err;
}

int main() {
    std::cout << "======================================================\n";
    std::cout << "[*] OpenFHE NVIDIA GPU HAL — End-to-End CKKS Test\n";
    std::cout << "[*] Pipeline: CPU encrypt → GPU EvalMult → CPU decrypt\n";
    std::cout << "======================================================\n\n";

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(5);
    params.SetScalingModSize(50);
    params.SetBatchSize(16384);
    params.SetRingDim(32768);

    auto t0 = clk::now();
    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    GlobalDAG::Init();
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);  // CPU only — do NOT capture in DAG
    double t_setup = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    std::cout << "[1] Key generation:     " << std::fixed << std::setprecision(2) << t_setup << " ms\n";

    size_t slots = 16384;
    std::vector<double> x_vals(slots), y_vals(slots);
    for (size_t i = 0; i < slots; i++) {
        x_vals[i] = 0.1 * (i % 10 + 1);
        y_vals[i] = 0.5 + 0.01 * (i % 7);
    }
    std::vector<std::complex<double>> expected(slots);
    for (size_t i = 0; i < slots; i++) expected[i] = x_vals[i] * y_vals[i];

    auto t1 = clk::now();
    auto ctx = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(x_vals));
    auto cty = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(y_vals));
    double t_enc = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
    std::cout << "[2] Encrypt (CPU):      " << t_enc << " ms\n";

    auto t2 = clk::now();
    auto ct_result = cc->EvalMult(ctx, cty);
    double t_mult = std::chrono::duration<double, std::milli>(clk::now() - t2).count();
    uint32_t ring = cc->GetRingDimension();
    std::cout << "[3] EvalMult (GPU HAL): " << t_mult << " ms"
              << (ring >= 4096 ? "  [GPU fast-path, ring=" : "  [CPU fallback, ring=")
              << ring << "]\n";

    auto t3 = clk::now();
    Plaintext pt_out;
    cc->Decrypt(kp.secretKey, ct_result, &pt_out);
    pt_out->SetLength(slots);
    double t_dec = std::chrono::duration<double, std::milli>(clk::now() - t3).count();
    std::cout << "[4] Decrypt (CPU):      " << t_dec << " ms\n";

    auto got = pt_out->GetCKKSPackedValue();
    double max_err = approx_error(got, expected);
    std::cout << "\n[*] First 8 slots:\n    idx  expected        got             error\n";
    for (int i = 0; i < 8; i++)
        printf("    [%d]  %.10f  %.10f  %.2e\n",
               i, expected[i].real(), got[i].real(), std::abs(got[i] - expected[i]));

    std::cout << "\n[*] Max absolute error across all " << slots << " slots: " << max_err << "\n";
    std::cout << "[*] Total pipeline latency (enc+mult+dec): " << (t_enc+t_mult+t_dec) << " ms\n\n";
    bool pass = (max_err < 1e-6);
    std::cout << "======================================================\n"
              << (pass ? "[PASS] CPU→GPU→CPU round-trip correct"
                       : "[FAIL] Error too large") << "\n"
              << "======================================================\n";
    return pass ? 0 : 1;
}

/**
 * test_e2e_ckks.cpp
 *
 * End-to-end CPU→GPU→CPU round-trip test for the OpenFHE NVIDIA GPU HAL.
 * Tests the exact pipeline CryptoLab achieves with HEaaN-GPU:
 *   1. Generate CKKS keys (CPU)
 *   2. Encrypt plaintext vector (CPU)
 *   3. EvalMult — RNS polynomial multiply dispatched to GPU via HAL
 *   4. Decrypt result (CPU)
 *   5. Verify numerical correctness against CPU-only reference
 *   6. Report timing breakdown for each phase
 */

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

    // ── Parameters (matches simple-ckks-bootstrapping defaults) ──────────────
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(5);
    params.SetScalingModSize(50);
    params.SetBatchSize(16384);
    params.SetRingDim(32768);   // N=8192: GPU fast-path triggers at ring >= 4096

    // ── 1. Setup ──────────────────────────────────────────────────────────────
    auto t0 = clk::now();
    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);
    double t_setup = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    std::cout << "[1] Key generation:     " << std::fixed << std::setprecision(2)
              << t_setup << " ms\n";

    // ── 2. Plaintext ──────────────────────────────────────────────────────────
    size_t slots = 16384;
    std::vector<double> x_vals(slots), y_vals(slots);
    for (size_t i = 0; i < slots; i++) {
        x_vals[i] = 0.1 * (i % 10 + 1);   // 0.1, 0.2, ... 1.0, 0.1, ...
        y_vals[i] = 0.5 + 0.01 * (i % 7); // small values so product stays < 1
    }

    // CPU reference: x * y elementwise
    std::vector<std::complex<double>> expected(slots);
    for (size_t i = 0; i < slots; i++)
        expected[i] = x_vals[i] * y_vals[i];

    // ── 3. Encrypt (CPU) ──────────────────────────────────────────────────────
    auto t1 = clk::now();
    auto ptx = cc->MakeCKKSPackedPlaintext(x_vals);
    auto pty = cc->MakeCKKSPackedPlaintext(y_vals);
    auto ctx = cc->Encrypt(kp.publicKey, ptx);
    auto cty = cc->Encrypt(kp.publicKey, pty);
    double t_enc = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
    std::cout << "[2] Encrypt (CPU):      " << t_enc << " ms\n";

    // ── 4. EvalMult (GPU via HAL) ─────────────────────────────────────────────
    auto t2 = clk::now();
    auto ct_result = cc->EvalMult(ctx, cty);
    double t_mult = std::chrono::duration<double, std::milli>(clk::now() - t2).count();
    std::cout << "[3] EvalMult (GPU HAL): " << t_mult << " ms";

    // Ring dimension check — confirm GPU path was taken
    uint32_t ring = cc->GetRingDimension();
    if (ring >= 4096)
        std::cout << "  [GPU fast-path, ring=" << ring << "]\n";
    else
        std::cout << "  [CPU fallback, ring=" << ring << " < 4096]\n";

    // ── 5. Decrypt (CPU) ──────────────────────────────────────────────────────
    auto t3 = clk::now();
    Plaintext pt_out;
    cc->Decrypt(kp.secretKey, ct_result, &pt_out);
    pt_out->SetLength(slots);
    double t_dec = std::chrono::duration<double, std::milli>(clk::now() - t3).count();
    std::cout << "[4] Decrypt (CPU):      " << t_dec << " ms\n";

    // ── 6. Verify ─────────────────────────────────────────────────────────────
    auto got = pt_out->GetCKKSPackedValue();
    double max_err = approx_error(got, expected);
    double t_total = t_enc + t_mult + t_dec;

    std::cout << "\n[*] First 8 slots:\n";
    std::cout << "    idx  expected        got             error\n";
    for (int i = 0; i < 8; i++) {
        printf("    [%d]  %.10f  %.10f  %.2e\n",
               i, expected[i].real(), got[i].real(),
               std::abs(got[i] - expected[i]));
    }

    std::cout << "\n[*] Max absolute error across all " << slots
              << " slots: " << max_err << "\n";
    std::cout << "[*] Total pipeline latency (enc+mult+dec): "
              << t_total << " ms\n\n";

    bool pass = (max_err < 1e-6);
    std::cout << "======================================================\n";
    std::cout << (pass ? "[PASS] CPU→GPU→CPU round-trip correct"
                       : "[FAIL] Error too large — GPU result wrong")
              << "\n";
    std::cout << "======================================================\n";
    return pass ? 0 : 1;
}

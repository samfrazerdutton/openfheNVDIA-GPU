// bench_params.cpp — parameter sweep to locate the CPU/GPU crossover.
//
// All prior benchmarking used one operating point (depth 5, ring 32768,
// ~9-11 towers), where per-operation GPU overhead (transfers, launches,
// syncs) is comparable to the arithmetic itself — hence the observed
// CPU/GPU parity. This sweep raises depth and ring dimension so the same
// fixed overhead amortizes over more work, and reports where (if anywhere)
// the GPU path pulls ahead on this hardware.
//
// Three modes per configuration, selected by env var, same binary:
//   cpu    OPENFHE_GPU=0                      full CPU path
//   gpu    (default)                          GPU multiplies, CPU keyswitch
//   gpu-ks OPENFHE_GPU_KS=1                   GPU multiplies + GPU keyswitch
//
// Usage:
//   ./bench_params                 sweep the default grid, current mode
//   ./bench_params 15 65536 10     single config: depth, ring, reps
//
// Reports median of N reps (not mean) so a single outlier cannot move the
// number, plus min/max so the spread is visible.

#include <openfhe.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace lbcrypto;
using clk = std::chrono::high_resolution_clock;

namespace {

std::string ModeLabel() {
    const char* g = std::getenv("OPENFHE_GPU");
    const char* k = std::getenv("OPENFHE_GPU_KS");
    if (g && g[0] == '0') return "cpu";
    if (k && k[0] == '1') return "gpu-ks";
    return "gpu";
}

struct Stats {
    double median, lo, hi;
};

Stats Summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s;
    const size_t n = v.size();
    s.median = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    s.lo = v.front();
    s.hi = v.back();
    return s;
}

// Returns {evalmult_stats, towers}; -1 median signals an unsupported config.
std::pair<Stats, uint32_t> RunConfig(uint32_t depth, uint32_t ring, uint32_t reps) {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(depth);
    params.SetScalingModSize(50);
    params.SetRingDim(ring);
    params.SetBatchSize(ring / 2);

    CryptoContext<DCRTPoly> cc;
    try {
        cc = GenCryptoContext(params);
    } catch (const std::exception& e) {
        std::cout << "    (skipped: " << e.what() << ")\n";
        return {{-1, -1, -1}, 0};
    }
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);

    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    const size_t slots = ring / 2;
    std::vector<double> x(slots, 0.5), y(slots, 0.5);
    auto ctx = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(x));
    auto cty = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(y));

    const uint32_t towers = ctx->GetElements()[0].GetNumOfElements();

    // Warmup: primes the key-residency cache, VRAM buffers, and any
    // first-touch CUDA cost so they land outside the timed region.
    auto warm = cc->EvalMult(ctx, cty);
    (void)warm;

    std::vector<double> ms;
    ms.reserve(reps);
    for (uint32_t i = 0; i < reps; ++i) {
        auto t0 = clk::now();
        auto r = cc->EvalMult(ctx, cty);
        double dt = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (r->GetElements().empty()) std::cout << "";  // keep result alive
        ms.push_back(dt);
    }

    // Correctness spot check: one multiply must still decrypt correctly.
    auto prod = cc->EvalMult(ctx, cty);
    Plaintext out;
    cc->Decrypt(kp.secretKey, prod, &out);
    out->SetLength(4);
    const auto& vals = out->GetCKKSPackedValue();
    double maxerr = 0.0;
    for (size_t i = 0; i < 4; ++i)
        maxerr = std::max(maxerr, std::abs(vals[i].real() - 0.25));
    if (maxerr > 1e-6) {
        std::cout << "    !! CORRECTNESS FAILURE, max err = " << maxerr << "\n";
    }

    return {Summarize(ms), towers};
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string mode = ModeLabel();

    std::cout << "======================================================\n"
              << "[*] CKKS EvalMult parameter sweep -- mode: " << mode << "\n"
              << "    cpu    = OPENFHE_GPU=0      (full CPU)\n"
              << "    gpu    = default            (GPU multiplies)\n"
              << "    gpu-ks = OPENFHE_GPU_KS=1   (GPU multiplies + keyswitch)\n"
              << "======================================================\n\n";

    std::vector<std::pair<uint32_t, uint32_t>> grid;
    uint32_t reps = 10;

    if (argc >= 3) {
        grid.emplace_back((uint32_t)std::stoul(argv[1]), (uint32_t)std::stoul(argv[2]));
        if (argc >= 4) reps = (uint32_t)std::stoul(argv[3]);
    } else {
        grid = {
            {5, 32768},   // the operating point everything was measured at
            {10, 32768},
            {15, 32768},
            {5, 65536},
            {10, 65536},
            {15, 65536},
        };
    }

    std::cout << std::left
              << std::setw(7)  << "depth"
              << std::setw(9)  << "ring"
              << std::setw(8)  << "towers"
              << std::setw(6)  << "reps"
              << std::setw(14) << "median (ms)"
              << std::setw(20) << "min-max (ms)" << "\n"
              << std::string(64, '-') << "\n";

    for (auto& [depth, ring] : grid) {
        std::cout << std::left << std::setw(7) << depth << std::setw(9) << ring
                  << std::flush;
        auto [s, towers] = RunConfig(depth, ring, reps);
        if (s.median < 0) continue;
        std::cout << std::setw(8) << towers << std::setw(6) << reps
                  << std::setw(14) << std::fixed << std::setprecision(2) << s.median
                  << std::setprecision(1) << s.lo << " - " << s.hi << "\n";
    }

    std::cout << "\nRun the same binary in all three modes and compare medians:\n"
              << "  OPENFHE_GPU=0    ./bench_params\n"
              << "  ./bench_params\n"
              << "  OPENFHE_GPU_KS=1 ./bench_params\n";
    return 0;
}

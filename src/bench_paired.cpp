// bench_paired.cpp — paired-difference CPU/GPU timing.
//
// WSL2 does not expose NVML clock control, so SM clocks cannot be locked and
// run-to-run variance on this laptop is ~±30%. Comparing medians from separate
// processes cannot resolve anything smaller than that, which is why every prior
// measurement showed "parity" regardless of what the kernels did.
//
// This harness instead interleaves modes within one process on one context and
// one pair of ciphertexts: rep i times CPU, then GPU, then GPU+KS, back to back.
// Thermal drift, scheduler noise, and background load are then common to all
// three samples in a pair and cancel in the per-rep difference. The reported
// statistic is the median of (gpu[i] - cpu[i]) with a bootstrap CI, which is
// sensitive to effects far below the marginal-noise floor.

#include <openfhe.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

extern "C" void gpu_set_mode(int gpu, int ks);

using namespace lbcrypto;
using clk = std::chrono::steady_clock;

namespace {

constexpr int MODE_CPU = 0, MODE_GPU = 1, MODE_KS = 2;
const char* kModeName[3] = {"cpu", "gpu", "gpu-ks"};

void SetMode(int m) {
    if (m == MODE_CPU) gpu_set_mode(0, 0);
    else if (m == MODE_GPU) gpu_set_mode(1, 0);
    else gpu_set_mode(1, 1);
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double Quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * (v.size() - 1);
    const size_t lo = (size_t)pos;
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}

// Bootstrap CI on the median of paired differences. Resampling pairs (not the
// raw timings) is what preserves the pairing that removes common-mode noise.
std::pair<double, double> BootstrapCI(const std::vector<double>& d, int iters = 4000) {
    if (d.size() < 3) return {0.0, 0.0};
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> pick(0, d.size() - 1);
    std::vector<double> meds;
    meds.reserve(iters);
    std::vector<double> sample(d.size());
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < d.size(); ++i) sample[i] = d[pick(rng)];
        meds.push_back(Median(sample));
    }
    return {Quantile(meds, 0.025), Quantile(meds, 0.975)};
}

struct Result {
    bool ok = false;
    uint32_t towers = 0;
    std::vector<double> t[3];
    double maxerr[3] = {0, 0, 0};
};

Result RunConfig(uint32_t depth, uint32_t ring, uint32_t reps, uint32_t warmup) {
    Result R;

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
        return R;
    }
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);

    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    const size_t slots = ring / 2;
    std::vector<double> x(slots, 0.5), y(slots, 0.5);
    auto ctx = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(x));
    auto cty = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(y));
    R.towers = ctx->GetElements()[0].GetNumOfElements();

    // Per-mode warmup. Each mode has its own first-touch costs (VRAM buffers,
    // eval-key upload, CPU cache state); a single shared warmup would leave
    // whichever mode ran second paying them inside the timed region.
    for (int m = 0; m < 3; ++m) {
        SetMode(m);
        for (uint32_t i = 0; i < warmup; ++i) {
            auto w = cc->EvalMult(ctx, cty);
            (void)w;
        }
    }

    // Correctness is checked per mode, not once at the end: a mode that is fast
    // because it is wrong would otherwise be reported as a win.
    for (int m = 0; m < 3; ++m) {
        SetMode(m);
        auto prod = cc->EvalMult(ctx, cty);
        Plaintext out;
        cc->Decrypt(kp.secretKey, prod, &out);
        out->SetLength(8);
        const auto& vals = out->GetCKKSPackedValue();
        double e = 0.0;
        for (size_t i = 0; i < 8; ++i) e = std::max(e, std::abs(vals[i].real() - 0.25));
        R.maxerr[m] = e;
        if (e > 1e-6)
            std::cout << "    !! " << kModeName[m] << " CORRECTNESS FAILURE err=" << e << "\n";
    }

    for (int m = 0; m < 3; ++m) R.t[m].reserve(reps);

    // Interleaved: all three modes sampled at the same thermal moment.
    for (uint32_t i = 0; i < reps; ++i) {
        for (int m = 0; m < 3; ++m) {
            SetMode(m);
            auto t0 = clk::now();
            auto r = cc->EvalMult(ctx, cty);
            double dt = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            if (r->GetElements().empty()) std::cout << "";  // keep result alive
            R.t[m].push_back(dt);
        }
    }

    R.ok = true;
    return R;
}

void ReportPair(const Result& R, int a, int b) {
    std::vector<double> d(R.t[a].size());
    for (size_t i = 0; i < d.size(); ++i) d[i] = R.t[b][i] - R.t[a][i];
    const double md = Median(d);
    auto [lo, hi] = BootstrapCI(d);
    const double base = Median(R.t[a]);
    const double pct = (base > 0) ? 100.0 * md / base : 0.0;
    const bool sig = (lo > 0.0) || (hi < 0.0);

    std::cout << "    " << std::left << std::setw(16)
              << (std::string(kModeName[b]) + " - " + kModeName[a])
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(9) << md << " ms  "
              << "[" << std::setw(7) << lo << "," << std::setw(7) << hi << "]  "
              << std::setw(7) << std::setprecision(1) << pct << "%   "
              << (sig ? (md < 0 ? "FASTER" : "SLOWER") : "not significant") << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    uint32_t reps = 50, warmup = 10;
    std::vector<std::pair<uint32_t, uint32_t>> grid;

    if (argc >= 3) {
        grid.emplace_back((uint32_t)std::stoul(argv[1]), (uint32_t)std::stoul(argv[2]));
        if (argc >= 4) reps = (uint32_t)std::stoul(argv[3]);
    } else {
        grid = {{5, 32768}, {10, 32768}, {5, 65536}, {10, 65536}, {15, 65536}};
    }

    std::cout << "======================================================\n"
              << "[*] CKKS EvalMult -- paired-difference benchmark\n"
              << "    modes interleaved per rep in one process; reported\n"
              << "    statistic is median(delta) with a 95% bootstrap CI.\n"
              << "    reps=" << reps << " warmup=" << warmup << "/mode\n"
              << "======================================================\n\n";

    for (auto& [depth, ring] : grid) {
        std::cout << "depth=" << depth << " ring=" << ring << std::flush;
        Result R = RunConfig(depth, ring, reps, warmup);
        if (!R.ok) continue;
        std::cout << " towers=" << R.towers << "\n";

        std::cout << "    " << std::left << std::setw(16) << "absolute"
                  << std::right << std::fixed << std::setprecision(2);
        for (int m = 0; m < 3; ++m)
            std::cout << kModeName[m] << "=" << Median(R.t[m]) << "ms  ";
        std::cout << "\n";

        std::cout << "    " << std::left << std::setw(16) << "IQR"
                  << std::right << std::setprecision(2);
        for (int m = 0; m < 3; ++m)
            std::cout << kModeName[m] << "="
                      << (Quantile(R.t[m], 0.75) - Quantile(R.t[m], 0.25)) << "ms  ";
        std::cout << "\n";

        ReportPair(R, MODE_CPU, MODE_GPU);
        ReportPair(R, MODE_CPU, MODE_KS);
        ReportPair(R, MODE_GPU, MODE_KS);
        std::cout << "\n";
    }

    std::cout << "A delta is meaningful only when its CI excludes zero. The\n"
              << "absolute medians are shown for scale; do not compare them\n"
              << "across separate runs of this binary.\n";
    return 0;
}

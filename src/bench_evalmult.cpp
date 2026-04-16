#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <numeric>
#include <random>

using namespace std;
using Clock = chrono::high_resolution_clock;

extern "C" void gpu_rns_mult_wrapper(const uint64_t* a, const uint64_t* b, uint64_t* res, uint64_t q, uint64_t mu_hi, uint32_t ring, uint32_t tower_idx);
extern "C" void gpu_synchronize_all();

double benchmark_gpu(uint32_t towers, uint32_t ring, int iters) {
    mt19937_64 rng(42);
    vector<uint64_t> moduli(towers);
    vector<uint64_t> mu_hi(towers);
    vector<vector<uint64_t>> ha(towers, vector<uint64_t>(ring));
    vector<vector<uint64_t>> hb(towers, vector<uint64_t>(ring));
    vector<vector<uint64_t>> hres(towers, vector<uint64_t>(ring));

    uint64_t base_prime = (1ULL << 60) - 57;
    for (uint32_t i = 0; i < towers; ++i) {
        moduli[i] = base_prime - i * 2;
        unsigned __int128 mu_128 = ((unsigned __int128)1 << 64) / moduli[i];
        mu_hi[i] = (uint64_t)mu_128;
        for (uint32_t j = 0; j < ring; ++j) {
            ha[i][j] = rng() % moduli[i];
            hb[i][j] = rng() % moduli[i];
        }
    }

    for (uint32_t i = 0; i < towers; ++i) {
        gpu_rns_mult_wrapper(ha[i].data(), hb[i].data(), hres[i].data(), moduli[i], mu_hi[i], ring, i);
    }
    gpu_synchronize_all();

    auto t0 = Clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        for (uint32_t i = 0; i < towers; ++i) {
            gpu_rns_mult_wrapper(ha[i].data(), hb[i].data(), hres[i].data(), moduli[i], mu_hi[i], ring, i);
        }
        gpu_synchronize_all();
    }
    auto t1 = Clock::now();
    return chrono::duration<double, milli>(t1 - t0).count() / iters;
}

int main() {
    cout << "========================================\n";
    cout << " CUDA HAL Benchmark (Encapsulated Pool)\n";
    cout << "========================================\n";
    struct TestCase { uint32_t towers; uint32_t ring; const char* label; };
    TestCase cases[] = {
        {16, 16384,  "16 towers, N=16k"},
        {16, 65536,  "16 towers, N=64k"},
    };
    for (auto& tc : cases) {
        double gpu_ms = benchmark_gpu(tc.towers, tc.ring, 20);
        printf("  %-25s  GPU: %6.2f ms\n", tc.label, gpu_ms);
    }
    return 0;
}

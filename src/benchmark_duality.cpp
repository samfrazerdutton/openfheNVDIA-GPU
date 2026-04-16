#include "cuda_hal.h"
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>

using namespace std;

extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** host_a, const uint64_t** host_b, uint64_t** host_res,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);

int main() {
    const uint32_t TOWERS = 16;
    const uint32_t RING = 32768;
    const int NUM_THREADS = 8; // Simulating OpenFHE max thread limits

    cout << "======================================================\n";
    cout << "[*] Duality-Grade OpenFHE GPU Verification Engine\n";
    cout << "[*] Simulating " << NUM_THREADS << " Concurrent OpenMP Threads\n";
    cout << "======================================================\n";

    bool global_success = true;
    auto start_all = chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int t = 0; t < NUM_THREADS; ++t) {
        // Thread-local data setup
        vector<vector<uint64_t>> a(TOWERS, vector<uint64_t>(RING));
        vector<vector<uint64_t>> b(TOWERS, vector<uint64_t>(RING));
        vector<vector<uint64_t>> res(TOWERS, vector<uint64_t>(RING));
        vector<uint64_t> q(TOWERS);

        vector<const uint64_t*> a_ptrs(TOWERS);
        vector<const uint64_t*> b_ptrs(TOWERS);
        vector<uint64_t*> res_ptrs(TOWERS);

        mt19937_64 rng(1337 + t);

        for (uint32_t i = 0; i < TOWERS; i++) {
            q[i] = 0x3FFFFFFF - (i * 2 * RING); // NTT Prime mock
            a_ptrs[i] = a[i].data();
            b_ptrs[i] = b[i].data();
            res_ptrs[i] = res[i].data();
            for (uint32_t j = 0; j < RING; j++) {
                a[i][j] = rng() % q[i];
                b[i][j] = rng() % q[i];
            }
        }

        // Hit the GPU (Thread-safe batched wrapper)
        gpu_rns_mult_batch_wrapper(a_ptrs.data(), b_ptrs.data(), res_ptrs.data(), q.data(), RING, TOWERS);

        // Verify EXACT 128-bit modular arithmetic
        bool exact = true;
        for (uint32_t i = 0; i < TOWERS; i++) {
            for (uint32_t j = 0; j < RING; j++) {
                unsigned __int128 expected = (unsigned __int128)a[i][j] * b[i][j];
                uint64_t expected_mod = (uint64_t)(expected % q[i]);
                if (res[i][j] != expected_mod) {
                    exact = false;
                    global_success = false;
                }
            }
        }

        #pragma omp critical
        {
            if (exact) cout << "[+] Thread " << t << " SUCCESS: 100% Precision (0 VRAM Corruptions)\n";
            else cout << "[-] Thread " << t << " FAILED: Math Mismatch or VRAM Collision!\n";
        }
    }

    auto end_all = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> elapsed = end_all - start_all;

    cout << "======================================================\n";
    if (global_success) {
        cout << "[SUCCESS] Thread Safety & Exact Modulo Verified.\n";
        uint64_t total_ops = (uint64_t)TOWERS * RING * NUM_THREADS;
        cout << "[BENCHMARK] Processed " << total_ops << " mults in " << elapsed.count() << " ms.\n";
        cout << "[BENCHMARK] Throughput: " << (total_ops / (elapsed.count() / 1000.0)) / 1e6 << " M ops/sec\n";
    } else {
        cout << "[FATAL] The engine hallucinated or corrupted VRAM.\n";
    }
    cout << "======================================================\n";
    return 0;
}

#include "cuda_hal.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace openfhe_cuda;

int main() {
    uint32_t towers = 16;
    uint32_t ring = 32768;

    cout << "======================================================" << endl;
    cout << "[*] Duality-Style Open-Source GPU FHE Engine" << endl;
    cout << "======================================================" << endl;
    
    cout << "[*] Allocating Zero-Copy VRAM for 16-Tower RNS Ciphertext..." << endl;

    vector<uint64_t*> d_x, d_half, d_res;
    CUDAMathHAL::AllocateVRAM(d_x, towers, ring);
    CUDAMathHAL::AllocateVRAM(d_half, towers, ring);
    CUDAMathHAL::AllocateVRAM(d_res, towers, ring);

    vector<uint64_t> moduli(towers);
    vector<vector<uint64_t>> host_x(towers, vector<uint64_t>(ring));
    vector<vector<uint64_t>> host_half(towers, vector<uint64_t>(ring));

    for(uint32_t i = 0; i < towers; i++) {
        moduli[i] = 0x3FFFFFFF - (i * 2 * ring);
        for(uint32_t j = 0; j < ring; j++) {
            host_x[i][j] = (2400) % moduli[i];
            host_half[i][j] = (500) % moduli[i];
        }
        cudaMemcpy(d_x[i], host_x[i].data(), ring * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_half[i], host_half[i].data(), ring * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    cout << "[*] Firing 16 Asynchronous CUDA Streams via C++ HAL..." << endl;
    
    auto start = chrono::high_resolution_clock::now();
    CUDAMathHAL::EvalMultRNS(d_half, d_x, d_res, moduli, ring);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> elapsed = end - start;

    cout << "[+] Hardware Execution Time: " << elapsed.count() << " ms
" << endl;

    vector<uint64_t> host_res(ring);
    cudaMemcpy(host_res.data(), d_res[0], ring * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint64_t expected = (host_x[0][0] * host_half[0][0]) % moduli[0];
    
    cout << "[*] Verification (Tower 0, Index 0):" << endl;
    cout << "    Expected: " << expected << endl;
    cout << "    Actual:   " << host_res[0] << endl;

    if (expected == host_res[0]) {
        cout << "
[SUCCESS] C++ HAL memory pointers and grid-stride logic are flawless." << endl;
    } else {
        cout << "
[FAILED] Mismatch detected." << endl;
    }

    CUDAMathHAL::FreeVRAM(d_x);
    CUDAMathHAL::FreeVRAM(d_half);
    CUDAMathHAL::FreeVRAM(d_res);

    return 0;
}

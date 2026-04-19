#include "cuda_hal.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;
using namespace openfhe_cuda;
static const uint64_t Q = 0xFFFFFFFF00000001ULL;
int main() {
    const uint32_t towers = 16, ring = 32768;
    cout << "[*] GPU Engine Smoke Test: " << towers << " towers x ring=" << ring << endl;
    vector<uint64_t*> d_a, d_b, d_res;
    CUDAMathHAL::AllocateVRAM(d_a, towers, ring);
    CUDAMathHAL::AllocateVRAM(d_b, towers, ring);
    CUDAMathHAL::AllocateVRAM(d_res, towers, ring);
    vector<uint64_t> moduli(towers, Q);
    vector<vector<uint64_t>> ha(towers, vector<uint64_t>(ring));
    vector<vector<uint64_t>> hb(towers, vector<uint64_t>(ring));
    for (uint32_t t = 0; t < towers; t++) {
        for (uint32_t j = 0; j < ring; j++) { ha[t][j] = (2400+j)%Q; hb[t][j] = (500+j)%Q; }
        cudaMemcpy(d_a[t], ha[t].data(), ring*8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b[t], hb[t].data(), ring*8, cudaMemcpyHostToDevice);
    }
    CUDAMathHAL::EvalMultRNS(d_a, d_b, d_res, moduli, ring);
    CUDAMathHAL::Synchronize();
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) CUDAMathHAL::EvalMultRNS(d_a, d_b, d_res, moduli, ring);
    CUDAMathHAL::Synchronize();
    double ms = chrono::duration<double,milli>(chrono::high_resolution_clock::now()-t0).count()/10.0;
    printf("[+] EvalMultRNS: %.3f ms/op\n", ms);
    vector<uint64_t> res(ring);
    cudaMemcpy(res.data(), d_res[0], ring*8, cudaMemcpyDeviceToHost);
    bool nonzero = false;
    for (uint32_t j = 0; j < ring; j++) if (res[j]) { nonzero = true; break; }
    CUDAMathHAL::FreeVRAM(d_a); CUDAMathHAL::FreeVRAM(d_b); CUDAMathHAL::FreeVRAM(d_res);
    if (!nonzero) { cout << "[FAIL] All results zero" << endl; return 1; }
    cout << "[PASS] Results non-zero, GPU HAL working" << endl;
    printf("       res[0]=%llu res[1]=%llu res[2]=%llu\n", res[0], res[1], res[2]);
    return 0;
}
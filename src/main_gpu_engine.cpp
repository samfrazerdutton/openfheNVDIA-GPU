#include "cuda_hal.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
using namespace std;
using namespace openfhe_cuda;
int main() {
    uint32_t towers = 16, ring = 32768;
    CUDAMathHAL::InitStreams(towers);
    vector<uint64_t*> d_x, d_res;
    CUDAMathHAL::AllocateVRAM(d_x, towers, ring);
    CUDAMathHAL::AllocateVRAM(d_res, towers, ring);
    vector<uint64_t> moduli(towers);
    vector<unsigned __int128> mu(towers);
    vector<vector<uint64_t>> host_x(towers, vector<uint64_t>(ring, 0));
    // Whiteboard X values scaled by 1000 to represent fixed-point float logic
    uint64_t wb_x[4] = {2400, 1200, 3400, 5600};
    for(uint32_t i = 0; i < towers; i++) {
        moduli[i] = 0x3FFFFFFF - (i * 2 * ring);
        mu[i] = ((unsigned __int128)1 << 64) / moduli[i];
        for(uint32_t j = 0; j < 4; j++) host_x[i][j] = wb_x[j];
        cudaMemcpy(d_x[i], host_x[i].data(), ring * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    cout << "======================================================" << endl;
    cout << "[*] Whiteboard Array Math Verification (X^2)" << endl;
    cout << "======================================================" << endl;
    CUDAMathHAL::EvalMultRNS(d_x, d_x, d_res, moduli, ring);
    CUDAMathHAL::Synchronize();
    vector<uint64_t> host_res(ring);
    cudaMemcpy(host_res.data(), d_res[0], ring * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint32_t errors = 0;
    for(int i = 0; i < 4; i++) {
        uint64_t expected = (wb_x[i] * wb_x[i]) % moduli[0];
        cout << "    Whiteboard X[" << i << "] = " << wb_x[i];
        cout << " | Expected X^2: " << expected;
        cout << " | GPU Output: " << host_res[i] << endl;
        if(expected != host_res[i]) errors++;
    }
    if (errors == 0) {
        cout << "\n[SUCCESS] The GPU correctly evaluated the whiteboard arrays!" << endl;
    } else {
        cout << "\n[FAILED] Math mismatch detected." << endl;
    }
    CUDAMathHAL::FreeVRAM(d_x); CUDAMathHAL::FreeVRAM(d_res);
    CUDAMathHAL::DestroyStreams();
    return 0;
}

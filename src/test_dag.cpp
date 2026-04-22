#include "fhe_compiler.h"
#include "phantom_registry.h"
#include "cuda_hal.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Testing Phase 2 DAG Compiler ===" << std::endl;

    FheCompiler compiler;
    PhantomRegistry phantom;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Mock parameters for a small polynomial
    uint32_t num_towers = 1;
    uint32_t ring_dim = 4096;
    size_t size_bytes = num_towers * ring_dim * sizeof(uint64_t);

    // 1. Allocate Phantom VRAM (No actual CPU memory used for intermediate data)
    void* dummy_a = phantom.AllocatePhantom(size_bytes);
    void* dummy_b = phantom.AllocatePhantom(size_bytes);
    void* dummy_res = phantom.AllocatePhantom(size_bytes);

    void* vram_a = phantom.GetVramPointer(dummy_a);
    void* vram_b = phantom.GetVramPointer(dummy_b);
    void* vram_res = phantom.GetVramPointer(dummy_res);

    // Host arrays to verify
    std::vector<uint64_t> host_a(num_towers * ring_dim, 2);
    std::vector<uint64_t> host_b(num_towers * ring_dim, 3);
    std::vector<uint64_t> host_res(num_towers * ring_dim, 0);

    // 2. Build the DAG
    // LOAD nodes
    DagNode* load_a = compiler.CreateNode(FheOpcode::LOAD, nullptr, nullptr, vram_a, host_a.data(), size_bytes, num_towers, ring_dim);
    DagNode* load_b = compiler.CreateNode(FheOpcode::LOAD, nullptr, nullptr, vram_b, host_b.data(), size_bytes, num_towers, ring_dim);

    // MULT node (depends on loads)
    DagNode* mult_op = compiler.CreateNode(FheOpcode::MULT_RNS, load_a, load_b, vram_res, nullptr, size_bytes, num_towers, ring_dim);

    // STORE node (depends on mult)
    DagNode* store_op = compiler.CreateNode(FheOpcode::STORE, mult_op, nullptr, vram_res, host_res.data(), size_bytes, num_towers, ring_dim);

    // 3. Compile and Execute
    compiler.CompileToCudaGraph(stream);
    std::cout << "Executing Graph..." << std::endl;
    compiler.ExecuteGraph(stream);
    
    // Sync to wait for the STORE operation to finish
    cudaStreamSynchronize(stream);

    // 4. Verify
    bool pass = true;
    for(int i = 0; i < ring_dim; i++) {
        if(host_res[i] != 6) { // 2 * 3 = 6
            pass = false;
            break;
        }
    }

    if(pass) std::cout << "[PASS] DAG Execution Successful. Result matched." << std::endl;
    else std::cout << "[FAIL] DAG Execution failed. Data mismatch." << std::endl;

    // Cleanup
    phantom.FreePhantom(dummy_a);
    phantom.FreePhantom(dummy_b);
    phantom.FreePhantom(dummy_res);
    cudaStreamDestroy(stream);

    return 0;
}

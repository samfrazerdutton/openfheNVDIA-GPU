#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdint>

// The supported operations in our FHE circuit
enum class FheOpcode {
    LOAD,       // Move data from CPU -> VRAM
    NTT_FWD,    // Forward NTT
    MULT_RNS,   // Pointwise Multiply
    ADD_RNS,    // Pointwise Add
    NTT_INV,    // Inverse NTT
    STORE       // Move data from VRAM -> CPU
};

// A node in the Directed Acyclic Graph
struct DagNode {
    FheOpcode op;
    int node_id;
    DagNode* left_operand;
    DagNode* right_operand;
    void* vram_ptr;       // Device memory location
    void* host_ptr;       // Host memory location (for LOAD/STORE)
    size_t size_bytes;    // Memory footprint
    uint32_t num_towers;  // RNS parameters
    uint32_t ring_dim;
};

class FheCompiler {
private:
    std::vector<DagNode*> node_pool;
    int next_id = 0;
    
    cudaGraph_t cuda_graph;
    cudaGraphExec_t graph_exec;
    bool graph_compiled;

public:
    FheCompiler();
    ~FheCompiler();

    // Create a node in the execution graph
    DagNode* CreateNode(FheOpcode op, DagNode* left, DagNode* right, 
                        void* vram_ptr, void* host_ptr, size_t size_bytes, 
                        uint32_t num_towers, uint32_t ring_dim);

    // Records the execution sequence into a CUDA Graph via Stream Capture
    void CompileToCudaGraph(cudaStream_t stream);

    // Fires the entire FHE circuit in one GPU call
    void ExecuteGraph(cudaStream_t stream);

    // Clears the graph for the next circuit evaluation
    void Reset();

private:
    // Recursively visits nodes to build the execution trace
    void TraverseAndRecord(DagNode* node, cudaStream_t stream);
};

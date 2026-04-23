#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>

enum class FheOpcode { LOAD, NTT_FWD, MULT_RNS, ADD_RNS, NTT_INV, STORE };

struct DagNode {
    FheOpcode op;
    int node_id;
    DagNode* left_operand;
    DagNode* right_operand;
    void* vram_ptr;
    void* host_ptr;
    size_t size_bytes;
    uint32_t num_towers;
    uint32_t ring_dim;
    uint64_t modulus;  // per-tower NTT prime
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
    DagNode* CreateNode(FheOpcode op, DagNode* left, DagNode* right,
                        void* vram_ptr, void* host_ptr, size_t size_bytes,
                        uint32_t num_towers, uint32_t ring_dim, uint64_t modulus = 0);
    void CompileToCudaGraph(cudaStream_t stream);
    void ExecuteGraph(cudaStream_t stream);
    void Reset();
private:
    void TraverseAndRecord(DagNode* node, cudaStream_t stream);
};

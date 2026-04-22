#pragma once
#include "fhe_compiler.h"
#include <unordered_map>
#include <cuda_runtime.h>

class GlobalDAG {
public:
    static FheCompiler compiler;
    static cudaStream_t stream;
    
    // Maps a CPU pointer (OpenFHE's array) to its latest DAG Node
    static std::unordered_map<void*, DagNode*> node_registry;
    
    // Toggles whether we are building a graph or eagerly executing
    static bool is_capturing;

    static void Init();
    static void ExecuteAndSync();
    
    // Resolves a CPU pointer to a node. If it doesn't exist, injects a LOAD node.
    static DagNode* GetOrLoadNode(void* host_ptr, size_t size_bytes, uint32_t num_towers, uint32_t ring_dim);
};

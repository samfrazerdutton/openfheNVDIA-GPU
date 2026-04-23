#pragma once
#include "fhe_compiler.h"
#include <unordered_map>
#include <cuda_runtime.h>

class GlobalDAG {
public:
    static FheCompiler compiler;
    static cudaStream_t stream;
    static std::unordered_map<void*, DagNode*> node_registry;
    static bool is_capturing;
    static void Init();
    static void ExecuteAndSync();
    static DagNode* GetOrLoadNode(void* host_ptr, size_t size_bytes,
                                   uint32_t num_towers, uint32_t ring_dim,
                                   uint64_t modulus = 0);
};

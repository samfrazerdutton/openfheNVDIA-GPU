#include "global_dag.h"
#include <iostream>

FheCompiler GlobalDAG::compiler;
cudaStream_t GlobalDAG::stream = nullptr;
std::unordered_map<void*, DagNode*> GlobalDAG::node_registry;
bool GlobalDAG::is_capturing = false;

void GlobalDAG::Init() {
    if (stream == nullptr) cudaStreamCreate(&stream);
}

DagNode* GlobalDAG::GetOrLoadNode(void* host_ptr, size_t size_bytes,
                                    uint32_t num_towers, uint32_t ring_dim,
                                    uint64_t modulus) {
    auto it = node_registry.find(host_ptr);
    if (it != node_registry.end()) return it->second;
    void* vram_ptr = nullptr;
    cudaMalloc(&vram_ptr, size_bytes);
    DagNode* node = compiler.CreateNode(FheOpcode::LOAD, nullptr, nullptr,
                                         vram_ptr, host_ptr, size_bytes,
                                         num_towers, ring_dim, modulus);
    node_registry[host_ptr] = node;
    return node;
}

void GlobalDAG::ExecuteAndSync() {
    if (node_registry.empty()) return;
    std::vector<void*> owned_vram;
    for (auto const& [host_ptr, node] : node_registry) {
        owned_vram.push_back(node->vram_ptr);
        compiler.CreateNode(FheOpcode::STORE, node, nullptr,
                             node->vram_ptr, host_ptr, node->size_bytes,
                             node->num_towers, node->ring_dim, node->modulus);
    }
    compiler.CompileToCudaGraph(stream);
    compiler.ExecuteGraph(stream);
    cudaStreamSynchronize(stream);
    for (void* ptr : owned_vram) cudaFree(ptr);
    node_registry.clear();
    compiler.Reset();  // was missing — caused node pool leak
}

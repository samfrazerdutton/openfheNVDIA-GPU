#include "global_dag.h"

FheCompiler GlobalDAG::compiler;
cudaStream_t GlobalDAG::stream;
std::unordered_map<void*, DagNode*> GlobalDAG::node_registry;
bool GlobalDAG::is_capturing = false;

void GlobalDAG::Init() {
    cudaStreamCreate(&stream);
}

DagNode* GlobalDAG::GetOrLoadNode(void* host_ptr, size_t size_bytes, uint32_t num_towers, uint32_t ring_dim) {
    if (node_registry.find(host_ptr) != node_registry.end()) {
        return node_registry[host_ptr]; // Node already exists in VRAM
    }
    
    // Node doesn't exist. We must LOAD it from the CPU to the GPU.
    void* vram_ptr;
    cudaMalloc(&vram_ptr, size_bytes);
    
    DagNode* load_node = compiler.CreateNode(FheOpcode::LOAD, nullptr, nullptr, vram_ptr, host_ptr, size_bytes, num_towers, ring_dim);
    node_registry[host_ptr] = load_node;
    
    return load_node;
}

void GlobalDAG::ExecuteAndSync() {
    if (node_registry.empty()) return;

    // Inject STORE nodes for all active pointers so OpenFHE gets its data back
    for (auto const& [host_ptr, node] : node_registry) {
        compiler.CreateNode(FheOpcode::STORE, node, nullptr, node->vram_ptr, host_ptr, node->size_bytes, node->num_towers, node->ring_dim);
    }

    compiler.CompileToCudaGraph(stream);
    compiler.ExecuteGraph(stream);
    cudaStreamSynchronize(stream);
    
    // Cleanup VRAM and Reset
    for (auto const& [host_ptr, node] : node_registry) {
        cudaFree(node->vram_ptr);
    }
    node_registry.clear();
    compiler.Reset();
}

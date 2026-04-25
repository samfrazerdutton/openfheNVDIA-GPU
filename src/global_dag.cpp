#include "global_dag.h"
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

FheCompiler GlobalDAG::compiler;
cudaStream_t GlobalDAG::stream = nullptr;
std::unordered_map<void*, DagNode*> GlobalDAG::node_registry;
bool GlobalDAG::is_capturing = false;

static std::mutex g_dag_mu;

void GlobalDAG::Init() {
    std::lock_guard<std::mutex> lk(g_dag_mu);
    if (stream != nullptr) return;
    cudaError_t e = cudaStreamCreate(&stream);
    if (e != cudaSuccess)
        throw std::runtime_error(
            std::string("[GlobalDAG] cudaStreamCreate: ") +
            cudaGetErrorString(e));
}

DagNode* GlobalDAG::GetOrLoadNode(void* host_ptr, size_t size_bytes,
                                   uint32_t num_towers, uint32_t ring_dim,
                                   uint64_t modulus) {
    std::lock_guard<std::mutex> lk(g_dag_mu);
    auto it = node_registry.find(host_ptr);
    if (it != node_registry.end()) return it->second;
    void*       vram_ptr = nullptr;
    cudaError_t e        = cudaMalloc(&vram_ptr, size_bytes);
    if (e != cudaSuccess)
        throw std::runtime_error(
            std::string("[GlobalDAG] cudaMalloc: ") +
            cudaGetErrorString(e));
    DagNode* node = compiler.CreateNode(
        FheOpcode::LOAD, nullptr, nullptr,
        vram_ptr, host_ptr, size_bytes,
        num_towers, ring_dim, modulus);
    node_registry[host_ptr] = node;
    return node;
}

void GlobalDAG::ExecuteAndSync() {
    std::lock_guard<std::mutex> lk(g_dag_mu);
    if (node_registry.empty()) return;
    std::vector<void*> owned;
    owned.reserve(node_registry.size());
    for (auto const& [hp, node] : node_registry) {
        owned.push_back(node->vram_ptr);
        compiler.CreateNode(FheOpcode::STORE, node, nullptr,
            node->vram_ptr, hp, node->size_bytes,
            node->num_towers, node->ring_dim, node->modulus);
    }
    compiler.CompileToCudaGraph(stream);
    compiler.ExecuteGraph(stream);
    cudaStreamSynchronize(stream);   // sync BEFORE freeing VRAM
    for (void* p : owned) cudaFree(p);
    node_registry.clear();
    compiler.Reset();
}

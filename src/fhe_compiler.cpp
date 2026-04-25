#include "fhe_compiler.h"
#include <iostream>
#include <stdexcept>
#include <string>

extern "C" void LaunchRNSMultMontgomery(
    const uint64_t* a, const uint64_t* b, uint64_t* r,
    uint64_t q, uint64_t q_inv, uint64_t R2,
    uint32_t n, cudaStream_t s);

static uint64_t calc_q_inv(uint64_t q) {
    uint64_t x = q;
    for (int i = 0; i < 5; ++i) x *= 2 - q * x;
    return -x;
}
static uint64_t calc_R2(uint64_t q) {
    unsigned __int128 R = ((unsigned __int128)1 << 64) %
                          (unsigned __int128)q;
    return (uint64_t)((R * R) % (unsigned __int128)q);
}

FheCompiler::FheCompiler()
    : graph_compiled(false), cuda_graph(nullptr), graph_exec(nullptr) {}

FheCompiler::~FheCompiler() { Reset(); }

DagNode* FheCompiler::CreateNode(
    FheOpcode op, DagNode* left, DagNode* right,
    void* vram_ptr, void* host_ptr, size_t size_bytes,
    uint32_t num_towers, uint32_t ring_dim, uint64_t modulus)
{
    DagNode* node = new DagNode{
        op, next_id++, left, right,
        vram_ptr, host_ptr, size_bytes,
        num_towers, ring_dim, modulus};
    node_pool.push_back(node);
    return node;
}

void FheCompiler::TraverseAndRecord(DagNode* node, cudaStream_t stream) {
    if (!node) return;
    TraverseAndRecord(node->left_operand,  stream);
    TraverseAndRecord(node->right_operand, stream);
    switch (node->op) {
        case FheOpcode::LOAD:
            if (!node->host_ptr || !node->vram_ptr)
                throw std::runtime_error("[FheCompiler] LOAD null ptr");
            cudaMemcpyAsync(node->vram_ptr, node->host_ptr,
                node->size_bytes, cudaMemcpyHostToDevice, stream);
            break;
        case FheOpcode::MULT_RNS: {
            if (!node->left_operand || !node->right_operand)
                throw std::runtime_error("[FheCompiler] MULT_RNS missing operand");
            uint64_t q = node->modulus ? node->modulus : 0xFFFFFFFF00000001ULL;
            LaunchRNSMultMontgomery(
                (const uint64_t*)node->left_operand->vram_ptr,
                (const uint64_t*)node->right_operand->vram_ptr,
                (uint64_t*)node->vram_ptr,
                q, calc_q_inv(q), calc_R2(q), node->ring_dim, stream);
            break;
        }
        case FheOpcode::STORE:
            if (!node->host_ptr || !node->vram_ptr)
                throw std::runtime_error("[FheCompiler] STORE null ptr");
            cudaMemcpyAsync(node->host_ptr, node->vram_ptr,
                node->size_bytes, cudaMemcpyDeviceToHost, stream);
            break;
        default:
            std::cerr << "[FheCompiler] unknown opcode node "
                      << node->node_id << "\n";
    }
}

void FheCompiler::CompileToCudaGraph(cudaStream_t stream) {
    if (node_pool.empty()) return;
    cudaError_t e;
    e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (e != cudaSuccess)
        throw std::runtime_error(
            std::string("[FheCompiler] BeginCapture: ") +
            cudaGetErrorString(e));
    TraverseAndRecord(node_pool.back(), stream);
    e = cudaStreamEndCapture(stream, &cuda_graph);
    if (e != cudaSuccess)
        throw std::runtime_error(
            std::string("[FheCompiler] EndCapture: ") +
            cudaGetErrorString(e));
    e = cudaGraphInstantiate(&graph_exec, cuda_graph, NULL, NULL, 0);
    if (e != cudaSuccess)
        throw std::runtime_error(
            std::string("[FheCompiler] Instantiate: ") +
            cudaGetErrorString(e));
    graph_compiled = true;
    std::cout << "[FheCompiler] compiled " << node_pool.size()
              << " nodes\n";
}

void FheCompiler::ExecuteGraph(cudaStream_t stream) {
    if (!graph_compiled) {
        std::cerr << "[FheCompiler] not compiled\n"; return;
    }
    cudaError_t e = cudaGraphLaunch(graph_exec, stream);
    if (e != cudaSuccess)
        throw std::runtime_error(
            std::string("[FheCompiler] Launch: ") +
            cudaGetErrorString(e));
}

void FheCompiler::Reset() {
    for (DagNode* n : node_pool) delete n;
    node_pool.clear();
    next_id = 0;
    if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
    if (cuda_graph) { cudaGraphDestroy(cuda_graph);     cuda_graph = nullptr; }
    graph_compiled = false;
}

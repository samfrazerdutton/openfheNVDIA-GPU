#pragma once

#include <vector>
#include <memory>

namespace fhe_compiler {

enum class FHEOpcode {
    ADD,
    MULT,
    MOD_REDUCE,
    KEY_SWITCH,
    NTT_FORWARD,
    NTT_INVERSE
};

/**
 * @brief Phase 2: Directed Acyclic Graph (DAG) Node
 * Represents a delayed FHE operation. Allows the system to compile
 * a massive FHE circuit and execute it entirely in VRAM without H2D syncs.
 */
struct DAGNode {
    FHEOpcode operation;
    std::vector<int> input_dependencies; 
    int output_id;
    bool requires_sync; 
};

class ExecutionGraph {
private:
    std::vector<DAGNode> instructions;

public:
    void QueueOperation(FHEOpcode op, const std::vector<int>& inputs, int out);
    
    // Analyzes the graph, allocates VRAM, and fires the sequence to the GPU
    void CompileAndExecute(class ShadowRegistry& reg, class StreamPool& pool);
};

} // namespace fhe_compiler

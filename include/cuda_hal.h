#pragma once
#include <vector>
#include <cstdint>

namespace openfhe_cuda {

    class CUDAMathHAL {
    public:
        // Allocates continuous VRAM for the RNS Towers
        static void AllocateVRAM(std::vector<uint64_t*>& d_ptrs, uint32_t towers, uint32_t ring_degree);
        
        // Frees the VRAM safely
        static void FreeVRAM(std::vector<uint64_t*>& d_ptrs);

        // Intercepts OpenFHE EvalMult and evaluates across 16 parallel CUDA streams
        static void EvalMultRNS(
            const std::vector<uint64_t*>& d_a, 
            const std::vector<uint64_t*>& d_b, 
            std::vector<uint64_t*>& d_res, 
            const std::vector<uint64_t>& moduli, 
            uint32_t ring_degree
        );
    };

} // namespace openfhe_cuda

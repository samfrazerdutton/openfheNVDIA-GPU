#pragma once
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace openfhe_cuda {

class CUDAMathHAL {
public:
    static void AllocateVRAM(std::vector<uint64_t*>& ptrs,
                              uint32_t towers, uint32_t ring);
    static void FreeVRAM(std::vector<uint64_t*>& ptrs);
    static void EvalMultRNS(const std::vector<uint64_t*>& d_a,
                             const std::vector<uint64_t*>& d_b,
                             std::vector<uint64_t*>&        d_res,
                             const std::vector<uint64_t>&   moduli,
                             uint32_t ring);
    static void InitStreams(uint32_t n);
    static void DestroyStreams();
    static void Synchronize();
};

} // namespace openfhe_cuda

extern "C" {
void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);

void gpu_poly_mult_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);

void gpu_rns_mult_wrapper(
    const uint64_t* ha, const uint64_t* hb, uint64_t* hr,
    uint64_t q, uint64_t mu_hi, uint32_t ring, uint32_t tower_idx);

void gpu_synchronize_all();
void gpu_sync_all_to_host();
}

#pragma once
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace openfhe_cuda {

// High-level HAL used by the benchmark and OpenFHE adapter.
class CUDAMathHAL {
public:
    // Allocate / free persistent VRAM blocks (one per RNS tower).
    static void AllocateVRAM(std::vector<uint64_t*>& ptrs,
                             uint32_t towers, uint32_t ring);
    static void FreeVRAM(std::vector<uint64_t*>& ptrs);

    // Pointwise RNS multiply: d_res[i][j] = d_a[i][j] * d_b[i][j] mod moduli[i]
    // Data must already be on device (use AllocateVRAM + cudaMemcpy).
    static void EvalMultRNS(const std::vector<uint64_t*>& d_a,
                            const std::vector<uint64_t*>& d_b,
                            std::vector<uint64_t*>&       d_res,
                            const std::vector<uint64_t>&  moduli,
                            uint32_t ring);

    static void InitStreams(uint32_t n);
    static void DestroyStreams();
    static void Synchronize();
};

} // namespace openfhe_cuda

// ── C-linkage wrappers (called from OpenFHE patch and benchmarks) ────────────

extern "C" {

// Pointwise RNS multiply: host arrays, PCIe upload per call.
// Each ha[i]/hb[i]/hr[i] is a host pointer to ring uint64_t values.
// Results are written back to hr[i] synchronously.
void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);

// Full negacyclic NTT polynomial multiply: NTT(a) * NTT(b) → INTT → hr.
// ha[i] and hb[i] are in coefficient domain; hr[i] receives the product.
void gpu_poly_mult_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);

// Single-tower RNS multiply (legacy shim for bench_evalmult).
void gpu_rns_mult_wrapper(
    const uint64_t* ha, const uint64_t* hb, uint64_t* hr,
    uint64_t q, uint64_t mu_hi, uint32_t ring, uint32_t tower_idx);

void gpu_synchronize_all();

} // extern "C"

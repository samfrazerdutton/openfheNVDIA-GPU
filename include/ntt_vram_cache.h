#pragma once
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <vector>

namespace openfhe_cuda {

struct NTTEntry {
    uint64_t* d_ntt_ptr;   // NTT-domain data in VRAM
    uint32_t  ring;
    uint64_t  modulus;
    bool      ntt_done;    // true = NTT-domain, false = coeff-domain
};

// Global (not thread_local) so all OMP threads share VRAM ownership correctly.
// Keyed on host pointer identity. Protected by a single mutex.
class NTTVRAMCache {
public:
    static NTTVRAMCache& Instance() {
        static NTTVRAMCache inst;
        return inst;
    }

    // Upload host poly, run forward NTT in-place, cache result.
    // Returns device pointer in NTT domain.
    uint64_t* UploadAndNTT(const uint64_t* host_ptr,
                           uint32_t ring,
                           uint64_t modulus,
                           const uint64_t* d_twiddles,
                           cudaStream_t stream);

    // Pointwise multiply two NTT-domain device ptrs → result stays in VRAM.
    void MultInNTTDomain(uint64_t* d_a, uint64_t* d_b,
                         uint64_t* d_res, uint64_t modulus,
                         uint32_t ring, cudaStream_t stream);

    // Run INTT on d_res and copy back to host.
    void INTTAndDownload(uint64_t* d_ptr, uint64_t* host_dst,
                         uint32_t ring, uint64_t modulus,
                         const uint64_t* d_inv_twiddles,
                         uint64_t n_inv_mod,          // N^{-1} mod q
                         cudaStream_t stream);

    void Evict(const uint64_t* host_ptr);
    void EvictAll();

private:
    NTTVRAMCache() = default;
    std::unordered_map<const uint64_t*, NTTEntry> cache_;
    std::mutex mu_;
};

} // namespace openfhe_cuda

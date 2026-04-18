#pragma once
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <vector>

namespace openfhe_cuda {

struct NTTEntry {
    uint64_t* d_ntt_ptr;
    uint32_t  ring;
    uint64_t  modulus;
    bool      ntt_done;
};

class NTTVRAMCache {
public:
    static NTTVRAMCache& Instance() {
        static NTTVRAMCache inst;
        return inst;
    }
    uint64_t* UploadAndNTT(const uint64_t* host_ptr, uint32_t ring, uint64_t modulus,
                           const uint64_t* d_twiddles, cudaStream_t stream);
    void MultInNTTDomain(uint64_t* d_a, uint64_t* d_b, uint64_t* d_res, uint64_t modulus,
                         uint32_t ring, cudaStream_t stream);
    void INTTAndDownload(uint64_t* d_ptr, uint64_t* host_dst, uint32_t ring, uint64_t modulus,
                         const uint64_t* d_inv_twiddles, uint64_t n_inv_mod, cudaStream_t stream);
    void Evict(const uint64_t* host_ptr);
    void EvictAll();
private:
    NTTVRAMCache() = default;
    std::unordered_map<const uint64_t*, NTTEntry> cache_;
    std::mutex mu_;
};

} // namespace openfhe_cuda


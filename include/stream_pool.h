#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <mutex>

namespace openfhe_cuda {

class StreamPool {
public:
    static StreamPool& Instance() {
        static StreamPool inst;
        return inst;
    }
    void Init(uint32_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!streams_.empty()) return;
        streams_.resize(n);
        for (auto& s : streams_) cudaStreamCreate(&s);
    }
    cudaStream_t Get(uint32_t tower_idx) {
        return streams_[tower_idx % streams_.size()];
    }
    void SyncAll() {
        for (auto s : streams_) cudaStreamSynchronize(s);
    }
    ~StreamPool() {
        for (auto s : streams_) cudaStreamDestroy(s);
    }
private:
    StreamPool() = default;
    std::vector<cudaStream_t> streams_;
    std::mutex mu_;
};

} // namespace openfhe_cuda


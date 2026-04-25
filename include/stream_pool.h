#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace openfhe_cuda {

class StreamPool {
public:
    static StreamPool& Instance() {
        static StreamPool inst;
        return inst;
    }

    void Init(uint32_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        if (streams_.size() >= n) return;
        uint32_t old = (uint32_t)streams_.size();
        streams_.resize(n);
        for (uint32_t i = old; i < n; i++) {
            cudaError_t e = cudaStreamCreate(&streams_[i]);
            if (e != cudaSuccess)
                throw std::runtime_error(
                    std::string("[StreamPool] cudaStreamCreate: ") +
                    cudaGetErrorString(e));
        }
    }

    cudaStream_t Get(uint32_t idx) {
        std::lock_guard<std::mutex> lk(mu_);
        if (streams_.empty())
            throw std::runtime_error("[StreamPool] not initialized");
        return streams_[idx % streams_.size()];
    }

    void SyncAll() {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto s : streams_) {
            cudaError_t e = cudaStreamSynchronize(s);
            if (e != cudaSuccess)
                fprintf(stderr, "[StreamPool] sync: %s\n",
                        cudaGetErrorString(e));
        }
    }

    ~StreamPool() {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto s : streams_) cudaStreamDestroy(s);
    }

private:
    StreamPool() = default;
    std::vector<cudaStream_t> streams_;
    std::mutex mu_;
};

} // namespace openfhe_cuda

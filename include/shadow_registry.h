#pragma once
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstring>

// ShadowRegistry — unified memory residency tracker.
// Uses cudaMallocManaged so host and device share the same pointer.

class ShadowRegistry {
public:
    static ShadowRegistry& Instance() {
        static ShadowRegistry inst;
        return inst;
    }

    uint64_t* GetDevicePtr(const void* h_ptr, size_t bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(h_ptr);
        if (it != map_.end()) {
            return it->second;
        }
        
        uint64_t* d_ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&d_ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("ShadowRegistry: cudaMallocManaged failed");
        }
        
        if (h_ptr) {
            std::memcpy(d_ptr, h_ptr, bytes);
        }
        
        map_[h_ptr] = d_ptr;
        return d_ptr;
    }

    void FreeAll() {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& pair : map_) {
            cudaFree(pair.second);
        }
        map_.clear();
    }

private:
    ShadowRegistry() = default;
    ~ShadowRegistry() { FreeAll(); }
    ShadowRegistry(const ShadowRegistry&) = delete;
    ShadowRegistry& operator=(const ShadowRegistry&) = delete;

    std::unordered_map<const void*, uint64_t*> map_;
    std::mutex mu_;
};

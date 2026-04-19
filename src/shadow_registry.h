#pragma once
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <stdexcept>

class ShadowRegistry {
private:
    struct Entry {
        uint64_t* d_ptr;
        size_t bytes;
        bool is_device_dirty; 
    };
    std::unordered_map<const void*, Entry> map_;
    std::mutex mu_;

public:
    static ShadowRegistry& Instance() {
        static ShadowRegistry inst;
        return inst;
    }

    // Retrieves an existing GPU pointer, or reallocates if capacity is insufficient
    uint64_t* GetDevicePtr(const void* h_ptr, size_t bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(h_ptr);
        
        if (it != map_.end()) {
            if (it->second.bytes >= bytes) {
                return it->second.d_ptr; // Existing capacity is sufficient
            } else {
                // OS reused the host pointer for a larger allocation. Reallocate VRAM.
                cudaFree(it->second.d_ptr);
                if (cudaMalloc(&it->second.d_ptr, bytes) != cudaSuccess) {
                    throw std::runtime_error("[ShadowRegistry] VRAM Reallocation Failed");
                }
                it->second.bytes = bytes;
                return it->second.d_ptr;
            }
        }
        
        // Brand new host pointer
        uint64_t* d_ptr = nullptr;
        if (cudaMalloc(&d_ptr, bytes) != cudaSuccess) {
            throw std::runtime_error("[ShadowRegistry] VRAM Exhausted");
        }
        map_[h_ptr] = {d_ptr, bytes, false};
        return d_ptr;
    }

    void MarkDeviceDirty(const void* h_ptr) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(h_ptr);
        if (it != map_.end()) {
            it->second.is_device_dirty = true;
        }
    }

    void SyncToHostIfNeeded(const void* h_ptr, cudaStream_t stream = 0) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(h_ptr);
        if (it != map_.end() && it->second.is_device_dirty) {
            cudaMemcpyAsync((void*)h_ptr, it->second.d_ptr, it->second.bytes, cudaMemcpyDeviceToHost, stream);
            it->second.is_device_dirty = false;
        }
    }

    void Purge() {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& pair : map_) {
            cudaFree(pair.second.d_ptr);
        }
        map_.clear();
    }
};

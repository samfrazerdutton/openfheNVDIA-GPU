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
    static constexpr int SHARDS = 16;
    std::unordered_map<const void*, Entry> map_[SHARDS];
    std::mutex mu_[SHARDS];

    int shard(const void* p) const {
        return (int)((uintptr_t)p >> 6 & (SHARDS - 1));
    }

public:
    static ShadowRegistry& Instance() {
        static ShadowRegistry inst;
        return inst;
    }

    uint64_t* GetDevicePtr(const void* h_ptr, size_t bytes) {
        int s = shard(h_ptr);
        std::lock_guard<std::mutex> lock(mu_[s]);
        auto& map_ = this->map_[s];
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
        int s = shard(h_ptr);
        std::lock_guard<std::mutex> lock(mu_[s]);
        auto it = map_[s].find(h_ptr);
        if (it != map_[s].end()) {
            it->second.is_device_dirty = true;
        }
    }

    void SyncToHostIfNeeded(const void* h_ptr, cudaStream_t stream = 0) {
        int s = shard(h_ptr);
        std::lock_guard<std::mutex> lock(mu_[s]);
        auto it = map_[s].find(h_ptr);
        if (it != map_[s].end() && it->second.is_device_dirty) {
            cudaMemcpyAsync((void*)h_ptr, it->second.d_ptr, it->second.bytes, cudaMemcpyDeviceToHost, stream);
            it->second.is_device_dirty = false;
        }
    }

    void Purge() {
        for (int s = 0; s < SHARDS; s++) {
            std::lock_guard<std::mutex> lock(mu_[s]);
            for (auto& pair : map_[s]) cudaFree(pair.second.d_ptr);
            map_[s].clear();
        }
    }
};

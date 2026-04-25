#pragma once
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <array>

class ShadowRegistry {
    static constexpr int SHARDS = 16;
    struct Entry { uint64_t* d_ptr; size_t bytes; };
    struct Shard {
        std::mutex mu;
        std::unordered_map<const void*, Entry> map;
    };
    std::array<Shard, SHARDS> shards_;
    int idx(const void* p) const { return (int)((uintptr_t)p >> 6 & (SHARDS-1)); }

public:
    static ShadowRegistry& Instance() {
        static ShadowRegistry inst;
        return inst;
    }

    uint64_t* GetDevicePtr(const void* h_ptr, size_t bytes) {
        if (!h_ptr) throw std::runtime_error("ShadowRegistry: null host ptr");
        auto& sh = shards_[idx(h_ptr)];
        std::lock_guard<std::mutex> lk(sh.mu);
        auto it = sh.map.find(h_ptr);
        
        if (it != sh.map.end()) {
            if (it->second.bytes >= bytes) {
                return it->second.d_ptr; // Existing VRAM buffer is large enough
            } else {
                cudaFree(it->second.d_ptr); // Too small, evict it
                sh.map.erase(it);
            }
        }
        
        uint64_t* d = nullptr;
        cudaError_t e = cudaMallocManaged(&d, bytes);
        if (e != cudaSuccess)
            throw std::runtime_error("ShadowRegistry cudaMallocManaged failed");
        
        sh.map[h_ptr] = {d, bytes};
        return d;
    }

    void Clear() {
        for(int i=0; i<SHARDS; i++) {
            std::lock_guard<std::mutex> lk(shards_[i].mu);
            for(auto& kv : shards_[i].map) cudaFree(kv.second.d_ptr);
            shards_[i].map.clear();
        }
    }
    
    size_t CacheSize() {
        size_t total = 0;
        for(int i=0; i<SHARDS; i++) {
            std::lock_guard<std::mutex> lk(shards_[i].mu);
            total += shards_[i].map.size();
        }
        return total;
    }
};

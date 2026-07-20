#pragma once
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <stdexcept>
#include <array>
#include <cstdint>

class ShadowRegistry {
    static constexpr int SHARDS = 16;
    struct Entry    { uint64_t* d_ptr; size_t bytes; };
    struct PinEntry { size_t bytes; };
    struct Shard {
        std::mutex mu;
        std::unordered_map<const void*, Entry>    map;
        std::unordered_map<const void*, PinEntry> pinned;
    };
    std::array<Shard, SHARDS> shards_;
    int idx(const void* p) const { return (int)((uintptr_t)p >> 6 & (SHARDS-1)); }

public:
    static ShadowRegistry& Instance() {
        static ShadowRegistry inst;
        return inst;
    }

    // Real device memory (cudaMalloc), cached by host pointer.
    // No more cudaMallocManaged: copies are explicit, kernels never page-fault.
    uint64_t* GetDevicePtr(const void* h_ptr, size_t bytes) {
        if (!h_ptr) throw std::runtime_error("ShadowRegistry: null host ptr");
        auto& sh = shards_[idx(h_ptr)];
        std::lock_guard<std::mutex> lk(sh.mu);
        auto it = sh.map.find(h_ptr);
        if (it != sh.map.end()) {
            if (it->second.bytes >= bytes) return it->second.d_ptr;
            cudaFree(it->second.d_ptr);
            sh.map.erase(it);
        }
        uint64_t* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess || !d)
            throw std::runtime_error("ShadowRegistry cudaMalloc failed (VRAM exhausted?)");
        sh.map[h_ptr] = {d, bytes};
        return d;
    }

    // Pin an OpenFHE-owned host buffer so async copies are true DMA.
    // Best-effort: if registration fails we proceed unpinned (slower, still correct).
    void PinHost(const void* h_ptr, size_t bytes) {
        if (!h_ptr) return;
        auto& sh = shards_[idx(h_ptr)];
        std::lock_guard<std::mutex> lk(sh.mu);
        auto it = sh.pinned.find(h_ptr);
        if (it != sh.pinned.end()) {
            if (it->second.bytes >= bytes) return;
            cudaHostUnregister(const_cast<void*>(h_ptr));  // same address, bigger buffer
            sh.pinned.erase(h_ptr);
        }
        cudaError_t e = cudaHostRegister(const_cast<void*>(h_ptr), bytes,
                                         cudaHostRegisterDefault);
        if (e == cudaSuccess) sh.pinned[h_ptr] = {bytes};
        else cudaGetLastError();  // clear sticky error; fall back to pageable
    }

    void Clear() {
        for (int i = 0; i < SHARDS; i++) {
            std::lock_guard<std::mutex> lk(shards_[i].mu);
            for (auto& kv : shards_[i].map) cudaFree(kv.second.d_ptr);
            shards_[i].map.clear();
            for (auto& kv : shards_[i].pinned)
                cudaHostUnregister(const_cast<void*>(kv.first));
            shards_[i].pinned.clear();
        }
    }

    size_t CacheSize() {
        size_t total = 0;
        for (int i = 0; i < SHARDS; i++) {
            std::lock_guard<std::mutex> lk(shards_[i].mu);
            total += shards_[i].map.size();
        }
        return total;
    }
};

#pragma once
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <stdexcept>
#include <array>
#include <vector>
#include <cstdint>

// Device-buffer cache: real cudaMalloc memory keyed by host pointer.
// Inputs are re-copied every call, so a stale host->device mapping can
// never serve stale data.
class ShadowRegistry {
    static constexpr int SHARDS = 16;
    struct Entry { uint64_t* d_ptr; size_t bytes; };
    struct Shard {
        std::mutex mu;
        std::unordered_map<const void*, Entry> map;
    };
    std::array<Shard, SHARDS> shards_;
    int idx(const void* p) const { return (int)((uintptr_t)p >> 6 & (SHARDS - 1)); }

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

    void Clear() {
        for (int i = 0; i < SHARDS; i++) {
            std::lock_guard<std::mutex> lk(shards_[i].mu);
            for (auto& kv : shards_[i].map) cudaFree(kv.second.d_ptr);
            shards_[i].map.clear();
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

// HAL-owned pinned (page-locked) staging buffers.
//
// Why this exists: cudaHostRegister on OpenFHE-owned buffers is unsafe —
// OpenFHE frees temporaries while still registered, leaving dangling pin
// state on reused addresses (observed: cudaMemcpyAsync "invalid argument").
// Instead the HAL owns pinned memory outright via cudaMallocHost, reuses it
// forever, and data flows host -> staging (CPU memcpy) -> device (true DMA).
//
// Acquire/Release are thread-safe. Buffers are never freed during the
// process lifetime (bounded: a handful of tower-sized buffers reused
// across every call). If cudaMallocHost fails, Acquire returns nullptr
// and callers fall back to pageable copies — slower, still correct.
class PinnedStagingPool {
    struct Buf { void* p; size_t bytes; };
    std::mutex mu_;
    std::vector<Buf> free_;
    std::unordered_map<void*, size_t> in_use_;

public:
    static PinnedStagingPool& Instance() {
        static PinnedStagingPool inst;
        return inst;
    }

    void* Acquire(size_t bytes) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (size_t i = 0; i < free_.size(); ++i) {
                if (free_[i].bytes >= bytes) {
                    Buf b = free_[i];
                    free_[i] = free_.back();
                    free_.pop_back();
                    in_use_[b.p] = b.bytes;
                    return b.p;
                }
            }
        }
        void* p = nullptr;
        if (cudaMallocHost(&p, bytes) != cudaSuccess || !p) {
            cudaGetLastError();  // clear sticky error; caller falls back to pageable
            return nullptr;
        }
        std::lock_guard<std::mutex> lk(mu_);
        in_use_[p] = bytes;
        return p;
    }

    void Release(void* p) {
        if (!p) return;
        std::lock_guard<std::mutex> lk(mu_);
        auto it = in_use_.find(p);
        if (it == in_use_.end()) return;
        free_.push_back({p, it->second});
        in_use_.erase(it);
    }
};

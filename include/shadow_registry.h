#pragma once
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ShadowRegistry — stateful VRAM residency tracker.
// Links host pointer addresses to persistent GPU allocations.
// Thread-safe: per-shard locking with 16 shards for OMP parallelism.
// Dirty/sync state uses atomic flags to prevent race between
// MarkDeviceDirty and SyncToHostIfNeeded from concurrent OMP threads.

class ShadowRegistry {
public:
    static ShadowRegistry& Instance() {
        static ShadowRegistry inst;
        return inst;
    }

    // Return (or allocate) a device pointer mirroring h_ptr.
    // If the existing slot is too small (OS recycled the address for a
    // larger allocation), it reallocates.
    uint64_t* GetDevicePtr(const void* h_ptr, size_t bytes) {
        int s = shard(h_ptr);
        std::lock_guard<std::mutex> lock(mu_[s]);
        auto& m = map_[s];
        auto  it = m.find(h_ptr);
        if (it != m.end()) {
            if (it->second.bytes >= bytes)
                return it->second.d_ptr;
            // OS reused the address for a larger allocation — reallocate.
            cudaFree(it->second.d_ptr);
            it->second.d_ptr = nullptr;
            if (cudaMalloc(&it->second.d_ptr, bytes) != cudaSuccess)
                throw std::runtime_error(
                    "[ShadowRegistry] VRAM realloc failed (" +
                    std::to_string(bytes) + " bytes)");
            it->second.bytes = bytes;
            it->second.dirty.store(false, std::memory_order_relaxed);
            return it->second.d_ptr;
        }
        uint64_t* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess)
            throw std::runtime_error(
                "[ShadowRegistry] VRAM exhausted (" +
                std::to_string(bytes) + " bytes)");
        m.emplace(h_ptr, Entry{d, bytes, false});
        return d;
    }

    // Mark result as needing sync back to host on next SyncToHostIfNeeded.
    void MarkDeviceDirty(const void* h_ptr) {
        int s = shard(h_ptr);
        std::lock_guard<std::mutex> lock(mu_[s]);
        auto it = map_[s].find(h_ptr);
        if (it != map_[s].end())
            it->second.dirty.store(true, std::memory_order_release);
    }

    // Async copy device → host if dirty, then clear dirty flag.
    void SyncToHostIfNeeded(const void* h_ptr, cudaStream_t stream = 0) {
        int s = shard(h_ptr);
        std::lock_guard<std::mutex> lock(mu_[s]);
        auto it = map_[s].find(h_ptr);
        if (it != map_[s].end() &&
            it->second.dirty.load(std::memory_order_acquire)) {
            cudaMemcpyAsync(const_cast<void*>(h_ptr),
                            it->second.d_ptr,
                            it->second.bytes,
                            cudaMemcpyDeviceToHost, stream);
            it->second.dirty.store(false, std::memory_order_release);
        }
    }

    // Free all VRAM and clear the registry.
    void Purge() {
        for (int s = 0; s < SHARDS; s++) {
            std::lock_guard<std::mutex> lock(mu_[s]);
            for (auto& kv : map_[s]) cudaFree(kv.second.d_ptr);
            map_[s].clear();
        }
    }

    ~ShadowRegistry() { Purge(); }

private:
    ShadowRegistry() = default;

    struct Entry {
        uint64_t*           d_ptr;
        size_t              bytes;
        std::atomic<bool>   dirty;
        // std::atomic is not copyable; provide move ctor for emplace.
        Entry(uint64_t* p, size_t b, bool d) : d_ptr(p), bytes(b), dirty(d) {}
        Entry(Entry&& o) noexcept
            : d_ptr(o.d_ptr), bytes(o.bytes),
              dirty(o.dirty.load(std::memory_order_relaxed)) {}
        Entry& operator=(Entry&&) = delete;
        Entry(const Entry&)       = delete;
        Entry& operator=(const Entry&) = delete;
    };

    static constexpr int SHARDS = 16;
    std::unordered_map<const void*, Entry> map_[SHARDS];
    std::mutex                             mu_[SHARDS];

    int shard(const void* p) const {
        return static_cast<int>(
            (reinterpret_cast<uintptr_t>(p) >> 6) & (SHARDS - 1));
    }
};

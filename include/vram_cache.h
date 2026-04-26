#pragma once
/**
 * Phase 3: Persistent VRAM Cache
 *
 * Ciphertexts remain GPU-resident across EvalMult calls.
 * PCIe transfer only happens on explicit Acquire/Release.
 * Implements LRU eviction when VRAM pressure exceeds threshold.
 *
 * Based on memory-centric optimization described in:
 *   Jung et al., "Over 100x Faster Bootstrapping in FHE through
 *   Memory-centric Optimization with GPUs", ePrint 2021/508
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <list>
#include <mutex>
#include <functional>

// Opaque handle for a GPU-resident polynomial tower set
using VramHandle = uint64_t;
constexpr VramHandle VRAM_INVALID = 0;

struct VramEntry {
    void*       d_ptr;        // device pointer
    size_t      size_bytes;
    uint32_t    num_towers;
    uint32_t    ring_dim;
    bool        dirty;        // device data differs from host
    VramHandle  handle;
    // LRU tracking
    std::list<VramHandle>::iterator lru_it;
};

class VramCache {
public:
    static VramCache& Instance();

    // Initialize with a fraction of available VRAM (default 0.75)
    void Init(double vram_fraction = 0.75, cudaStream_t stream = nullptr);

    // Upload host buffer to GPU, returns persistent handle
    // If already resident, returns existing handle (no upload)
    VramHandle Acquire(const void* host_ptr, size_t size_bytes,
                       uint32_t num_towers, uint32_t ring_dim);

    // Mark handle as containing updated device data (after in-place GPU op)
    void MarkDirty(VramHandle handle);

    // Sync dirty handle back to host_ptr synchronously
    void Writeback(VramHandle handle, void* host_ptr);

    // Explicitly evict handle, sync dirty data back to host_ptr
    void Release(VramHandle handle, void* host_ptr);

    // Get raw device pointer for kernel launch
    void* DevPtr(VramHandle handle);

    // Evict LRU entries until free_bytes are available
    void Evict(size_t free_bytes, std::function<void*(VramHandle)> get_host);

    // Stats
    size_t BytesResident() const { return bytes_resident_; }
    size_t Capacity()      const { return capacity_bytes_; }
    size_t HitCount()      const { return hits_; }
    size_t MissCount()     const { return misses_; }

private:
    VramCache() = default;
    VramHandle NextHandle();

    mutable std::mutex mu_;
    cudaStream_t stream_ = nullptr;

    std::unordered_map<VramHandle, VramEntry>   entries_;
    std::unordered_map<const void*, VramHandle> host_index_;  // host ptr → handle
    std::list<VramHandle>                        lru_list_;

    size_t capacity_bytes_  = 0;
    size_t bytes_resident_  = 0;
    uint64_t next_handle_   = 1;
    size_t hits_   = 0;
    size_t misses_ = 0;
};

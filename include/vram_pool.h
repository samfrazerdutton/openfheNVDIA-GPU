#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <string>

namespace openfhe_cuda {

// A fixed pool of pre-allocated device buffers, each of size slot_bytes.
// Threads check out a slot, use it, and return it. Blocks if all slots are taken.
// Safe under arbitrary OMP parallelism — no thread_local, no global map races.
class VRAMPool {
public:
    static VRAMPool& Instance() {
        static VRAMPool inst;
        return inst;
    }

    // Call once before any GPU work. Safe to call multiple times (no-op after first).
    void Init(uint32_t num_slots, size_t slot_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        if (initialised_) return;
        slot_bytes_ = slot_bytes;
        slots_.resize(num_slots, nullptr);
        free_.resize(num_slots);
        for (uint32_t i = 0; i < num_slots; i++) {
            cudaError_t e = cudaMalloc(&slots_[i], slot_bytes);
            if (e != cudaSuccess)
                throw std::runtime_error(
                    std::string("[VRAMPool] cudaMalloc failed: ") + cudaGetErrorString(e));
            free_[i] = slots_[i];
        }
        initialised_ = true;
    }

    // Check out a slot. Blocks until one is available.
    // Returns a device pointer valid for slot_bytes bytes.
    uint64_t* Checkout() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !free_.empty(); });
        uint64_t* ptr = reinterpret_cast<uint64_t*>(free_.back());
        free_.pop_back();
        return ptr;
    }

    // Return a slot to the pool.
    void Return(uint64_t* ptr) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            free_.push_back(ptr);
        }
        cv_.notify_one();
    }

    size_t SlotBytes() const { return slot_bytes_; }

    ~VRAMPool() {
        for (auto p : slots_) if (p) cudaFree(p);
    }

    // Non-copyable
    VRAMPool(const VRAMPool&)            = delete;
    VRAMPool& operator=(const VRAMPool&) = delete;

private:
    VRAMPool() = default;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::vector<void*>      slots_;
    std::vector<void*>      free_;
    size_t                  slot_bytes_ = 0;
    bool                    initialised_ = false;
};

// RAII guard: checks out a slot on construction, returns it on destruction.
// Use this so slots are always returned even if an exception is thrown.
struct VRAMSlot {
    uint64_t* ptr;
    explicit VRAMSlot() : ptr(VRAMPool::Instance().Checkout()) {}
    ~VRAMSlot()           { VRAMPool::Instance().Return(ptr); }
    VRAMSlot(const VRAMSlot&)            = delete;
    VRAMSlot& operator=(const VRAMSlot&) = delete;
};

} // namespace openfhe_cuda

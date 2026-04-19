#pragma once
#include <cuda_runtime.h>
#include <omp.h>
#include <cstdint>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <string>

namespace openfhe_cuda {

// FIX: Slot count formula.
// Old: (max_threads * 3) + 16
//   With 1 OMP thread and 16 towers: 19 slots available, but the new
//   concurrent batch wrapper checks out 16*3 = 48 slots up front → deadlock.
//
// New: slots = max(num_towers_hint, max_threads) * 3 + 8 buffer.
//   Since we don't know num_towers at Init() time, we use a generous
//   default of 32 towers * 3 slots + 8 = 104 slots, regardless of OMP.
//   Each slot at MAX_RING=65536 is 512 KB; 104 slots = ~52 MB VRAM, fine.
//
// If you need more towers, call Init() with a larger num_slots override
// or increase MAX_TOWERS_HINT below.
static constexpr uint32_t MAX_TOWERS_HINT = 16;   // towers per call
static constexpr uint32_t MAX_OMP_THREADS  = 8;    // match benchmark_duality
static constexpr uint32_t SLOTS_PER_TOWER  = 3;    // a, b, result per tower
static constexpr uint32_t POOL_BUFFER      = 16;   // headroom

class VRAMPool {
public:
    static VRAMPool& Instance() {
        static VRAMPool inst;
        return inst;
    }

    void Init(size_t slot_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        if (initialised_) {
            if (slot_bytes <= slot_bytes_) return;
            // Requested larger slots -- re-initialize.
            for (auto p : slots_) if (p) cudaFree(p);
            slots_.clear(); free_.clear();
            initialised_ = false;
        }

        // Must cover worst case: all OMP threads simultaneously in-flight,
        // each holding SLOTS_PER_TOWER slots for every tower in their batch.
        uint32_t num_slots = MAX_OMP_THREADS * MAX_TOWERS_HINT * SLOTS_PER_TOWER + POOL_BUFFER;

        slot_bytes_ = slot_bytes;
        slots_.resize(num_slots, nullptr);
        free_.resize(num_slots);
        for (uint32_t i = 0; i < num_slots; i++) {
            cudaError_t e = cudaMalloc(&slots_[i], slot_bytes);
            if (e != cudaSuccess)
                throw std::runtime_error(
                    "[VRAMPool] cudaMalloc failed at slot " + std::to_string(i) +
                    ": " + cudaGetErrorString(e));
            free_[i] = slots_[i];
        }
        initialised_ = true;
    }

    uint64_t* Checkout() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !free_.empty(); });
        uint64_t* ptr = reinterpret_cast<uint64_t*>(free_.back());
        free_.pop_back();
        return ptr;
    }

    void Return(uint64_t* ptr) {
        { std::lock_guard<std::mutex> lk(mu_); free_.push_back(ptr); }
        cv_.notify_one();
    }

    ~VRAMPool() { for (auto p : slots_) if (p) cudaFree(p); }

private:
    VRAMPool() = default;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::vector<void*>      slots_;
    std::vector<void*>      free_;
    size_t                  slot_bytes_  = 0;
    bool                    initialised_ = false;
};

struct VRAMSlot {
    uint64_t* ptr;
    explicit VRAMSlot() : ptr(VRAMPool::Instance().Checkout()) {}
    ~VRAMSlot()                          { VRAMPool::Instance().Return(ptr); }
    VRAMSlot(const VRAMSlot&)            = delete;
    VRAMSlot(VRAMSlot&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    VRAMSlot& operator=(VRAMSlot&& o) noexcept { if (this != &o) { if (ptr) VRAMPool::Instance().Return(ptr); ptr = o.ptr; o.ptr = nullptr; } return *this; }
    VRAMSlot& operator=(const VRAMSlot&) = delete;
};

} // namespace openfhe_cuda


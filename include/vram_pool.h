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
        }
        
        // Dynamically scale VRAM slots based on available CPU threads
        int max_threads = omp_get_max_threads();
        uint32_t num_slots = (max_threads * 3) + 16; // 3 per thread + 16 buffer
        
        slot_bytes_ = slot_bytes;
        slots_.resize(num_slots, nullptr);
        free_.resize(num_slots);
        for (uint32_t i = 0; i < num_slots; i++) {
            cudaError_t e = cudaMalloc(&slots_[i], slot_bytes);
            if (e != cudaSuccess) throw std::runtime_error("[VRAMPool] cudaMalloc failed");
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
        {
            std::lock_guard<std::mutex> lk(mu_);
            free_.push_back(ptr);
        }
        cv_.notify_one();
    }

    ~VRAMPool() {
        for (auto p : slots_) if (p) cudaFree(p);
    }

private:
    VRAMPool() = default;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::vector<void*>      slots_;
    std::vector<void*>      free_;
    size_t                  slot_bytes_ = 0;
    bool                    initialised_ = false;
};

struct VRAMSlot {
    uint64_t* ptr;
    explicit VRAMSlot() : ptr(VRAMPool::Instance().Checkout()) {}
    ~VRAMSlot()           { VRAMPool::Instance().Return(ptr); }
    VRAMSlot(const VRAMSlot&)            = delete;
    VRAMSlot& operator=(const VRAMSlot&) = delete;
};

} // namespace openfhe_cuda

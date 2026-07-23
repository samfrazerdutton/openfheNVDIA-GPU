#pragma once
// residency_tracker.h — Fix B: cross-operation device residency.
//
// Problem this solves:
//   Every HAL wrapper currently does upload -> compute -> download on every
//   call. In a chain like EvalMult -> Relinearize(keyswitch), the multiply's
//   result is downloaded to host and then immediately re-uploaded by the
//   keyswitch. At depth-5 params that is ~4.6 MB of pure round-trip waste
//   per chain, and it compounds with every additional op.
//
// What this does:
//   Tracks, per host buffer, whether the device copy is CURRENT (device and
//   host agree), DIRTY (device has newer data that host has not seen), or
//   ABSENT. Wrappers consult it to:
//     - skip an upload when the device copy is already current or dirty
//       (the data is already there from the previous op's output), and
//     - defer a download, leaving results in VRAM marked dirty.
//
// Safety model (deliberately conservative):
//   The danger is CPU-side OpenFHE code reading a host buffer whose true
//   value sits dirty in VRAM. Because we cannot intercept every host read,
//   this layer is OPT-IN (OPENFHE_GPU_FUSE=1) and provides FlushAll(),
//   which the HAL calls before any operation that can hand data back to
//   CPU code paths. When in doubt, flush: a redundant download costs
//   microseconds, a missed one corrupts ciphertexts.
//
// Correctness invariant:
//   A buffer is only left dirty when the very next consumer of that host
//   pointer is another GPU wrapper in the same chain. Any other exit path
//   (decrypt, serialization, CPU fallback, unknown callers) goes through
//   FlushAll() first.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

class ResidencyTracker {
public:
    enum class State : uint8_t {
        Absent = 0,   // no device copy known
        Current,      // device copy matches host
        Dirty         // device copy is newer; host is stale
    };

    struct Entry {
        void*  d_ptr = nullptr;
        size_t bytes = 0;
        State  state = State::Absent;
    };

    static ResidencyTracker& Instance() {
        static ResidencyTracker inst;
        return inst;
    }

    static bool Enabled() {
        static const bool on = [] {
            const char* e = std::getenv("OPENFHE_GPU_FUSE");
            return e && e[0] == '1';
        }();
        return on;
    }

    static bool Logging() {
        static const bool on = [] {
            const char* e = std::getenv("OPENFHE_GPU_LOG");
            return e && e[0] == '1';
        }();
        return on;
    }

    // Does the device already hold valid data for this host buffer?
    // If true, the caller may skip its upload.
    bool DeviceHasCurrent(const void* h_ptr, void* d_ptr, size_t bytes) {
        if (!Enabled()) return false;
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(h_ptr);
        if (it == map_.end()) return false;
        if (it->second.d_ptr != d_ptr || it->second.bytes != bytes) return false;
        bool ok = (it->second.state != State::Absent);
        if (ok) ++uploads_skipped_;
        return ok;
    }

    // Record that a host->device upload just happened (device == host).
    void MarkUploaded(const void* h_ptr, void* d_ptr, size_t bytes) {
        if (!Enabled()) return;
        std::lock_guard<std::mutex> lk(mu_);
        map_[h_ptr] = Entry{d_ptr, bytes, State::Current};
    }

    // Record that a kernel wrote this buffer on the device; host is stale.
    // Returns true if the caller may skip the download for now.
    bool MarkDeviceWritten(const void* h_ptr, void* d_ptr, size_t bytes) {
        if (!Enabled()) return false;
        std::lock_guard<std::mutex> lk(mu_);
        map_[h_ptr] = Entry{d_ptr, bytes, State::Dirty};
        ++downloads_deferred_;
        return true;
    }

    // Record that the host copy was just refreshed from the device.
    void MarkSynced(const void* h_ptr) {
        if (!Enabled()) return;
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(h_ptr);
        if (it != map_.end()) it->second.state = State::Current;
    }

    // Write back every dirty buffer. Called before any path that may hand
    // data to CPU-side code. Safe to call redundantly.
    void FlushAll() {
        if (!Enabled()) return;
        std::vector<std::pair<void*, Entry>> pending;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& kv : map_) {
                if (kv.second.state == State::Dirty) {
                    pending.emplace_back(const_cast<void*>(kv.first), kv.second);
                    kv.second.state = State::Current;
                }
            }
        }
        for (auto& p : pending) {
            cudaError_t e = cudaMemcpy(p.first, p.second.d_ptr, p.second.bytes,
                                       cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) {
                // Never leave a buffer silently stale: surface it loudly.
                std::fprintf(stderr, "[FUSE] FlushAll writeback failed: %s\n",
                             cudaGetErrorString(e));
            }
            ++flushes_;
        }
    }

    // Forget a buffer entirely (its device allocation was freed/reused).
    void Invalidate(const void* h_ptr) {
        if (!Enabled()) return;
        std::lock_guard<std::mutex> lk(mu_);
        map_.erase(h_ptr);
    }

    void Stats(uint64_t& skipped, uint64_t& deferred, uint64_t& flushed) {
        std::lock_guard<std::mutex> lk(mu_);
        skipped  = uploads_skipped_;
        deferred = downloads_deferred_;
        flushed  = flushes_;
    }

    void ResetStats() {
        std::lock_guard<std::mutex> lk(mu_);
        uploads_skipped_ = downloads_deferred_ = flushes_ = 0;
    }

private:
    std::mutex mu_;
    std::unordered_map<const void*, Entry> map_;
    uint64_t uploads_skipped_   = 0;
    uint64_t downloads_deferred_ = 0;
    uint64_t flushes_           = 0;
};

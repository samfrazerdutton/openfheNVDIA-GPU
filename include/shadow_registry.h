#pragma once
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <stdexcept>
#include <array>
#include <vector>
#include <cstdint>
#include <cstring>

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

// ── Fix 3: KeyResidencyCache ─────────────────────────────────────────────
//
// Eval-key towers (a/b vectors) are immutable after keygen, yet the naive
// keyswitch path re-uploaded them on every call — 2/3 of all keyswitch
// PCIe traffic. This cache uploads each key tower ONCE, pre-scaled into
// the Montgomery domain (bR[i] = b[i] * 2^64 mod q), and reuses the
// device copy forever.
//
// Address-reuse guard: entries store a fingerprint (size, modulus, and 8
// sampled words of the host buffer). A recycled host address holding a
// different key fails the fingerprint check and triggers a fresh
// transform+upload — a stale key can never be served silently.
class KeyResidencyCache {
    struct Entry {
        uint64_t* d_ptr;
        size_t bytes;
        uint64_t q;
        uint64_t fp[8];
    };
    std::mutex mu_;
    std::unordered_map<const void*, Entry> map_;
    uint64_t uploads_ = 0, hits_ = 0;

    static void Fingerprint(const uint64_t* h, size_t n_elems, uint64_t out[8]) {
        for (int k = 0; k < 8; ++k) {
            size_t pos = (n_elems > 1) ? (size_t)((n_elems - 1) * (uint64_t)k / 7) : 0;
            out[k] = h[pos];
        }
    }

public:
    static KeyResidencyCache& Instance() {
        static KeyResidencyCache inst;
        return inst;
    }

    // Returns the device pointer of the R-scaled key tower, uploading and
    // transforming on first sight (or on fingerprint mismatch).
    // staging: optional pinned buffer of >= bytes for the transformed copy
    // (pass nullptr to use a pageable temporary). was_hit: optional out.
    uint64_t* GetScaledKey(const uint64_t* h_key, size_t bytes, uint64_t q,
                           void* staging, cudaStream_t s, bool* was_hit = nullptr) {
        if (!h_key) throw std::runtime_error("KeyResidencyCache: null host ptr");
        const size_t n = bytes / sizeof(uint64_t);
        uint64_t fp[8];
        Fingerprint(h_key, n, fp);

        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = map_.find(h_key);
            if (it != map_.end() && it->second.bytes == bytes && it->second.q == q &&
                std::memcmp(it->second.fp, fp, sizeof(fp)) == 0) {
                ++hits_;
                if (was_hit) *was_hit = true;
                return it->second.d_ptr;
            }
        }

        // Miss (new key, or recycled address): transform to bR = b * 2^64 mod q
        // and upload. Host transform: n 128-bit mulmods, once per key tower.
        const uint64_t Rmodq = (uint64_t)((((unsigned __int128)1) << 64) % q);
        std::vector<uint64_t> tmp;
        uint64_t* dst = (uint64_t*)staging;
        if (!dst) {
            tmp.resize(n);
            dst = tmp.data();
        }
        for (size_t i = 0; i < n; ++i)
            dst[i] = (uint64_t)(((unsigned __int128)h_key[i] * Rmodq) % q);

        uint64_t* d = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = map_.find(h_key);
            if (it != map_.end()) {
                if (it->second.bytes >= bytes) {
                    d = it->second.d_ptr;  // reuse allocation, refresh content
                } else {
                    cudaFree(it->second.d_ptr);
                    map_.erase(it);
                }
            }
        }
        if (!d) {
            if (cudaMalloc(&d, bytes) != cudaSuccess || !d)
                throw std::runtime_error("KeyResidencyCache cudaMalloc failed");
        }
        cudaError_t e = staging
            ? cudaMemcpyAsync(d, dst, bytes, cudaMemcpyHostToDevice, s)
            : cudaMemcpy(d, dst, bytes, cudaMemcpyHostToDevice);
        if (e != cudaSuccess)
            throw std::runtime_error("KeyResidencyCache upload failed");
        if (staging) cudaStreamSynchronize(s);  // staging is reused by caller

        {
            std::lock_guard<std::mutex> lk(mu_);
            Entry en;
            en.d_ptr = d;
            en.bytes = bytes;
            en.q = q;
            std::memcpy(en.fp, fp, sizeof(fp));
            map_[h_key] = en;
            ++uploads_;
        }
        if (was_hit) *was_hit = false;
        return d;
    }

    void Stats(uint64_t& uploads, uint64_t& hits) {
        std::lock_guard<std::mutex> lk(mu_);
        uploads = uploads_;
        hits = hits_;
    }

    void Clear() {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& kv : map_) cudaFree(kv.second.d_ptr);
        map_.clear();
    }
};

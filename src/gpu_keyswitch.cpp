// gpu_keyswitch.cpp — GPU hybrid key-switch host-side orchestration (Fix 3).
//
// Inner product of EvalFastKeySwitchCoreExt on the GPU:
//   acc0[i] += c[j][i] * b[j][i]   (mod q_i)
//   acc1[i] += c[j][i] * a[j][i]   (mod q_i)
//
// Fix 3 over Fix 2:
//  - Eval-key towers (a/b: immutable after keygen) are uploaded ONCE via
//    KeyResidencyCache, pre-scaled into the Montgomery domain, and reused
//    from VRAM on every subsequent keyswitch. Per-call uploads drop from
//    3 buffers per (digit, tower) to 1 (the digit only): ~14 MB -> ~4.6 MB.
//  - The MAC kernel is Montgomery (LaunchRNSMacMont) — no 128-bit modulo
//    in the hot loop. mont_reduce(c * bR) = c*b mod q, plain domain out.
//
// Interface is unchanged from Fix 2, so the patched OpenFHE needs no
// rebuild — only the HAL.

#include "stream_pool.h"
#include "shadow_registry.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>

#define KS_CUDA_CHECK(call)                                                   \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
            throw std::runtime_error(std::string("[CUDA KS] " #call ": ") +   \
                                     cudaGetErrorString(_e));                 \
    } while (0)

extern "C" void LaunchRNSMacMont(const uint64_t* d_c, const uint64_t* d_bR,
                                 uint64_t* d_out, uint64_t q, uint64_t q_inv,
                                 uint32_t n, cudaStream_t s);

extern "C" void gpu_keyswitch_sync() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
}

// q_inv = -q^{-1} mod 2^64 (Newton iteration), matching ks_mont_reduce.
static uint64_t ks_calc_q_inv(uint64_t q) {
    uint64_t x = q;                 // q odd => invertible mod 2^64
    for (int i = 0; i < 6; ++i)
        x *= 2 - q * x;             // x -> q^{-1} mod 2^64
    return (uint64_t)(0) - x;       // -q^{-1} mod 2^64
}

// c, b, a: flattened [limit * towers] arrays of host tower pointers,
//          indexed c[j * towers + i].
// out0, out1: [towers] host pointers to the accumulator towers.
// q: [towers] moduli. ring: coefficients per tower.
extern "C" void gpu_keyswitch_inner_product(
    const uint64_t** c, const uint64_t** b, const uint64_t** a,
    uint64_t** out0, uint64_t** out1,
    const uint64_t* q, uint32_t ring, uint32_t limit, uint32_t towers)
{
    static const bool log_calls = [] {
        const char* e = std::getenv("OPENFHE_GPU_LOG");
        return e && e[0] == '1';
    }();
    static std::atomic<uint64_t> call_no{0};

    openfhe_cuda::StreamPool::Instance().Init(32);
    const size_t bytes = (size_t)ring * sizeof(uint64_t);
    auto& reg  = ShadowRegistry::Instance();
    auto& pool = PinnedStagingPool::Instance();
    auto& keys = KeyResidencyCache::Instance();

    std::vector<uint64_t> qinv(towers);
    for (uint32_t i = 0; i < towers; i++) qinv[i] = ks_calc_q_inv(q[i]);

    // Device accumulators, zeroed once, cached by output host pointers.
    std::vector<uint64_t*> dacc0(towers), dacc1(towers);
    for (uint32_t i = 0; i < towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        dacc0[i] = reg.GetDevicePtr(out0[i], bytes);
        dacc1[i] = reg.GetDevicePtr(out1[i], bytes);
        KS_CUDA_CHECK(cudaMemsetAsync(dacc0[i], 0, bytes, s));
        KS_CUDA_CHECK(cudaMemsetAsync(dacc1[i], 0, bytes, s));
    }

    // One shared staging buffer for key transforms (synchronous per upload,
    // first-call only), plus per-(j,i) staging for the streamed digits.
    void* key_staging = pool.Acquire(bytes);
    uint64_t key_uploads_this_call = 0;

    std::vector<void*> held;
    held.reserve((size_t)limit * towers);

    for (uint32_t j = 0; j < limit; j++) {
        for (uint32_t i = 0; i < towers; i++) {
            cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
            const uint64_t* hc = c[(size_t)j * towers + i];
            const uint64_t* hb = b[(size_t)j * towers + i];
            const uint64_t* ha = a[(size_t)j * towers + i];

            // Keys: resident, R-scaled, uploaded once ever.
            bool hit_b = false, hit_a = false;
            uint64_t* dbR = keys.GetScaledKey(hb, bytes, q[i], key_staging, s, &hit_b);
            uint64_t* daR = keys.GetScaledKey(ha, bytes, q[i], key_staging, s, &hit_a);
            if (!hit_b) ++key_uploads_this_call;
            if (!hit_a) ++key_uploads_this_call;

            // Digit: streams in per call (it changes every keyswitch).
            uint64_t* dc = reg.GetDevicePtr(hc, bytes);
            void* sc = pool.Acquire(bytes);
            if (sc) {
                std::memcpy(sc, hc, bytes);
                KS_CUDA_CHECK(cudaMemcpyAsync(dc, sc, bytes, cudaMemcpyHostToDevice, s));
                held.push_back(sc);
            } else {
                KS_CUDA_CHECK(cudaMemcpyAsync(dc, hc, bytes, cudaMemcpyHostToDevice, s));
            }

            LaunchRNSMacMont(dc, dbR, dacc0[i], q[i], qinv[i], ring, s);
            LaunchRNSMacMont(dc, daR, dacc1[i], q[i], qinv[i], ring, s);
        }
    }

    openfhe_cuda::StreamPool::Instance().SyncAll();
    for (void* p : held) pool.Release(p);
    pool.Release(key_staging);

    for (uint32_t i = 0; i < towers; i++) {
        KS_CUDA_CHECK(cudaMemcpy(out0[i], dacc0[i], bytes, cudaMemcpyDeviceToHost));
        KS_CUDA_CHECK(cudaMemcpy(out1[i], dacc1[i], bytes, cudaMemcpyDeviceToHost));
    }

    if (log_calls) {
        uint64_t up = 0, hits = 0;
        keys.Stats(up, hits);
        printf("[KS_LOG] call=%llu digits=%u towers=%u ring=%u key_uploads_now=%llu key_cache{uploads=%llu hits=%llu}\n",
               (unsigned long long)++call_no, limit, towers, ring,
               (unsigned long long)key_uploads_this_call,
               (unsigned long long)up, (unsigned long long)hits);
    }
}

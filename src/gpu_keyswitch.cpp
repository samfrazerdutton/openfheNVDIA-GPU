// gpu_keyswitch.cpp — GPU hybrid key-switch host-side orchestration (Fix 3b).
//
// Inner product of EvalFastKeySwitchCoreExt on the GPU:
//   acc0[i] += c[j][i] * b[j][i]   (mod q_i)
//   acc1[i] += c[j][i] * a[j][i]   (mod q_i)
//
// Evolution:
//   Fix 2  — device-resident accumulators across digits.
//   Fix 3  — eval keys uploaded once (KeyResidencyCache), Montgomery MAC.
//   Fix 3b — FUSED launches: one kernel per (digit, operand) covering all
//            towers, instead of one per (digit, tower). At depth-5 params
//            that is 4 launches per keyswitch instead of 36. Launch
//            overhead (~10 us each under WSL) dominated the ~4 us of real
//            kernel work, so this is the change that attacks the actual
//            bottleneck.
//
// Interface unchanged — the patched OpenFHE needs no rebuild, HAL only.

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

extern "C" void LaunchRNSMacMontFused(
    const uint64_t* const* d_c, const uint64_t* const* d_kR,
    uint64_t* const* d_acc, const uint64_t* d_q, const uint64_t* d_qinv,
    uint32_t n, uint32_t towers, cudaStream_t s);

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

// Scratch device arrays holding per-tower pointers/params for the fused
// kernel. Grown on demand, reused across calls (no per-call cudaMalloc).
namespace {
struct FusedArgBuffers {
    uint64_t** d_c   = nullptr;   // const uint64_t** in kernel
    uint64_t** d_kR  = nullptr;
    uint64_t** d_acc = nullptr;
    uint64_t*  d_q    = nullptr;
    uint64_t*  d_qinv = nullptr;
    uint32_t   cap    = 0;

    void Ensure(uint32_t towers) {
        if (towers <= cap) return;
        Free();
        KS_CUDA_CHECK(cudaMalloc(&d_c,    towers * sizeof(uint64_t*)));
        KS_CUDA_CHECK(cudaMalloc(&d_kR,   towers * sizeof(uint64_t*)));
        KS_CUDA_CHECK(cudaMalloc(&d_acc,  towers * sizeof(uint64_t*)));
        KS_CUDA_CHECK(cudaMalloc(&d_q,    towers * sizeof(uint64_t)));
        KS_CUDA_CHECK(cudaMalloc(&d_qinv, towers * sizeof(uint64_t)));
        cap = towers;
    }
    void Free() {
        if (d_c)    cudaFree(d_c);
        if (d_kR)   cudaFree(d_kR);
        if (d_acc)  cudaFree(d_acc);
        if (d_q)    cudaFree(d_q);
        if (d_qinv) cudaFree(d_qinv);
        d_c = d_kR = d_acc = nullptr;
        d_q = d_qinv = nullptr;
        cap = 0;
    }
};

// Two sets: one for the b-accumulation, one for the a-accumulation, so both
// fused launches can be in flight without clobbering each other's args.
FusedArgBuffers& ArgsB() { static FusedArgBuffers b; return b; }
FusedArgBuffers& ArgsA() { static FusedArgBuffers a; return a; }
}  // namespace

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

    // Everything runs on ONE stream: the fused kernels touch all towers, so
    // per-tower stream parallelism no longer applies, and a single ordered
    // stream gives correct accumulate-after-upload semantics for free.
    cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(0);

    std::vector<uint64_t> qinv(towers);
    for (uint32_t i = 0; i < towers; i++) qinv[i] = ks_calc_q_inv(q[i]);

    // Device accumulators, zeroed once, cached by output host pointers.
    std::vector<uint64_t*> hacc0(towers), hacc1(towers);
    for (uint32_t i = 0; i < towers; i++) {
        hacc0[i] = reg.GetDevicePtr(out0[i], bytes);
        hacc1[i] = reg.GetDevicePtr(out1[i], bytes);
        KS_CUDA_CHECK(cudaMemsetAsync(hacc0[i], 0, bytes, s));
        KS_CUDA_CHECK(cudaMemsetAsync(hacc1[i], 0, bytes, s));
    }

    auto& argsB = ArgsB();
    auto& argsA = ArgsA();
    argsB.Ensure(towers);
    argsA.Ensure(towers);

    // Per-tower params are identical for both operands and constant across
    // digits: upload once per call.
    KS_CUDA_CHECK(cudaMemcpyAsync(argsB.d_q, q, towers * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice, s));
    KS_CUDA_CHECK(cudaMemcpyAsync(argsB.d_qinv, qinv.data(),
                                  towers * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice, s));
    KS_CUDA_CHECK(cudaMemcpyAsync(argsB.d_acc, hacc0.data(),
                                  towers * sizeof(uint64_t*),
                                  cudaMemcpyHostToDevice, s));
    KS_CUDA_CHECK(cudaMemcpyAsync(argsA.d_acc, hacc1.data(),
                                  towers * sizeof(uint64_t*),
                                  cudaMemcpyHostToDevice, s));

    void* key_staging = pool.Acquire(bytes);
    uint64_t key_uploads_this_call = 0;
    std::vector<void*> held;
    held.reserve((size_t)limit * towers);

    std::vector<uint64_t*> hc_dev(towers), hbR(towers), haR(towers);

    for (uint32_t j = 0; j < limit; j++) {
        for (uint32_t i = 0; i < towers; i++) {
            const uint64_t* hc = c[(size_t)j * towers + i];
            const uint64_t* hb = b[(size_t)j * towers + i];
            const uint64_t* ha = a[(size_t)j * towers + i];

            bool hit_b = false, hit_a = false;
            hbR[i] = keys.GetScaledKey(hb, bytes, q[i], key_staging, s, &hit_b);
            haR[i] = keys.GetScaledKey(ha, bytes, q[i], key_staging, s, &hit_a);
            if (!hit_b) ++key_uploads_this_call;
            if (!hit_a) ++key_uploads_this_call;

            uint64_t* dc = reg.GetDevicePtr(hc, bytes);
            void* sc = pool.Acquire(bytes);
            if (sc) {
                std::memcpy(sc, hc, bytes);
                KS_CUDA_CHECK(cudaMemcpyAsync(dc, sc, bytes,
                                              cudaMemcpyHostToDevice, s));
                held.push_back(sc);
            } else {
                KS_CUDA_CHECK(cudaMemcpyAsync(dc, hc, bytes,
                                              cudaMemcpyHostToDevice, s));
            }
            hc_dev[i] = dc;
        }

        // Pointer arrays for this digit, then TWO launches for all towers.
        KS_CUDA_CHECK(cudaMemcpyAsync(argsB.d_c, hc_dev.data(),
                                      towers * sizeof(uint64_t*),
                                      cudaMemcpyHostToDevice, s));
        KS_CUDA_CHECK(cudaMemcpyAsync(argsB.d_kR, hbR.data(),
                                      towers * sizeof(uint64_t*),
                                      cudaMemcpyHostToDevice, s));
        KS_CUDA_CHECK(cudaMemcpyAsync(argsA.d_kR, haR.data(),
                                      towers * sizeof(uint64_t*),
                                      cudaMemcpyHostToDevice, s));

        LaunchRNSMacMontFused((const uint64_t* const*)argsB.d_c,
                              (const uint64_t* const*)argsB.d_kR,
                              argsB.d_acc, argsB.d_q, argsB.d_qinv,
                              ring, towers, s);
        LaunchRNSMacMontFused((const uint64_t* const*)argsB.d_c,
                              (const uint64_t* const*)argsA.d_kR,
                              argsA.d_acc, argsB.d_q, argsB.d_qinv,
                              ring, towers, s);
        // Both launches read argsB.d_c on the same stream before the next
        // digit overwrites it — stream ordering guarantees safety.
        KS_CUDA_CHECK(cudaStreamSynchronize(s));
    }

    for (void* p : held) pool.Release(p);
    pool.Release(key_staging);

    for (uint32_t i = 0; i < towers; i++) {
        KS_CUDA_CHECK(cudaMemcpy(out0[i], hacc0[i], bytes, cudaMemcpyDeviceToHost));
        KS_CUDA_CHECK(cudaMemcpy(out1[i], hacc1[i], bytes, cudaMemcpyDeviceToHost));
    }

    if (log_calls) {
        uint64_t up = 0, hits = 0;
        keys.Stats(up, hits);
        printf("[KS_LOG] call=%llu digits=%u towers=%u ring=%u launches=%u "
               "key_uploads_now=%llu key_cache{uploads=%llu hits=%llu}\n",
               (unsigned long long)++call_no, limit, towers, ring, limit * 2,
               (unsigned long long)key_uploads_this_call,
               (unsigned long long)up, (unsigned long long)hits);
    }
}

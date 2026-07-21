// gpu_keyswitch.cpp — GPU hybrid key-switch host-side orchestration (Fix 2).
//
// Implements the inner product of EvalFastKeySwitchCoreExt on the GPU:
//   acc0[i] += c[j][i] * b[j][i]   (mod q_i)
//   acc1[i] += c[j][i] * a[j][i]   (mod q_i)
// for all digits j and towers i, using the rns_mac_kernel in
// kernels/cuda_keyswitch.cu.
//
// Key design point: the accumulators live in DEVICE memory across all
// digit iterations — they are zeroed once, accumulated on-device j times,
// and downloaded once at the end. Only the inputs stream in per digit.
//
// Transfers use the HAL-owned PinnedStagingPool (true DMA) with a
// pageable fallback, same as gpu_rns_mult_batch_wrapper.

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

extern "C" void LaunchRNSMac(const uint64_t* d_a, const uint64_t* d_b,
                             uint64_t* d_out, uint64_t q, uint32_t n,
                             cudaStream_t s);

extern "C" void gpu_keyswitch_sync() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
}

// c, b, a: flattened [limit * towers] arrays of host tower pointers,
//          indexed c[j * towers + i].
// out0, out1: [towers] host pointers to the (zero-initialized) accumulator
//          towers of elements[0] / elements[1].
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
    if (log_calls)
        printf("[KS_LOG] call=%llu digits=%u towers=%u ring=%u\n",
               (unsigned long long)++call_no, limit, towers, ring);

    openfhe_cuda::StreamPool::Instance().Init(32);
    const size_t bytes = (size_t)ring * sizeof(uint64_t);
    auto& reg  = ShadowRegistry::Instance();
    auto& pool = PinnedStagingPool::Instance();

    // Device accumulators, zeroed once. Cached by the real output host
    // pointers so repeat keyswitches reuse the same VRAM.
    std::vector<uint64_t*> dacc0(towers), dacc1(towers);
    for (uint32_t i = 0; i < towers; i++) {
        cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
        dacc0[i] = reg.GetDevicePtr(out0[i], bytes);
        dacc1[i] = reg.GetDevicePtr(out1[i], bytes);
        KS_CUDA_CHECK(cudaMemsetAsync(dacc0[i], 0, bytes, s));
        KS_CUDA_CHECK(cudaMemsetAsync(dacc1[i], 0, bytes, s));
    }

    // Stream the inputs per (digit, tower); accumulate entirely on-device.
    // Staging buffers must stay alive until SyncAll (async copies read them),
    // so collect and release at the end.
    std::vector<void*> held;
    held.reserve((size_t)limit * towers * 3);

    for (uint32_t j = 0; j < limit; j++) {
        for (uint32_t i = 0; i < towers; i++) {
            cudaStream_t s = openfhe_cuda::StreamPool::Instance().Get(i);
            const uint64_t* hc = c[(size_t)j * towers + i];
            const uint64_t* hb = b[(size_t)j * towers + i];
            const uint64_t* ha = a[(size_t)j * towers + i];
            uint64_t* dc = reg.GetDevicePtr(hc, bytes);
            uint64_t* db = reg.GetDevicePtr(hb, bytes);
            uint64_t* da = reg.GetDevicePtr(ha, bytes);

            void* sc = pool.Acquire(bytes);
            void* sb = pool.Acquire(bytes);
            void* sa = pool.Acquire(bytes);
            if (sc && sb && sa) {
                std::memcpy(sc, hc, bytes);
                std::memcpy(sb, hb, bytes);
                std::memcpy(sa, ha, bytes);
                KS_CUDA_CHECK(cudaMemcpyAsync(dc, sc, bytes, cudaMemcpyHostToDevice, s));
                KS_CUDA_CHECK(cudaMemcpyAsync(db, sb, bytes, cudaMemcpyHostToDevice, s));
                KS_CUDA_CHECK(cudaMemcpyAsync(da, sa, bytes, cudaMemcpyHostToDevice, s));
                held.push_back(sc);
                held.push_back(sb);
                held.push_back(sa);
            } else {
                // Fallback: pageable copies (slower, still correct).
                pool.Release(sc);
                pool.Release(sb);
                pool.Release(sa);
                KS_CUDA_CHECK(cudaMemcpyAsync(dc, hc, bytes, cudaMemcpyHostToDevice, s));
                KS_CUDA_CHECK(cudaMemcpyAsync(db, hb, bytes, cudaMemcpyHostToDevice, s));
                KS_CUDA_CHECK(cudaMemcpyAsync(da, ha, bytes, cudaMemcpyHostToDevice, s));
            }

            LaunchRNSMac(dc, db, dacc0[i], q[i], ring, s);
            LaunchRNSMac(dc, da, dacc1[i], q[i], ring, s);
        }
    }

    openfhe_cuda::StreamPool::Instance().SyncAll();
    for (void* p : held) pool.Release(p);

    // Download the finished accumulators once.
    for (uint32_t i = 0; i < towers; i++) {
        KS_CUDA_CHECK(cudaMemcpy(out0[i], dacc0[i], bytes, cudaMemcpyDeviceToHost));
        KS_CUDA_CHECK(cudaMemcpy(out1[i], dacc1[i], bytes, cudaMemcpyDeviceToHost));
    }
}

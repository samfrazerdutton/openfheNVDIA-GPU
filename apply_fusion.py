#!/usr/bin/env python3
"""Fix B: wire cross-operation device residency into the HAL wrappers.

Changes (all gated behind OPENFHE_GPU_FUSE=1; default behaviour identical):

  src/cuda_hal.cpp
    - gpu_rns_mult_batch_wrapper: skip uploads whose device copy is already
      current; leave results in VRAM marked dirty instead of downloading.
    - export gpu_flush_all() for the decrypt/CPU-exit path.

  src/gpu_keyswitch.cpp
    - gpu_keyswitch_inner_product: same upload-skip logic for the streamed
      digits (their producer is usually the multiply that just ran), and
      deferred writeback for the accumulators.
    - FUSE stats appended to [KS_LOG].

Run from the repo root:  python3 apply_fusion.py
"""
import sys, os

def die(msg):
    sys.exit("[!] " + msg)

if not os.path.exists("src/cuda_hal.cpp"):
    die("run from the repo root")

# ── 1. cuda_hal.cpp ──────────────────────────────────────────────────────
P = "src/cuda_hal.cpp"
src = open(P).read()

if "ResidencyTracker" in src:
    print("[=] cuda_hal.cpp already fused")
else:
    if '#include "shadow_registry.h"' not in src:
        die("cuda_hal.cpp: shadow_registry.h include not found")
    src = src.replace('#include "shadow_registry.h"',
                      '#include "shadow_registry.h"\n#include "residency_tracker.h"', 1)

    # Upload side: consult the tracker before staging/copying inputs.
    OLD_UP = """        if (staged) {
            std::memcpy(st_a[i], ha[i], bytes);
            std::memcpy(st_b[i], hb[i], bytes);
            CUDA_CHECK(cudaMemcpyAsync(da, st_a[i], bytes, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(db, st_b[i], bytes, cudaMemcpyHostToDevice, s));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(da, ha[i], bytes, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(db, hb[i], bytes, cudaMemcpyHostToDevice, s));
        }"""
    NEW_UP = """        auto& rt = ResidencyTracker::Instance();
        const bool skip_a = rt.DeviceHasCurrent(ha[i], da, bytes);
        const bool skip_b = rt.DeviceHasCurrent(hb[i], db, bytes);
        if (staged) {
            if (!skip_a) {
                std::memcpy(st_a[i], ha[i], bytes);
                CUDA_CHECK(cudaMemcpyAsync(da, st_a[i], bytes, cudaMemcpyHostToDevice, s));
                rt.MarkUploaded(ha[i], da, bytes);
            }
            if (!skip_b) {
                std::memcpy(st_b[i], hb[i], bytes);
                CUDA_CHECK(cudaMemcpyAsync(db, st_b[i], bytes, cudaMemcpyHostToDevice, s));
                rt.MarkUploaded(hb[i], db, bytes);
            }
        } else {
            if (!skip_a) {
                CUDA_CHECK(cudaMemcpyAsync(da, ha[i], bytes, cudaMemcpyHostToDevice, s));
                rt.MarkUploaded(ha[i], da, bytes);
            }
            if (!skip_b) {
                CUDA_CHECK(cudaMemcpyAsync(db, hb[i], bytes, cudaMemcpyHostToDevice, s));
                rt.MarkUploaded(hb[i], db, bytes);
            }
        }"""
    if OLD_UP not in src:
        die("cuda_hal.cpp: upload anchor not found (was Fix 1 applied?)")
    src = src.replace(OLD_UP, NEW_UP, 1)

    # Download side: defer when fusion is on.
    OLD_DL = """        if (staged)
            CUDA_CHECK(cudaMemcpyAsync(st_r[i], d_out[i], bytes, cudaMemcpyDeviceToHost, s));
        else
            CUDA_CHECK(cudaMemcpyAsync(hr[i], d_out[i], bytes, cudaMemcpyDeviceToHost, s));"""
    NEW_DL = """        if (!ResidencyTracker::Instance().MarkDeviceWritten(hr[i], d_out[i], bytes)) {
            if (staged)
                CUDA_CHECK(cudaMemcpyAsync(st_r[i], d_out[i], bytes, cudaMemcpyDeviceToHost, s));
            else
                CUDA_CHECK(cudaMemcpyAsync(hr[i], d_out[i], bytes, cudaMemcpyDeviceToHost, s));
        }"""
    if OLD_DL not in src:
        die("cuda_hal.cpp: download anchor not found")
    src = src.replace(OLD_DL, NEW_DL, 1)

    # Final host copy of staged results must respect the deferral too.
    OLD_FIN = """    for (uint32_t i = 0; i < num_towers; i++) {
        if (staged) std::memcpy(hr[i], st_r[i], bytes);"""
    NEW_FIN = """    const bool fused = ResidencyTracker::Enabled();
    for (uint32_t i = 0; i < num_towers; i++) {
        if (staged && !fused) std::memcpy(hr[i], st_r[i], bytes);"""
    if OLD_FIN not in src:
        die("cuda_hal.cpp: final-copy anchor not found")
    src = src.replace(OLD_FIN, NEW_FIN, 1)

    # Exported flush for CPU-exit paths.
    if "gpu_flush_all" not in src:
        src += """

// Fix B: force every deferred device-side result back to host memory.
// Called before any path that can hand data to CPU-side OpenFHE code.
extern "C" void gpu_flush_all() {
    openfhe_cuda::StreamPool::Instance().SyncAll();
    ResidencyTracker::Instance().FlushAll();
}
"""
    open(P, "w").write(src)
    print("[+] cuda_hal.cpp: upload-skip + deferred writeback + gpu_flush_all")

# ── 2. gpu_keyswitch.cpp ─────────────────────────────────────────────────
P2 = "src/gpu_keyswitch.cpp"
if not os.path.exists(P2):
    die("src/gpu_keyswitch.cpp not found")
ks = open(P2).read()

if "ResidencyTracker" in ks:
    print("[=] gpu_keyswitch.cpp already fused")
else:
    ks = ks.replace('#include "shadow_registry.h"',
                    '#include "shadow_registry.h"\n#include "residency_tracker.h"', 1)

    OLD_DIG = """            uint64_t* dc = reg.GetDevicePtr(hc, bytes);
            void* sc = pool.Acquire(bytes);
            if (sc) {
                std::memcpy(sc, hc, bytes);
                KS_CUDA_CHECK(cudaMemcpyAsync(dc, sc, bytes,
                                              cudaMemcpyHostToDevice, s));
                held.push_back(sc);
            } else {
                KS_CUDA_CHECK(cudaMemcpyAsync(dc, hc, bytes,
                                              cudaMemcpyHostToDevice, s));
            }"""
    NEW_DIG = """            uint64_t* dc = reg.GetDevicePtr(hc, bytes);
            auto& rt = ResidencyTracker::Instance();
            if (!rt.DeviceHasCurrent(hc, dc, bytes)) {
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
                rt.MarkUploaded(hc, dc, bytes);
            }"""
    if OLD_DIG not in src and OLD_DIG not in ks:
        die("gpu_keyswitch.cpp: digit-upload anchor not found")
    ks = ks.replace(OLD_DIG, NEW_DIG, 1)

    OLD_KDL = """    for (uint32_t i = 0; i < towers; i++) {
        KS_CUDA_CHECK(cudaMemcpy(out0[i], hacc0[i], bytes, cudaMemcpyDeviceToHost));
        KS_CUDA_CHECK(cudaMemcpy(out1[i], hacc1[i], bytes, cudaMemcpyDeviceToHost));
    }"""
    NEW_KDL = """    {
        auto& rt = ResidencyTracker::Instance();
        for (uint32_t i = 0; i < towers; i++) {
            if (!rt.MarkDeviceWritten(out0[i], hacc0[i], bytes))
                KS_CUDA_CHECK(cudaMemcpy(out0[i], hacc0[i], bytes, cudaMemcpyDeviceToHost));
            if (!rt.MarkDeviceWritten(out1[i], hacc1[i], bytes))
                KS_CUDA_CHECK(cudaMemcpy(out1[i], hacc1[i], bytes, cudaMemcpyDeviceToHost));
        }
    }"""
    if OLD_KDL not in ks:
        die("gpu_keyswitch.cpp: accumulator-download anchor not found")
    ks = ks.replace(OLD_KDL, NEW_KDL, 1)

    # Extend the KS_LOG line with fusion counters.
    OLD_LOG = 'printf("[KS_LOG] call=%llu digits=%u towers=%u ring=%u launches=%u "'
    NEW_LOG = ('uint64_t sk = 0, df = 0, fl = 0;\n'
               '        ResidencyTracker::Instance().Stats(sk, df, fl);\n'
               '        printf("[FUSE] uploads_skipped=%llu downloads_deferred=%llu flushes=%llu\\n",\n'
               '               (unsigned long long)sk, (unsigned long long)df, (unsigned long long)fl);\n'
               '        printf("[KS_LOG] call=%llu digits=%u towers=%u ring=%u launches=%u "')
    if OLD_LOG in ks:
        ks = ks.replace(OLD_LOG, NEW_LOG, 1)

    open(P2, "w").write(ks)
    print("[+] gpu_keyswitch.cpp: digit upload-skip + deferred accumulator writeback")

print("\nNext: rebuild the HAL, then patch OpenFHE's decrypt path to call")
print("gpu_flush_all() (apply_flush_patch.py), then A/B with OPENFHE_GPU_FUSE=1.")

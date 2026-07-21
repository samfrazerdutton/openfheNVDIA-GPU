#!/usr/bin/env python3
"""Patch KeySwitchHYBRID::EvalFastKeySwitchCoreExt to run its inner product
on the GPU (gpu_keyswitch_inner_product in libopenfhe_cuda_hal.so).

- Gated on GPU_HAL_PATCHED_V8 (already defined by the patched dcrtpoly.h,
  which keyswitch-hybrid.cpp includes transitively) and the OPENFHE_GPU
  runtime kill switch.
- CPU loop retained verbatim as the fallback path.
- Accumulators are written directly into the freshly-allocated, zeroed
  towers of elements[0]/elements[1] — same backing-storage pattern the
  Times()/operator*= hooks already use.

Usage:  python3 apply_keyswitch_patch.py ~/openfhe-development
"""
import sys, os

if len(sys.argv) < 2:
    sys.exit("Usage: python3 apply_keyswitch_patch.py /path/to/openfhe-development")

P = os.path.join(sys.argv[1], "src/pke/lib/keyswitch/keyswitch-hybrid.cpp")
if not os.path.exists(P):
    sys.exit(f"not found: {P}")

src = open(P).read()

if "gpu_keyswitch_inner_product" in src:
    sys.exit("[=] already patched — nothing to do")

# ── 1. extern "C" declaration before the namespace ──
DECL = """
#ifdef GPU_HAL_PATCHED_V8
#include <cstdlib>
#include <cstdint>
#include <vector>
extern "C" void gpu_keyswitch_inner_product(
    const uint64_t** c, const uint64_t** b, const uint64_t** a,
    uint64_t** out0, uint64_t** out1,
    const uint64_t* q, uint32_t ring, uint32_t limit, uint32_t towers);
#endif // GPU_HAL_PATCHED_V8
"""
NS = "namespace lbcrypto {"
if NS not in src:
    sys.exit("[!] anchor 'namespace lbcrypto {' not found")
src = src.replace(NS, DECL + "\n" + NS, 1)

# ── 2. GPU dispatch in front of the inner-product loop ──
OLD = """    for (uint32_t j = 0; j < limit; ++j) {
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(sizeQlP))
        for (uint32_t i = 0; i < sizeQlP; ++i) {
            const auto idx  = (i >= sizeQl) ? i + delta : i;
            const auto& cji = (*digits)[j].GetElementAtIndex(i);
            const auto& bji = bv[j].GetElementAtIndex(idx);
            const auto& aji = av[j].GetElementAtIndex(idx);
            elements[0].SetElementAtIndex(i, elements[0].GetElementAtIndex(i) + cji * bji);
            elements[1].SetElementAtIndex(i, elements[1].GetElementAtIndex(i) + cji * aji);
        }
    }"""

NEW = """#ifdef GPU_HAL_PATCHED_V8
    {
        static const bool gpu_ks_on = [] {
            const char* e = std::getenv("OPENFHE_GPU_KS");
            return (e && e[0] == '1');
        }();
        const uint32_t ringDim = (uint32_t)(*digits)[0].GetRingDimension();
        if (gpu_ks_on && ringDim >= 4096 && limit >= 1 && sizeQlP >= 1) {
            std::vector<const uint64_t*> pc((size_t)limit * sizeQlP);
            std::vector<const uint64_t*> pb((size_t)limit * sizeQlP);
            std::vector<const uint64_t*> pa((size_t)limit * sizeQlP);
            std::vector<uint64_t*> po0(sizeQlP), po1(sizeQlP);
            std::vector<uint64_t> mods(sizeQlP);
            for (uint32_t j = 0; j < limit; ++j) {
                for (uint32_t i = 0; i < sizeQlP; ++i) {
                    const auto idx = (i >= sizeQl) ? i + delta : i;
                    pc[(size_t)j * sizeQlP + i] = reinterpret_cast<const uint64_t*>(
                        &(*digits)[j].GetElementAtIndex(i).GetValues()[0]);
                    pb[(size_t)j * sizeQlP + i] = reinterpret_cast<const uint64_t*>(
                        &bv[j].GetElementAtIndex(idx).GetValues()[0]);
                    pa[(size_t)j * sizeQlP + i] = reinterpret_cast<const uint64_t*>(
                        &av[j].GetElementAtIndex(idx).GetValues()[0]);
                }
            }
            for (uint32_t i = 0; i < sizeQlP; ++i) {
                po0[i] = const_cast<uint64_t*>(reinterpret_cast<const uint64_t*>(
                    &elements[0].GetElementAtIndex(i).GetValues()[0]));
                po1[i] = const_cast<uint64_t*>(reinterpret_cast<const uint64_t*>(
                    &elements[1].GetElementAtIndex(i).GetValues()[0]));
                mods[i] = (*digits)[0].GetElementAtIndex(i).GetModulus().ConvertToInt();
            }
            gpu_keyswitch_inner_product(pc.data(), pb.data(), pa.data(),
                                        po0.data(), po1.data(), mods.data(),
                                        ringDim, limit, sizeQlP);
            return result;
        }
    }
#endif // GPU_HAL_PATCHED_V8
""" + OLD

if OLD not in src:
    print("[!] ANCHOR NOT FOUND. Inner-product region for diagnosis:")
    k = src.find("EvalFastKeySwitchCoreExt")
    print(src[k:k + 2600] if k >= 0 else "(function not found at all)")
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
open(P, "w").write(src)
print("[+] keyswitch-hybrid.cpp patched: inner product dispatches to GPU (fallback intact)")

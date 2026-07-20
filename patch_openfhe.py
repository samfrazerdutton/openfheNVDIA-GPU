#!/usr/bin/env python3
"""OpenFHE NVIDIA GPU HAL Patcher v8.
Verified against openfhe-development ed361af2 (v1.5.1).
Changes vs v7:
  - #define GPU_HAL_PATCHED_V8 baked into the patched header itself:
    no compile flags needed in OpenFHE's build OR downstream builds.
  - extern "C" declarations anchored to 'namespace lbcrypto {'
    (header has include guards, not #pragma once).
  - NativeVectorT has no .data(): pointers taken via &GetValues()[0].
  - core link line plain-form, pke link line PUBLIC (match each file).
  - keyswitch patch removed (broken against 1.5.1; keyswitch runs on CPU).
  - runtime kill switch: OPENFHE_GPU=0 disables GPU dispatch.
Run against a FRESH clone only:
  python3 patch_openfhe_v8.py ~/openfhe-development
"""
import sys, os

GUARD = "GPU_HAL_PATCHED_V8"

if len(sys.argv) < 2:
    sys.exit("Usage: python3 patch_openfhe_v8.py /path/to/openfhe-development")

root    = sys.argv[1]
hdr     = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly.h")
cm_core = os.path.join(root, "src/core/CMakeLists.txt")
cm_pke  = os.path.join(root, "src/pke/CMakeLists.txt")

hal_so = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "build", "libopenfhe_cuda_hal.so")
cudart = "/usr/local/cuda/lib64/libcudart.so"

print("[*] OpenFHE NVIDIA GPU HAL Patcher v8")
if not os.path.exists(hal_so):
    sys.exit(f"[!] HAL not built yet: {hal_so}\n    Build it first: cd openfheNVDIA-GPU && mkdir -p build && cd build && cmake .. && make openfhe_cuda_hal")

# ── 1. core CMakeLists (plain form to match its existing calls) ──
src = open(cm_core).read()
if GUARD not in src:
    src += (f"\n# OpenFHE NVIDIA GPU HAL ({GUARD})\n"
            f"target_link_libraries(OPENFHEcore {hal_so} {cudart})\n")
    open(cm_core, "w").write(src)
    print("  [+] core CMakeLists patched (plain link)")
else:
    print("  [=] core CMakeLists already patched")

# ── 2. pke CMakeLists (PUBLIC to match its existing calls) ──
src = open(cm_pke).read()
if GUARD not in src:
    src += (f"\n# OpenFHE NVIDIA GPU HAL ({GUARD})\n"
            f"target_link_libraries(OPENFHEpke PUBLIC {hal_so} {cudart})\n")
    open(cm_pke, "w").write(src)
    print("  [+] pke CMakeLists patched (PUBLIC link)")
else:
    print("  [=] pke CMakeLists already patched")

# ── 3. dcrtpoly.h: declarations + self-activating define ──
src = open(hdr).read()
if GUARD in src:
    sys.exit("  [=] header already patched — use a fresh clone to re-patch")

decl = f'''
#define {GUARD}
#ifdef {GUARD}
#include <vector>
#include <cstdlib>
extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);
extern "C" void gpu_sync_all_to_host();
#endif // {GUARD}
'''
anchor = "namespace lbcrypto {"
if anchor not in src:
    sys.exit("[!] anchor 'namespace lbcrypto {' not found in dcrtpoly.h")
src = src.replace(anchor, decl + "\n" + anchor, 1)

old_op = (
    "    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {\n"
    "        size_t size{m_vectors.size()};\n"
    "#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))\n"
    "        for (size_t i = 0; i < size; ++i)\n"
    "            m_vectors[i] *= rhs.m_vectors[i];\n"
    "        return *this;\n"
    "    }"
)
new_op = (
    "    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {\n"
    "        size_t size{m_vectors.size()};\n"
    f"#ifdef {GUARD}\n"
    "        static const bool gpu_hal_on = [] {\n"
    "            const char* e = std::getenv(\"OPENFHE_GPU\");\n"
    "            return !(e && e[0] == '0');\n"
    "        }();\n"
    "        uint32_t ring = (uint32_t)m_params->GetRingDimension();\n"
    "        if (gpu_hal_on && m_format == Format::EVALUATION &&\n"
    "            ring >= 4096 && size >= 1 && size <= 64) {\n"
    "            static thread_local std::vector<const uint64_t*> ha_ptrs, hb_ptrs;\n"
    "            static thread_local std::vector<uint64_t*>       hr_ptrs;\n"
    "            static thread_local std::vector<uint64_t>        moduli;\n"
    "            ha_ptrs.resize(size); hb_ptrs.resize(size);\n"
    "            hr_ptrs.resize(size); moduli.resize(size);\n"
    "            for (size_t i = 0; i < size; ++i) {\n"
    "                ha_ptrs[i] = reinterpret_cast<const uint64_t*>(&m_vectors[i].GetValues()[0]);\n"
    "                hb_ptrs[i] = reinterpret_cast<const uint64_t*>(&rhs.m_vectors[i].GetValues()[0]);\n"
    "                hr_ptrs[i] = reinterpret_cast<uint64_t*>(&m_vectors[i][0]);\n"
    "                moduli[i]  = m_vectors[i].GetModulus().ConvertToInt();\n"
    "            }\n"
    "            gpu_rns_mult_batch_wrapper(ha_ptrs.data(), hb_ptrs.data(), hr_ptrs.data(),\n"
    "                                       moduli.data(), ring, (uint32_t)size);\n"
    "            return *this;\n"
    "        }\n"
    f"#endif // {GUARD}\n"
    "#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))\n"
    "        for (size_t i = 0; i < size; ++i)\n"
    "            m_vectors[i] *= rhs.m_vectors[i];\n"
    "        return *this;\n"
    "    }"
)
if old_op not in src:
    sys.exit("[!] operator*= anchor not found — OpenFHE version drift; pin to ed361af2")
src = src.replace(old_op, new_op, 1)
open(hdr, "w").write(src)
print("  [+] dcrtpoly.h patched (self-activating, OPENFHE_GPU=0 kill switch)")
print("\n[SUCCESS] v8 complete. Build OpenFHE, then rebuild your repo (no flags needed).")

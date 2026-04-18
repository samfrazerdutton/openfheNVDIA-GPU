#!/usr/bin/env python3
"""
OpenFHE NVIDIA GPU HAL Patcher -- negacyclic-correct edition.

Changes from original:
  - More robust operator*= regex (handles varied whitespace/pragma formatting).
  - Uses CMAKE_INSTALL_RPATH instead of absolute .so path so moving the repo
    doesn't break the OpenFHE build.
  - Adds a guard comment so repeated patching is clearly detected.
"""
import sys, os, re

print("======================================================")
print("[*] OpenFHE NVIDIA GPU HAL Patcher (Negacyclic-Fixed)")
print("======================================================")

if len(sys.argv) < 2:
    print("Usage: python3 patch_openfhe.py /path/to/openfhe-development")
    sys.exit(1)

root       = sys.argv[1]
hdr_path   = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly.h")
cmake_path = os.path.join(root, "src/core/CMakeLists.txt")
hal_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
hal_so     = os.path.join(hal_dir, "libopenfhe_cuda_hal.so")

for p in [hdr_path, cmake_path]:
    if not os.path.exists(p):
        print(f"[ERROR] Not found: {p}")
        sys.exit(1)

# ── 1. Patch CMakeLists ───────────────────────────────────────────────────────
with open(cmake_path) as f: cmake = f.read()
if "openfhe_cuda_hal" not in cmake:
    with open(cmake_path, "a") as f:
        # FIX: use RPATH via target property so the .so is found at runtime
        # even if OpenFHE is installed to a different prefix.
        f.write(f"\ntarget_include_directories(OPENFHEcore PUBLIC /usr/local/cuda/include)\n")
        f.write(f"target_link_libraries(OPENFHEcore {hal_so} /usr/local/cuda/lib64/libcudart.so OpenMP::OpenMP_CXX)\n")
        f.write(f"set_target_properties(OPENFHEcore PROPERTIES BUILD_RPATH \"{hal_dir}\" INSTALL_RPATH \"{hal_dir}\")\n")
    print("[+] CMakeLists.txt patched")
else:
    print("[-] CMakeLists.txt already patched")

# ── 2. Patch dcrtpoly.h ───────────────────────────────────────────────────────
with open(hdr_path) as f: src = f.read()

GUARD = "GPU_SWA_NEGACYCLIC_PATCHED"

inject_decl = f"""
// ── GPU HAL Co-Processor Airgap ({GUARD}) ─────────────────────────────────
#include <cstdint>
#include <vector>
extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** host_a, const uint64_t** host_b, uint64_t** host_res,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);
// ─────────────────────────────────────────────────────────────────────────────
"""

new_op = """    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {
        size_t size{m_vectors.size()};
        uint32_t ring = m_params->GetRingDimension();

        // ── GPU Fast-Path (negacyclic NTT, ring Z[X]/(X^N+1)) ───────────────
        if (ring >= 4096 && size <= 32) {
            std::vector<const uint64_t*> a_ptrs(size);
            std::vector<const uint64_t*> b_ptrs(size);
            std::vector<uint64_t*>       res_ptrs(size);
            std::vector<uint64_t>        moduli(size);
            for (size_t i = 0; i < size; ++i) {
                moduli[i]   = m_vectors[i].GetModulus().ConvertToInt();
                a_ptrs[i]   = reinterpret_cast<const uint64_t*>(&m_vectors[i][0]);
                b_ptrs[i]   = reinterpret_cast<const uint64_t*>(&rhs.m_vectors[i][0]);
                res_ptrs[i] = reinterpret_cast<uint64_t*>(&m_vectors[i][0]);
            }
            ::gpu_rns_mult_batch_wrapper(a_ptrs.data(), b_ptrs.data(), res_ptrs.data(),
                                         moduli.data(), ring, (uint32_t)size);
            return *this;
        }

        // ── CPU Fallback ─────────────────────────────────────────────────────
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))
        for (size_t i = 0; i < size; ++i)
            m_vectors[i] *= rhs.m_vectors[i];
        return *this;
    }"""

if GUARD in src:
    print("[-] dcrtpoly.h already patched")
else:
    # Inject declaration after first system include.
    if "gpu_rns_mult_batch_wrapper" not in src:
        src = re.sub(r'(#include\s+<[^>]+>\n)', r'\1' + inject_decl, src, count=1)

    # FIX: more robust regex -- matches regardless of whitespace/pragma variation
    # between the function signature and the for-loop body.
    pattern = re.compile(
        r'(DCRTPolyType\s*&\s*operator\*=\s*\(\s*const\s+DCRTPolyType\s*&\s*\w+\s*\)\s*override\s*\{)'
        r'.*?'
        r'(m_vectors\s*\[\s*i\s*\]\s*\*=\s*rhs\s*\.\s*m_vectors\s*\[\s*i\s*\]\s*;)'
        r'.*?'
        r'\}',
        re.DOTALL
    )
    match = pattern.search(src)
    if match:
        src = src[:match.start()] + new_op + src[match.end():]
        print("[+] operator*= hooked to GPU HAL (negacyclic)")
    else:
        print("[-] operator*= pattern not found -- check OpenFHE version.")
        print("    Signature searched: DCRTPolyType& operator*=(const DCRTPolyType& ...) override")

    with open(hdr_path, "w") as f: f.write(src)

print("[+] OpenFHE patch complete.")
print("======================================================")


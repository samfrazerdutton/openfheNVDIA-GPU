#!/usr/bin/env python3
"""
OpenFHE NVIDIA GPU HAL Patcher  (v3 — negacyclic-correct)

Hooks operator*= on DCRTPolyImpl so that EvalMult / KeySwitch dispatch to:

  gpu_rns_mult_batch_wrapper   — when ciphertexts are ALREADY in NTT (evaluation) domain
                                 (pointwise multiply, no NTT round-trip needed)

  gpu_poly_mult_wrapper        — when ciphertexts are in COEFFICIENT domain
                                 (full NTT + pointwise + INTT)

OpenFHE CKKS operates in EVALUATION format by default, so the fast RNS path
activates for virtually all EvalMult calls. The poly path is kept for the rare
cases where a ciphertext is in COEFFICIENT format (e.g., post-bootstrap).

GPU fast-path activates when ring >= 4096 and num_towers <= 32.
"""
import sys, os, re

GUARD = "GPU_HAL_PATCHED_V3"

print("=" * 60)
print("[*] OpenFHE NVIDIA GPU HAL Patcher v3")
print("=" * 60)

if len(sys.argv) < 2:
    print("Usage: python3 patch_openfhe.py /path/to/openfhe-development")
    sys.exit(1)

root       = sys.argv[1]
hdr_path   = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly.h")
cmake_path = os.path.join(root, "src/core/CMakeLists.txt")
hal_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
hal_so     = os.path.join(hal_dir, "libopenfhe_cuda_hal.so")
hal_inc    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")

for p in [hdr_path, cmake_path, hal_so]:
    if not os.path.exists(p):
        print(f"[ERROR] Not found: {p}")
        if p == hal_so:
            print("        Build the HAL first:")
            print("          mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)")
        sys.exit(1)

# ── 1. Patch CMakeLists ───────────────────────────────────────────────────────
with open(cmake_path) as f: cmake = f.read()
if "openfhe_cuda_hal" not in cmake:
    with open(cmake_path, "a") as f:
        f.write(f"\n# ── OpenFHE NVIDIA GPU HAL ({GUARD}) ──\n")
        f.write(f"target_include_directories(OPENFHEcore PUBLIC\n")
        f.write(f"    /usr/local/cuda/include\n")
        f.write(f"    {hal_inc}\n")
        f.write(f")\n")
        f.write(f"target_link_libraries(OPENFHEcore\n")
        f.write(f"    {hal_so}\n")
        f.write(f"    /usr/local/cuda/lib64/libcudart.so\n")
        f.write(f"    OpenMP::OpenMP_CXX\n")
        f.write(f")\n")
        f.write(f"set_target_properties(OPENFHEcore PROPERTIES\n")
        f.write(f"    BUILD_RPATH   \"{hal_dir}\"\n")
        f.write(f"    INSTALL_RPATH \"{hal_dir}\"\n")
        f.write(f")\n")
    print("[+] CMakeLists.txt patched")
else:
    print("[-] CMakeLists.txt already patched")

# ── 2. Patch dcrtpoly.h ───────────────────────────────────────────────────────
with open(hdr_path) as f: src = f.read()

if GUARD in src:
    print(f"[-] dcrtpoly.h already patched ({GUARD})")
    print("[+] Patch complete.")
    sys.exit(0)

# Warn about old patches — don't silently overwrite.
for old in ["GPU_HAL_PATCHED_V2", "GPU_HAL_PATCHED", "GPU_SWA_NEGACYCLIC_PATCHED"]:
    if old in src:
        print(f"[!] Old patch detected ({old}).")
        print("    Remove the old patch manually from dcrtpoly.h, then re-run.")
        print(f"    Search for '{old}' in {hdr_path}")
        sys.exit(1)

gpu_decl = f"""
// ── OpenFHE NVIDIA GPU HAL ({GUARD}) ─────────────────────────────────────────
#include <cstdint>
#include <vector>
extern "C" {{
    // Pointwise RNS multiply (NTT domain, no round-trip).
    void gpu_rns_mult_batch_wrapper(
        const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
        const uint64_t* q, uint32_t ring, uint32_t num_towers);
    // Full polynomial multiply (coeff domain → NTT → pointwise → INTT).
    void gpu_poly_mult_wrapper(
        const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
        const uint64_t* q, uint32_t ring, uint32_t num_towers);
}}
// ─────────────────────────────────────────────────────────────────────────────
"""

new_op = """    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {
        size_t   ntowers = m_vectors.size();
        uint32_t ring    = m_params->GetRingDimension();

        if (ring >= 4096 && ntowers <= 32) {
            std::vector<const uint64_t*> a_ptrs(ntowers);
            std::vector<const uint64_t*> b_ptrs(ntowers);
            std::vector<uint64_t*>       r_ptrs(ntowers);
            std::vector<uint64_t>        moduli(ntowers);
            for (size_t i = 0; i < ntowers; ++i) {
                moduli[i] = m_vectors[i].GetModulus().ConvertToInt();
                a_ptrs[i] = reinterpret_cast<const uint64_t*>(
                                m_vectors[i].GetValues().data());
                b_ptrs[i] = reinterpret_cast<const uint64_t*>(
                                rhs.m_vectors[i].GetValues().data());
                r_ptrs[i] = reinterpret_cast<uint64_t*>(
                                m_vectors[i].GetValues().data());
            }
            // Choose kernel based on polynomial format:
            // EVALUATION = NTT domain → pointwise multiply only (fast path).
            // COEFFICIENT = coeff domain → full NTT round-trip needed.
            if (m_format == Format::EVALUATION) {
                ::gpu_rns_mult_batch_wrapper(
                    a_ptrs.data(), b_ptrs.data(), r_ptrs.data(),
                    moduli.data(), ring, static_cast<uint32_t>(ntowers));
            } else {
                ::gpu_poly_mult_wrapper(
                    a_ptrs.data(), b_ptrs.data(), r_ptrs.data(),
                    moduli.data(), ring, static_cast<uint32_t>(ntowers));
            }
            return *this;
        }

        // CPU fallback for small rings.
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(ntowers))
        for (size_t i = 0; i < ntowers; ++i)
            m_vectors[i] *= rhs.m_vectors[i];
        return *this;
    }"""

# Inject extern "C" block after the first #include in the file.
src = re.sub(r'(#include\s+[<"][^>"]+[>"]\n)', r'\\1' + gpu_decl, src, count=1)

# Match operator*= across OpenFHE 1.x and 2.x source layouts.
patterns = [
    re.compile(
        r'DCRTPolyType\s*&\s*operator\*=\s*\(\s*const\s+DCRTPolyType\s*&\s*\w+\s*\)\s*override\s*\{'
        r'.*?m_vectors\s*\[\s*i\s*\]\s*\*=\s*\w+\.m_vectors\s*\[\s*i\s*\]\s*;.*?\n\s*\}',
        re.DOTALL),
    re.compile(
        r'DCRTPolyType\s*&\s*operator\*=\s*\(\s*const\s+DCRTPolyType\s*&\s*\w+\s*\)\s*override\s*\{'
        r'.*?ParallelFor\s*\(.*?m_vectors\s*\[\s*\w+\s*\]\s*\*=.*?\}\s*\).*?\n\s*\}',
        re.DOTALL),
    re.compile(
        r'DCRTPolyType\s*&\s*operator\*=\s*\(\s*const\s+DCRTPolyType\s*&\s*\w+\s*\)\s*override\s*\{'
        r'.*?std::transform\s*\(.*?\n\s*\}',
        re.DOTALL),
]
m = None
for idx, pat in enumerate(patterns):
    m = pat.search(src)
    if m:
        print(f"[+] operator*= matched by pattern {idx + 1}")
        break

if not m:
    print("[ERROR] Could not match operator*= in dcrtpoly.h.")
    print(f"        Manual inspection: grep -n 'operator\\*=' {hdr_path}")
    sys.exit(1)

src = src[:m.start()] + new_op + src[m.end():]
with open(hdr_path, "w") as f:
    f.write(src)
print("[+] operator*= hooked to GPU HAL (RNS + poly paths)")

print("[+] Patch complete.")
print()
print("    Rebuild OpenFHE:")
print("      cd /path/to/openfhe-development/build")
print("      make -j$(nproc)")
print("=" * 60)

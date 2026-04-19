#!/usr/bin/env python3
"""
OpenFHE NVIDIA GPU HAL Patcher

Hooks two points in OpenFHE:
  1. DCRTPolyImpl::operator*=  -- pointwise RNS multiply (coefficient-wise)
     → calls gpu_rns_mult_batch_wrapper (Montgomery, no NTT)
     → used by BGV/BFV EvalMult and CKKS rescaling

  2. DCRTPolyImpl::operator*=  in EVALUATION format
     → same hook covers both since OpenFHE calls *= in NTT domain

The GPU path activates when ring >= 4096 and towers <= 32.
Smaller rings fall through to the CPU path automatically.
"""
import sys, os, re

print("=" * 54)
print("[*] OpenFHE NVIDIA GPU HAL Patcher")
print("=" * 54)

if len(sys.argv) < 2:
    print("Usage: python3 patch_openfhe.py /path/to/openfhe-development")
    sys.exit(1)

root       = sys.argv[1]
hdr_path   = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly.h")
cmake_path = os.path.join(root, "src/core/CMakeLists.txt")
hal_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
hal_so     = os.path.join(hal_dir, "libopenfhe_cuda_hal.so")

for p in [hdr_path, cmake_path, hal_so]:
    if not os.path.exists(p):
        print(f"[ERROR] Not found: {p}")
        print(f"        Build the HAL first: cd build && make -j$(nproc)")
        sys.exit(1)

# ── 1. Patch CMakeLists ───────────────────────────────────────────────────────
with open(cmake_path) as f: cmake = f.read()
if "openfhe_cuda_hal" not in cmake:
    with open(cmake_path, "a") as f:
        f.write(f"\ntarget_include_directories(OPENFHEcore PUBLIC /usr/local/cuda/include)\n")
        f.write(f"target_link_libraries(OPENFHEcore {hal_so} /usr/local/cuda/lib64/libcudart.so OpenMP::OpenMP_CXX)\n")
        f.write(f"set_target_properties(OPENFHEcore PROPERTIES BUILD_RPATH \"{hal_dir}\" INSTALL_RPATH \"{hal_dir}\")\n")
    print("[+] CMakeLists.txt patched")
else:
    print("[-] CMakeLists.txt already patched")

# ── 2. Patch dcrtpoly.h ───────────────────────────────────────────────────────
with open(hdr_path) as f: src = f.read()

GUARD = "GPU_HAL_PATCHED_V2"

gpu_decl = f"""
// ── GPU HAL ({GUARD}) ────────────────────────────────────────────────────────
#include <cstdint>
#include <vector>
extern "C" {{
    void gpu_rns_mult_batch_wrapper(
        const uint64_t** ha, const uint64_t** hb, uint64_t** hr,
        const uint64_t* q, uint32_t ring, uint32_t num_towers);
}}
// ─────────────────────────────────────────────────────────────────────────────
"""

new_op = """    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {
        size_t   ntowers = m_vectors.size();
        uint32_t ring    = m_params->GetRingDimension();

        // GPU fast-path: pointwise RNS multiply in NTT domain.
        // Activates for ring >= 4096 (covers all standard CKKS/BGV params).
        if (ring >= 4096 && ntowers <= 32) {
            std::vector<const uint64_t*> a_ptrs(ntowers);
            std::vector<const uint64_t*> b_ptrs(ntowers);
            std::vector<uint64_t*>       r_ptrs(ntowers);
            std::vector<uint64_t>        moduli(ntowers);
            for (size_t i = 0; i < ntowers; ++i) {
                moduli[i] = m_vectors[i].GetModulus().ConvertToInt();
                a_ptrs[i] = reinterpret_cast<const uint64_t*>(m_vectors[i].GetValues().data());
                b_ptrs[i] = reinterpret_cast<const uint64_t*>(rhs.m_vectors[i].GetValues().data());
                r_ptrs[i] = reinterpret_cast<uint64_t*>(m_vectors[i].GetValues().data());
            }
            ::gpu_rns_mult_batch_wrapper(
                a_ptrs.data(), b_ptrs.data(), r_ptrs.data(),
                moduli.data(), ring, static_cast<uint32_t>(ntowers));
            return *this;
        }

        // CPU fallback for small rings or unusual tower counts.
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(ntowers))
        for (size_t i = 0; i < ntowers; ++i)
            m_vectors[i] *= rhs.m_vectors[i];
        return *this;
    }"""

if GUARD in src:
    print("[-] dcrtpoly.h already patched (V2)")
else:
    # Remove old V1 patch if present
    if "GPU_SWA_NEGACYCLIC_PATCHED" in src or "GPU_HAL_PATCHED" in src:
        print("[!] Old patch detected -- remove it manually from dcrtpoly.h first,")
        print("    then re-run this script. Skipping header patch.")
    else:
        # Inject extern "C" declaration after first #include
        if "gpu_rns_mult_batch_wrapper" not in src:
            src = re.sub(r'(#include\s+[<"][^>"]+[>"]\n)', r'\1' + gpu_decl, src, count=1)

        # Replace operator*=
        # Try multiple patterns to handle OpenFHE 1.x and 2.x source layouts.
        PATTERNS = [
            # OpenFHE <= 1.1: simple #pragma omp for with direct *= in loop body
            re.compile(
                r'DCRTPolyType\s*&\s*operator\*=\s*\(\s*const\s+DCRTPolyType\s*&\s*\w+\s*\)\s*override\s*\{'
                r'.*?'
                r'm_vectors\s*\[\s*i\s*\]\s*\*=\s*\w+\.m_vectors\s*\[\s*i\s*\]\s*;'
                r'.*?'
                r'\n\s*\}',
                re.DOTALL),
            # OpenFHE >= 1.2: lambda-based parallel_for form
            re.compile(
                r'DCRTPolyType\s*&\s*operator\*=\s*\(\s*const\s+DCRTPolyType\s*&\s*\w+\s*\)\s*override\s*\{'
                r'.*?'
                r'ParallelFor\s*\(.*?m_vectors\s*\[\s*\w+\s*\]\s*\*=.*?\}\s*\)'
                r'.*?'
                r'\n\s*\}',
                re.DOTALL),
            # OpenFHE >= 1.3: std::transform variant
            re.compile(
                r'DCRTPolyType\s*&\s*operator\*=\s*\(\s*const\s+DCRTPolyType\s*&\s*\w+\s*\)\s*override\s*\{'
                r'.*?'
                r'std::transform\s*\('
                r'.*?'
                r'\n\s*\}',
                re.DOTALL),
        ]
        m = None
        for i, pat in enumerate(PATTERNS):
            m = pat.search(src)
            if m:
                print(f"[+] operator*= matched by pattern {i+1}")
                break
        if m:
            src = src[:m.start()] + new_op + src[m.end():]
            with open(hdr_path, "w") as f: f.write(src)
            print("[+] operator*= hooked to GPU HAL")
        else:
            print("[ERROR] No operator*= pattern matched across all 3 variants.")
            print("        Manual inspection required:")
            print(f"        grep -n 'operator\*=' {hdr_path}")
            sys.exit(1)

print("[+] Patch complete.")
print("    Rebuild OpenFHE: cd /path/to/openfhe-development/build && make -j$(nproc)")
print("=" * 54)

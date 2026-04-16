import sys, os, re

print("[*] OpenFHE NVIDIA GPU HAL Patcher v2")

if len(sys.argv) < 2:
    print("Usage: python3 patch_openfhe.py /path/to/openfhe-development")
    sys.exit(1)

root      = sys.argv[1]
hdr_path  = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly.h")
cmake_path= os.path.join(root, "src/core/CMakeLists.txt")
hal_so    = "/mnt/c/Users/samca/openfhe-cuda/build/libopenfhe_cuda_hal.so"

# ── 1. Patch CMakeLists ───────────────────────────────────────────────────────
with open(cmake_path) as f: cmake = f.read()
if "openfhe_cuda_hal" not in cmake:
    with open(cmake_path, "a") as f:
        f.write(f"""
target_include_directories(OPENFHEcore PUBLIC /usr/local/cuda/include)
target_link_libraries(OPENFHEcore {hal_so} /usr/local/cuda/lib64/libcudart.so)
""")
    print("[+] CMakeLists.txt patched")
else:
    print("[-] CMakeLists.txt already patched")

# ── 2. Patch dcrtpoly.h ───────────────────────────────────────────────────────
with open(hdr_path) as f: src = f.read()

if "gpu_rns_mult_wrapper" in src:
    print("[-] dcrtpoly.h already patched")
    sys.exit(0)

# Inject C declaration at top of file (after first #pragma once / include)
inject_decl = """
// ── GPU HAL C-Airgap ─────────────────────────────────────────────────────────
#include <cstdint>
extern "C" void gpu_rns_mult_wrapper(
    const uint64_t* a, const uint64_t* b, uint64_t* res,
    uint64_t q, uint32_t ring, uint32_t tower_idx);
// ─────────────────────────────────────────────────────────────────────────────

"""

# Insert after the first #include line
src = re.sub(r'(#include\s+<[^>]+>\n)', r'\1' + inject_decl, src, count=1)

# Replace operator*= — exact pattern from OpenFHE source
old_op = """    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {
        size_t size{m_vectors.size()};
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))
        for (size_t i = 0; i < size; ++i)
            m_vectors[i] *= rhs.m_vectors[i];
        return *this;
    }"""

new_op = """    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {
        size_t size{m_vectors.size()};
        uint32_t ring = m_params->GetRingDimension();
        // ── GPU fast-path for large rings (bootstrapping workloads) ──────────
        if (ring >= 4096) {
            for (size_t i = 0; i < size; ++i) {
                uint64_t q = m_vectors[i].GetModulus().ConvertToInt();
                std::vector<uint64_t> ha(ring), hb(ring), hres(ring);
                for (uint32_t j = 0; j < ring; ++j) {
                    ha[j]  = m_vectors[i][j].ConvertToInt();
                    hb[j]  = rhs.m_vectors[i][j].ConvertToInt();
                }
                ::gpu_rns_mult_wrapper(ha.data(), hb.data(), hres.data(),
                                       q, ring, (uint32_t)i);
                for (uint32_t j = 0; j < ring; ++j)
                    m_vectors[i][j] = hres[j];
            }
            return *this;
        }
        // ── CPU fallback for small rings ─────────────────────────────────────
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))
        for (size_t i = 0; i < size; ++i)
            m_vectors[i] *= rhs.m_vectors[i];
        return *this;
    }"""

if old_op in src:
    src = src.replace(old_op, new_op)
    print("[+] operator*= patched with GPU fast-path (ring >= 4096)")
else:
    print("[!] WARNING: operator*= pattern not found — OpenFHE version mismatch?")
    print("    Check dcrtpoly.h manually and adjust old_op string in this script.")
    sys.exit(1)

with open(hdr_path, "w") as f: f.write(src)
print("[+] dcrtpoly.h written")
print("[SUCCESS] Run: cd openfhe-development/build && cmake .. && make -j$(nproc)")

import sys, os, re

print("======================================================")
print("[*] OpenFHE NVIDIA GPU SWA Patcher (Final)")
print("======================================================")

if len(sys.argv) < 2:
    print("Usage: python3 patch_openfhe.py /path/to/openfhe-development")
    sys.exit(1)

root      = sys.argv[1]
hdr_path  = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly.h")
cmake_path= os.path.join(root, "src/core/CMakeLists.txt")
hal_so    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "libopenfhe_cuda_hal.so")

# ── 1. Patch CMakeLists ───────────────────────────────────────────────────────
with open(cmake_path) as f: cmake = f.read()
if "openfhe_cuda_hal" not in cmake:
    with open(cmake_path, "a") as f:
        f.write(f"\ntarget_include_directories(OPENFHEcore PUBLIC /usr/local/cuda/include)\n")
        f.write(f"target_link_libraries(OPENFHEcore {hal_so} /usr/local/cuda/lib64/libcudart.so OpenMP::OpenMP_CXX)\n")
    print("[+] CMakeLists.txt patched")
else:
    print("[-] CMakeLists.txt already patched")

# ── 2. Patch dcrtpoly.h (Injecting SWA) ───────────────────────────────────────
with open(hdr_path) as f: src = f.read()

inject_decl = """
// ── GPU SWA Co-Processor Airgap ──────────────────────────────────────────────
#include <cstdint>
#include <vector>
extern "C" void gpu_rns_mult_batch_wrapper(
    const uint64_t** host_a, const uint64_t** host_b, uint64_t** host_res,
    const uint64_t* q, uint32_t ring, uint32_t num_towers);
// ─────────────────────────────────────────────────────────────────────────────
"""

if "gpu_rns_mult_batch_wrapper" not in src:
    src = re.sub(r'(#include\s+<[^>]+>\n)', r'\1' + inject_decl, src, count=1)

new_op = """    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {
        size_t size{m_vectors.size()};
        uint32_t ring = m_params->GetRingDimension();

        // ── SWA GPU Fast-Path ────────────────────────────────────────────────
        if (ring >= 4096 && size <= 32) {
            std::vector<const uint64_t*> a_ptrs(size);
            std::vector<const uint64_t*> b_ptrs(size);
            std::vector<uint64_t*> res_ptrs(size);
            std::vector<uint64_t> moduli(size);

            for (size_t i = 0; i < size; ++i) {
                moduli[i] = m_vectors[i].GetModulus().ConvertToInt();
                // Extract raw host pointers for the SWA Cache
                a_ptrs[i] = reinterpret_cast<const uint64_t*>(&m_vectors[i][0]);
                b_ptrs[i] = reinterpret_cast<const uint64_t*>(&rhs.m_vectors[i][0]);
                res_ptrs[i] = reinterpret_cast<uint64_t*>(&m_vectors[i][0]);
            }

            // The HAL will check if these pointers exist in VRAM. 
            // If they do, PCIe transfer is SKIPPED!
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

pattern = re.compile(
    r'(DCRTPolyType&\s+operator\*=\s*\(const DCRTPolyType&\s+rhs\)\s+override\s*\{[^}]+?for\s*\(size_t i[^}]+?m_vectors\[i\]\s*\*=\s*rhs\.m_vectors\[i\];[^}]+?\}\s*\})',
    re.DOTALL
)

match = pattern.search(src)
if match:
    src = src[:match.start()] + new_op + src[match.end():]
    print("[+] OpenFHE operator*= dynamically hooked to SWA GPU Cache")
else:
    print("[-] operator*= pattern not found. May already be patched.")

with open(hdr_path, "w") as f: f.write(src)
print("[+] OpenFHE core successfully injected.")
print("======================================================")

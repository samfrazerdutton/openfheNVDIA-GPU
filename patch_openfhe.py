import sys
import os

print("[*] OpenFHE NVIDIA GPU HAL Installer")

if len(sys.argv) < 2:
    print("Usage: python3 patch_openfhe.py </path/to/openfhe-development>")
    sys.exit(1)

openfhe_path = sys.argv[1]
header_path = os.path.join(openfhe_path, "src/core/include/lattice/hal/default/dcrtpoly.h")
cmake_path = os.path.join(openfhe_path, "src/core/CMakeLists.txt")

# Patch CMake
with open(cmake_path, "r") as f:
    cmake_content = f.read()
if "libopenfhe_cuda_hal.so" not in cmake_content:
    with open(cmake_path, "a") as f:
        f.write('\ntarget_include_directories(OPENFHEcore PUBLIC /usr/local/cuda/include)\n')
        f.write('target_link_libraries(OPENFHEcore /mnt/c/Users/samca/openfhe-cuda/build/libopenfhe_cuda_hal.so /usr/local/cuda/lib64/libcudart.so)\n')
    print("[+] CMakeLists.txt patched with CUDA libraries.")

# Patch dcrtpoly.h
with open(header_path, "r") as f:
    content = f.read()

if "gpu_rns_mult_wrapper" not in content:
    top_inject = """#include <cstdint>\n#include <vector>\nextern "C" void gpu_rns_mult_wrapper(const uint64_t* a, const uint64_t* b, uint64_t* res, uint64_t q, unsigned __int128 mu, uint32_t ring);\n\n"""
    content = top_inject + content
    
    old_op = """    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {\n        size_t size{m_vectors.size()};\n#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))\n        for (size_t i = 0; i < size; ++i)\n            m_vectors[i] *= rhs.m_vectors[i];\n        return *this;\n    }"""
    
    new_op = """    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {\n        size_t size{m_vectors.size()};\n        uint32_t ring = m_params->GetRingDimension();\n        for (size_t i = 0; i < size; ++i) {\n            uint64_t q = m_vectors[i].GetModulus().ConvertToInt();\n            unsigned __int128 mu = ((unsigned __int128)1 << 64) / q;\n            std::vector<uint64_t> ha(ring), hb(ring), hres(ring);\n            for (uint32_t j = 0; j < ring; ++j) {\n                ha[j] = m_vectors[i][j].ConvertToInt();\n                hb[j] = rhs.m_vectors[i][j].ConvertToInt();\n            }\n            ::gpu_rns_mult_wrapper(ha.data(), hb.data(), hres.data(), q, mu, ring);\n            for (uint32_t j = 0; j < ring; ++j) {\n                m_vectors[i][j] = hres[j];\n            }\n        }\n        return *this;\n    }"""
    
    content = content.replace(old_op, new_op)
    with open(header_path, "w") as f:
        f.write(content)
    print("[+] dcrtpoly.h patched with CUDA airgap wrapper.")
else:
    print("[-] GPU wrapper already present in dcrtpoly.h")

print("[SUCCESS] OpenFHE is now GPU accelerated. Rebuild OpenFHE to apply.")

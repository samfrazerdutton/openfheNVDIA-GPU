#!/usr/bin/env python3
import sys, os, shutil

GUARD_CMAKE  = "GPU_HAL_PATCHED_V5"
GUARD_HDR    = "GPU_HAL_PATCHED_V5"

if len(sys.argv) < 2:
    print("Usage: python3 patch_openfhe.py /path/to/openfhe-development")
    sys.exit(1)

root      = sys.argv[1]
hdr_path  = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly.h")
cmake_path= os.path.join(root, "src/core/CMakeLists.txt")

hal_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
hal_so    = os.path.join(hal_dir, "libopenfhe_cuda_hal.so")
hal_inc   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")

print("[*] OpenFHE NVIDIA GPU HAL Patcher v5 (DAG Edition)")

# ── 1. CMakeLists ──
cmake = open(cmake_path).read()
if GUARD_CMAKE not in cmake:
    with open(cmake_path, "a") as f:
        f.write(f"\n# ── OpenFHE NVIDIA GPU HAL ({GUARD_CMAKE}) ──────────────────\n")
        f.write(f"target_compile_definitions(OPENFHEcore PUBLIC {GUARD_CMAKE})\n")
        f.write(f"target_include_directories(OPENFHEcore PUBLIC /usr/local/cuda/include {hal_inc})\n")
        f.write(f"target_link_libraries(OPENFHEcore {hal_so} /usr/local/cuda/lib64/libcudart.so OpenMP::OpenMP_CXX)\n")

# ── 2. dcrtpoly.h ──
src = open(hdr_path).read()
if GUARD_HDR not in src:
    shutil.copy(hdr_path, hdr_path + ".bak")
    
    old = (
        "    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {\n"
        "        size_t size{m_vectors.size()};\n"
        "#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))\n"
        "        for (size_t i = 0; i < size; ++i)\n"
        "            m_vectors[i] *= rhs.m_vectors[i];\n"
        "        return *this;\n"
        "    }"
    )
    new = (
        "    DCRTPolyType& operator*=(const DCRTPolyType& rhs) override {\n"
        "        size_t size{m_vectors.size()};\n"
        "#ifdef GPU_HAL_PATCHED_V5\n"
        "        uint32_t ring = m_params->GetRingDimension();\n"
        "        if (m_format == Format::EVALUATION && ring >= 4096 && size <= 32) {\n"
        "            for (size_t i = 0; i < size; ++i) {\n"
        "                void* host_a = (void*)m_vectors[i].GetValues().data();\n"
        "                void* host_b = (void*)rhs.m_vectors[i].GetValues().data();\n"
        "                size_t bytes = ring * sizeof(uint64_t);\n"
        "                \n"
        "                DagNode* node_a = GlobalDAG::GetOrLoadNode(host_a, bytes, 1, ring);\n"
        "                DagNode* node_b = GlobalDAG::GetOrLoadNode(host_b, bytes, 1, ring);\n"
        "                \n"
        "                void* vram_res;\n"
        "                cudaMalloc(&vram_res, bytes);\n"
        "                \n"
        "                DagNode* mult_node = GlobalDAG::compiler.CreateNode(FheOpcode::MULT_RNS, node_a, node_b, vram_res, host_a, bytes, 1, ring);\n"
        "                GlobalDAG::node_registry[host_a] = mult_node;\n"
        "            }\n"
        "            if (!GlobalDAG::is_capturing) { GlobalDAG::ExecuteAndSync(); }\n"
        "            return *this;\n"
        "        }\n"
        "#endif\n"
        "#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))\n"
        "        for (size_t i = 0; i < size; ++i)\n"
        "            m_vectors[i] *= rhs.m_vectors[i];\n"
        "        return *this;\n"
        "    }"
    )
    
    inject = "\n#ifdef GPU_HAL_PATCHED_V5\n#include \"global_dag.h\"\n#endif\n"
    src = src.replace("#pragma once", "#pragma once" + inject, 1)
    src = src.replace(old, new)
    open(hdr_path, "w").write(src)

print("[SUCCESS] OpenFHE Patcher updated to DAG Engine (v5).")

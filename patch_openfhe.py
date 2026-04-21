#!/usr/bin/env python3
"""
OpenFHE NVIDIA GPU HAL Patcher v4
Tested against OpenFHE v1.4.x / v1.5.0 + openfheNVDIA-GPU main.

Usage:
  python3 patch_openfhe.py /path/to/openfhe-development

What it patches:
  1. src/core/CMakeLists.txt         — links libopenfhe_cuda_hal.so + CUDA
  2. src/core/.../dcrtpoly.h         — operator*= GPU fast-path (ring>=4096, towers<=32)
  3. src/core/.../dcrtpoly-impl.h    — DropLastElementAndScale marker

Safe to re-run: all patches are guarded.
"""
import sys, os, shutil

GUARD_CMAKE  = "GPU_HAL_PATCHED_V4"
GUARD_HDR    = "GPU_HAL_PATCHED_V3"
GUARD_IMPL   = "GPU_RESCALE_PATCHED"

if len(sys.argv) < 2:
    print("Usage: python3 patch_openfhe.py /path/to/openfhe-development")
    sys.exit(1)

root      = sys.argv[1]
hdr_path  = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly.h")
impl_path = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly-impl.h")
cmake_path= os.path.join(root, "src/core/CMakeLists.txt")

# resolve HAL .so relative to this script
hal_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
hal_so    = os.path.join(hal_dir, "libopenfhe_cuda_hal.so")
hal_inc   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")

print("=" * 60)
print("[*] OpenFHE NVIDIA GPU HAL Patcher v4")
print(f"    OpenFHE root : {root}")
print(f"    HAL .so      : {hal_so}")
print("=" * 60)

for p in [hdr_path, impl_path, cmake_path]:
    if not os.path.exists(p):
        print(f"[ERROR] Not found: {p}")
        sys.exit(1)
if not os.path.exists(hal_so):
    print(f"[ERROR] HAL not built yet: {hal_so}")
    print("        Run: mkdir -p build && cd build && cmake .. && make -j$(nproc)")
    sys.exit(1)

# ── 1. CMakeLists ─────────────────────────────────────────────────────────────
cmake = open(cmake_path).read()
if GUARD_CMAKE in cmake:
    print("[=] CMakeLists.txt already patched")
else:
    shutil.copy(cmake_path, cmake_path + ".bak")
    with open(cmake_path, "a") as f:
        f.write(f"""
# ── OpenFHE NVIDIA GPU HAL ({GUARD_CMAKE}) ──────────────────
target_compile_definitions(OPENFHEcore PUBLIC GPU_HAL_PATCHED_V3)
target_include_directories(OPENFHEcore PUBLIC
    /usr/local/cuda/include
    {hal_inc}
)
target_link_libraries(OPENFHEcore
    {hal_so}
    /usr/local/cuda/lib64/libcudart.so
    OpenMP::OpenMP_CXX
)
""")
    print("[+] CMakeLists.txt patched")

# ── 2. dcrtpoly.h — operator*= GPU fast-path ─────────────────────────────────
src = open(hdr_path).read()
if GUARD_HDR in src:
    print("[=] dcrtpoly.h already patched")
else:
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
        "#ifdef GPU_HAL_PATCHED_V3\n"
        "        uint32_t ring = m_params->GetRingDimension();\n"
        "        if (m_format == Format::EVALUATION && ring >= 4096 && size <= 32) {\n"
        "            std::vector<const uint64_t*> ha(size), hb(size);\n"
        "            std::vector<uint64_t*>       hr(size);\n"
        "            std::vector<uint64_t>        mods(size);\n"
        "            for (size_t i = 0; i < size; ++i) {\n"
        "                ha[i]   = reinterpret_cast<const uint64_t*>(m_vectors[i].GetValues().data());\n"
        "                hb[i]   = reinterpret_cast<const uint64_t*>(rhs.m_vectors[i].GetValues().data());\n"
        "                hr[i]   = reinterpret_cast<uint64_t*>(m_vectors[i].GetValues().data());\n"
        "                mods[i] = m_vectors[i].GetModulus().ConvertToInt();\n"
        "            }\n"
        "            gpu_rns_mult_batch_wrapper(ha.data(), hb.data(), hr.data(),\n"
        "                                       mods.data(), ring, (uint32_t)size);\n"
        "            return *this;\n"
        "        }\n"
        "#endif\n"
        "#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))\n"
        "        for (size_t i = 0; i < size; ++i)\n"
        "            m_vectors[i] *= rhs.m_vectors[i];\n"
        "        return *this;\n"
        "    }"
    )
    if old not in src:
        print("[ERROR] operator*= block not found in dcrtpoly.h")
        print("        OpenFHE version may have changed. Check the block manually.")
        sys.exit(1)

    inject = (
        "\n#ifdef GPU_HAL_PATCHED_V3\n"
        "#include \"cuda_hal.h\"\n"
        "extern \"C\" void gpu_rns_mult_batch_wrapper(\n"
        "    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,\n"
        "    const uint64_t* q, uint32_t ring, uint32_t num_towers);\n"
        "#endif\n"
    )
    src = src.replace("#pragma once", "#pragma once" + inject, 1)
    src = src.replace(old, new)
    open(hdr_path, "w").write(src)
    print("[+] dcrtpoly.h patched — operator*= GPU fast-path active for ring>=4096")

# ── 3. dcrtpoly-impl.h — DropLastElementAndScale marker ──────────────────────
impl = open(impl_path).read()
if GUARD_IMPL in impl:
    print("[=] dcrtpoly-impl.h already patched")
else:
    shutil.copy(impl_path, impl_path + ".bak")
    old = (
        "void DCRTPolyImpl<VecType>::DropLastElementAndScale(const std::vector<NativeInteger>& QlQlInvModqlDivqlModq,\n"
        "                                                    const std::vector<NativeInteger>& qlInvModq) {\n"
        "    auto lastPoly(m_vectors.back());\n"
        "    lastPoly.SetFormat(Format::COEFFICIENT);\n"
        "    this->DropLastElement();\n"
        "    uint32_t size(m_vectors.size());\n"
        "\n"
        "#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))\n"
        "    for (uint32_t i = 0; i < size; ++i) {\n"
        "        auto tmp = lastPoly;\n"
        "        tmp.SwitchModulus(m_vectors[i].GetModulus(), m_vectors[i].GetRootOfUnity(), 0, 0);\n"
        "        tmp *= QlQlInvModqlDivqlModq[i];\n"
        "        if (m_format == Format::EVALUATION)\n"
        "            tmp.SwitchFormat();\n"
        "        m_vectors[i] *= qlInvModq[i];\n"
        "        m_vectors[i] += tmp;\n"
        "        if (m_format == Format::COEFFICIENT)\n"
        "            m_vectors[i].SwitchFormat();\n"
        "    }\n"
        "}"
    )
    if old not in impl:
        print("[ERROR] DropLastElementAndScale block not found in dcrtpoly-impl.h")
        print("        OpenFHE version may differ. Check manually.")
        sys.exit(1)
    new = "// GPU_RESCALE_PATCHED\n" + old
    impl = impl.replace(old, new)
    open(impl_path, "w").write(impl)
    print("[+] dcrtpoly-impl.h patched — DropLastElementAndScale marked")

print()
print("=" * 60)
print("[SUCCESS] All patches applied.")
print()
print("Next steps:")
print("  cd", os.path.join(root, "build"))
print("  make -j$(nproc) OPENFHEcore")
print("=" * 60)

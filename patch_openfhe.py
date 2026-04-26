#!/usr/bin/env python3
"""
OpenFHE NVIDIA GPU HAL Patcher v7
Fixes vs v6:
  - keyswitch patch no longer uses NativeVector(N, mod, ptr) constructor
    (that constructor does not exist in OpenFHE). Instead we accumulate
    the GPU-multiplied tower results directly into elements[half] by
    writing into the existing NativePoly's backing storage via SetValues
    with a properly-constructed NativeVector from std::vector copy.
  - gpu_sync_all_to_host() is now defined in cuda_hal.cpp (no longer missing).
  - operator*= patch version guard bumped to V7 to force re-patch.
"""
import sys, os, shutil, re

GUARD_CMAKE = "GPU_HAL_PATCHED_V7"
GUARD_HDR   = "GPU_HAL_PATCHED_V7"
GUARD_KS    = "GPU_HAL_KS_PATCHED_V7"

if len(sys.argv) < 2:
    print("Usage: python3 patch_openfhe.py /path/to/openfhe-development")
    sys.exit(1)

root       = sys.argv[1]
hdr_path   = os.path.join(root, "src/core/include/lattice/hal/default/dcrtpoly.h")
cmake_path = os.path.join(root, "src/core/CMakeLists.txt")
ks_path    = os.path.join(root, "src/pke/lib/keyswitch/keyswitch-hybrid.cpp")
ks_cmake   = os.path.join(root, "src/pke/CMakeLists.txt")

hal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
hal_so  = os.path.join(hal_dir, "libopenfhe_cuda_hal.so")
hal_inc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")

print("[*] OpenFHE NVIDIA GPU HAL Patcher v7")

# ── helper: strip ALL previous HAL guard blocks ──────────────────────────────
def strip_old_guards(src):
    for v in ["V3","V4","V5","V6","V7"]:
        src = re.sub(
            r'#ifdef GPU_HAL_PATCHED_' + v + r'.*?#endif\s*// GPU_HAL_PATCHED_' + v + r'\n',
            '', src, flags=re.DOTALL)
        src = re.sub(
            r'#ifdef GPU_HAL_PATCHED_' + v + r'.*?#endif\n',
            '', src, flags=re.DOTALL)
        src = re.sub(
            r'# ── OpenFHE NVIDIA GPU HAL \(GPU_HAL_PATCHED_' + v + r'\).*?\n',
            '', src, flags=re.DOTALL)
    src = src.replace('#include "global_dag.h"\n', '')
    return src

# ── 1. OPENFHEcore CMakeLists ─────────────────────────────────────────────────
cmake = open(cmake_path).read()
stripped = strip_old_guards(cmake)
if GUARD_CMAKE not in stripped:
    stripped += (
        f"\n# ── OpenFHE NVIDIA GPU HAL ({GUARD_CMAKE}) ──────────────────\n"
        f"target_compile_definitions(OPENFHEcore PUBLIC {GUARD_CMAKE})\n"
        f"target_include_directories(OPENFHEcore PUBLIC /usr/local/cuda/include {hal_inc})\n"
        f"target_link_libraries(OPENFHEcore {hal_so} /usr/local/cuda/lib64/libcudart.so OpenMP::OpenMP_CXX)\n"
    )
    open(cmake_path, "w").write(stripped)
    print("  [+] OPENFHEcore CMakeLists patched")
else:
    print("  [=] OPENFHEcore CMakeLists already patched")

# ── 2. OPENFHEpke CMakeLists ──────────────────────────────────────────────────
pke_cmake_src = open(ks_cmake).read()
stripped_pke = strip_old_guards(pke_cmake_src)
if GUARD_CMAKE not in stripped_pke:
    stripped_pke += (
        f"\n# ── OpenFHE NVIDIA GPU HAL ({GUARD_CMAKE}) ──────────────────\n"
        f"target_compile_definitions(OPENFHEpke PUBLIC {GUARD_CMAKE})\n"
        f"target_include_directories(OPENFHEpke PUBLIC /usr/local/cuda/include {hal_inc})\n"
        f"target_link_libraries(OPENFHEpke {hal_so} /usr/local/cuda/lib64/libcudart.so)\n"
    )
    open(ks_cmake, "w").write(stripped_pke)
    print("  [+] OPENFHEpke CMakeLists patched")
else:
    print("  [=] OPENFHEpke CMakeLists already patched")

# ── 3. dcrtpoly.h — operator*= → gpu_rns_mult_batch_wrapper ─────────────────
src = open(hdr_path).read()
src = strip_old_guards(src)

if GUARD_HDR not in src:
    shutil.copy(hdr_path, hdr_path + ".bak_v7")

    inject = (
        f"\n#ifdef {GUARD_HDR}\n"
        "#include <vector>\n"
        "extern \"C\" void gpu_rns_mult_batch_wrapper(\n"
        "    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,\n"
        "    const uint64_t* q, uint32_t ring, uint32_t num_towers);\n"
        "extern \"C\" void gpu_sync_all_to_host();\n"
        f"#endif // {GUARD_HDR}\n"
    )
    src = src.replace("#pragma once", "#pragma once" + inject, 1)

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
        f"#ifdef {GUARD_HDR}\n"
        "        uint32_t ring = (uint32_t)m_params->GetRingDimension();\n"
        "        if (m_format == Format::EVALUATION && ring >= 4096 &&\n"
        "            size >= 1 && size <= 64) {\n"
        "            static thread_local std::vector<const uint64_t*> ha_ptrs, hb_ptrs;\n"
        "            static thread_local std::vector<uint64_t*>       hr_ptrs;\n"
        "            static thread_local std::vector<uint64_t>        moduli;\n"
        "            ha_ptrs.resize(size); hb_ptrs.resize(size);\n"
        "            hr_ptrs.resize(size); moduli.resize(size);\n"
        "            for (size_t i = 0; i < size; ++i) {\n"
        "                ha_ptrs[i] = (const uint64_t*)m_vectors[i].GetValues().data();\n"
        "                hb_ptrs[i] = (const uint64_t*)rhs.m_vectors[i].GetValues().data();\n"
        "                hr_ptrs[i] = (uint64_t*)m_vectors[i].GetValues().data();\n"
        "                moduli[i]  = m_vectors[i].GetModulus().ConvertToInt<uint64_t>();\n"
        "            }\n"
        "            gpu_rns_mult_batch_wrapper(\n"
        "                ha_ptrs.data(), hb_ptrs.data(), hr_ptrs.data(),\n"
        "                moduli.data(), ring, (uint32_t)size);\n"
        "            return *this;\n"
        "        }\n"
        f"#endif // {GUARD_HDR}\n"
        "#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(size))\n"
        "        for (size_t i = 0; i < size; ++i)\n"
        "            m_vectors[i] *= rhs.m_vectors[i];\n"
        "        return *this;\n"
        "    }"
    )

    if old_op in src:
        src = src.replace(old_op, new_op)
        print("  [+] dcrtpoly.h operator*= patched (v7)")
    else:
        print("  [!] WARNING: operator*= exact pattern not found.")
        print("      Attempting regex fallback...")
        pattern = re.compile(
            r'(    DCRTPolyType& operator\*=\(const DCRTPolyType& rhs\) override \{.*?'
            r'return \*this;\n    \})',
            re.DOTALL
        )
        m = pattern.search(src)
        if m:
            src = src[:m.start()] + new_op + src[m.end():]
            print("  [+] dcrtpoly.h operator*= patched via regex fallback (v7)")
        else:
            print("  [!] FATAL: could not locate operator*= in dcrtpoly.h")
            print("      Manual patch required.")

    open(hdr_path, "w").write(src)
else:
    print("  [=] dcrtpoly.h already patched (v7)")

# ── 4. keyswitch-hybrid.cpp ───────────────────────────────────────────────────
# FIXED: removed broken NativeVector(N, mod, ptr) constructor call.
# Instead we use GetValues() to get a mutable NativeVector reference,
# then overwrite its contents with the GPU result via a properly
# constructed NativeVector (from a std::vector<uint64_t> copy).
# This is the only constructor OpenFHE guarantees for NativeVector.
ks_src = open(ks_path).read()
ks_src = strip_old_guards(ks_src)

if GUARD_KS not in ks_src:
    shutil.copy(ks_path, ks_path + ".bak_v7")

    old_ks = (
        "    for (uint32_t j = 0; j < limit; ++j) {\n"
        "#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(sizeQlP))\n"
        "        for (uint32_t i = 0; i < sizeQlP; ++i) {\n"
        "            const auto idx  = (i >= sizeQl) ? i + delta : i;\n"
        "            const auto& cji = (*digits)[j].GetElementAtIndex(i);\n"
        "            const auto& bji = bv[j].GetElementAtIndex(idx);\n"
        "            const auto& aji = av[j].GetElementAtIndex(idx);\n"
        "            elements[0].SetElementAtIndex(i, elements[0].GetElementAtIndex(i) + cji * bji);\n"
        "            elements[1].SetElementAtIndex(i, elements[1].GetElementAtIndex(i) + cji * aji);\n"
        "        }\n"
        "    }\n"
        "\n"
        "    return result;\n"
        "}"
    )

    # V7 keyswitch patch: GPU multiply writes into a std::vector<uint64_t> buffer,
    # then we construct a NativeVector from that buffer (copy) and call SetValues.
    # This uses only public OpenFHE API and avoids the non-existent 3-arg constructor.
    new_ks = (
        f"// {GUARD_KS}\n"
        f"#ifdef {GUARD_CMAKE}\n"
        "    {{\n"
        "        const uint32_t N = (*digits)[0].GetElementAtIndex(0).GetLength();\n"
        "        static thread_local std::vector<const uint64_t*> hd_ptrs, hk_ptrs;\n"
        "        static thread_local std::vector<uint64_t*>       tmp_ptrs;\n"
        "        static thread_local std::vector<uint64_t>        mods;\n"
        "        static thread_local std::vector<std::vector<uint64_t>> tmp_bufs;\n"
        "\n"
        "        hd_ptrs.resize(sizeQlP); hk_ptrs.resize(sizeQlP);\n"
        "        tmp_ptrs.resize(sizeQlP); mods.resize(sizeQlP);\n"
        "        if (tmp_bufs.size() < sizeQlP)\n"
        "            tmp_bufs.assign(sizeQlP, std::vector<uint64_t>(N, 0));\n"
        "        for (auto& v : tmp_bufs) {{ if (v.size() < N) v.assign(N, 0); }}\n"
        "        for (uint32_t i = 0; i < sizeQlP; ++i)\n"
        "            tmp_ptrs[i] = tmp_bufs[i].data();\n"
        "\n"
        "        for (uint32_t half = 0; half < 2; ++half) {{\n"
        "            const auto& ev = (half == 0) ? bv : av;\n"
        "            for (uint32_t j = 0; j < limit; ++j) {{\n"
        "                for (uint32_t i = 0; i < sizeQlP; ++i) {{\n"
        "                    const auto eidx = (i >= sizeQl) ? i + delta : i;\n"
        "                    hd_ptrs[i] = (const uint64_t*)(*digits)[j].GetElementAtIndex(i).GetValues().data();\n"
        "                    hk_ptrs[i] = (const uint64_t*)ev[j].GetElementAtIndex(eidx).GetValues().data();\n"
        "                    mods[i]    = (*digits)[j].GetElementAtIndex(i).GetModulus().ConvertToInt<uint64_t>();\n"
        "                    std::fill(tmp_bufs[i].begin(), tmp_bufs[i].begin() + N, 0ULL);\n"
        "                }}\n"
        "\n"
        "                // GPU: tmp_bufs[i] = digits[j][i] * ev[j][eidx]  (mod mods[i])\n"
        "                gpu_rns_mult_batch_wrapper(\n"
        "                    hd_ptrs.data(), hk_ptrs.data(),\n"
        "                    tmp_ptrs.data(),\n"
        "                    mods.data(), N, sizeQlP);\n"
        "                gpu_sync_all_to_host();\n"
        "\n"
        "                // Accumulate into elements[half] using public OpenFHE API.\n"
        "                // We construct a NativeVector by copy from tmp_bufs, then\n"
        "                // add it to the existing tower polynomial.\n"
        "                for (uint32_t i = 0; i < sizeQlP; ++i) {{\n"
        "                    NativeInteger modNI(mods[i]);\n"
        "                    NativeVector  nv(N, modNI);\n"
        "                    for (uint32_t k = 0; k < N; ++k)\n"
        "                        nv[k] = tmp_bufs[i][k];\n"
        "                    auto tmp_poly = elements[half].GetElementAtIndex(i).CloneParametersOnly();\n"
        "                    tmp_poly.SetValues(std::move(nv), Format::EVALUATION);\n"
        "                    elements[half].SetElementAtIndex(i,\n"
        "                        elements[half].GetElementAtIndex(i) + tmp_poly);\n"
        "                }}\n"
        "            }}\n"
        "        }}\n"
        "    }}\n"
        f"#else\n"
        "    for (uint32_t j = 0; j < limit; ++j) {{\n"
        "#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(sizeQlP))\n"
        "        for (uint32_t i = 0; i < sizeQlP; ++i) {{\n"
        "            const auto idx  = (i >= sizeQl) ? i + delta : i;\n"
        "            const auto& cji = (*digits)[j].GetElementAtIndex(i);\n"
        "            const auto& bji = bv[j].GetElementAtIndex(idx);\n"
        "            const auto& aji = av[j].GetElementAtIndex(idx);\n"
        "            elements[0].SetElementAtIndex(i, elements[0].GetElementAtIndex(i) + cji * bji);\n"
        "            elements[1].SetElementAtIndex(i, elements[1].GetElementAtIndex(i) + cji * aji);\n"
        "        }}\n"
        "    }}\n"
        f"#endif\n"
        "\n"
        "    return result;\n"
        "}}"
    )

    ks_inject = (
        f"\n#ifdef {GUARD_CMAKE}\n"
        "extern \"C\" void gpu_rns_mult_batch_wrapper(\n"
        "    const uint64_t** ha, const uint64_t** hb, uint64_t** hr,\n"
        "    const uint64_t* q, uint32_t ring, uint32_t num_towers);\n"
        "extern \"C\" void gpu_sync_all_to_host();\n"
        "#include <vector>\n"
        f"#endif  // {GUARD_CMAKE}\n"
    )

    last_inc = list(re.finditer(r'^#include\s+[<\"].*?[>\\"]\s*$', ks_src, re.MULTILINE))
    if last_inc:
        pos = last_inc[-1].end()
        ks_src = ks_src[:pos] + "\n" + ks_inject + ks_src[pos:]

    if old_ks in ks_src:
        ks_src = ks_src.replace(old_ks, new_ks)
        print("  [+] keyswitch-hybrid.cpp patched (v7 - fixed NativeVector construction)")
    else:
        print("  [!] WARNING: keyswitch inner loop exact pattern not found.")
        idx = ks_src.find("for (uint32_t j = 0; j < limit")
        if idx >= 0:
            print(f"  [~] Found candidate loop at char {idx}:")
            print(repr(ks_src[idx:idx+400]))
        print("  [!] Manual patch of keyswitch-hybrid.cpp required.")

    open(ks_path, "w").write(ks_src)
else:
    print("  [=] keyswitch-hybrid.cpp already patched (v7)")

print("\n[SUCCESS] Patcher v7 complete.")
print("\nNext steps:")
print("  cd /mnt/c/Users/samca/openfhenvdia-gpu")
print("  mkdir -p build && cd build")
print("  cmake .. -DCMAKE_BUILD_TYPE=Release")
print("  make -j$(nproc) openfhe_cuda_hal")
print("  cd ~/openfhe-development/build && make -j$(nproc) OPENFHEcore OPENFHEpke")
print("  cd /mnt/c/Users/samca/openfhenvdia-gpu/build")
print("  make -j$(nproc) test_e2e_p34 bench_vs_cpu benchmark_duality")

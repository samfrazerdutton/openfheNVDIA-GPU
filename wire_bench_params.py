#!/usr/bin/env python3
"""Wire up bench_params and fix the misleading bench_vs_cpu banner.

1. Adds a bench_params target to CMakeLists.txt, mirroring how the other
   OpenFHE-linked e2e targets are declared.
2. bench_vs_cpu printed "CPU (Native OpenMP) Benchmark" in every mode --
   including when GPU multiplies and GPU keyswitch were active, since its
   --gpu flag only ever changed the banner. It now reports the mode that
   is actually in effect, read from the same env vars the HAL uses.

Run from the repo root:  python3 wire_bench_params.py
"""
import re, sys, os

if not os.path.exists("CMakeLists.txt"):
    sys.exit("run from the repo root (CMakeLists.txt not found)")

# ── 1. CMake target ──────────────────────────────────────────────────────
cm = open("CMakeLists.txt").read()
if "bench_params" in cm:
    print("[=] CMakeLists already has bench_params")
else:
    # Clone the declaration block of an existing OpenFHE-linked target.
    m = re.search(r'^([ \t]*)add_executable\(\s*bench_vs_cpu\b.*$', cm, re.M)
    if not m:
        sys.exit("[!] could not find the bench_vs_cpu target to model; "
                 "add bench_params manually")
    indent = m.group(1)
    start = m.start()
    # Collect every line mentioning bench_vs_cpu (add_executable, link,
    # includes, properties) so bench_params gets the same treatment.
    block = [ln for ln in cm[start:].splitlines() if "bench_vs_cpu" in ln]
    if not block:
        sys.exit("[!] no bench_vs_cpu lines found")
    new_block = "\n".join(
        ln.replace("bench_vs_cpu", "bench_params")
          .replace("src/bench_vs_cpu.cpp", "src/bench_params.cpp")
        for ln in block
    )
    cm = cm.rstrip() + "\n\n" + indent + "# Parameter sweep: locates the CPU/GPU crossover\n" + new_block + "\n"
    open("CMakeLists.txt", "w").write(cm)
    print("[+] CMakeLists: bench_params target added (modeled on bench_vs_cpu)")

# ── 2. Honest banner in bench_vs_cpu ─────────────────────────────────────
p = "src/bench_vs_cpu.cpp"
if not os.path.exists(p):
    print("[=] src/bench_vs_cpu.cpp not found; skipping banner fix")
    sys.exit(0)

src = open(p).read()
if "OPENFHE_GPU_KS" in src:
    print("[=] bench_vs_cpu banner already honest")
    sys.exit(0)

OLD_FLAG = '    bool is_gpu = (argc > 1 && std::string(argv[1]) == "--gpu");'
NEW_FLAG = '''    // The old --gpu flag only changed this banner; dispatch is decided by
    // the HAL's env vars. Report what is actually in effect.
    const char* env_gpu = std::getenv("OPENFHE_GPU");
    const char* env_ks  = std::getenv("OPENFHE_GPU_KS");
    const bool cpu_only = (env_gpu && env_gpu[0] == '0');
    const bool ks_on    = (env_ks && env_ks[0] == '1');
    std::string mode_label =
        cpu_only ? "CPU only (OPENFHE_GPU=0)"
                 : (ks_on ? "GPU multiplies + GPU keyswitch (OPENFHE_GPU_KS=1)"
                          : "GPU multiplies, CPU keyswitch (default)");'''

OLD_BANNER = '    std::cout << (is_gpu ? "[*] GPU (RTX 2060) Benchmark" : "[*] CPU (Native OpenMP) Benchmark") << "\\n";'
NEW_BANNER = '    std::cout << "[*] CKKS EvalMult Benchmark -- " << mode_label << "\\n";'

ok = True
if OLD_FLAG in src:
    src = src.replace(OLD_FLAG, NEW_FLAG, 1)
else:
    print("[!] flag anchor not found; banner left as-is")
    ok = False
if OLD_BANNER in src:
    src = src.replace(OLD_BANNER, NEW_BANNER, 1)
else:
    print("[!] banner anchor not found; banner left as-is")
    ok = False

if ok:
    if "#include <cstdlib>" not in src:
        src = src.replace("#include <openfhe.h>", "#include <openfhe.h>\n#include <cstdlib>", 1)
    if "#include <string>" not in src:
        src = src.replace("#include <openfhe.h>", "#include <openfhe.h>\n#include <string>", 1)
    open(p, "w").write(src)
    print("[+] bench_vs_cpu: banner now reports the real dispatch mode")

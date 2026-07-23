#!/usr/bin/env python3
"""Fix B safety net: flush deferred GPU results before CPU-side reads.

With OPENFHE_GPU_FUSE=1 the HAL may leave results in VRAM (host copy
stale). Any CPU code that reads those coefficients must see current data.
This patch calls gpu_flush_all() at the entry of the decrypt path, which
is where ciphertext coefficients cross back into CPU-only arithmetic.

Deliberately conservative: flushing costs microseconds, missing a flush
corrupts plaintext. If in doubt, add more call sites, not fewer.

Usage:  python3 apply_flush_patch.py ~/openfhe-development
"""
import sys, os

if len(sys.argv) < 2:
    sys.exit("Usage: python3 apply_flush_patch.py /path/to/openfhe-development")

root = sys.argv[1]
targets = [
    os.path.join(root, "src/pke/lib/schemerns/rns-pke.cpp"),
]

DECL = """
#ifdef GPU_HAL_PATCHED_V8
extern "C" void gpu_flush_all();
#endif
"""

patched_any = False
for P in targets:
    if not os.path.exists(P):
        print(f"[=] not found, skipping: {P}")
        continue
    src = open(P).read()
    if "gpu_flush_all" in src:
        print(f"[=] already patched: {os.path.basename(P)}")
        patched_any = True
        continue

    NS = "namespace lbcrypto {"
    if NS not in src:
        print(f"[!] no namespace anchor in {os.path.basename(P)}; skipping")
        continue
    src = src.replace(NS, DECL + "\n" + NS, 1)

    # Insert a flush at the top of every DecryptCore definition found.
    import re
    pattern = re.compile(
        r'(DecryptCore\([^)]*\)\s*const\s*\{)', re.S)
    matches = list(pattern.finditer(src))
    if not matches:
        print(f"[!] no DecryptCore body found in {os.path.basename(P)}")
        print("    Inspect manually:  grep -n 'DecryptCore' " + P)
        continue

    out = []
    last = 0
    for m in matches:
        out.append(src[last:m.end()])
        out.append("\n#ifdef GPU_HAL_PATCHED_V8\n"
                   "    gpu_flush_all();  // Fix B: no stale host reads\n"
                   "#endif\n")
        last = m.end()
    out.append(src[last:])
    src = "".join(out)

    open(P, "w").write(src)
    print(f"[+] {os.path.basename(P)}: gpu_flush_all() at {len(matches)} DecryptCore site(s)")
    patched_any = True

if not patched_any:
    sys.exit("[!] nothing patched — locate the decrypt path manually before "
             "enabling OPENFHE_GPU_FUSE=1")

print("\nRebuild OpenFHE, then verify BOTH modes decrypt correctly:")
print("  ./test_e2e_ckks                          # fusion off")
print("  OPENFHE_GPU_FUSE=1 ./test_e2e_ckks       # fusion on -- must still PASS")

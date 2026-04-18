#!/usr/bin/env bash
# =============================================================================
# build_and_verify.sh
# Run these commands one section at a time from the repo root.
# Expected output is annotated in comments.
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$REPO_ROOT/build"

echo "============================================================"
echo " Step 1: Clean build directory"
echo "============================================================"
rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 2: CMake configure"
echo "============================================================"
# Expected: -- Build files written to .../build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee cmake_configure.log
grep -E "(CUDA|Error|error)" cmake_configure.log || true

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 3: Build (all targets)"
echo "============================================================"
# Expected: [100%] Linking CXX executable benchmark_duality
make -j"$(nproc)" 2>&1 | tee make.log
# Verify the shared library and executables exist.
ls -lh libopenfhe_cuda_hal.so benchmark_duality benchmark

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 4: Verify GPU device is available"
echo "============================================================"
# Expected: at least one CUDA device listed
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 5: Run twiddle sanity check (static binary test)"
echo "============================================================"
# A small standalone C++ program that exercises BuildTwiddleTable directly
# and checks psi^N = -1 mod q.  Compile and run in one shot.
cat > /tmp/twiddle_check.cpp << 'TWEOF'
#include <cstdint>
#include <iostream>
#include <stdexcept>
// Pull in the twiddle source directly via include path.
// Adjust -I path if your include dir differs.
#include "twiddle_gen.h"

static uint64_t powmod(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r=1; b%=m;
    while(e>0){if(e&1)r=(uint64_t)(((unsigned __int128)r*b)%m);b=(uint64_t)(((unsigned __int128)b*b)%m);e>>=1;}
    return r;
}

int main() {
    // Use a known negacyclic-NTT-friendly prime: q = 576460752303423489
    // q - 1 = 2^59 * ... check: (q-1) % (2*16) = 0
    // Instead, generate a small known case: N=16, use first prime from gen.
    // We test with q = 7681 which satisfies (q-1) % 32 = 0 (7680 / 32 = 240).
    uint64_t q = 7681, N = 16;
    if ((q-1) % (2*N) != 0) { std::cerr << "q not negacyclic-friendly\n"; return 1; }

    TwiddleTable tt = BuildTwiddleTable(q, N);

    // Check table sizes.
    if (tt.forward.size() != N || tt.inverse.size() != N) {
        std::cerr << "FAIL: table size wrong\n"; return 1;
    }

    // Verify forward[0] = psi^1 (the first twisted twiddle is psi^bit_rev(0)+1 = psi^1).
    // And that psi = forward[0] satisfies psi^(2N) = 1, psi^N = q-1.
    uint64_t psi_candidate = tt.forward[0];
    if (powmod(psi_candidate, 2*N, q) != 1) {
        std::cerr << "FAIL: psi^(2N) != 1, psi=" << psi_candidate << "\n"; return 1;
    }
    if (powmod(psi_candidate, N, q) != q-1) {
        std::cerr << "FAIL: psi^N != -1 mod q\n"; return 1;
    }

    // Verify N^{-1} * N = 1 mod q.
    if (((__uint128_t)tt.n_inv * N) % q != 1) {
        std::cerr << "FAIL: n_inv wrong\n"; return 1;
    }

    std::cout << "[PASS] twiddle_gen: negacyclic roots verified (q=" << q << ", N=" << N << ")\n";
    std::cout << "  psi=" << psi_candidate
              << "  psi^N mod q=" << powmod(psi_candidate,N,q)
              << " (expect " << q-1 << ")\n";
    std::cout << "  n_inv=" << tt.n_inv << "\n";
    return 0;
}
TWEOF

g++ -std=c++17 -O2 \
    -I "$REPO_ROOT/include" \
    /tmp/twiddle_check.cpp \
    "$REPO_ROOT/src/twiddle_gen.cpp" \
    -o /tmp/twiddle_check

/tmp/twiddle_check
# Expected:
#   [PASS] twiddle_gen: negacyclic roots verified (q=7681, N=16)
#   psi=...  psi^N mod q=7680 (expect 7680)

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 6: Run full duality benchmark (GPU correctness + perf)"
echo "============================================================"
cd "$BUILD"
# LD_LIBRARY_PATH ensures the freshly-built .so is found.
LD_LIBRARY_PATH="$BUILD:$LD_LIBRARY_PATH" ./benchmark_duality
# Expected output (abridged):
#   [*] Negacyclic NTT GPU Verification Engine
#   [TEST 1] ... Thread N pointwise OK  (all threads)
#   [TEST 2] ...
#     psi^16 mod q = <q-1>  (expect <q-1>)        ← negacyclic check
#     [+] CPU negacyclic NTT self-consistent
#     [+] GPU negacyclic NTT convolution OK
#   [TEST 3] Throughput ...  X.XX ms/op
#   [PASS] All tests passed

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Step 7: Run legacy VRAM-pool benchmark"
echo "============================================================"
LD_LIBRARY_PATH="$BUILD:$LD_LIBRARY_PATH" ./benchmark
# Expected:
#   [BENCH] 16 towers x 32768 ring: X.XXX ms/op
#   [BENCH] throughput: XXXX.X M coeff-mults/sec

echo ""
echo "============================================================"
echo " All steps complete."
echo " If [PASS] appeared above, the negacyclic NTT fix is verified."
echo "============================================================"


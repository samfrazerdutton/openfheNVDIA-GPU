# OpenFHE NVIDIA GPU HAL

A CUDA hardware-abstraction layer that accelerates [OpenFHE](https://github.com/openfheorg/openfhe-development)'s
lattice-crypto primitives (NTT, RNS pointwise ops, key-switching) on NVIDIA GPUs, plus **Dumbo Protocol** —
a three-node encrypted edge-failover demo built on top of the HAL.

Verified working end-to-end on an RTX 2060 (sm_75) under WSL2 — CUDA 13.2, CMake 3.28, GCC 13.3.

## What's actually in this repo

| Piece | What it is |
|---|---|
| `openfhe_cuda_hal` (`src/`, `kernels/`, `include/`) | The GPU HAL itself: CUDA kernels for negacyclic NTT, RNS pointwise multiply, key-switching, plus a small DAG compiler (`fhe_compiler.cpp`/`global_dag.cpp`) that schedules GPU ops. |
| `patches/` + `patch_openfhe.py` | Patches applied to an existing OpenFHE source checkout so its `DCRTPoly`/`NativePoly` operators can call out to the GPU HAL instead of the stock CPU path. |
| `dumbo/` | Python demo: three independently-runnable services (`edge`, `hub`, `failover`) that simulate an edge node encrypting telemetry, an untrusted hub relaying it, and a failover node decrypting + routing — see [`DUMBO_PROTOCOL.md`](./DUMBO_PROTOCOL.md) for the full writeup. |
| `dumbo_ext/` | A pybind11 wrapper (`dumbo_cuda`) exposing the CUDA NTT/encoder kernels directly to Python, used by the demo and `dumbo_showcase.py`. |
| `dumbo_setup.sh` | Bootstraps the `dumbo/` Python package tree + venv from scratch. |

The GPU HAL and the Dumbo Protocol demo are two layers of the same project: the HAL is the general-purpose
CUDA acceleration layer, Dumbo is one concrete application of it (encrypted telemetry handoff), used here
mainly to exercise and showcase the HAL under a realistic workload.

## Prerequisites

- NVIDIA GPU, CUDA-capable (tested on RTX 2060 / sm_75; CMake also targets sm_80, sm_86)
- CUDA Toolkit 12.x+ (tested against 13.2) — installed *inside* WSL if you're on Windows, not just the Windows-side driver
- CMake 3.18+
- GCC/G++ with C++17 and OpenMP support
- An OpenFHE build installed at `/usr/local` (`libOPENFHEpke.so`, `libOPENFHEcore.so`, `libOPENFHEbinfhe.so` + headers under `/usr/local/include/openfhe`) — required for the end-to-end (`test_e2e_*`, `bench_vs_cpu`) targets. Without it, CMake still builds the HAL and the DAG/NTT test targets, just skips the OpenFHE-integrated ones.
- Python 3.12+ (only needed for the `dumbo/` demo and `dumbo_ext` pybind module)

Check your toolchain before building:
```bash
nvidia-smi          # confirms the GPU is visible to WSL
nvcc --version       # confirms the CUDA toolkit is installed inside WSL
cmake --version
```

## Building the GPU HAL

Work inside the Linux filesystem (`~/`), not `/mnt/c/...` — much faster and avoids permission issues.

```bash
git clone https://github.com/samfrazerdutton/openfheNVDIA-GPU.git
cd openfheNVDIA-GPU
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75   # 75 for RTX 20-series; 80/86 for Ampere
make -j$(nproc)
```

This produces `libopenfhe_cuda_hal.so` plus the test/benchmark executables below.

### Patching OpenFHE itself (optional, needed for e2e targets)

If you want the `test_e2e_*` / `bench_vs_cpu` targets to link against a *patched* OpenFHE that routes through
the GPU HAL rather than stock CPU code:

```bash
python3 patch_openfhe.py /path/to/openfhe-development
```

This rewrites `DCRTPoly`'s multiply/keyswitch operators to call into `openfhe_cuda_hal`, guarded by a version
marker (currently `GPU_HAL_PATCHED_V7`) so re-running the patcher against an already-patched tree is a no-op
rather than double-patching.

## Test & benchmark targets

| Binary | What it checks |
|---|---|
| `test_dag` | DAG compiler correctness — builds and executes a small op graph, checks the result. |
| `test_e2e_ckks` | Full CKKS round trip (keygen → encrypt → GPU EvalMult → decrypt) against expected plaintext values. |
| `test_e2e_p34` | Same idea, targeting the P34 parameter set. |
| `benchmark` | CKKS-pipeline throughput (coeff-mults/sec) at a given ring degree/tower count. |
| `benchmark_duality` | Re-validates negacyclic NTT correctness (GPU vs CPU-reference vs analytic reference, bit-exact) then reports pointwise-RNS and NTT-multiply throughput in isolation. |
| `bench_vs_cpu` | Same operation, CPU-only (OpenMP) path — the number to compare GPU results against. |
| `bench_evalmult` | GPU latency scaling across ring sizes (16k/32k/64k). |
| `run_benchmark.sh` | Convenience script: builds `bench_vs_cpu` and runs it both without and with the GPU HAL preloaded (`LD_PRELOAD=libopenfhe_cuda_hal.so`). |

Run them from `build/`:
```bash
./test_dag && ./test_e2e_ckks
./benchmark
./bench_vs_cpu
./benchmark_duality
./bench_evalmult
```

### Measured results (RTX 2060, sm_75, 32768-degree ring unless noted)

- **Correctness**: DAG execution, CKKS round-trip (error ~1e-11 to 1e-12 — expected CKKS approximation noise, not a bug), and negacyclic NTT (bit-exact GPU vs CPU vs analytic reference) all pass.
- **CKKS pipeline throughput**: 16 towers → 0.229 ms/op, ~2285 M coeff-mults/sec.
- **CPU-only comparison** (`bench_vs_cpu`, 11 towers): 75.16 ms mean latency — roughly two orders of magnitude slower than the GPU path above.
- **Isolated primitives** (`benchmark_duality`): pointwise RNS multiply ~42-45 M coeff-mults/sec; full NTT-based polynomial multiply ~9.3-9.5 M coeff-mults/sec. These are lower than the CKKS-pipeline number above because they measure cheaper, more granular operations — not a regression.
- **Latency scaling** (`bench_evalmult`): 16 towers — 40.79 ms (N=16k), 88.50 ms (N=32k), 132.22 ms (N=64k).

Numbers will vary by GPU; re-run the suite on your own hardware to get comparable figures.

## Dumbo Protocol demo

Three-node encrypted telemetry handoff: an edge node encrypts stress telemetry, an untrusted hub relays the
ciphertext without being able to read it, and a failover node decrypts and makes routing decisions.

```bash
./dumbo_setup.sh          # builds the dumbo/ package tree + Python venv
source dumbo/venv/bin/activate
export PYTHONPATH=.
python3 dumbo_showcase.py
```

Two modes, controlled by `DUMBO_FHE_REAL`:

- **Mode A — Real BFV (`DUMBO_FHE_REAL=1`, default)**: genuine OpenFHE BFV encryption. The hub only ever sees
  opaque ciphertext; it cannot recover telemetry even with the mixing structure, since it lacks the secret key.
- **Mode B — Plaintext polynomial packing (`DUMBO_FHE_REAL=0`)**: uses the GPU NTT encoder to pack telemetry
  into a polynomial with a *public* mixing matrix — **not encryption**. Anyone who knows the polynomial degree
  and modulus can invert it. This mode exists purely to benchmark the GPU HAL's raw packing performance without
  BFV overhead, and is labeled `mode=plaintext_polynomial` in all log output. Don't use it where telemetry
  privacy actually matters.

See [`DUMBO_PROTOCOL.md`](./DUMBO_PROTOCOL.md) for the full architecture writeup, including exactly which CUDA
kernels run per encode call and how the modular arithmetic (`mulmod64` via `__umul64hi` + Barrett reduction) is
implemented.

## Repository layout
CMakeLists.txt          # GPU HAL + test/benchmark build targets
src/                    # HAL source (cuda_hal, DAG compiler, keyswitch, evalmult, benchmarks)
kernels/                # CUDA kernels (NTT, RNS math, keyswitch)
include/                # HAL headers (VRAM cache/pool, stream pool, DAG registries)
patches/                # Patches applied to an OpenFHE checkout by patch_openfhe.py
patch_openfhe.py         # Applies the above patches, idempotently
dumbo/                  # Edge / hub / failover Python services + shared FHE utilities
dumbo_ext/               # pybind11 wrapper exposing CUDA kernels to Python
dumbo_setup.sh           # Bootstraps dumbo/ + its venv
dumbo_showcase.py         # End-to-end demo runner
run_benchmark.sh          # bench_vs_cpu with/without GPU HAL preloaded
DUMBO_PROTOCOL.md         # Dumbo Protocol architecture deep-dive
`build/` is gitignored — always a fresh `mkdir build && cd build && cmake ..`, never committed.

## Known issues

- `gpu_engine.cpp` has a handful of harmless `-Wformat` warnings (`%llu` vs `%lu` for `size_t` args) — cosmetic, doesn't affect correctness on 64-bit platforms.
- `CMAKE_CUDA_ARCHITECTURES` defaults to `"75;80;86"`; override with `-DCMAKE_CUDA_ARCHITECTURES=<yours>` to cut build time on a single-GPU dev box.
- The end-to-end targets (`test_e2e_*`, `bench_vs_cpu`) only build if OpenFHE is found at `/usr/local` — check the CMake configure output for "OpenFHE found — building e2e targets" vs "OpenFHE not found — skipping e2e targets".

## License

MIT License. Created by Sam Frazer Dutton (Billinghurst).

# openfheNVDIA-GPU

**CUDA acceleration HAL for [OpenFHE](https://github.com/openfheorg/openfhe-development)** — routes RNS polynomial arithmetic (Montgomery multiply, negacyclic NTT, keyswitch kernels) through custom CUDA kernels, with a patcher that hooks them into OpenFHE's CKKS pipeline.

Built and verified end-to-end on a 65 W laptop **RTX 2060 (sm_75)** under WSL2 — the goal is bringing consumer and older Turing+ NVIDIA GPUs into FHE workloads, reproducibly, for anyone who clones this repo.

## Verified results

RTX 2060 Max-Q (65 W laptop), locked clocks, median of 7 reps. CUDA 13.2, driver 595.97, WSL2 Ubuntu 24.04, OpenFHE v1.5.1 @ `ed361af2`.

| Metric | Result |
|---|---|
| Raw kernel: pointwise RNS multiply, 16 towers x N=32768 | ~0.20 ms/op (~2.5-2.7 G coeff-mults/s) |
| Kernel time per launch (nsys, cached device buffers) | ~4.2 us (was 117 us with managed memory) |
| HAL EvalMult path, 16 towers, N=32k | ~16 ms (4x faster after ShadowRegistry rewrite) |
| CKKS e2e EvalMult, GPU vs CPU | statistical tie (~50-100 ms both) — see Status |
| CKKS e2e Decrypt, GPU vs CPU | parity (~20-35 ms both) |
| CKKS correctness (all stages through GPU) | max error ~1e-12, verified vs plaintext |

GPU execution in every CKKS stage (keygen / encrypt / EvalMult tensor product / decrypt) is profiler-verified (`nsys` + built-in call tracing), not assumed.

**Honest status:** end-to-end EvalMult is currently at CPU parity, not ahead. The multiplies run on the GPU, but each dispatch pays pageable host<->device transfer cost, and keyswitch/relinearization — the largest share of EvalMult — still runs on CPU. Both are active roadmap items below. Numbers on a desktop GPU outside WSL should be materially better; benchmark PRs welcome (see Contributing).

## Requirements

- NVIDIA GPU, **Turing (sm_75) or newer** (CUDA 13 dropped pre-Turing support; older GPUs would need CUDA 12.x — untested)
- CUDA toolkit installed inside Linux/WSL (`nvcc` must work), CMake >= 3.24, GCC 13
- OpenFHE **v1.5.1 at commit `ed361af2`** — the patcher pins to this; other versions will be rejected

## Install

Order matters: the HAL must be built **before** patching OpenFHE (the patched build links `build/libopenfhe_cuda_hal.so` by absolute path — don't move this repo afterwards).

```bash
# 1. Build the HAL
git clone https://github.com/samfrazerdutton/openfheNVDIA-GPU.git
cd openfheNVDIA-GPU && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
make -j$(nproc) openfhe_cuda_hal

# 2. Clone, pin, patch, and build OpenFHE (~20-40 min)
cd ~ && git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development && git checkout ed361af2
python3 ~/openfheNVDIA-GPU/patch_openfhe.py ~/openfhe-development
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
make -j$(nproc) && sudo make install && sudo ldconfig

# 3. Build everything in this repo and verify
cd ~/openfheNVDIA-GPU/build && cmake .. && make -j$(nproc)
./test_dag && ./test_e2e_ckks    # both must print [PASS]
```

## Runtime controls

| Env var | Effect |
|---|---|
| `OPENFHE_GPU=0` | Kill switch — full CPU path, same binary. This is how the CPU/GPU A/B comparisons are made. |
| `OPENFHE_GPU_LOG=1` | Print every GPU dispatch (call #, towers, ring) to verify what actually runs on the GPU. |

## Benchmarking

`scripts/bench_true.sh` runs the full suite 7x with GPU telemetry logging and writes median tables to `results/`. Methodology: lock GPU clocks first (`nvidia-smi -lgc 960,1350` from Windows admin PowerShell for WSL), run on AC power from a cool machine, report medians, publish the temperature range alongside the numbers. WSL adds run-to-run variance that locked clocks do not fully remove — treat single runs as noise.

For the honest e2e comparison:

```bash
for i in 1 2 3 4 5; do OPENFHE_GPU=0 ./test_e2e_ckks | grep EvalMult; done   # CPU
for i in 1 2 3 4 5; do ./test_e2e_ckks | grep EvalMult; done                 # GPU
```

## Repository layout

```
CMakeLists.txt        HAL + test/benchmark build targets
src/                  HAL source (cuda_hal, DAG compiler, keyswitch, benchmarks)
kernels/              CUDA kernels (Montgomery RNS math, negacyclic NTT, keyswitch)
include/              ShadowRegistry (device-buffer cache), stream pool, DAG registries
patch_openfhe.py      Applies GPU hooks to a pinned OpenFHE checkout (v9)
scripts/bench_true.sh Benchmark harness with telemetry + median reporting
results/              Published benchmark summaries (RESULTS_*.md)
dumbo/, dumbo_ext/    Edge/hub Python services + pybind11 CUDA bindings (Dumbo Protocol)
```

## Architecture notes

- The patcher injects GPU dispatch into `DCRTPolyImpl::operator*=` and `DCRTPolyImpl::Times()` — the latter is the path CKKS EvalMult/keygen/encrypt actually use. Dispatch requires eval format, ring >= 4096, and <= 64 towers.
- `ShadowRegistry` caches real `cudaMalloc` device buffers keyed by host pointer; inputs are re-copied per call, so stale mappings can never serve stale data.
- A library constructor initializes the CUDA context + stream pool at load, keeping the ~300 ms first-touch cost (WSL) out of every timed operation.
- Host `cudaHostRegister` pinning was tried and reverted: OpenFHE frees temporaries while registered, leaving dangling pin state on reused addresses. The planned fix is HAL-owned pinned staging buffers.

## Roadmap

1. ~~HAL-owned pinned staging buffers~~ — **done** (PinnedStagingPool, pageable fallback).
2. **Keyswitch inner product on GPU** — done, correct, opt-in via `OPENFHE_GPU_KS=1` (regresses at small params: transfer volume exceeds the CPU inner-product cost; needs eval-key VRAM residency + Barrett MAC to win).
3. **VRAM residency across operations** — keep ciphertexts on-device through op chains via the DAG compiler.
4. CI (compile matrix), CUDA 12.x support for pre-Turing GPUs, Dockerfile.

## Contributing

Benchmark reports from other GPUs are especially welcome — run `scripts/bench_true.sh`, note your GPU/driver/CUDA/cooling, and open a PR adding your `RESULTS_*.md`. Desktop cards and non-WSL Linux numbers are the most valuable missing data.

## License

MIT — see [LICENSE](LICENSE). Created by Sam Frazer Dutton.

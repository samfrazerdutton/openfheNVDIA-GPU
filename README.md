# OpenFHE NVIDIA GPU HAL (Phase 2: DAG Edition)

High-performance Hardware Abstraction Layer for OpenFHE, targeting NVIDIA RTX 20-series (Turing) and above. Implements a JIT DAG compiler that captures FHE circuits and dispatches them to CUDA Graphs, eliminating redundant CPU-GPU synchronization.

## Benchmarks
*NVIDIA RTX 2060 Laptop / WSL2 Ubuntu 22.04 / CKKS N=32768*

| Measurement | Time | Notes |
|---|---|---|
| GPU kernel (16 towers, N=16k) | **4.35 ms** | Data already in VRAM, pure compute |
| GPU kernel (16 towers, N=64k) | **8.58 ms** | Pure compute |
| EvalMult end-to-end (100-rep mean) | **84.2 ms** | Includes PCIe + CPU key-switch |
| Full pipeline (enc + mult + dec) | **147 ms** | CKKS N=32768, 11 towers |
| Numerical error | **< 1e-11** | All 16384 slots verified |

The gap between 4ms kernel time and 84ms wall time is PCIe transfer overhead and CPU-side key-switching (relinearization). Eliminating that is Phase 3.

## Architecture
OpenFHE operator*=
│
▼
dcrtpoly.h patch (GPU_HAL_PATCHED_V6)
│
▼
gpu_rns_mult_batch_wrapper
│  per-tower modulus array
\
▼
ShadowRegistry  ──►  VRAM
│
▼
LaunchRNSMultMontgomery (cuda_math.cu)
Montgomery reduction, __uint128_t
│
▼
Results written back to host
**DAG Compiler** (`fhe_compiler.h/cpp`): Captures FHE operations into a `cudaGraph_t` via stream capture. Nodes carry per-tower moduli so each RNS prime is correct.

**GlobalDAG** (`global_dag.h/cpp`): Singleton managing the OpenFHE→GPU intercept. Allocates VRAM for each host pointer, injects STORE nodes, compiles and launches the graph, then frees VRAM. No leaks across repeated EvalMult calls.

**NTT Kernels** (`cuda_ntt.cu`): Negacyclic NTT using `psi = g^((q-1)/2N)` with `psi^N = -1 mod q`. Requires `(q-1) % 2N == 0` primes.

**Montgomery RNS** (`cuda_math.cu`): Exact 128-bit modular multiply, no approximation error.

## Getting Started

### 1. Build the HAL
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. Patch OpenFHE
```bash
python3 patch_openfhe.py ~/openfhe-development
cd ~/openfhe-development/build && make -j$(nproc) OPENFHEcore
```

### 3. Run end-to-end test
```bash
cd build
LD_PRELOAD=$PWD/libopenfhe_cuda_hal.so ./test_e2e_ckks
```

### 4. Run benchmarks
```bash
./bench_evalmult                                          # pure GPU kernel
LD_PRELOAD=$PWD/libopenfhe_cuda_hal.so ./bench_vs_cpu    # full pipeline
```

## Roadmap

- **Phase 2 (done):** DAG compiler, per-tower modulus correctness, zero VRAM leaks
- **Phase 3:** Persistent VRAM — keep ciphertexts GPU-resident between operations, eliminating PCIe upload/download per EvalMult
- **Phase 4:** GPU-resident relinearization keys + key-switch acceleration — the dominant remaining cost

## Author
Sam Frazer Dutton

# OpenFHE NVIDIA GPU HAL (Phase 2: DAG Edition)

This repository provides a high-performance **Hardware Abstraction Layer (HAL)** for the OpenFHE library, specifically optimized for **NVIDIA RTX 20-series (Turing)** architecture and above. 

By implementing a **Just-In-Time (JIT) Directed Acyclic Graph (DAG) Compiler**, this project bypasses the traditional "PCIe Bottleneck" in Fully Homomorphic Encryption (FHE). Instead of eager execution, it captures FHE circuits into optimized CUDA Graphs and executes them in a single shot on the GPU.

## 🚀 Performance Benchmarks
*Tested on: NVIDIA RTX 2060 Laptop GPU / WSL2 (Ubuntu 22.04)* *Parameters: CKKS, Ring Dimension $N=32768$*

```text
======================================================
[*] OpenFHE NVIDIA GPU HAL — End-to-End CKKS Test
[*] Pipeline: CPU encrypt → GPU EvalMult → CPU decrypt
======================================================

[1] Key generation:      535.61 ms
[2] Encrypt (CPU):       88.94 ms
[3] EvalMult (GPU HAL):  57.41 ms  [GPU Fast-path, ring=32768]
[4] Decrypt (CPU):       25.27 ms

[*] Max absolute error across all 16384 slots: 0.00
[*] Total pipeline latency (enc+mult+dec): 171.63 ms

======================================================
[PASS] CPU→GPU→CPU round-trip correct
======================================================
🛠 Architecture: The "Lazy" DAG CompilerTraditional FHE GPU acceleration suffers from high latency due to constant CPU-GPU synchronization. This HAL solves that via:Instruction Capture: Intercepts OpenFHE operator*= calls and builds a Directed Acyclic Graph instead of executing math immediately.Phantom Memory Management: Maps OpenFHE host pointers to VRAM-resident "Phantom" buffers, ensuring intermediate data never leaves the GPU.CUDA Graph JIT: Compiles the execution tree into a native cudaGraph_t, launching complex circuits (like $y = (a \times b) + (c \times d)$) as a single kernel sequence.Negacyclic NTT Acceleration: Optimized Montgomery-reduction based NTT kernels specifically tuned for Turing-generation tensor/core-adjacent memory paths.📦 Componentsinclude/fhe_compiler.h: Core DAG tree logic and CUDA Graph recorder.include/global_dag.h: Singleton engine that manages the OpenFHE-to-GPU hook.include/phantom_registry.h: Tracking system for GPU-resident polynomials.src/cuda_hal.cpp: Hardware wrappers for RNS multiplication and NTT.patch_openfhe.py: Automated Python patcher to inject the HAL into OpenFHE's source.🚀 Getting Started1. Patch OpenFHEPoint the patcher to your OpenFHE-development directory to inject the v5 DAG hooks.
./patch_openfhe.py ../openfhe-development
2. Build the HAL
mkdir -p build && cd build
cmake ..
make -j$(nproc)
3. Run the E2E Test
To ensure the dynamic linker uses the local optimized HAL, use LD_PRELOAD:
LD_PRELOAD=./libopenfhe_cuda_hal.so ./test_e2e_ckks
👤 Author
Sam Cade Billinghurst a.k.a Sam Frazer Dutton

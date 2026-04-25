# OpenFHE NVIDIA GPU Hardware Abstraction Layer (HAL)

A high-performance NVIDIA CUDA Hardware Abstraction Layer (HAL) for the [OpenFHE](https://github.com/openfheorg/openfhe-development) Fully Homomorphic Encryption library. This engine accelerates polynomial arithmetic (NTT, INTT, and RNS multiplications) for schemes like CKKS, enabling rapid Evaluation Multiplication (`EvalMult`) and Zero-Trust AI inference.

## 🚀 Features

* **High-Throughput Polynomial Math:** Reaches ~67.2 Million coeff-mults/s for large-ring (N=65536) Pointwise RNS and ~9.8 Million coeff-mults/s for full Negacyclic NTT convolutions on consumer GPUs (e.g., RTX 2060).
* **Persistent VRAM Caching:** Features a highly optimized, size-aware `ShadowRegistry` that dynamically caches OpenFHE host pointers in GPU managed memory, eliminating redundant PCIe transfers across evaluation circuits.
* **Thread-Safe & OpenMP Ready:** The memory registry and stream pools use a 16-shard mutex architecture to safely handle concurrent OpenFHE CPU threads during batch evaluations.
* **Seamless E2E Integration:** Hooks directly into OpenFHE's CPU-side decryption and evaluation pipelines (Phase 3+4 CKKS compatible).

## 🛠 Prerequisites

* **OS:** Ubuntu 24.04 LTS (WSL2 supported)
* **Hardware:** NVIDIA GPU with Compute Capability 7.5, 8.0, or 8.6 (e.g., Turing, Ampere)
* **Toolkit:** CUDA Toolkit 13.2+
* **Compiler:** GCC/G++ 13.3+ and CMake 3.18+
* **OpenFHE:** OpenFHE development branch installed globally (headers in `/usr/local/include/openfhe`)

## 🏗 Build Instructions

Clone the repository and build using standard CMake:

```bash
git clone [https://github.com/samfrazerdutton/openfheNVDIA-GPU.git](https://github.com/samfrazerdutton/openfheNVDIA-GPU.git)
cd openfheNVDIA-GPU
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)



🧪 Testing & Benchmarks
The build pipeline generates several test suites to verify math correctness and measure throughput.

Duality & OpenMP Benchmark: Tests thread contention and verifies GPU Negacyclic NTT against a CPU reference.
OMP_NUM_THREADS=8 ./benchmark_duality
Evaluation Multiplication Benchmark: Raw throughput tests for RNS multiplication at various ring dimensions.
./bench_evalmult
End-to-End CKKS Integration: Verifies the full OpenFHE pipeline (KeyGen -> CPU Encrypt -> GPU EvalMult -> CPU Decrypt).
./test_e2e_ckks
./test_e2e_p34
Architecture Overview
The core of the integration relies on intercepting OpenFHE's polynomial multiplications.

gpu_rns_mult_batch_wrapper: The primary C-linkage hook that intercepts EvalMult.

ShadowRegistry: Manages cudaMallocManaged memory. It maps OpenFHE's host-side pointers to GPU VRAM and dynamically resizes them if the ring dimension expands, preventing aliasing and out-of-memory faults.

📄 License
This project is open-source and intended for academic, research, and defense technology development.

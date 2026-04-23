# OpenFHE NVIDIA GPU HAL 

A high-performance, transparent Hardware Abstraction Layer (HAL) for the [OpenFHE](https://openfhe.org/) library. This project solves the notorious "PCIe Bottleneck" in Fully Homomorphic Encryption (FHE) by implementing persistent VRAM caching and fully offloading RNS key-switching to NVIDIA GPUs.

This HAL is designed as a drop-in accelerator. By utilizing dynamic linking (`LD_PRELOAD`) and an automated source patcher, developers can accelerate their existing OpenFHE applications **without rewriting any high-level cryptographic logic**.

##  Performance Benchmarks

*Hardware: NVIDIA RTX 2060 Laptop GPU (Turing) vs. Multi-core CPU (OpenMP)* *Parameters: CKKS, Ring Dimension $N=32768$, 11 Towers, Multiplicative Depth 5*

### Head-to-Head Latency (EvalMult)
Despite running on a highly constrained 2019-era laptop GPU, the HAL successfully beats a native, OpenMP-optimized CPU execution on pure latency by keeping ciphertexts resident in VRAM.

| Architecture | Mean Latency | Max Error | Notes |
| :--- | :--- | :--- | :--- |
| **Native CPU (OpenMP)** | 43.36 ms | - | Standard OpenFHE execution |
| **NVIDIA RTX 2060** | **40.80 ms** | < 1.36e-11 | **Phase 3+4 GPU HAL** |

*Note: FHE is heavily memory-bandwidth bound. Scaling this architecture from an RTX 2060 (336 GB/s) to datacenter hardware like an NVIDIA H100 (3+ TB/s) is expected to yield sub-5ms latencies.*

---

##  Architecture & Features

This project was built in four phases to systematically eliminate CPU-GPU synchronization overhead:

1. **Exact Montgomery RNS & Negacyclic NTT:** Low-level CUDA kernels (`LaunchRNSMultMontgomery`, `LaunchNTT`) providing flawless 128-bit exact modular arithmetic. Zero precision loss compared to the CPU.
2. **DAG Compiler (Lazy Execution):** Captures OpenFHE operations into a Directed Acyclic Graph, dispatching massive parallel circuits via `cudaGraph_t` to minimize kernel launch overhead.
3. **Persistent VRAM (`ShadowRegistry`):** Eliminates the PCIe bottleneck. The HAL tracks host-pointer lifecycle, uploading ciphertexts to the GPU exactly once. Subsequent `EvalMult` operations execute entirely in VRAM (0 ms PCIe cost).
4. **GPU Hybrid Key-Switching:** The evaluation key (EVK) is uploaded once during initialization. The massively parallel inner-product operations of `EvalFastKeySwitchCoreExt` are batched and executed entirely on the GPU, removing the final ~35ms CPU bottleneck.

##  Getting Started

### Prerequisites
* CUDA Toolkit 13.0+
* OpenFHE (Development branch / v1.1.x)
* CMake 3.15+

### 1. Build the HAL
```bash
git clone [https://github.com/samfrazerdutton/openfheNVDIA-GPU.git](https://github.com/samfrazerdutton/openfheNVDIA-GPU.git)
cd openfheNVDIA-GPU
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) openfhe_cuda_hal
2. Patch OpenFHE (v6 Patcher)
The included Python script automatically injects the ShadowRegistry and GPU Key-Switching hooks into your local OpenFHE installation.
cd openfheNVDIA-GPU
python3 patch_openfhe.py /path/to/your/openfhe-development

# Rebuild OpenFHE core and pke
cd /path/to/your/openfhe-development/build
make -j$(nproc) OPENFHEcore OPENFHEpke
3. Run Applications
No code changes are required for your OpenFHE executable. Simply force the dynamic linker to use the HAL:
# Run the E2E benchmark
cd openfheNVDIA-GPU/build
make bench_vs_cpu
LD_PRELOAD=$PWD/libopenfhe_cuda_hal.so ./bench_vs_cpu --gpu
 -Roadmap
Dynamic Modulus Extension: Support for dynamic resizing of the RNS towers during modulus switching.

Datacenter Scaling: Benchmarking on NVIDIA 6G / Datacenter hardware (A100/H100).

Multi-GPU Swarm: Partitioning RNS towers across multiple GPUs over NVLink.

- Author
Sam Frazer Dutton Mutton Industries

Built to bring accessible, high-performance open-source Fully Homomorphic Encryption to the American tech ecosystem.

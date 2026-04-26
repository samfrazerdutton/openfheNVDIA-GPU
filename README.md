# OpenFHE NVIDIA GPU HAL

A CUDA Hardware Abstraction Layer that accelerates [OpenFHE](https://github.com/openfheorg/openfhe-development) fully homomorphic encryption by offloading RNS polynomial arithmetic to the GPU. Targets the CKKS scheme for machine learning workloads.

Tested on an **NVIDIA RTX 2060** against an **8-thread OpenMP CPU baseline**.

---

## How it works

OpenFHE's `DCRTPolyImpl::operator*=` is patched to call `gpu_rns_mult_batch_wrapper`, which uploads coefficient towers to VRAM via a `ShadowRegistry` cache, executes batched Montgomery-form pointwise multiplications across all RNS towers in parallel CUDA streams, and writes results back. The key-switch inner product in `EvalFastKeySwitchCoreExt` is similarly patched to run on-device. Once ciphertext data is resident in VRAM it stays there across repeated EvalMult calls, eliminating redundant PCIe transfers in iterative workloads.

---

## Performance

### RNS pointwise multiply throughput (`benchmark_duality`)

| Ring dim | Towers | Throughput |
|---|---|---|
| N=32768 | 16 | **35.7 M coeff-mults/s** (14.69 ms/op) |
| N=65536 | 16 | **47.5 M coeff-mults/s** (22.07 ms/op) |

### NTT polynomial multiply throughput (`benchmark_duality`)

| Ring dim | Towers | Throughput |
|---|---|---|
| N=32768 | 16 | 12.1 M coeff-mults/s (43.47 ms/op) |
| N=65536 | 16 | 17.4 M coeff-mults/s (60.32 ms/op) |

### CPU vs GPU EvalMult (`bench_vs_cpu`, N=32768, 11 towers, 25 reps)

| Backend | Mean latency |
|---|---|
| CPU — 8-thread OpenMP | 54.79 ms |
| GPU — RTX 2060 | **44.55 ms** |
| **Speedup** | **1.23x** |

### GPU-resident EvalMult chain (`test_e2e_p34`, N=32768, 10 warm iters)

After one cold upload the ciphertext stays in VRAM. Subsequent EvalMult calls skip PCIe:

| | Latency |
|---|---|
| Cold (PCIe upload + key-switch) | 43.25 ms |
| Warm mean | **43.72 ms** |
| Warm min | **33.60 ms** |
| Warm max | 61.40 ms |

### End-to-end CKKS pipeline (`test_e2e_ckks`, N=32768)

| Stage | Latency |
|---|---|
| Key generation | 480.07 ms |
| Encrypt (CPU) | 83.43 ms |
| EvalMult (GPU HAL) | **45.55 ms** |
| Decrypt (CPU) | 34.51 ms |
| **Total enc+mult+dec** | **163.49 ms** |

---

## Correctness

All results verified bit-exact against CPU reference for NTT convolution. CKKS numerical error is at floating-point noise floor:

| Test | Result |
|---|---|
| GPU negacyclic NTT convolution | ✅ bit-exact vs CPU reference |
| Phase 3 GPU-resident EvalMult | ✅ max error 1.54e-11 |
| CKKS CPU→GPU→CPU round-trip (16384 slots) | ✅ max absolute error 0.00 |

---

## Dependencies

- OpenFHE (tested against v1.5.1)
- CUDA 12.x + cuBLAS
- GCC with OpenMP
- CMake 3.20+

---

## Build

```bash
# 1. Build the HAL
git clone https://github.com/samfrazerdutton/openfheNVDIA-GPU
cd openfheNVDIA-GPU
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$(pwd)"
make -j$(nproc) openfhe_cuda_hal

# 2. Patch and build OpenFHE
cd ..
python3 patch_openfhe.py ~/openfhe-development
cd ~/openfhe-development/build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=$HOME/openfhe-local
make -j$(nproc) OPENFHEcore OPENFHEpke
make install

# 3. Build test and benchmark binaries
cd /path/to/openfheNVDIA-GPU/build
make -j$(nproc) benchmark_duality bench_vs_cpu test_e2e_p34 test_e2e_ckks
```

---

## Run all tests and benchmarks

```bash
cd build
OMP_NUM_THREADS=8 ./benchmark_duality
./bench_vs_cpu
LD_PRELOAD=$PWD/libopenfhe_cuda_hal.so ./bench_vs_cpu --gpu
./test_e2e_p34
./test_e2e_ckks
```

Or in one shot with a log:

```bash
OMP_NUM_THREADS=8 ./benchmark_duality && \
./bench_vs_cpu && \
LD_PRELOAD=$PWD/libopenfhe_cuda_hal.so ./bench_vs_cpu --gpu && \
./test_e2e_p34 && \
./test_e2e_ckks 2>&1 | tee results.txt
```

---

## Hardware tested

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 2060 |
| CPU | 8-thread baseline (OpenMP) |
| Ring dim | N = 32768 / 65536 |
| RNS towers | 11–16 |
| OpenFHE version | v1.5.1 |

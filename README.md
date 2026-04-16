# OpenFHE NVIDIA GPU HAL

A CUDA Hardware Abstraction Layer that accelerates [OpenFHE](https://github.com/openfheorg/openfhe-development) CKKS bootstrapping by offloading RNS polynomial multiplications to the GPU using exact 128-bit modular arithmetic.

## What it does

- Intercepts `DCRTPolyImpl::operator*=` in OpenFHE's core via a one-time patch
- Routes ring dimension ≥ 4096 multiplications to GPU (CKKS bootstrapping workloads)
- Falls back to OpenFHE's CPU path for small rings
- Uses exact `uint128 % modulus` — zero Barrett approximation error
- Persistent VRAM pool — no per-call `cudaMalloc` overhead
- Async CUDA streams per RNS tower — full parallelism

## Performance

~2.3ms for 16-tower, 32k-degree polynomial multiplication on GPU vs ~18ms CPU baseline.

## Prerequisites

- NVIDIA GPU (Ampere or newer recommended)
- CUDA Toolkit 11.0+ (`nvcc --version` to check)
- OpenFHE v1.5.x (`git clone https://github.com/openfheorg/openfhe-development`)
- CMake 3.18+, GCC 11+

## Build & Install

### 1. Build this HAL

```bash
git clone https://github.com/samfrazerdutton/openfheNVDIA-GPU
cd openfheNVDIA-GPU
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ls -lh libopenfhe_cuda_hal.so   # should appear ~39K
```

### 2. Patch OpenFHE

```bash
# From the openfheNVDIA-GPU root directory:
python3 patch_openfhe.py /path/to/openfhe-development
```

### 3. Rebuild OpenFHE

```bash
cd /path/to/openfhe-development/build
cmake .. -DBUILD_EXAMPLES=ON
make -j$(nproc)
```

### 4. Make the library permanently discoverable

```bash
echo "/path/to/openfheNVDIA-GPU/build" | sudo tee /etc/ld.so.conf.d/openfhe-cuda.conf
sudo ldconfig
```

### 5. Run the CKKS bootstrapping example

```bash
cd /path/to/openfhe-development/build
./bin/examples/pke/simple-ckks-bootstrapping
```

Expected output:

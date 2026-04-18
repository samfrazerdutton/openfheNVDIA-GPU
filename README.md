# OpenFHE NVIDIA GPU HAL

A CUDA Hardware Abstraction Layer that accelerates [OpenFHE](https://github.com/openfheorg/openfhe-development) CKKS/BGV schemes by offloading RNS polynomial multiplications and Number Theoretic Transforms (NTTs) to the GPU using exact 128-bit modular arithmetic.

## What it does

- **Dynamic Hooking:** Intercepts `DCRTPolyImpl::operator*=` in OpenFHE's core via a one-time patch.
- **Selective Offloading:** Routes ring dimension ≥ 4096 multiplications to the GPU, falling back to the CPU for smaller rings.
- **Exact Mathematics:** Uses exact `uint128 % modulus` — zero Barrett approximation error, ensuring strict cryptologic noise budget integrity.
- **Thread-Safe Concurrency:** Utilizes explicit asynchronous CUDA streams per RNS tower, backed by race-free per-call VRAM allocation (`cudaMalloc`/`cudaFree`) and stream synchronization to cleanly support OpenMP threading.
- **Native GPU NTT:** Implements Cooley-Tukey Decimation-In-Time (DIT) algorithms with explicit bit-reversal permutations and mathematically strict cyclic primitive $N$-th roots.

## Prerequisites

- NVIDIA GPU (Ampere or newer recommended)
- CUDA Toolkit 11.0+ (`nvcc --version` to check)
- OpenFHE v1.5.x (`git clone https://github.com/openfheorg/openfhe-development`)
- CMake 3.18+, GCC 11+

## Build & Verify

### 1. Build the HAL

```bash
git clone [https://github.com/samfrazerdutton/openfheNVDIA-GPU](https://github.com/samfrazerdutton/openfheNVDIA-GPU)
cd openfheNVDIA-GPU
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
2. Run the Duality-Grade Verification Engine
Before integrating with OpenFHE, verify hardware consistency, CPU-to-GPU exactness, and OMP thread-safety:

Bash
./benchmark_duality
Expect to see [PASS] All tests passed confirming the NTT trace and pointwise multipliers.

OpenFHE Integration
3. Patch OpenFHE
Inject the co-processor airgap into OpenFHE's core:

Bash
# From the openfheNVDIA-GPU root directory:
python3 patch_openfhe.py /path/to/openfhe-development
4. Rebuild OpenFHE
Bash
cd /path/to/openfhe-development/build
cmake .. -DBUILD_EXAMPLES=ON
make -j$(nproc)
5. Make the library permanently discoverable
Bash
echo "/path/to/openfheNVDIA-GPU/build" | sudo tee /etc/ld.so.conf.d/openfhe-cuda.conf
sudo ldconfig
6. Run an OpenFHE Bootstrapping Example
Bash
cd /path/to/openfhe-development/build
./bin/examples/pke/simple-ckks-bootstrapping

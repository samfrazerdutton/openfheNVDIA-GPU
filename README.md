# OpenFHE NVIDIA GPU HAL

A CUDA Hardware Abstraction Layer that accelerates OpenFHE's RNS polynomial
arithmetic on NVIDIA GPUs. Targets CKKS bootstrapping for ML workloads.

## What's in here

| Path | Purpose |
|------|---------|
| `include/` | Public headers (cuda_hal.h, vram_pool.h, stream_pool.h, twiddle_gen.h) |
| `kernels/cuda_math.cu` | Montgomery RNS multiply kernel |
| `kernels/cuda_ntt.cu` | Negacyclic NTT / INTT kernels (pre/post twist + cyclic stages) |
| `src/cuda_hal.cpp` | Host-side HAL: batch RNS multiply, poly multiply via NTT |
| `src/twiddle_gen.cpp` | Negacyclic twiddle table builder (2N layout: twist section + cyclic section) |
| `src/benchmark_duality.cpp` | Correctness + throughput benchmark |
| `patch_openfhe.py` | Patches OpenFHE core to call the GPU HAL for EvalMult |
| `CMakeLists.txt` | Build system |

## Hardware tested

- NVIDIA GeForce RTX 2060 Max-Q (6 GB, Turing/SM75)
- CUDA 13.2, GCC 13.3, Ubuntu 24 (WSL2)

## Performance (RTX 2060 Max-Q)

| Operation | N | Towers | Throughput |
|-----------|---|--------|-----------|
| Pointwise RNS multiply | 32768 | 16 | ~69 M coeff-mults/s |
| NTT poly multiply | 32768 | 16 | via gpu_poly_mult_wrapper |

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH ./benchmark_duality
```

## Integrate with OpenFHE

```bash
python3 patch_openfhe.py /path/to/openfhe-development
cd /path/to/openfhe-development/build
cmake .. -DBUILD_EXAMPLES=ON && make -j$(nproc)
./bin/examples/pke/simple-ckks-bootstrapping
```

## Twiddle table layout

Forward and inverse tables are each `2*N` entries:
- `[0..N-1]` — psi^k / psi_inv^k for pre/post twist
- `[N..2N-1]` — w^j / w_inv^j (w = psi^2) for cyclic NTT stages

## Status

- [x] Pointwise RNS multiply (Montgomery, multi-tower, concurrent streams)
- [x] Negacyclic NTT polynomial multiply (GPU matches CPU reference exactly)
- [x] OpenFHE CKKS bootstrapping verified end-to-end
- [ ] Larger ring benchmarks (N=65536, N=131072)
- [ ] Full OpenFHE EvalMult hook (replace CPU path)

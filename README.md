# OpenFHE NVIDIA GPU HAL (Stateful & Validated)

A high-performance, stateful CUDA abstraction layer for OpenFHE CKKS/BGV. This engine utilizes a Shadow Registry to maintain VRAM residency, bypassing PCIe bottlenecks for sequential homomorphic operations.

## Architecture Highlights
- **Stateful Residency:** Implements a `ShadowRegistry` to link CPU memory addresses with persistent GPU allocations, only syncing back to host when explicitly requested.
- **Dynamic Reallocation:** Safely handles OS memory address recycling with capacity-aware VRAM management.
- **Arithmetic:** Branchless, PTX-native 64-bit interleaved modular reduction.
- **Negacyclic NTT:** Split-memory 2N twiddle layout with exact DIT/DIF sequence synchronization.

## Performance Benchmarks (RTX 2060 Max-Q)
- **Pointwise RNS (N=32768):** 5.61 ms/op (93.5 M coeff-mults/s)
- **Pointwise RNS (N=65536):** 7.56 ms/op (138.7 M coeff-mults/s)
- **NTT Poly Mult (N=65536):** 30.65 ms/op (34.2 M coeff-mults/s)

## Verification
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH ./benchmark_duality

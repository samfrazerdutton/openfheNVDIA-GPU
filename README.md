# OpenFHE NVIDIA GPU HAL (Validated)

High-performance CUDA abstraction layer for OpenFHE CKKS/BGV schemes, heavily optimized for NVIDIA Turing (RTX 2060) architecture.

## Technical Specifications
- **Architecture:** Bypasses PCIe bottlenecks via asynchronous stream-ordered memory management.
- **Arithmetic:** Branchless, PTX-safe 64-bit interleaved modular reduction ensuring exact polynomial convolutions without invoking unsupported `__int128` modulo instructions.
- **Negacyclic NTT:** Implements a split-memory $2N$ twiddle layout ($N$ twist factors + $N$ cyclic roots) with an exact DIT Forward / DIF Inverse sequence.
- **Performance:** Achieves up to ~65M coeff-mults/sec for RNS operations on consumer-grade laptop hardware.

## Verified Benchmarks (RTX 2060 Max-Q)
- **Pointwise RNS (N=32768, 16 towers):** 10.50 ms/op (50.0 M coeff-mults/s)
- **Pointwise RNS (N=65536, 16 towers):** 16.90 ms/op (62.1 M coeff-mults/s)
- **NTT Poly Mult (N=32768, 16 towers):** 25.48 ms/op (20.6 M coeff-mults/s)
- **NTT Poly Mult (N=65536, 16 towers):** 40.33 ms/op (26.0 M coeff-mults/s)

## Verification
To verify the engine against the OpenFHE CPU reference:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH ./benchmark_duality

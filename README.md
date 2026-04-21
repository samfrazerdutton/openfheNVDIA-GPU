# OpenFHE NVIDIA GPU HAL

A CUDA Hardware Abstraction Layer that accelerates OpenFHE's RNS polynomial
multiplication for the CKKS scheme. Tested on OpenFHE v1.4.x / v1.5.0.

## What it accelerates

- `operator*=` on `DCRTPolyImpl` — every CKKS `EvalMult` (ring ≥ 4096, towers ≤ 32)
- GPU negacyclic NTT polynomial multiply (16-tower, N=32768 in ~15ms)
- Pointwise RNS multiply: 3007 M coeff-mults/sec on RTX-class hardware

## Requirements

- CUDA 12+ and an NVIDIA GPU 
- OpenFHE v1.4.x or v1.5.0 installed at `/usr/local` or built from source
- CMake 3.18+, GCC 13+, OpenMP

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Verify

```bash
cd build
LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH ./benchmark_duality
LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH ./test_e2e_ckks
```

Expected: `[PASS] All tests passed` and `[PASS] CPU→GPU→CPU round-trip correct`

## Patch OpenFHE

```bash
python3 patch_openfhe.py /path/to/openfhe-development
cd /path/to/openfhe-development/build
make -j$(nproc) OPENFHEcore
```

The patcher is safe to re-run. It backs up all modified files as `.bak`.
The `patches/` directory contains the exact patched headers from the tested build.

## Performance (RTX 2080, N=32768, 16 towers)

| Operation | Latency |
|---|---|
| Pointwise RNS multiply | 0.174 ms |
| GPU EvalMult (full e2e) | 38–53 ms |
| NTT polynomial multiply | 15.2 ms |

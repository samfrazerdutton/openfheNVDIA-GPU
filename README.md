# OpenFHE NVIDIA GPU HAL

An open-source, high-performance CUDA Hardware Abstraction Layer for OpenFHE. This engine accelerates the Residue Number System (RNS) polynomial multiplications required for Fully Homomorphic Encryption (FHE), specifically targeting the CKKS scheme for Machine Learning workloads.

## Performance
Evaluates 16-tower, 32k-degree polynomials natively on the GPU in ~2.3ms.

## Build Instructions
```bash
mkdir build && cd build
cmake ..
make -j
./gpu_engine
```

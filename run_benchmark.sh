#!/bin/bash
cd build
make -j$(nproc) bench_vs_cpu > /dev/null

./bench_vs_cpu
LD_PRELOAD=$PWD/libopenfhe_cuda_hal.so ./bench_vs_cpu --gpu

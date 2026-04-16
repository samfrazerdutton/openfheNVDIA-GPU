cmake_minimum_required(VERSION 3.18)
project(OpenFHE_CUDA_HAL CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_library(openfhe_cuda_hal SHARED
    src/cuda_hal.cpp
    kernels/cuda_math.cu
)

target_include_directories(openfhe_cuda_hal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(openfhe_cuda_hal PRIVATE CUDA::cudart OpenMP::OpenMP_CXX)

set_target_properties(openfhe_cuda_hal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# The Duality Verification Benchmark
add_executable(benchmark_duality src/benchmark_duality.cpp)
target_include_directories(benchmark_duality PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(benchmark_duality PRIVATE openfhe_cuda_hal CUDA::cudart OpenMP::OpenMP_CXX)

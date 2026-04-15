cmake_minimum_required(VERSION 3.18)
project(OpenFHE_CUDA_HAL CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Auto-detect and optimize for your 2020 laptop GPU
set(CMAKE_CUDA_ARCHITECTURES "native")

# Explicitly find the CUDA toolkit so g++ knows where cuda_runtime.h is
find_package(CUDAToolkit REQUIRED)

add_library(openfhe_cuda_hal SHARED 
    src/cuda_hal.cpp 
    kernels/cuda_math.cu
)

target_include_directories(openfhe_cuda_hal PUBLIC include)

# Link the CUDA runtime library to the C++ files
target_link_libraries(openfhe_cuda_hal PRIVATE CUDA::cudart)

set_target_properties(openfhe_cuda_hal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(gpu_engine src/main_gpu_engine.cpp)
target_link_libraries(gpu_engine PRIVATE openfhe_cuda_hal CUDA::cudart)

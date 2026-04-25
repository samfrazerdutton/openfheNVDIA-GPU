cmake_minimum_required(VERSION 3.18)
project(OpenFHE_CUDA_HAL CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA arch")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(OPENFHE_INCLUDE_DIRS
    /usr/local/include/openfhe
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/binfhe
)
set(OPENFHE_LIBS
    /usr/local/lib/libOPENFHEpke.so
    /usr/local/lib/libOPENFHEcore.so
    /usr/local/lib/libOPENFHEbinfhe.so
)

add_library(openfhe_cuda_hal SHARED
    src/cuda_hal.cpp
    src/global_dag.cpp
    src/fhe_compiler.cpp
    src/twiddle_gen.cpp
    src/vram_cache.cpp
    src/gpu_keyswitch.cpp
    src/gpu_evalmult_p34.cpp
    kernels/cuda_math.cu
    kernels/cuda_ntt.cu
    kernels/cuda_keyswitch.cu
)
target_include_directories(openfhe_cuda_hal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(openfhe_cuda_hal PRIVATE CUDA::cudart OpenMP::OpenMP_CXX)
set_target_properties(openfhe_cuda_hal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(openfhe_cuda_hal PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

function(hal_executable name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${name} PRIVATE openfhe_cuda_hal CUDA::cudart OpenMP::OpenMP_CXX)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)
endfunction()

hal_executable(benchmark_duality src/benchmark_duality.cpp)
hal_executable(benchmark         src/benchmark.cpp)
hal_executable(bench_evalmult    src/bench_evalmult.cpp)
hal_executable(gpu_engine        src/gpu_engine.cpp)

add_executable(test_dag
    src/test_dag.cpp src/fhe_compiler.cpp src/cuda_hal.cpp src/twiddle_gen.cpp
    src/vram_cache.cpp src/gpu_keyswitch.cpp src/gpu_evalmult_p34.cpp
    kernels/cuda_math.cu kernels/cuda_ntt.cu kernels/cuda_keyswitch.cu
)
target_include_directories(test_dag PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_dag PRIVATE CUDA::cudart OpenMP::OpenMP_CXX)
set_target_properties(test_dag PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

function(e2e_executable name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OPENFHE_INCLUDE_DIRS}
    )
    target_link_libraries(${name} PRIVATE
        openfhe_cuda_hal
        ${OPENFHE_LIBS}
        CUDA::cudart
        OpenMP::OpenMP_CXX
    )
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)
endfunction()

if(EXISTS /usr/local/lib/libOPENFHEpke.so)
    message(STATUS "OpenFHE found — building e2e targets")
    e2e_executable(test_e2e_ckks  src/test_e2e_ckks.cpp)
    e2e_executable(bench_vs_cpu   src/bench_vs_cpu.cpp)
    e2e_executable(test_e2e_p34   src/test_e2e_p34.cpp)
else()
    message(STATUS "OpenFHE not found — skipping e2e targets")
endif()

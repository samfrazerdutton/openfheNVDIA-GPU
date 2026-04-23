cmake_minimum_required(VERSION 3.18)
project(OpenFHE_CUDA_HAL CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Target the GPU you have. Add more architectures as needed:
# 75 = Turing (RTX 20xx), 80 = Ampere (A100/RTX 30xx), 86 = RTX 30xx mobile
set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA compute capabilities")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP      REQUIRED)

# ── Core HAL shared library ───────────────────────────────────────────────────
add_library(openfhe_cuda_hal SHARED
    src/cuda_hal.cpp
    src/global_dag.cpp
    src/fhe_compiler.cpp
    src/twiddle_gen.cpp
    kernels/cuda_math.cu
    kernels/cuda_ntt.cu
)
target_include_directories(openfhe_cuda_hal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include    # cuda_hal.h, shadow_registry.h, etc.
)
target_link_libraries(openfhe_cuda_hal PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
)
set_target_properties(openfhe_cuda_hal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)
target_compile_options(openfhe_cuda_hal PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

# ── Macro to register a benchmark/test binary ─────────────────────────────────
function(hal_executable name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(${name} PRIVATE
        openfhe_cuda_hal
        CUDA::cudart
        OpenMP::OpenMP_CXX
    )
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    )
endfunction()

hal_executable(benchmark_duality src/benchmark_duality.cpp)
hal_executable(benchmark         src/benchmark.cpp)
hal_executable(bench_evalmult    src/bench_evalmult.cpp)
hal_executable(gpu_engine        src/gpu_engine.cpp)

# ── End-to-end CKKS test (requires OpenFHE installed) ─────────────────────────
# Build only when OpenFHE is found on the system.
find_package(OpenFHE QUIET)
if(OpenFHE_FOUND)
    message(STATUS "OpenFHE found — building test_e2e_ckks")
    add_executable(test_e2e_ckks src/test_e2e_ckks.cpp)
    target_include_directories(test_e2e_ckks PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        /usr/local/include/openfhe /usr/local/include/openfhe/pke /usr/local/include/openfhe/core /usr/local/include/openfhe/binfhe
    )
    target_link_libraries(test_e2e_ckks PRIVATE
        openfhe_cuda_hal
        OPENFHEcore OPENFHEpke OPENFHEbinfhe
        CUDA::cudart
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "OpenFHE not found — skipping test_e2e_ckks (run patch_openfhe.py first)")
endif()

# ── Head-to-head CPU vs GPU benchmark ────────────────────────────────────────
if(OpenFHE_FOUND)
    add_executable(bench_vs_cpu src/bench_vs_cpu.cpp)
    target_include_directories(bench_vs_cpu PRIVATE
        /usr/local/include/openfhe
        /usr/local/include/openfhe/pke
        /usr/local/include/openfhe/core
        /usr/local/include/openfhe/binfhe
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(bench_vs_cpu PRIVATE
        openfhe_cuda_hal
        OPENFHEpke
        OPENFHEcore
        OPENFHEbinfhe
        CUDA::cudart
        OpenMP::OpenMP_CXX
    )
endif()

# Phase 2 DAG Test
add_executable(test_dag src/test_dag.cpp src/fhe_compiler.cpp src/cuda_hal.cpp src/twiddle_gen.cpp kernels/cuda_math.cu kernels/cuda_ntt.cu)
target_include_directories(test_dag PRIVATE include)
target_link_libraries(test_dag PRIVATE ${CUDAToolkit_LIBRARY_DIR}/libcudart.so)
target_include_directories(test_dag PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} /usr/local/cuda/include)

# ─── Phase 3+4 additions ────────────────────────────────────────────────────

# Phase 3: Persistent VRAM cache
target_sources(openfhe_cuda_hal PRIVATE
    ${CMAKE_SOURCE_DIR}/src/vram_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu_keyswitch.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu_evalmult_p34.cpp
    ${CMAKE_SOURCE_DIR}/kernels/cuda_keyswitch.cu
)
target_include_directories(openfhe_cuda_hal PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Phase 3+4 end-to-end test
add_executable(test_e2e_p34
    src/test_e2e_p34.cpp
)
target_link_libraries(test_e2e_p34
    openfhe_cuda_hal
    OPENFHEcore OPENFHEpke OPENFHEbinfhe
    ${CUDA_LIBRARIES}
)
target_include_directories(test_e2e_p34 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    /usr/local/include/openfhe
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/binfhe
)

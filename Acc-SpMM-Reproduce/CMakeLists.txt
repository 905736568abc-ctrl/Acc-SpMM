cmake_minimum_required(VERSION 3.24)

project(AccSpMMReproduce VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

option(ACC_SPMM_BUILD_BENCHMARK "Build benchmark executable." ON)
option(ACC_SPMM_ENABLE_NATIVE "Enable host-native optimizations for CPU-only helpers." OFF)

set(ACC_SPMM_SOURCES
    baseline/cusparse_spmm.cu
    dtc/dtc_spmm.cu
    acc/pipeline.cu
    acc/reorder.cpp
    benchmark/matrix_loader.cpp
)

set(ACC_SPMM_HEADERS
    baseline/cusparse_spmm.hpp
    dtc/dtc_spmm.hpp
    acc/bittcf.hpp
    acc/pipeline.hpp
    acc/reorder.hpp
    benchmark/matrix_loader.hpp
)

add_library(acc_spmm_core STATIC
    ${ACC_SPMM_SOURCES}
    ${ACC_SPMM_HEADERS}
)

target_include_directories(acc_spmm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(acc_spmm_core
    PUBLIC
        CUDA::cudart
        CUDA::cusparse
)

set_target_properties(acc_spmm_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

if(ACC_SPMM_ENABLE_NATIVE)
    target_compile_options(acc_spmm_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
    )
endif()

if(ACC_SPMM_BUILD_BENCHMARK)
    add_executable(acc_spmm_bench
        benchmark/benchmark.cpp
    )
    target_link_libraries(acc_spmm_bench PRIVATE acc_spmm_core)
    target_include_directories(acc_spmm_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

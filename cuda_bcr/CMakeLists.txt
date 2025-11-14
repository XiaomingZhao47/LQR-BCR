cmake_minimum_required(VERSION 3.24)
project(CUDA_BCR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)  

# Find packages
find_package(CUDAToolkit 12.8 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# CUDA BCR library
add_library(cuda_bcr_lib STATIC
    src/bcr_solver.cu
    src/block_operations.cu
    src/riccati_kernels.cu
)

target_link_libraries(cuda_bcr_lib PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)

set_target_properties(cuda_bcr_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Optimization flags
target_compile_options(cuda_bcr_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -lineinfo
        --ptxas-options=-v
        --maxrregcount=64
    >
)

# Python module
pybind11_add_module(_cuda_bcr src/python_bindings.cpp)
target_link_libraries(_cuda_bcr PRIVATE cuda_bcr_lib)

# Installation
install(TARGETS _cuda_bcr
    LIBRARY DESTINATION cuda_bcr
)

# C++ test executable (optional)
add_executable(test_cuda_bcr_cpp
    tests/test_cpp.cu
)
target_link_libraries(test_cuda_bcr_cpp PRIVATE cuda_bcr_lib)
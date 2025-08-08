cmake_minimum_required(VERSION 3.18)
project(CUDANet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
add_executable(cudanet src/main.cpp src/conv2d.cu)
set_property(TARGET cudanet PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(cudanet PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
)

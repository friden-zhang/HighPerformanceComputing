cmake_minimum_required(VERSION 3.22)

project(HighPerformanceComputing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86 --extended-lambda")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(vector_add_kernel)
add_subdirectory(sgemm_kernel)
add_subdirectory(reduction)
cmake_minimum_required(VERSION 3.22)

project(HighPerformanceComputing LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HPC_ENABLE_CUTLASS "Enable CUTLASS baseline benchmarks" OFF)
set(CUTLASS_DIR "" CACHE PATH "Path to CUTLASS root (expects include/cutlass/...)")
option(HPC_FETCH_CUTLASS "Fetch CUTLASS from GitHub when not found" ON)
set(HPC_CUTLASS_GIT_REPOSITORY "https://github.com/NVIDIA/cutlass.git"
  CACHE STRING "CUTLASS Git repository to fetch when HPC_FETCH_CUTLASS is ON")
set(HPC_CUTLASS_GIT_TAG "v3.5.0"
  CACHE STRING "CUTLASS Git tag/commit to fetch when HPC_FETCH_CUTLASS is ON")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86 89)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --extended-lambda")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(BEFORE ${CUDAToolkit_INCLUDE_DIRS})

if(HPC_ENABLE_CUTLASS)
  set(_cutlass_root "")
  set(_cutlass_candidates "")
  if(CUTLASS_DIR)
    list(APPEND _cutlass_candidates "${CUTLASS_DIR}")
  endif()
  list(APPEND _cutlass_candidates "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")

  foreach(_candidate IN LISTS _cutlass_candidates)
    if(EXISTS "${_candidate}/include/cutlass/gemm/device/gemm.h")
      set(_cutlass_root "${_candidate}")
      break()
    endif()
  endforeach()

  if(NOT _cutlass_root AND HPC_FETCH_CUTLASS)
    include(FetchContent)
    FetchContent_Declare(hpc_cutlass
      GIT_REPOSITORY "${HPC_CUTLASS_GIT_REPOSITORY}"
      GIT_TAG "${HPC_CUTLASS_GIT_TAG}"
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE
    )

    FetchContent_GetProperties(hpc_cutlass)
    if(NOT hpc_cutlass_POPULATED)
      message(STATUS
        "CUTLASS not found locally; fetching ${HPC_CUTLASS_GIT_REPOSITORY}@${HPC_CUTLASS_GIT_TAG}")
      FetchContent_Populate(hpc_cutlass)
    endif()

    if(EXISTS "${hpc_cutlass_SOURCE_DIR}/include/cutlass/gemm/device/gemm.h")
      set(_cutlass_root "${hpc_cutlass_SOURCE_DIR}")
    else()
      message(FATAL_ERROR
        "Fetched CUTLASS, but expected header not found at "
        "${hpc_cutlass_SOURCE_DIR}/include/cutlass/gemm/device/gemm.h")
    endif()
  endif()

  if(_cutlass_root)
    add_library(cutlass_headers INTERFACE)
    target_include_directories(cutlass_headers SYSTEM INTERFACE "${_cutlass_root}/include")
  else()
    message(FATAL_ERROR
      "CUTLASS not found. Set -DCUTLASS_DIR=/path/to/cutlass (with include/cutlass/...), "
      "place it at ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass, "
      "or enable auto-fetch with -DHPC_FETCH_CUTLASS=ON.")
  endif()
endif()

add_subdirectory(vector_add_kernel)
add_subdirectory(sgemm_kernel)
add_subdirectory(reduction)

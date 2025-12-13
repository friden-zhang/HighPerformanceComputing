project(sgemm_kernel CUDA CXX)

include_directories(./)

# add test
add_executable(sgemm_kernel_test
    gtest/sgemm_kernel_test.cu
)

target_link_libraries(sgemm_kernel_test
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)

enable_testing()

add_test(NAME sgemm_kernel_test COMMAND sgemm_kernel_test)

add_executable(sgemm_kernel_benchmark
    benchmark/sgemm_kernel_benchmark.cu
)

target_link_libraries(sgemm_kernel_benchmark
    PRIVATE
    CUDA::cudart
    benchmark::benchmark
)

if(TARGET cutlass_headers)
  target_link_libraries(sgemm_kernel_benchmark PRIVATE cutlass_headers)
  target_compile_definitions(sgemm_kernel_benchmark PRIVATE HPC_USE_CUTLASS=1)
  target_compile_options(sgemm_kernel_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()

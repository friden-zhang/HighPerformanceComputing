project(gemv CUDA CXX)

include_directories(./)

add_executable(gemv_test
    gtest/gemv_test.cu
)

target_link_libraries(gemv_test
    PRIVATE
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)

enable_testing()
add_test(NAME gemv_test COMMAND gemv_test)

add_executable(gemv_benchmark
    benchmark/gemv_benchmark.cu
)

target_link_libraries(gemv_benchmark
    PRIVATE
    CUDA::cudart
    benchmark::benchmark
)

if(TARGET cutlass_headers)
  target_link_libraries(gemv_benchmark PRIVATE cutlass_headers)
  target_compile_definitions(gemv_benchmark PRIVATE HPC_USE_CUTLASS=1)
  target_compile_options(gemv_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()

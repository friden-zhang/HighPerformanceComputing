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
    CUDA::cudart
    benchmark::benchmark
)
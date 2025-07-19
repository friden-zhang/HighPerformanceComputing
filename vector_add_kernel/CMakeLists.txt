project(vector_add_kernel CUDA CXX)

include_directories(./)

# add test
add_executable(vector_add_kernel_test
    gtest/vector_add_kernel_test.cu
)

add_executable(vector_add_kernel_benchmark
    benchmark/vector_add_kernel_benchmark.cu
)

target_link_libraries(vector_add_kernel_test
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)

target_link_libraries(vector_add_kernel_benchmark
    CUDA::cudart
    benchmark::benchmark
)

enable_testing()

add_test(NAME vector_add_kernel_test COMMAND vector_add_kernel_test)
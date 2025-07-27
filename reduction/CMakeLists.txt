project(reduction CUDA CXX)

include_directories(./)

# add test
add_executable(reduction_test
    gtest/reduction_test.cu
)

target_link_libraries(reduction_test
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)

add_executable(reduction_benchmark
    benchmark/reduction_benchmark.cu
)

target_link_libraries(reduction_benchmark
    CUDA::cudart
    benchmark::benchmark
)

enable_testing()

add_test(NAME reduction_test COMMAND reduction_test)
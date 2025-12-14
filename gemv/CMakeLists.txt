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


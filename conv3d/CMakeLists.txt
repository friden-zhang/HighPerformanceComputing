project(conv3d CUDA CXX)

include_directories(./)

add_executable(conv3d_test
    gtest/conv3d_test.cu
)

target_link_libraries(conv3d_test
    PRIVATE
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)

enable_testing()
add_test(NAME conv3d_test COMMAND conv3d_test)

add_executable(conv3d_benchmark
    benchmark/conv3d_benchmark.cu
)

target_link_libraries(conv3d_benchmark
    PRIVATE
    CUDA::cudart
    benchmark::benchmark
)

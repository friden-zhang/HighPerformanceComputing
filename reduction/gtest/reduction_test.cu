#include <gtest/gtest.h>

#include "kernel/reduction.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

static void cpu_sum(float *input, float *output, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += input[i];
  }
  *output = sum;
}

class ReductionTest : public ::testing::Test {
public:
  void SetUp() override {
    data = thrust::host_vector<float>(2048);
    for (int i = 0; i < 2048; i++) {
      data[i] = i / 2;
    }
    cpu_result = thrust::host_vector<float>(1);
    cpu_sum(data.data(), cpu_result.data(), 2048);
    d_data = data;
  }

  thrust::host_vector<float> data;
  thrust::host_vector<float> cpu_result;
  thrust::device_vector<float> d_data;
};

TEST_F(ReductionTest, NativeSumKernel) {
  thrust::device_vector<float> d_result(1);
  auto err = launch_native_sum_kernel<128>(
      thrust::raw_pointer_cast(d_data.data()),
      thrust::raw_pointer_cast(d_result.data()), 2048, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_result(d_result);
  EXPECT_NEAR(h_result[0], cpu_result[0], 1e-6);
}

TEST_F(ReductionTest, ControlDivergenceKernel) {
  thrust::device_vector<float> d_result(1);
  auto err = launch_control_divergence_kernel<128>(
      thrust::raw_pointer_cast(d_data.data()),
      thrust::raw_pointer_cast(d_result.data()), 2048, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_result(d_result);
  EXPECT_NEAR(h_result[0], cpu_result[0], 1e-6);
}

TEST_F(ReductionTest, SharedMemoryReductionKernel) {
  thrust::device_vector<float> d_result(1);
  auto err = launch_shared_memory_reduction_kernel<128>(
      thrust::raw_pointer_cast(d_data.data()),
      thrust::raw_pointer_cast(d_result.data()), 2048, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_result(d_result);
  EXPECT_NEAR(h_result[0], cpu_result[0], 1e-6);
}

TEST_F(ReductionTest, ThreadCoarseningKernel) {
  thrust::device_vector<float> d_result(1);
  auto err = launch_thread_coarsening_kernel<128, 3>(
      thrust::raw_pointer_cast(d_data.data()),
      thrust::raw_pointer_cast(d_result.data()), 2048, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_result(d_result);
  EXPECT_NEAR(h_result[0], cpu_result[0], 1e-6);
}

TEST_F(ReductionTest, WarpReductionKernel) {
  thrust::device_vector<float> d_result(1);
  auto err = launch_warp_reduction_kernel<128>(
      thrust::raw_pointer_cast(d_data.data()),
      thrust::raw_pointer_cast(d_result.data()), 2048, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_result(d_result);
  EXPECT_NEAR(h_result[0], cpu_result[0], 1e-6);
}
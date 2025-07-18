#include <gtest/gtest.h>

#include "kernel/vector_add_kernel.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

static void cpu_vector_add(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

class VectorAddKernelTest : public ::testing::Test {
public:
  void SetUp() override {
    a = thrust::host_vector<float>(1024 * 2048);
    b = thrust::host_vector<float>(1024 * 2048);
    for (int i = 0; i < 1024 * 2048; i++) {
      a[i] = i / 1024.0f;
      b[i] = i / 1024.0f;
    }
    cpu_c = thrust::host_vector<float>(1024 * 2048);
    cpu_vector_add(a.data(), b.data(), cpu_c.data(), 1024 * 2048);
    d_a = a;
    d_b = b;
  }

  thrust::host_vector<float> a;
  thrust::host_vector<float> b;
  thrust::host_vector<float> cpu_c;
  thrust::device_vector<float> d_a;
  thrust::device_vector<float> d_b;
};

TEST_F(VectorAddKernelTest, NormalKernel) {
  thrust::device_vector<float> d_c(1024 * 2048);
  cudaError_t err = launch_vector_add_kernel<32>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1024 * 2048, nullptr);
  EXPECT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(d_c);
  for (int i = 0; i < 1024 * 2048; i++) {
    EXPECT_NEAR(h_c[i], cpu_c[i], 1e-6);
  }
}

TEST_F(VectorAddKernelTest, SharedKernel) {
  thrust::device_vector<float> d_c(1024 * 2048);
  cudaError_t err = launch_vector_add_shared_kernel<32>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1024 * 2048, nullptr);
  EXPECT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(d_c);
  for (int i = 0; i < 1024 * 2048; i++) {
    EXPECT_NEAR(h_c[i], cpu_c[i], 1e-6);
  }
}

TEST_F(VectorAddKernelTest, CUBLoadStriped) {
  thrust::device_vector<float> d_c(1024 * 2048);
  cudaError_t err = launch_vector_add_cub_load_striped<32, 8>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1024 * 2048, nullptr);
  EXPECT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(d_c);
  for (int i = 0; i < 1024 * 2048; i++) {
    EXPECT_NEAR(h_c[i], cpu_c[i], 1e-6);
  }
}

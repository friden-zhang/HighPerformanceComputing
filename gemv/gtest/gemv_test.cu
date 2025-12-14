#include <gtest/gtest.h>

#include "kernel/native.cuh"
#include "kernel/wrap_reduce.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

static void cpu_sgemv(const float *A, const float *x, float *y, float alpha,
                      float beta, int M, int N) {
  for (int row = 0; row < M; row++) {
    float sum = 0.0f;
    for (int col = 0; col < N; col++) {
      sum += A[row * N + col] * x[col];
    }
    y[row] = alpha * sum + beta * y[row];
  }
}

class SGEMVKernelTest : public ::testing::Test {
public:
  void SetUp() override {
    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count == 0) {
      if (device_err == cudaSuccess) {
        GTEST_SKIP() << "No CUDA device available";
      }
      GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(device_err);
    }
    cudaSetDevice(0);

    A = thrust::host_vector<float>(M * N);
    x = thrust::host_vector<float>(N);
    y_init = thrust::host_vector<float>(M);

    for (int i = 0; i < M * N; i++) {
      A[i] = rand() % 5 / 5.0f;
    }
    for (int i = 0; i < N; i++) {
      x[i] = rand() % 5 / 5.0f;
    }
    for (int i = 0; i < M; i++) {
      y_init[i] = rand() % 5 / 5.0f;
    }

    cpu_y = y_init;
    cpu_sgemv(A.data(), x.data(), cpu_y.data(), alpha, beta, M, N);

    d_a = A;
    d_x = x;
  }

  int M = 128;
  int N = 512;
  float alpha = 0.7f;
  float beta = 0.3f;

  thrust::host_vector<float> A;
  thrust::host_vector<float> x;
  thrust::host_vector<float> y_init;
  thrust::host_vector<float> cpu_y;

  thrust::device_vector<float> d_a;
  thrust::device_vector<float> d_x;
};

TEST_F(SGEMVKernelTest, NativeKernel) {
  thrust::device_vector<float> d_y = y_init;
  cudaError_t err = launch_sgemv_native<256>(
      thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()), alpha, beta, M, N, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_y = d_y;
  for (int i = 0; i < M; i++) {
    ASSERT_NEAR(h_y[i], cpu_y[i], 1e-2);
  }
}

TEST_F(SGEMVKernelTest, WarpReduceKernel) {
  thrust::device_vector<float> d_y = y_init;
  cudaError_t err = launch_sgemv_wrap_reduce<256>(
      thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()), alpha, beta, M, N, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_y = d_y;
  for (int i = 0; i < M; i++) {
    ASSERT_NEAR(h_y[i], cpu_y[i], 1e-2);
  }
}

#include <gtest/gtest.h>

#include "kernel/mdspan_linear_nk_parallel.cuh"
#include "kernel/mdspan_spatial_nk_serial.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

static void cpu_conv3d(const float *input, const float *kernel, float *output,
                       int N, int C, int D, int H, int W, int K, int KD,
                       int KH, int KW) {
  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int od = 0; od < OD; ++od) {
        for (int oh = 0; oh < OH; ++oh) {
          for (int ow = 0; ow < OW; ++ow) {
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
              for (int kd = 0; kd < KD; ++kd) {
                for (int kh = 0; kh < KH; ++kh) {
                  for (int kw = 0; kw < KW; ++kw) {
                    const size_t in_idx =
                        ((((static_cast<size_t>(n) * C + c) * D + (od + kd)) *
                              H +
                          (oh + kh)) *
                             W +
                         (ow + kw));
                    const size_t kernel_idx =
                        ((((static_cast<size_t>(k) * C + c) * KD + kd) * KH +
                          kh) *
                             KW +
                         kw);
                    sum += input[in_idx] * kernel[kernel_idx];
                  }
                }
              }
            }
            const size_t out_idx =
                ((((static_cast<size_t>(n) * K + k) * OD + od) * OH + oh) *
                     OW +
                 ow);
            output[out_idx] = sum;
          }
        }
      }
    }
  }
}

class Conv3DTest : public ::testing::Test {
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

    input = thrust::host_vector<float>(N * C * D * H * W);
    kernel = thrust::host_vector<float>(K * C * KD * KH * KW);

    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = rand() % 5 / 5.0f;
    }
    for (size_t i = 0; i < kernel.size(); ++i) {
      kernel[i] = rand() % 5 / 5.0f;
    }

    output_ref = thrust::host_vector<float>(N * K * OD * OH * OW);
    cpu_conv3d(input.data(), kernel.data(), output_ref.data(), N, C, D, H, W, K,
               KD, KH, KW);

    d_input = input;
    d_kernel = kernel;
  }

  int N = 2;
  int C = 3;
  int D = 7;
  int H = 7;
  int W = 7;
  int K = 2;
  int KD = 3;
  int KH = 3;
  int KW = 3;
  int OD = D - KD + 1;
  int OH = H - KH + 1;
  int OW = W - KW + 1;

  thrust::host_vector<float> input;
  thrust::host_vector<float> kernel;
  thrust::host_vector<float> output_ref;
  thrust::device_vector<float> d_input;
  thrust::device_vector<float> d_kernel;
};

TEST_F(Conv3DTest, MdspanSpatialNKSerial) {
  thrust::device_vector<float> d_output(N * K * OD * OH * OW);
  cudaError_t err = launch_conv3d_mdspan_spatial_nk_serial<4, 4, 4>(
      thrust::raw_pointer_cast(d_input.data()),
      thrust::raw_pointer_cast(d_kernel.data()),
      thrust::raw_pointer_cast(d_output.data()), N, C, D, H, W, K, KD, KH, KW,
      nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_output = d_output;
  for (size_t i = 0; i < h_output.size(); ++i) {
    ASSERT_NEAR(h_output[i], output_ref[i], 1e-2);
  }
}

TEST_F(Conv3DTest, MdspanLinearNKParallel) {
  thrust::device_vector<float> d_output(N * K * OD * OH * OW);
  cudaError_t err = launch_conv3d_mdspan_linear_nk_parallel<256>(
      thrust::raw_pointer_cast(d_input.data()),
      thrust::raw_pointer_cast(d_kernel.data()),
      thrust::raw_pointer_cast(d_output.data()), N, C, D, H, W, K, KD, KH, KW,
      nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_output = d_output;
  for (size_t i = 0; i < h_output.size(); ++i) {
    ASSERT_NEAR(h_output[i], output_ref[i], 1e-2);
  }
}

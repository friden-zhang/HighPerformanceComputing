#pragma once

#include <cuda/std/mdspan>
#include <cuda_runtime.h>

// Conv3D: mdspan indexing, linearized output index with N/K parallelism.
__global__ void conv3d_mdspan_linear_nk_parallel(
    const float *__restrict__ input, const float *__restrict__ kernel,
    float *__restrict__ output, int N, int C, int D, int H, int W, int K,
    int KD, int KH, int KW) {

  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  const size_t total = static_cast<size_t>(N) * K * OD * OH * OW;
  const size_t linear =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (linear >= total)
    return;

  size_t idx = linear;
  const int ow = static_cast<int>(idx % OW);
  idx /= OW;
  const int oh = static_cast<int>(idx % OH);
  idx /= OH;
  const int od = static_cast<int>(idx % OD);
  idx /= OD;
  const int k = static_cast<int>(idx % K);
  const int n = static_cast<int>(idx / K);

  constexpr auto dyn = cuda::std::dynamic_extent;
  using ext_t = cuda::std::extents<int, dyn, dyn, dyn, dyn, dyn>;
  using mdspan_input_t = cuda::std::mdspan<const float, ext_t>;
  using mdspan_kernel_t = cuda::std::mdspan<const float, ext_t>;
  using mdspan_output_t = cuda::std::mdspan<float, ext_t>;

  ext_t ext_input(N, C, D, H, W);
  ext_t ext_kernel(K, C, KD, KH, KW);
  ext_t ext_output(N, K, OD, OH, OW);

  mdspan_input_t mdspan_input{input, ext_input};
  mdspan_kernel_t mdspan_kernel{kernel, ext_kernel};
  mdspan_output_t mdspan_output{output, ext_output};

  float sum = 0.0f;
  for (int c = 0; c < C; ++c) {
    for (int kd = 0; kd < KD; ++kd) {
      for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
          sum += mdspan_input(n, c, od + kd, oh + kh, ow + kw) *
                 mdspan_kernel(k, c, kd, kh, kw);
        }
      }
    }
  }
  mdspan_output(n, k, od, oh, ow) = sum;
}

template <int BLOCK_SIZE>
cudaError_t launch_conv3d_mdspan_linear_nk_parallel(
    const float *__restrict__ input, const float *__restrict__ kernel,
    float *__restrict__ output, int N, int C, int D, int H, int W, int K,
    int KD, int KH, int KW, cudaStream_t stream = nullptr) {
  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  if (OD <= 0 || OH <= 0 || OW <= 0) {
    return cudaErrorInvalidValue;
  }

  const size_t total = static_cast<size_t>(N) * K * OD * OH * OW;
  if (total == 0) {
    return cudaErrorInvalidValue;
  }

  const size_t grid_x = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 block(BLOCK_SIZE);
  dim3 grid(static_cast<unsigned int>(grid_x));

  if (stream) {
    conv3d_mdspan_linear_nk_parallel<<<grid, block, 0, stream>>>(
        input, kernel, output, N, C, D, H, W, K, KD, KH, KW);
  } else {
    conv3d_mdspan_linear_nk_parallel<<<grid, block>>>(
        input, kernel, output, N, C, D, H, W, K, KD, KH, KW);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}

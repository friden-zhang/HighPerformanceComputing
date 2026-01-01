#pragma once

#include <cuda/std/mdspan>
#include <cuda_runtime.h>

// Conv3D baseline: mdspan indexing, thread maps to (od, oh, ow) and serializes
// N/K.
__global__ void conv3d_mdspan_spatial_nk_serial(
    const float *__restrict__ input, const float *__restrict__ kernel,
    float *__restrict__ output, int N, int C, int D, int H, int W, int K,
    int KD, int KH, int KW) {

  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  const int ow = blockIdx.x * blockDim.x + threadIdx.x;
  const int oh = blockIdx.y * blockDim.y + threadIdx.y;
  const int od = blockIdx.z * blockDim.z + threadIdx.z;

  if (ow >= OW || oh >= OH || od >= OD)
    return;

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

  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
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
  }
}

template <int BLOCK_X, int BLOCK_Y, int BLOCK_Z>
cudaError_t launch_conv3d_mdspan_spatial_nk_serial(
    const float *__restrict__ input, const float *__restrict__ kernel,
    float *__restrict__ output, int N, int C, int D, int H, int W, int K,
    int KD, int KH, int KW, cudaStream_t stream = nullptr) {
  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  if (OD <= 0 || OH <= 0 || OW <= 0) {
    return cudaErrorInvalidValue;
  }

  dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
  dim3 grid((OW + BLOCK_X - 1) / BLOCK_X, (OH + BLOCK_Y - 1) / BLOCK_Y,
            (OD + BLOCK_Z - 1) / BLOCK_Z);

  if (stream) {
    conv3d_mdspan_spatial_nk_serial<<<grid, block, 0, stream>>>(
        input, kernel, output, N, C, D, H, W, K, KD, KH, KW);
  } else {
    conv3d_mdspan_spatial_nk_serial<<<grid, block>>>(
        input, kernel, output, N, C, D, H, W, K, KD, KH, KW);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}

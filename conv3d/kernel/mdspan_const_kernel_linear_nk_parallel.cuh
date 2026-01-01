#pragma once

#include <cuda/std/mdspan>
#include <cuda_runtime.h>

#include "kernel/mdspan_linear_nk_parallel.cuh"

constexpr int kConv3dConstKernelMaxFloats = 16384;

static __device__ __constant__ float kConv3dConstKernel[kConv3dConstKernelMaxFloats];

// Conv3D: mdspan indexing, linearized output index with N/K parallelism and
// kernel weights in constant memory.
template <int BLOCK_SIZE>
__global__ void conv3d_mdspan_const_kernel_linear_nk_parallel(
    const float *__restrict__ input, const float *__restrict__ kernel,
    float *__restrict__ output, int N, int C, int D, int H, int W, int K,
    int KD, int KH, int KW) {

  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  const size_t total =
      static_cast<size_t>(N) * K * OD * OH * OW;
  const size_t linear =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (linear >= total)
    return;

  (void)kernel;

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
  using mdspan_output_t = cuda::std::mdspan<float, ext_t>;

  ext_t ext_input(N, C, D, H, W);
  ext_t ext_output(N, K, OD, OH, OW);

  mdspan_input_t mdspan_input{input, ext_input};
  mdspan_output_t mdspan_output{output, ext_output};

  float sum = 0.0f;
  const int kernel_stride_c = KD * KH * KW;
  const int kernel_stride_k = C * kernel_stride_c;
  const int kernel_base_k = k * kernel_stride_k;

  for (int c = 0; c < C; ++c) {
    const int kernel_base_c = kernel_base_k + c * kernel_stride_c;
    for (int kd = 0; kd < KD; ++kd) {
      const int kernel_base_kd = kernel_base_c + kd * KH * KW;
      for (int kh = 0; kh < KH; ++kh) {
        const int kernel_base_kh = kernel_base_kd + kh * KW;
        for (int kw = 0; kw < KW; ++kw) {
          sum += mdspan_input(n, c, od + kd, oh + kh, ow + kw) *
                 kConv3dConstKernel[kernel_base_kh + kw];
        }
      }
    }
  }
  mdspan_output(n, k, od, oh, ow) = sum;
}

inline cudaError_t prepare_conv3d_const_kernel(const float *__restrict__ kernel,
                                               size_t elements,
                                               cudaStream_t stream = nullptr) {
  if (elements > static_cast<size_t>(kConv3dConstKernelMaxFloats)) {
    return cudaErrorInvalidValue;
  }

  return cudaMemcpyToSymbolAsync(kConv3dConstKernel, kernel,
                                 elements * sizeof(float), 0,
                                 cudaMemcpyDeviceToDevice, stream);
}

template <int BLOCK_SIZE>
cudaError_t launch_conv3d_mdspan_const_kernel_linear_nk_parallel_prepared(
    const float *__restrict__ input, const float *__restrict__ kernel,
    float *__restrict__ output, int N, int C, int D, int H, int W, int K,
    int KD, int KH, int KW, cudaStream_t stream = nullptr) {
  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  if (OD <= 0 || OH <= 0 || OW <= 0) {
    return cudaErrorInvalidValue;
  }

  const size_t total =
      static_cast<size_t>(N) * K * OD * OH * OW;
  if (total == 0) {
    return cudaErrorInvalidValue;
  }

  const size_t grid_x = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 block(BLOCK_SIZE);
  dim3 grid(static_cast<unsigned int>(grid_x));

  if (stream) {
    conv3d_mdspan_const_kernel_linear_nk_parallel<BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(input, kernel, output, N, C, D, H, W, K,
                                     KD, KH, KW);
  } else {
    conv3d_mdspan_const_kernel_linear_nk_parallel<BLOCK_SIZE>
        <<<grid, block>>>(input, kernel, output, N, C, D, H, W, K, KD, KH, KW);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}

template <int BLOCK_SIZE>
cudaError_t launch_conv3d_mdspan_const_kernel_linear_nk_parallel(
    const float *__restrict__ input, const float *__restrict__ kernel,
    float *__restrict__ output, int N, int C, int D, int H, int W, int K,
    int KD, int KH, int KW, cudaStream_t stream = nullptr) {
  const size_t kernel_elems =
      static_cast<size_t>(K) * C * KD * KH * KW;
  if (kernel_elems > static_cast<size_t>(kConv3dConstKernelMaxFloats)) {
    return launch_conv3d_mdspan_linear_nk_parallel<BLOCK_SIZE>(
        input, kernel, output, N, C, D, H, W, K, KD, KH, KW, stream);
  }

  cudaError_t prep = prepare_conv3d_const_kernel(kernel, kernel_elems, stream);
  if (prep != cudaSuccess) {
    return prep;
  }

  return launch_conv3d_mdspan_const_kernel_linear_nk_parallel_prepared<
      BLOCK_SIZE>(input, kernel, output, N, C, D, H, W, K, KD, KH, KW, stream);
}

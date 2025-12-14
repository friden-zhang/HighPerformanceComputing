#pragma once

#include <cuda/std/mdspan>
#include <cuda_runtime.h>

__global__ void sgemv_native(const float *A, const float *x, float *y,
                             float alpha, float beta, int M, int N) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) {
    return;
  }

  constexpr auto dyn = cuda::std::dynamic_extent;
  using ext2d_t = cuda::std::extents<int, dyn, dyn>;
  using ext1d_t = cuda::std::extents<int, dyn>;
  using A_mdspan_t = cuda::std::mdspan<const float, ext2d_t>;
  using x_mdspan_t = cuda::std::mdspan<const float, ext1d_t>;
  using y_mdspan_t = cuda::std::mdspan<float, ext1d_t>;

  A_mdspan_t mdA{A, ext2d_t(M, N)};
  x_mdspan_t mdx{x, ext1d_t(N)};
  y_mdspan_t mdy{y, ext1d_t(M)};

  float sum = 0.0f;
  for (int col = 0; col < N; col++) {
    sum += mdA(row, col) * mdx(col);
  }
  mdy(row) = alpha * sum + beta * mdy(row);
}

template <int BLOCK_SIZE>
cudaError_t launch_sgemv_native(const float *A, const float *x, float *y,
                                float alpha, float beta, int M, int N,
                                cudaStream_t stream = nullptr) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  if (stream) {
    sgemv_native<<<grid, block, 0, stream>>>(A, x, y, alpha, beta, M, N);
  } else {
    sgemv_native<<<grid, block>>>(A, x, y, alpha, beta, M, N);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}

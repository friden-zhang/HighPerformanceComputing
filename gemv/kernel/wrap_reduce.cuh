#pragma once

#include <cub/cub.cuh>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void sgemv_wrap_reduce(const float *A, const float *x, float *y,
                                  float alpha, float beta, int M, int N) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = BLOCK_SIZE / kWarpSize;

  using WarpReduce = cub::WarpReduce<float>;
  __shared__ typename WarpReduce::TempStorage temp_storage[kWarpsPerBlock];

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp_id;
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

  float thread_sum = 0.0f;
  for (int col = lane_id; col < N; col += kWarpSize) {
    thread_sum += mdA(row, col) * mdx(col);
  }

  const float sum = WarpReduce(temp_storage[warp_id]).Sum(thread_sum);
  if (lane_id == 0) {
    mdy(row) = alpha * sum + beta * mdy(row);
  }
}

template <int BLOCK_SIZE>
cudaError_t launch_sgemv_wrap_reduce(const float *A, const float *x, float *y,
                                    float alpha, float beta, int M, int N,
                                    cudaStream_t stream = nullptr) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
  constexpr int kWarpsPerBlock = BLOCK_SIZE / 32;

  dim3 block(BLOCK_SIZE);
  dim3 grid((M + kWarpsPerBlock - 1) / kWarpsPerBlock);

  if (stream) {
    sgemv_wrap_reduce<BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(A, x, y, alpha, beta, M, N);
  } else {
    sgemv_wrap_reduce<BLOCK_SIZE><<<grid, block>>>(A, x, y, alpha, beta, M, N);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}

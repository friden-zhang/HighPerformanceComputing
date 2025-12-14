#pragma once

#include <cub/cub.cuh>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
}

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
__global__ void sgemv_wrap_reduce_x_shared(const float *A, const float *x,
                                          float *y, float alpha, float beta,
                                          int M, int N) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = BLOCK_SIZE / kWarpSize;

  extern __shared__ float s_x[];

  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    s_x[col] = x[col];
  }
  __syncthreads();

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
  x_mdspan_t mdx{s_x, ext1d_t(N)};
  y_mdspan_t mdy{y, ext1d_t(M)};

  float thread_sum = 0.0f;
  for (int col = lane_id; col < N; col += kWarpSize) {
    thread_sum += mdA(row, col) * mdx(col);
  }

  const float sum = warp_reduce_sum(thread_sum);
  if (lane_id == 0) {
    mdy(row) = alpha * sum + beta * mdy(row);
  }
}

template <int BLOCK_SIZE>
__global__ void sgemv_wrap_reduce_vec4(const float *A, const float *x, float *y,
                                       float alpha, float beta, int M, int N) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = BLOCK_SIZE / kWarpSize;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp_id;
  if (row >= M) {
    return;
  }

  const int n4 = N / 4;
  const int rem = N % 4;

  const float4 *A4 = reinterpret_cast<const float4 *>(A);
  const float4 *x4 = reinterpret_cast<const float4 *>(x);

  float thread_sum = 0.0f;

  const int row_base4 = row * n4;
  for (int col4 = lane_id; col4 < n4; col4 += kWarpSize) {
    float4 a4 = A4[row_base4 + col4];
    float4 v4 = x4[col4];
    thread_sum += a4.x * v4.x + a4.y * v4.y + a4.z * v4.z + a4.w * v4.w;
  }

  if (rem) {
    const int scalar_base = n4 * 4;
    for (int col = scalar_base + lane_id; col < N; col += kWarpSize) {
      thread_sum += A[row * N + col] * x[col];
    }
  }

  const float sum = warp_reduce_sum(thread_sum);
  if (lane_id == 0) {
    y[row] = alpha * sum + beta * y[row];
  }
}

template <int BLOCK_SIZE, int WARPS_PER_ROW>
__global__ void sgemv_warp_group_reduce(const float *A, const float *x, float *y,
                                        float alpha, float beta, int M, int N) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = BLOCK_SIZE / kWarpSize;
  static_assert(kWarpsPerBlock % WARPS_PER_ROW == 0,
                "WARPS_PER_ROW must divide WARPS_PER_BLOCK");
  constexpr int kRowsPerBlock = kWarpsPerBlock / WARPS_PER_ROW;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  const int warp_in_row = warp_id % WARPS_PER_ROW;
  const int row_in_block = warp_id / WARPS_PER_ROW;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  const bool valid_row = row < M;

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
  if (valid_row) {
    for (int col = warp_in_row * kWarpSize + lane_id; col < N;
         col += WARPS_PER_ROW * kWarpSize) {
      thread_sum += mdA(row, col) * mdx(col);
    }
  }

  const float warp_sum = warp_reduce_sum(thread_sum);

  __shared__ float partial[kRowsPerBlock][WARPS_PER_ROW];
  if (lane_id == 0) {
    partial[row_in_block][warp_in_row] = warp_sum;
  }

  __syncthreads();

  if (valid_row && lane_id == 0 && warp_in_row == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS_PER_ROW; w++) {
      sum += partial[row_in_block][w];
    }
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

template <int BLOCK_SIZE>
cudaError_t launch_sgemv_wrap_reduce_x_shared(const float *A, const float *x,
                                             float *y, float alpha, float beta,
                                             int M, int N,
                                             cudaStream_t stream = nullptr) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
  constexpr int kWarpsPerBlock = BLOCK_SIZE / 32;

  dim3 block(BLOCK_SIZE);
  dim3 grid((M + kWarpsPerBlock - 1) / kWarpsPerBlock);
  size_t shared_bytes = static_cast<size_t>(N) * sizeof(float);

  if (stream) {
    sgemv_wrap_reduce_x_shared<BLOCK_SIZE>
        <<<grid, block, shared_bytes, stream>>>(A, x, y, alpha, beta, M, N);
  } else {
    sgemv_wrap_reduce_x_shared<BLOCK_SIZE>
        <<<grid, block, shared_bytes>>>(A, x, y, alpha, beta, M, N);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}

template <int BLOCK_SIZE>
cudaError_t launch_sgemv_wrap_reduce_vec4(const float *A, const float *x,
                                         float *y, float alpha, float beta,
                                         int M, int N,
                                         cudaStream_t stream = nullptr) {
  if (N % 4 != 0) {
    return launch_sgemv_wrap_reduce<BLOCK_SIZE>(A, x, y, alpha, beta, M, N,
                                                stream);
  }

  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
  constexpr int kWarpsPerBlock = BLOCK_SIZE / 32;

  dim3 block(BLOCK_SIZE);
  dim3 grid((M + kWarpsPerBlock - 1) / kWarpsPerBlock);

  if (stream) {
    sgemv_wrap_reduce_vec4<BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(A, x, y, alpha, beta, M, N);
  } else {
    sgemv_wrap_reduce_vec4<BLOCK_SIZE><<<grid, block>>>(A, x, y, alpha, beta, M,
                                                        N);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}

template <int BLOCK_SIZE, int WARPS_PER_ROW>
cudaError_t launch_sgemv_warp_group_reduce(const float *A, const float *x,
                                          float *y, float alpha, float beta,
                                          int M, int N,
                                          cudaStream_t stream = nullptr) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
  constexpr int kWarpsPerBlock = BLOCK_SIZE / 32;
  static_assert(kWarpsPerBlock % WARPS_PER_ROW == 0,
                "WARPS_PER_ROW must divide WARPS_PER_BLOCK");
  constexpr int kRowsPerBlock = kWarpsPerBlock / WARPS_PER_ROW;

  dim3 block(BLOCK_SIZE);
  dim3 grid((M + kRowsPerBlock - 1) / kRowsPerBlock);

  if (stream) {
    sgemv_warp_group_reduce<BLOCK_SIZE, WARPS_PER_ROW>
        <<<grid, block, 0, stream>>>(A, x, y, alpha, beta, M, N);
  } else {
    sgemv_warp_group_reduce<BLOCK_SIZE, WARPS_PER_ROW>
        <<<grid, block>>>(A, x, y, alpha, beta, M, N);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}

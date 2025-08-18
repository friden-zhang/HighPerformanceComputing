#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>

template <int BlockSize, int TileSize>
__global__ void sgemm_shared_mem_blocking_tile_strict_vec(float *A, float *B,
                                                          float *C, float alpha,
                                                          float beta, int M,
                                                          int N, int K) {
  const uint block_row = blockIdx.y;
  const uint block_col = blockIdx.x;

  const uint row = block_row * BlockSize + threadIdx.y;
  const uint col = block_col * BlockSize + threadIdx.x * TileSize;

  __shared__ float shared_A[BlockSize][BlockSize + 1];
  __shared__ float shared_B[BlockSize][BlockSize + 1];

  using float_2d_mds = cuda::std::mdspan<float, cuda::std::dextents<int, 2>>;
  float_2d_mds shared_A_mds{&shared_A[0][0], BlockSize, BlockSize + 1};
  float_2d_mds shared_B_mds{&shared_B[0][0], BlockSize, BlockSize + 1};

  constexpr auto dyn = cuda::std::dynamic_extent;
  using ext_t = cuda::std::extents<int, dyn, dyn>;
  using mdspan_t = cuda::std::mdspan<float, ext_t>;

  ext_t ext_A(M, N);
  ext_t ext_B(N, K);
  ext_t ext_C(M, K);

  mdspan_t mdspan_A{A, ext_A};
  mdspan_t mdspan_B{B, ext_B};
  mdspan_t mdspan_C{C, ext_C};

  float sum[TileSize] = {0.0f};
  const int N_blocks = (N + BlockSize - 1) / BlockSize;

  for (uint index = 0; index < N_blocks; ++index) {
    const int a_col = index * BlockSize + threadIdx.x * TileSize;
    const int b_row = index * BlockSize + threadIdx.y;

    // load A
#pragma unroll
    for (int i = 0; i < TileSize; i += 4) {
      const int a_col_i = a_col + i;
      float4 vec_a = *reinterpret_cast<float4 *>(
          &mdspan_A(row, a_col_i)); // load 4 floats at once
      shared_A_mds(threadIdx.y, threadIdx.x * TileSize + i) = vec_a.x;
      shared_A_mds(threadIdx.y, threadIdx.x * TileSize + i + 1) = vec_a.y;
      shared_A_mds(threadIdx.y, threadIdx.x * TileSize + i + 2) = vec_a.z;
      shared_A_mds(threadIdx.y, threadIdx.x * TileSize + i + 3) = vec_a.w;
    }

    // load B
#pragma unroll
    for (int i = 0; i < TileSize; i += 4) {
      const int b_col_i = col + i;

      float4 vec_b = *reinterpret_cast<float4 *>(
          &mdspan_B(b_row, b_col_i)); // load 4 floats at once
      shared_B_mds(threadIdx.y, threadIdx.x * TileSize + i) = vec_b.x;
      shared_B_mds(threadIdx.y, threadIdx.x * TileSize + i + 1) = vec_b.y;
      shared_B_mds(threadIdx.y, threadIdx.x * TileSize + i + 2) = vec_b.z;
      shared_B_mds(threadIdx.y, threadIdx.x * TileSize + i + 3) = vec_b.w;
    }
    __syncthreads();
#pragma unroll
    for (uint k = 0; k < BlockSize; ++k) {
      const float a = shared_A_mds(threadIdx.y, k);
#pragma unroll
      for (int i = 0; i < TileSize; ++i) {
        const float b = shared_B_mds(k, threadIdx.x * TileSize + i);
        sum[i] = fmaf(a, b, sum[i]);
      }
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < TileSize; i += 4) {
      uint col_i = col + i;
      float4& out = *reinterpret_cast<float4*>(&mdspan_C(row, col_i));
      // mdspan_C(row, col_i) = alpha * sum[i] + beta * ;
      out.x = alpha * sum[i] + beta * out.x;
      out.y = alpha * sum[i + 1] + beta * out.y;
      out.z = alpha * sum[i + 2] + beta * out.z;
      out.w = alpha * sum[i + 3] + beta * out.w;
    }
  }
}

template <int BlockSize, int TileSize>
cudaError_t launch_sgemm_shared_mem_blocking_tile_strict_vec(
    float *A, float *B, float *C, float alpha, float beta, int M, int N, int K,
    cudaStream_t stream = nullptr) {

  static_assert(BlockSize > 0, "BlockSize must be greater than 0");
  static_assert(TileSize > 0, "TileSize must be greater than 0");
  static_assert(
      BlockSize % TileSize == 0,
      "BlockSize must be divisible by TileSize"); // BlockSize / TileSize
  static_assert(TileSize % 4 == 0,
                "TileSize must be divisible by 4"); // BlockSize / 4
  assert(N % (BlockSize * TileSize) == 0);
  assert(K % (BlockSize * TileSize) == 0);
  assert(M % (BlockSize * TileSize) == 0);

  dim3 block(BlockSize / TileSize, BlockSize);
  // Grid size based on K and M dimensions, it decided by c dimension
  dim3 grid((K + BlockSize - 1) / BlockSize, (M + BlockSize - 1) / BlockSize);

  if (stream) {
    sgemm_shared_mem_blocking_tile_strict_vec<BlockSize, TileSize>
        <<<grid, block, 0, stream>>>(A, B, C, alpha, beta, M, N, K);
  } else {
    sgemm_shared_mem_blocking_tile_strict_vec<BlockSize, TileSize>
        <<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
  }
  return cudaGetLastError();
}
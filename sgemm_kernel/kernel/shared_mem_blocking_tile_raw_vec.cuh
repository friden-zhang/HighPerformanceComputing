#pragma once

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

template <int BlockSize, int TileSize>
__global__ void sgemm_shared_mem_blocking_tile_raw_vec(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, float alpha, float beta, int M, int N, int K) {
  static_assert(BlockSize > 0, "BlockSize must be greater than 0");
  static_assert(TileSize > 0, "TileSize must be greater than 0");
  static_assert(BlockSize % TileSize == 0,
                "BlockSize must be divisible by TileSize");
  static_assert(TileSize % 4 == 0, "TileSize must be divisible by 4");

  const int row = static_cast<int>(blockIdx.y) * BlockSize +
                  static_cast<int>(threadIdx.y);
  const int col = static_cast<int>(blockIdx.x) * BlockSize +
                  static_cast<int>(threadIdx.x) * TileSize;

  __shared__ alignas(16) float shared_A[BlockSize][BlockSize];
  __shared__ alignas(16) float shared_B[BlockSize][BlockSize];

  float sum[TileSize] = {0.0f};
  const int N_tiles = N / BlockSize;

  for (int tile = 0; tile < N_tiles; ++tile) {
    const int a_col = tile * BlockSize + static_cast<int>(threadIdx.x) * TileSize;
    const int b_row = tile * BlockSize + static_cast<int>(threadIdx.y);

    const float *a_ptr = A + row * N + a_col;
    const float *b_ptr = B + b_row * K + col;

#pragma unroll
    for (int i = 0; i < TileSize; i += 4) {
      const float4 vec_a = *reinterpret_cast<const float4 *>(a_ptr + i);
      *reinterpret_cast<float4 *>(
          &shared_A[threadIdx.y][static_cast<int>(threadIdx.x) * TileSize + i]) =
          vec_a;

      const float4 vec_b = *reinterpret_cast<const float4 *>(b_ptr + i);
      *reinterpret_cast<float4 *>(
          &shared_B[threadIdx.y][static_cast<int>(threadIdx.x) * TileSize + i]) =
          vec_b;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BlockSize; ++k) {
      const float a = shared_A[threadIdx.y][k];
#pragma unroll
      for (int i = 0; i < TileSize; ++i) {
        const float b =
            shared_B[k][static_cast<int>(threadIdx.x) * TileSize + i];
        sum[i] = fmaf(a, b, sum[i]);
      }
    }

    __syncthreads();
  }

  float *c_ptr = C + row * K + col;
  if (beta == 0.0f) {
#pragma unroll
    for (int i = 0; i < TileSize; i += 4) {
      float4 out;
      out.x = alpha * sum[i];
      out.y = alpha * sum[i + 1];
      out.z = alpha * sum[i + 2];
      out.w = alpha * sum[i + 3];
      *reinterpret_cast<float4 *>(c_ptr + i) = out;
    }
  } else {
#pragma unroll
    for (int i = 0; i < TileSize; i += 4) {
      float4 out = *reinterpret_cast<float4 *>(c_ptr + i);
      out.x = alpha * sum[i] + beta * out.x;
      out.y = alpha * sum[i + 1] + beta * out.y;
      out.z = alpha * sum[i + 2] + beta * out.z;
      out.w = alpha * sum[i + 3] + beta * out.w;
      *reinterpret_cast<float4 *>(c_ptr + i) = out;
    }
  }
}

template <int BlockSize, int TileSize>
cudaError_t launch_sgemm_shared_mem_blocking_tile_raw_vec(
    float *A, float *B, float *C, float alpha, float beta, int M, int N, int K,
    cudaStream_t stream = nullptr) {
  static_assert(BlockSize > 0, "BlockSize must be greater than 0");
  static_assert(TileSize > 0, "TileSize must be greater than 0");
  static_assert(BlockSize % TileSize == 0,
                "BlockSize must be divisible by TileSize");
  static_assert(TileSize % 4 == 0, "TileSize must be divisible by 4");

  assert(M % BlockSize == 0);
  assert(N % BlockSize == 0);
  assert(K % BlockSize == 0);
  assert(N % 4 == 0);
  assert(K % 4 == 0);

  assert((reinterpret_cast<uintptr_t>(A) & 0xF) == 0);
  assert((reinterpret_cast<uintptr_t>(B) & 0xF) == 0);
  assert((reinterpret_cast<uintptr_t>(C) & 0xF) == 0);

  dim3 block(BlockSize / TileSize, BlockSize);
  dim3 grid(K / BlockSize, M / BlockSize);

  const float *A_const = A;
  const float *B_const = B;

  if (stream) {
    sgemm_shared_mem_blocking_tile_raw_vec<BlockSize, TileSize>
        <<<grid, block, 0, stream>>>(A_const, B_const, C, alpha, beta, M, N, K);
  } else {
    sgemm_shared_mem_blocking_tile_raw_vec<BlockSize, TileSize>
        <<<grid, block>>>(A_const, B_const, C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}


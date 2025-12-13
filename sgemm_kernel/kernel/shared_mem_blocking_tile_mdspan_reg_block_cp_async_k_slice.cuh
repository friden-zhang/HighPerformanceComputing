#pragma once

#include <cassert>
#include <cstdint>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>

#include "shared_mem_blocking_tile_mdspan_reg_block_cp_async.cuh"

template <int BlockSize, int RowTile, int ColTile, int KSlice>
__global__ void sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, float alpha, float beta, int M, int N, int K) {
  static_assert(BlockSize > 0, "BlockSize must be greater than 0");
  static_assert(RowTile > 0, "RowTile must be greater than 0");
  static_assert(ColTile > 0, "ColTile must be greater than 0");
  static_assert(KSlice > 0, "KSlice must be greater than 0");
  static_assert(BlockSize % RowTile == 0,
                "BlockSize must be divisible by RowTile");
  static_assert(BlockSize % ColTile == 0,
                "BlockSize must be divisible by ColTile");
  static_assert(ColTile == 4, "This kernel currently supports ColTile == 4");
  static_assert(KSlice % 4 == 0, "KSlice must be divisible by 4");

  constexpr auto dyn = cuda::std::dynamic_extent;
  using ext_t = cuda::std::extents<int, dyn, dyn>;
  using mdspan_a_t = cuda::std::mdspan<const float, ext_t>;
  using mdspan_b_t = cuda::std::mdspan<const float, ext_t>;
  using mdspan_c_t = cuda::std::mdspan<float, ext_t>;

  const ext_t ext_A(M, N);
  const ext_t ext_B(N, K);
  const ext_t ext_C(M, K);

  mdspan_a_t mdspan_A{A, ext_A};
  mdspan_b_t mdspan_B{B, ext_B};
  mdspan_c_t mdspan_C{C, ext_C};

  const int block_row0 = static_cast<int>(blockIdx.y) * BlockSize;
  const int block_col0 = static_cast<int>(blockIdx.x) * BlockSize;

  const int row_local0 = static_cast<int>(threadIdx.y) * RowTile;
  const int col_local0 = static_cast<int>(threadIdx.x) * ColTile;

  const int row0 = block_row0 + row_local0;
  const int col0 = block_col0 + col_local0;

  __shared__ alignas(16) float shared_A[2][BlockSize][KSlice];
  __shared__ alignas(16) float shared_B[2][KSlice][BlockSize];

  float acc[RowTile][ColTile] = {0.0f};

  const int tid = static_cast<int>(threadIdx.y) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
  const int num_threads =
      static_cast<int>(blockDim.x) * static_cast<int>(blockDim.y);

  constexpr int VecWidth = 4;
  constexpr int AVecPerRow = KSlice / VecWidth;
  constexpr int BVecPerRow = BlockSize / VecWidth;
  constexpr int NumAVec = BlockSize * AVecPerRow;
  constexpr int NumBVec = KSlice * BVecPerRow;

  auto prefetch_slice = [&](int slice, int stage) {
    // A slice: [BlockSize x KSlice]
    for (int idx = tid; idx < NumAVec; idx += num_threads) {
      const int row = idx / AVecPerRow;
      const int vec = idx - row * AVecPerRow;
      const int col = vec * VecWidth;

      const float *src = &mdspan_A(block_row0 + row, slice * KSlice + col);
      void *dst = &shared_A[stage][row][col];
      hpc::detail::cp_async_cg_16(dst, src);
    }

    // B slice: [KSlice x BlockSize]
    for (int idx = tid; idx < NumBVec; idx += num_threads) {
      const int row = idx / BVecPerRow;
      const int vec = idx - row * BVecPerRow;
      const int col = vec * VecWidth;

      const float *src = &mdspan_B(slice * KSlice + row, block_col0 + col);
      void *dst = &shared_B[stage][row][col];
      hpc::detail::cp_async_cg_16(dst, src);
    }
    hpc::detail::cp_async_commit_group();
  };

  const int k_slices = N / KSlice;
  int stage = 0;

  prefetch_slice(0, stage);
  hpc::detail::cp_async_wait_all();
  __syncthreads();

  for (int slice = 0; slice < k_slices; ++slice) {
    const int next_stage = stage ^ 1;
    if (slice + 1 < k_slices) {
      prefetch_slice(slice + 1, next_stage);
    }

#pragma unroll
    for (int kk = 0; kk < KSlice; ++kk) {
      const float4 b_vec = *reinterpret_cast<const float4 *>(
          &shared_B[stage][kk][col_local0]);

#pragma unroll
      for (int ry = 0; ry < RowTile; ++ry) {
        const float a = shared_A[stage][row_local0 + ry][kk];
        acc[ry][0] = fmaf(a, b_vec.x, acc[ry][0]);
        acc[ry][1] = fmaf(a, b_vec.y, acc[ry][1]);
        acc[ry][2] = fmaf(a, b_vec.z, acc[ry][2]);
        acc[ry][3] = fmaf(a, b_vec.w, acc[ry][3]);
      }
    }

    if (slice + 1 < k_slices) {
      hpc::detail::cp_async_wait_all();
      __syncthreads();
      stage = next_stage;
    }
  }

  if (beta == 0.0f) {
#pragma unroll
    for (int ry = 0; ry < RowTile; ++ry) {
      float *c_ptr = &mdspan_C(row0 + ry, col0);
      float4 out;
      out.x = alpha * acc[ry][0];
      out.y = alpha * acc[ry][1];
      out.z = alpha * acc[ry][2];
      out.w = alpha * acc[ry][3];
      *reinterpret_cast<float4 *>(c_ptr) = out;
    }
  } else {
#pragma unroll
    for (int ry = 0; ry < RowTile; ++ry) {
      float *c_ptr = &mdspan_C(row0 + ry, col0);
      float4 out = *reinterpret_cast<float4 *>(c_ptr);
      out.x = alpha * acc[ry][0] + beta * out.x;
      out.y = alpha * acc[ry][1] + beta * out.y;
      out.z = alpha * acc[ry][2] + beta * out.z;
      out.w = alpha * acc[ry][3] + beta * out.w;
      *reinterpret_cast<float4 *>(c_ptr) = out;
    }
  }
}

template <int BlockSize, int RowTile, int ColTile, int KSlice>
cudaError_t launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice(
    float *A, float *B, float *C, float alpha, float beta, int M, int N, int K,
    cudaStream_t stream = nullptr) {
  static_assert(BlockSize > 0, "BlockSize must be greater than 0");
  static_assert(RowTile > 0, "RowTile must be greater than 0");
  static_assert(ColTile > 0, "ColTile must be greater than 0");
  static_assert(KSlice > 0, "KSlice must be greater than 0");
  static_assert(BlockSize % RowTile == 0,
                "BlockSize must be divisible by RowTile");
  static_assert(BlockSize % ColTile == 0,
                "BlockSize must be divisible by ColTile");
  static_assert(ColTile == 4, "This kernel currently supports ColTile == 4");
  static_assert(KSlice % 4 == 0, "KSlice must be divisible by 4");

  assert(M % BlockSize == 0);
  assert(N % KSlice == 0);
  assert(K % BlockSize == 0);
  assert(N % 4 == 0);
  assert(K % 4 == 0);

  assert((reinterpret_cast<uintptr_t>(A) & 0xF) == 0);
  assert((reinterpret_cast<uintptr_t>(B) & 0xF) == 0);
  assert((reinterpret_cast<uintptr_t>(C) & 0xF) == 0);

  dim3 block(BlockSize / ColTile, BlockSize / RowTile);
  dim3 grid(K / BlockSize, M / BlockSize);

  const float *A_const = A;
  const float *B_const = B;

  if (stream) {
    sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
        BlockSize, RowTile, ColTile, KSlice>
        <<<grid, block, 0, stream>>>(A_const, B_const, C, alpha, beta, M, N, K);
  } else {
    sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
        BlockSize, RowTile, ColTile, KSlice>
        <<<grid, block>>>(A_const, B_const, C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}


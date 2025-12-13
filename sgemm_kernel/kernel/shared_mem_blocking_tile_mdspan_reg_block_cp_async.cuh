#pragma once

#include <cassert>
#include <cstdint>
#include <cuda/std/mdspan>
#include <cuda_runtime.h>

namespace hpc {
namespace detail {

__device__ __forceinline__ void cp_async_cg_16(void *shared_dst,
                                               const void *global_src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const unsigned int shared_addr =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(shared_addr), "l"(global_src)
               : "memory");
#else
  *reinterpret_cast<float4 *>(shared_dst) =
      *reinterpret_cast<const float4 *>(global_src);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

} // namespace detail
} // namespace hpc

template <int BlockSize, int RowTile, int ColTile>
__global__ void sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, float alpha, float beta, int M, int N, int K) {
  static_assert(BlockSize > 0, "BlockSize must be greater than 0");
  static_assert(RowTile > 0, "RowTile must be greater than 0");
  static_assert(ColTile > 0, "ColTile must be greater than 0");
  static_assert(BlockSize % RowTile == 0,
                "BlockSize must be divisible by RowTile");
  static_assert(BlockSize % ColTile == 0,
                "BlockSize must be divisible by ColTile");
  static_assert(ColTile == 4, "This kernel currently supports ColTile == 4");

  const int row0 = static_cast<int>(blockIdx.y) * BlockSize +
                   static_cast<int>(threadIdx.y) * RowTile;
  const int col0 = static_cast<int>(blockIdx.x) * BlockSize +
                   static_cast<int>(threadIdx.x) * ColTile;

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

  __shared__ alignas(16) float shared_A[2][BlockSize][BlockSize];
  __shared__ alignas(16) float shared_B[2][BlockSize][BlockSize];

  float acc[RowTile][ColTile] = {0.0f};

  const int a_col0 = static_cast<int>(threadIdx.x) * ColTile;
  const int b_row0 = static_cast<int>(threadIdx.y) * RowTile;
  const int N_tiles = N / BlockSize;

  auto prefetch_tile = [&](int tile, int stage) {
#pragma unroll
    for (int ry = 0; ry < RowTile; ++ry) {
      const float *src_a = &mdspan_A(row0 + ry, tile * BlockSize + a_col0);
      void *dst_a = &shared_A[stage][b_row0 + ry][a_col0];
      hpc::detail::cp_async_cg_16(dst_a, src_a);

      const float *src_b = &mdspan_B(tile * BlockSize + b_row0 + ry, col0);
      void *dst_b = &shared_B[stage][b_row0 + ry][a_col0];
      hpc::detail::cp_async_cg_16(dst_b, src_b);
    }
    hpc::detail::cp_async_commit_group();
  };

  int stage = 0;
  prefetch_tile(0, stage);
  hpc::detail::cp_async_wait_all();
  __syncthreads();

  for (int tile = 0; tile < N_tiles; ++tile) {
    const int next_stage = stage ^ 1;
    if (tile + 1 < N_tiles) {
      prefetch_tile(tile + 1, next_stage);
    }

#pragma unroll
    for (int k = 0; k < BlockSize; ++k) {
      const float4 b_vec = *reinterpret_cast<const float4 *>(
          &shared_B[stage][k][a_col0]);
#pragma unroll
      for (int ry = 0; ry < RowTile; ++ry) {
        const float a = shared_A[stage][b_row0 + ry][k];
        acc[ry][0] = fmaf(a, b_vec.x, acc[ry][0]);
        acc[ry][1] = fmaf(a, b_vec.y, acc[ry][1]);
        acc[ry][2] = fmaf(a, b_vec.z, acc[ry][2]);
        acc[ry][3] = fmaf(a, b_vec.w, acc[ry][3]);
      }
    }

    if (tile + 1 < N_tiles) {
      hpc::detail::cp_async_wait_all();
    }
    __syncthreads();
    stage = next_stage;
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

template <int BlockSize, int RowTile, int ColTile>
cudaError_t launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async(
    float *A, float *B, float *C, float alpha, float beta, int M, int N, int K,
    cudaStream_t stream = nullptr) {
  static_assert(BlockSize > 0, "BlockSize must be greater than 0");
  static_assert(RowTile > 0, "RowTile must be greater than 0");
  static_assert(ColTile > 0, "ColTile must be greater than 0");
  static_assert(BlockSize % RowTile == 0,
                "BlockSize must be divisible by RowTile");
  static_assert(BlockSize % ColTile == 0,
                "BlockSize must be divisible by ColTile");
  static_assert(ColTile == 4, "This kernel currently supports ColTile == 4");

  assert(M % BlockSize == 0);
  assert(N % BlockSize == 0);
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
    sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async<BlockSize, RowTile,
                                                             ColTile>
        <<<grid, block, 0, stream>>>(A_const, B_const, C, alpha, beta, M, N, K);
  } else {
    sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async<BlockSize, RowTile,
                                                             ColTile>
        <<<grid, block>>>(A_const, B_const, C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}


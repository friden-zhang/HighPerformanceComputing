#pragma once

#include <cuda/std/mdspan>
#include <cuda_runtime.h>
#include <type_traits>

template <int BLOCK_SIZE>
__global__ void sgemm_shared_mem_blocking(float *A, float *B, float *C,
                                          float alpha, float beta, int M, int N,
                                          int K) {
  const uint block_row = blockIdx.y;
  const uint block_col = blockIdx.x;

  const uint row = block_row * BLOCK_SIZE + threadIdx.y;
  const uint col = block_col * BLOCK_SIZE + threadIdx.x;

  __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE + 1];
  __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE + 1];

  using float_2d_mds = cuda::std::mdspan<float, cuda::std::dextents<int, 2>>;
  float_2d_mds shared_A_mds{&shared_A[0][0], BLOCK_SIZE, BLOCK_SIZE + 1};
  float_2d_mds shared_B_mds{&shared_B[0][0], BLOCK_SIZE, BLOCK_SIZE + 1};

  constexpr auto dyn = cuda::std::dynamic_extent;
  using ext_t = cuda::std::extents<int, dyn, dyn>;
  using mdspan_t = cuda::std::mdspan<float, ext_t>;

  ext_t ext_A(M, N);
  ext_t ext_B(N, K);
  ext_t ext_C(M, K);

  mdspan_t mdspan_A{A, ext_A};
  mdspan_t mdspan_B{B, ext_B};
  mdspan_t mdspan_C{C, ext_C};

  float sum = 0.0f;

  const int N_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (uint index = 0; index < N_blocks; index++) {

    const int a_col = index * BLOCK_SIZE + threadIdx.x;
    const int a_row = index * BLOCK_SIZE + threadIdx.y;

    shared_A_mds(threadIdx.y, threadIdx.x) =
        a_col < N && row < M ? mdspan_A(row, a_col) : 0.0f;

    shared_B_mds(threadIdx.y, threadIdx.x) =
        a_row < N && col < K ? mdspan_B(a_row, col) : 0.0f;

    __syncthreads();
#pragma unroll
    for (uint k = 0; k < BLOCK_SIZE; k++) {
      sum += shared_A_mds(threadIdx.y, k) * shared_B_mds(k, threadIdx.x);
    }
    __syncthreads();
  }
  if (row >= M || col >= K)
    return;
  mdspan_C(row, col) = alpha * sum + beta * mdspan_C(row, col);
}

template <int BLOCK_SIZE>
cudaError_t launch_sgemm_shared_mem_blocking(float *A, float *B, float *C,
                                             float alpha, float beta, int M,
                                             int N, int K,
                                             cudaStream_t stream = nullptr) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  if (stream) {
    sgemm_shared_mem_blocking<BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(A, B, C, alpha, beta, M, N, K);
  } else {
    sgemm_shared_mem_blocking<BLOCK_SIZE>
        <<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}
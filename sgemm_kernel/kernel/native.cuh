#pragma once

#include <cuda/std/mdspan>
#include <cuda_runtime.h>


__global__ void sgemm_native(float *A, float *B, float *C, float alpha,
                             float beta, int M, int N, int K) {
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= K || row >= M)
    return;

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
#pragma unroll
  for (uint index = 0; index < N; index++) {
    // sum += A[row * N + index] * B[index * K + col];
    sum += mdspan_A(row, index) * mdspan_B(index, col);
  }

  // C[row * K + col] = alpha * sum + beta * C[row * K + col];
  mdspan_C(row, col) = alpha * sum + beta * mdspan_C(row, col);
}

template <int BLOCK_SIZEX, int BLOCK_SIZEY>
cudaError_t launch_sgemm_native(float *A, float *B, float *C, float alpha,
                                float beta, int M, int N, int K,
                                cudaStream_t stream = nullptr) {
  dim3 block(BLOCK_SIZEX, BLOCK_SIZEY);
  dim3 grid((K + BLOCK_SIZEX - 1) / BLOCK_SIZEX,
            (M + BLOCK_SIZEY - 1) / BLOCK_SIZEY);

  if (stream) {
    sgemm_native<<<grid, block, 0, stream>>>(A, B, C, alpha, beta, M, N, K);
  } else {
    sgemm_native<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}
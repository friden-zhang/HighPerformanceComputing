#pragma once

#include <cuda_runtime.h>

__global__ void sgemm_native(float *A, float *B, float *C, float alpha,
                             float beta, int M, int N, int K) {
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= K || row >= M)
    return;

  float sum = 0.0f;
  for (uint index = 0; index < N; index++) {
    sum += A[row * N + index] * B[index * K + col];
  }

  C[row * K + col] = alpha * sum + beta * C[row * K + col];
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
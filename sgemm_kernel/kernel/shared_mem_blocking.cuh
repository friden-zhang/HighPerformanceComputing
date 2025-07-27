#pragma once

#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void sgemm_shared_mem_blocking(float *A, float *B, float *C,
                                          float alpha, float beta, int M, int N,
                                          int K) {
  const uint block_row = blockIdx.y;
  const uint block_col = blockIdx.x;

  const uint row = block_row * BLOCK_SIZE + threadIdx.y;
  const uint col = block_col * BLOCK_SIZE + threadIdx.x;

  if (row >= M || col >= K)
    return;

  __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0.0f;

  for (uint index = 0; index < N / BLOCK_SIZE; index++) {
    shared_A[threadIdx.y][threadIdx.x] =
        A[row * N + index * BLOCK_SIZE + threadIdx.x];
    shared_B[threadIdx.y][threadIdx.x] =
        B[(threadIdx.y + index * BLOCK_SIZE) * K + col];

    __syncthreads();

    for (uint k = 0; k < BLOCK_SIZE; k++) {
      sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
    }
    __syncthreads();
  }
  C[row * K + col] = alpha * sum + beta * C[row * K + col];
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
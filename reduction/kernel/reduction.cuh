#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <typename T> static T CeilDiv(T x, T y) { return (x + y - 1) / y; }

/// ----------------------------------------------------------------------------
/// native sum kernel
/// ----------------------------------------------------------------------------
template <int BlockDim>
__global__ void native_sum_kernel(float *input, float *output, int n) {
  int data_index = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (data_index > n) {
    return;
  }
  for (int stride = 1; stride <= BlockDim; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[data_index] +=
          data_index + stride < n ? input[data_index + stride] : 0;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, input[2 * blockIdx.x * blockDim.x]);
  }
}

// launch native sum kernel
template <int BlockDim>
cudaError_t launch_native_sum_kernel(float *input, float *output, int n,
                                     cudaStream_t stream) {
  dim3 block(BlockDim);
  dim3 grid(CeilDiv(n, BlockDim * 2));
  if (stream == nullptr) {
    native_sum_kernel<BlockDim><<<grid, block>>>(input, output, n);
    cudaDeviceSynchronize();
  } else {
    native_sum_kernel<BlockDim><<<grid, block, 0, stream>>>(input, output, n);
  }
  return cudaGetLastError();
}

/// ----------------------------------------------------------------------------
/// Control divergence kernel
/// ----------------------------------------------------------------------------
template <int BlockDim>
__global__ void control_divergence_kernel(float *input, float *output, int n) {
  int data_index = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
  if (data_index > n) {
    return;
  }
  for (uint32_t stride = BlockDim; stride >= 1; stride >>= 1) {
    if (threadIdx.x < stride) {
      input[data_index] +=
          data_index + stride < n ? input[data_index + stride] : 0;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, input[2 * blockIdx.x * blockDim.x]);
  }
}

// launch control divergence kernel
template <int BlockDim>
cudaError_t launch_control_divergence_kernel(float *input, float *output, int n,
                                             cudaStream_t stream) {
  dim3 block(BlockDim);
  dim3 grid(CeilDiv(n, BlockDim * 2));
  if (stream == nullptr) {
    control_divergence_kernel<BlockDim><<<grid, block>>>(input, output, n);
    cudaDeviceSynchronize();
  } else {
    control_divergence_kernel<BlockDim>
        <<<grid, block, 0, stream>>>(input, output, n);
  }
  return cudaGetLastError();
}

/// ----------------------------------------------------------------------------
/// shared memory reduction kernel
/// ----------------------------------------------------------------------------
template <int BlockDim>
__global__ void shared_memory_reduction_kernel(float *input, float *output,
                                               int n) {
  __shared__ float shared_data[BlockDim];
  int data_index = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
  if (data_index > n) {
    return;
  }
  shared_data[threadIdx.x] =
      input[data_index] +
      (data_index + BlockDim < n ? input[data_index + BlockDim] : 0);
  __syncthreads();
  for (uint32_t stride = BlockDim / 2; stride >= 1; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, shared_data[0]);
  }
}

// launch shared memory reduction kernel
template <int BlockDim>
cudaError_t launch_shared_memory_reduction_kernel(float *input, float *output,
                                                  int n, cudaStream_t stream) {
  dim3 block(BlockDim);
  dim3 grid(CeilDiv(n, BlockDim * 2));
  if (stream == nullptr) {
    shared_memory_reduction_kernel<BlockDim><<<grid, block>>>(input, output, n);
    cudaDeviceSynchronize();
  } else {
    shared_memory_reduction_kernel<BlockDim>
        <<<grid, block, 0, stream>>>(input, output, n);
  }
  return cudaGetLastError();
}

/// ----------------------------------------------------------------------------
/// Thread Coarsening
/// ----------------------------------------------------------------------------
template <int BlockDim, int ThreadCoarseningSize>
__global__ void thread_coarsening_kernel(float *input, float *output, int n) {
  __shared__ float shared_data[BlockDim];
  int data_index =
      (blockIdx.x * blockDim.x) * ThreadCoarseningSize + threadIdx.x;
  if (data_index > n) {
    return;
  }
  float sum = 0.0f;
  for (int i = 0; i < ThreadCoarseningSize; i++) {
    if (data_index + i * BlockDim > n) {
      break;
    }
    sum += input[data_index + i * BlockDim];
  }
  shared_data[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = BlockDim / 2; stride >= 1; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, shared_data[0]);
  }
}

// launch thread coarsening kernel
template <int BlockDim, int ThreadCoarseningSize>
cudaError_t launch_thread_coarsening_kernel(float *input, float *output, int n,
                                            cudaStream_t stream) {
  dim3 block(BlockDim);
  dim3 grid(CeilDiv(n, BlockDim * ThreadCoarseningSize));
  if (stream == nullptr) {
    thread_coarsening_kernel<BlockDim, ThreadCoarseningSize>
        <<<grid, block>>>(input, output, n);
    cudaDeviceSynchronize();
  } else {
    thread_coarsening_kernel<BlockDim, ThreadCoarseningSize>
        <<<grid, block, 0, stream>>>(input, output, n);
  }
  return cudaGetLastError();
}
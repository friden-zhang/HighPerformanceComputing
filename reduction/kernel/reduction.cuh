#include <cub/cub.cuh>
#include <cuda/__atomic/atomic.h>
#include <cuda_runtime.h>

#include <cuda/atomic>

template <typename T> static T CeilDiv(T x, T y) { return (x + y - 1) / y; }

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, 32);
  }
  return x;
}

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
    cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
    output_ref.fetch_add(input[2 * blockIdx.x * blockDim.x],
                         cuda::std::memory_order_relaxed);
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
    cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
    output_ref.fetch_add(input[2 * blockIdx.x * blockDim.x],
                         cuda::std::memory_order_relaxed);
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
    cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
    output_ref.fetch_add(shared_data[0], cuda::std::memory_order_relaxed);
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
    cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
    output_ref.fetch_add(shared_data[0], cuda::std::memory_order_relaxed);
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

/// ----------------------------------------------------------------------------
/// Warp Reduction
/// ----------------------------------------------------------------------------
template <int BlockDim>
__global__ void warp_reduction_kernel(float *input, float *output, int n) {
  constexpr int kWarpThreadNum = 32;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float data = tid < n ? input[tid] : 0.0f;
  data = warp_reduce_sum(data);
  if (BlockDim > kWarpThreadNum) {
    static_assert(BlockDim / kWarpThreadNum <= 32,
                  "BlockDim / kWarpThreadNum <= 32");
    __shared__ float shared_data[32];
    int wrap_id = threadIdx.x / kWarpThreadNum;
    int lane_id = threadIdx.x % kWarpThreadNum;
    if (lane_id == 0) {
      shared_data[wrap_id] = data;
    }
    __syncthreads();
    if (wrap_id == 0) {
      data = threadIdx.x < BlockDim / kWarpThreadNum ? shared_data[threadIdx.x]
                                                     : 0.0f;
      data = warp_reduce_sum(data);
      if (lane_id == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
        output_ref.fetch_add(data, cuda::std::memory_order_relaxed);
      }
    }
  }
}

// launch warp reduction kernel
template <int BlockDim>
cudaError_t launch_warp_reduction_kernel(float *input, float *output, int n,
                                         cudaStream_t stream) {
  dim3 block(BlockDim);
  dim3 grid(CeilDiv(n, BlockDim));
  if (stream == nullptr) {
    warp_reduction_kernel<BlockDim><<<grid, block>>>(input, output, n);
    cudaDeviceSynchronize();
  } else {
    warp_reduction_kernel<BlockDim>
        <<<grid, block, 0, stream>>>(input, output, n);
  }
  return cudaGetLastError();
}
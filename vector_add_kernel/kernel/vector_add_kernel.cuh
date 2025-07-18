#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

/// ----------------------------------------------------------------------------
/// normal vector add kernel
/// ----------------------------------------------------------------------------
__global__ void vector_add_kernel(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

template <int BLOCK_SIZE>
cudaError_t launch_vector_add_kernel(float *a, float *b, float *c, int n,
                                     cudaStream_t stream) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  if (stream == nullptr) {
    vector_add_kernel<<<grid, block>>>(a, b, c, n);
    cudaDeviceSynchronize();
  } else {
    vector_add_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
  }
  return cudaGetLastError();
}

/// ----------------------------------------------------------------------------
/// shared memory vector add kernel
/// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void vector_add_kernel_shared(float *a, float *b, float *c, int n) {
  __shared__ float shared_a[BLOCK_SIZE];
  __shared__ float shared_b[BLOCK_SIZE];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    shared_a[threadIdx.x] = a[idx];
    shared_b[threadIdx.x] = b[idx];
    __syncthreads();
    c[idx] = shared_a[threadIdx.x] + shared_b[threadIdx.x];
  }
}

template <int BLOCK_SIZE>
cudaError_t launch_vector_add_shared_kernel(float *a, float *b, float *c, int n,
                                            cudaStream_t stream) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  if (stream == nullptr) {
    vector_add_kernel_shared<BLOCK_SIZE>
        <<<grid, block>>>(a, b, c, n);
    cudaDeviceSynchronize();
  } else {
    vector_add_kernel_shared<BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(a, b, c, n);
  }
  return cudaGetLastError();
}

/// ----------------------------------------------------------------------------
/// cub with thread tile s
/// ----------------------------------------------------------------------------
template <int BLOCK_SIZE, int THREAD_TILE_SIZE>
__global__ void vector_add_cub_load_striped(float *a, float *b, float *c,
                                              int n) {
  using BlockLoad = cub::BlockLoad<float, BLOCK_SIZE, THREAD_TILE_SIZE, cub::BLOCK_LOAD_STRIPED>;
  using BlockStore = cub::BlockStore<float, BLOCK_SIZE, THREAD_TILE_SIZE, cub::BLOCK_STORE_STRIPED>;

  __shared__ typename BlockLoad::TempStorage load_a_storage;
  __shared__ typename BlockLoad::TempStorage load_b_storage;
  __shared__ typename BlockStore::TempStorage store_storage;

  float a_items[THREAD_TILE_SIZE];
  float b_items[THREAD_TILE_SIZE];
  float c_items[THREAD_TILE_SIZE];

  const int TILE_ITEMS = BLOCK_SIZE * THREAD_TILE_SIZE;
  int block_offset = blockIdx.x * TILE_ITEMS;

  const float* a_ptr = a + block_offset;
  const float* b_ptr = b + block_offset;
  float* c_ptr = c + block_offset;

  BlockLoad(load_a_storage).Load(a_ptr, a_items, n - block_offset, 0.0f);
  BlockLoad(load_b_storage).Load(b_ptr, b_items, n - block_offset, 0.0);

#pragma unroll
  for (int i = 0; i < THREAD_TILE_SIZE; i++) {
    c_items[i] = a_items[i] + b_items[i];
  }

  BlockStore(store_storage).Store(c_ptr, c_items, n - block_offset);
}

template <int BLOCK_SIZE, int THREAD_TILE_SIZE>
cudaError_t launch_vector_add_cub_load_striped(float *a, float *b, float *c,
                                                 int n,
                                                 cudaStream_t stream) {
  int blocks = CEIL_DIV(n, BLOCK_SIZE * THREAD_TILE_SIZE);
  blocks = std::max(blocks, 1);
  dim3 grid(blocks), block(BLOCK_SIZE);
  cudaError_t err;
  if (stream == nullptr) {
    vector_add_cub_load_striped<BLOCK_SIZE, THREAD_TILE_SIZE>
        <<<grid, block>>>(a, b, c, n);
    err = cudaDeviceSynchronize();
  } else {
    vector_add_cub_load_striped<BLOCK_SIZE, THREAD_TILE_SIZE>
        <<<grid, block, 0, stream>>>(a, b, c, n);
    err = cudaGetLastError();
  }
  return err;
}
#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "kernel/native.cuh"
#include "kernel/shared_mem_blocking.cuh"
#include "kernel/shared_mem_blocking_tile.cuh"
#include "kernel/shared_mem_blocking_tile_vec.cuh"
#include "kernel/shared_mem_blocking_tile_strict_vec.cuh"

int M = 1024;
int N = 2048;
int K = 3072;

template <int BLOCK_SIZEX, int BLOCK_SIZEY>
static void BM_SGEMMKernel(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);

  for (auto _ : state) {
    auto ret = launch_sgemm_native<BLOCK_SIZEX, BLOCK_SIZEY>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

template <int BLOCK_SIZE>
static void BM_SGEMMSharedMemBlocking(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);

  for (auto _ : state) {
    auto ret = launch_sgemm_shared_mem_blocking<BLOCK_SIZE>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

template <int BLOCK_SIZE, int TILE_SIZE>
static void BM_SGEMMSharedMemBlockingTile(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret = launch_sgemm_shared_mem_blocking_tile<BLOCK_SIZE, TILE_SIZE>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

template <int BLOCK_SIZE, int TILE_SIZE>
static void BM_SGEMMSharedMemBlockingTileVec(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret = launch_sgemm_shared_mem_blocking_tile_vec<BLOCK_SIZE, TILE_SIZE>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

template <int BLOCK_SIZE, int TILE_SIZE>
static void BM_SGEMMSharedMemBlockingTileStrictVec(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret = launch_sgemm_shared_mem_blocking_tile_strict_vec<BLOCK_SIZE, TILE_SIZE>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

BENCHMARK(BM_SGEMMKernel<16, 8>);
BENCHMARK(BM_SGEMMKernel<16, 16>);
BENCHMARK(BM_SGEMMKernel<32, 16>);
BENCHMARK(BM_SGEMMKernel<32, 32>);

BENCHMARK(BM_SGEMMSharedMemBlocking<16>);
BENCHMARK(BM_SGEMMSharedMemBlocking<32>);

BENCHMARK(BM_SGEMMSharedMemBlockingTile<16, 2>);
BENCHMARK(BM_SGEMMSharedMemBlockingTile<16, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTile<16, 8>);
BENCHMARK(BM_SGEMMSharedMemBlockingTile<32, 2>);
BENCHMARK(BM_SGEMMSharedMemBlockingTile<32, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTile<32, 8>);

BENCHMARK(BM_SGEMMSharedMemBlockingTileVec<16, 2>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileVec<16, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileVec<16, 8>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileVec<32, 2>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileVec<32, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileVec<32, 8>);


BENCHMARK(BM_SGEMMSharedMemBlockingTileStrictVec<16, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileStrictVec<16, 8>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileStrictVec<32, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileStrictVec<32, 8>);

BENCHMARK_MAIN();
#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "kernel/native.cuh"
#include "kernel/wrap_reduce.cuh"

int M = 10240;
int N = 2048;

template <int BLOCK_SIZE>
static void BM_SGEMVNative(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(M);

  for (auto _ : state) {
    auto ret = launch_sgemv_native<BLOCK_SIZE>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()), 1.0f, 0.0f, M, N, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

template <int BLOCK_SIZE>
static void BM_SGEMVWrapReduce(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(M);

  for (auto _ : state) {
    auto ret = launch_sgemv_wrap_reduce<BLOCK_SIZE>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()), 1.0f, 0.0f, M, N, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

BENCHMARK(BM_SGEMVNative<128>);
BENCHMARK(BM_SGEMVNative<256>);
BENCHMARK(BM_SGEMVNative<512>);

BENCHMARK(BM_SGEMVWrapReduce<128>);
BENCHMARK(BM_SGEMVWrapReduce<256>);
BENCHMARK(BM_SGEMVWrapReduce<512>);

BENCHMARK_MAIN();


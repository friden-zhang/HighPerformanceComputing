#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "kernel/native.cuh"
#include "kernel/shared_mem_blocking.cuh"

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
    launch_sgemm_native<BLOCK_SIZEX, BLOCK_SIZEY>(thrust::raw_pointer_cast(d_a.data()),
                                thrust::raw_pointer_cast(d_b.data()),
                                thrust::raw_pointer_cast(d_c.data()), 1.0f,
                                0.0f, M, N, K, nullptr);
  }
}

template <int BLOCK_SIZE>
static void BM_SGEMMSharedMemBlocking(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);

  for (auto _ : state) {
    launch_sgemm_shared_mem_blocking<BLOCK_SIZE>(thrust::raw_pointer_cast(d_a.data()),
                                thrust::raw_pointer_cast(d_b.data()),
                                thrust::raw_pointer_cast(d_c.data()), 1.0f,
                                0.0f, M, N, K, nullptr);
  }
}

BENCHMARK(BM_SGEMMKernel<16, 8>);
BENCHMARK(BM_SGEMMKernel<16, 16>);
BENCHMARK(BM_SGEMMKernel<32, 16>);
BENCHMARK(BM_SGEMMKernel<32, 32>);

BENCHMARK(BM_SGEMMSharedMemBlocking<16>);
BENCHMARK(BM_SGEMMSharedMemBlocking<32>);

BENCHMARK_MAIN();
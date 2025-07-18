#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "kernel/vector_add_kernel.cuh"

static void BM_VectorAddKernel(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(10240 * 2048);
  thrust::device_vector<float> d_b(10240 * 2048);
  thrust::device_vector<float> d_c(10240 * 2048);

  for (auto _ : state) {
    launch_vector_add_kernel<16>(thrust::raw_pointer_cast(d_a.data()),
                                 thrust::raw_pointer_cast(d_b.data()),
                                 thrust::raw_pointer_cast(d_c.data()),
                                 10240 * 2048, nullptr);
  }
}

static void BM_VectorAddSharedKernel(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(10240 * 2048);
  thrust::device_vector<float> d_b(10240 * 2048);
  thrust::device_vector<float> d_c(10240 * 2048);

  for (auto _ : state) {
    launch_vector_add_shared_kernel<16>(thrust::raw_pointer_cast(d_a.data()),
                                        thrust::raw_pointer_cast(d_b.data()),
                                        thrust::raw_pointer_cast(d_c.data()),
                                        10240 * 2048, nullptr);
  }
}

template <int BLOCK_SIZE, int THREAD_TILE_SIZE>
static void BM_VectorAddCUBLoadStriped(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(10240 * 2048);
  thrust::device_vector<float> d_b(10240 * 2048);
  thrust::device_vector<float> d_c(10240 * 2048);

  for (auto _ : state) {
    launch_vector_add_cub_load_striped<BLOCK_SIZE, THREAD_TILE_SIZE>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), 10240 * 2048, nullptr);
  }
}

BENCHMARK(BM_VectorAddKernel);
BENCHMARK(BM_VectorAddSharedKernel);

BENCHMARK(BM_VectorAddCUBLoadStriped<16, 8>);
BENCHMARK(BM_VectorAddCUBLoadStriped<32, 8>);
BENCHMARK(BM_VectorAddCUBLoadStriped<64, 8>);
BENCHMARK(BM_VectorAddCUBLoadStriped<128, 8>);
BENCHMARK(BM_VectorAddCUBLoadStriped<256, 8>);
BENCHMARK(BM_VectorAddCUBLoadStriped<512, 8>);
BENCHMARK(BM_VectorAddCUBLoadStriped<1024, 8>);
BENCHMARK(BM_VectorAddCUBLoadStriped<16, 16>);
BENCHMARK(BM_VectorAddCUBLoadStriped<32, 16>);
BENCHMARK(BM_VectorAddCUBLoadStriped<64, 16>);
BENCHMARK(BM_VectorAddCUBLoadStriped<128, 16>);
BENCHMARK(BM_VectorAddCUBLoadStriped<256, 16>);
BENCHMARK(BM_VectorAddCUBLoadStriped<512, 16>);
BENCHMARK(BM_VectorAddCUBLoadStriped<1024, 16>);
BENCHMARK(BM_VectorAddCUBLoadStriped<16, 32>);
BENCHMARK(BM_VectorAddCUBLoadStriped<32, 32>);
BENCHMARK(BM_VectorAddCUBLoadStriped<64, 32>);
BENCHMARK(BM_VectorAddCUBLoadStriped<128, 32>);
BENCHMARK(BM_VectorAddCUBLoadStriped<256, 32>);
BENCHMARK(BM_VectorAddCUBLoadStriped<512, 32>);

BENCHMARK_MAIN();
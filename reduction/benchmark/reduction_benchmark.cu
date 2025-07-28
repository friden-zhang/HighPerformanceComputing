#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

#include "kernel/reduction.cuh"

template <int BlockDim>
static void BM_NativeSumKernel(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::host_vector<float> data(10240 * 2048);
  thrust::random::default_random_engine rng(42);
  thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
  thrust::generate(data.begin(), data.end(),
                   [&]() { return dist(rng); });
  thrust::device_vector<float> d_data = data;
  thrust::device_vector<float> d_result(1);

  for (auto _ : state) {
    launch_native_sum_kernel<BlockDim>(
        thrust::raw_pointer_cast(d_data.data()),
        thrust::raw_pointer_cast(d_result.data()), 10240 * 2048, nullptr);
  }
}

template <int BlockDim>
static void BM_ControlDivergenceKernel(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::host_vector<float> data(10240 * 2048);
  thrust::random::default_random_engine rng(42);
  thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
  thrust::generate(data.begin(), data.end(),
                   [&]() { return dist(rng); });
  thrust::device_vector<float> d_data = data;
  thrust::device_vector<float> d_result(1);

  for (auto _ : state) {
    launch_control_divergence_kernel<BlockDim>(
        thrust::raw_pointer_cast(d_data.data()),
        thrust::raw_pointer_cast(d_result.data()), 10240 * 2048, nullptr);
  }
}

template <int BlockDim>
static void BM_SharedMemoryReductionKernel(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::host_vector<float> data(10240 * 2048);
  thrust::random::default_random_engine rng(42);
  thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
  thrust::generate(data.begin(), data.end(),
                   [&]() { return dist(rng); });
  thrust::device_vector<float> d_data = data;
  thrust::device_vector<float> d_result(1);

  for (auto _ : state) {
    launch_shared_memory_reduction_kernel<BlockDim>(
        thrust::raw_pointer_cast(d_data.data()),
        thrust::raw_pointer_cast(d_result.data()), 10240 * 2048, nullptr);
  }
}

template <int BlockDim, int ThreadCoarseningSize>
static void BM_ThreadCoarseningKernel(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::host_vector<float> data(10240 * 2048);
  thrust::random::default_random_engine rng(42);
  thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
  thrust::generate(data.begin(), data.end(),
                   [&]() { return dist(rng); });
  thrust::device_vector<float> d_data = data;
  thrust::device_vector<float> d_result(1);

  for (auto _ : state) {
    launch_thread_coarsening_kernel<BlockDim, ThreadCoarseningSize>(
        thrust::raw_pointer_cast(d_data.data()),
        thrust::raw_pointer_cast(d_result.data()), 10240 * 2048, nullptr);
  }
}

template <int BlockDim>
static void BM_WarpReductionKernel(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::host_vector<float> data(10240 * 2048);
  thrust::random::default_random_engine rng(42);
  thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
  thrust::generate(data.begin(), data.end(),
                   [&]() { return dist(rng); });
  thrust::device_vector<float> d_data = data;
  thrust::device_vector<float> d_result(1);

  for (auto _ : state) {
    launch_warp_reduction_kernel<BlockDim>(
        thrust::raw_pointer_cast(d_data.data()),
        thrust::raw_pointer_cast(d_result.data()), 10240 * 2048, nullptr);
  }
}

BENCHMARK(BM_NativeSumKernel<128>);
BENCHMARK(BM_NativeSumKernel<256>);
BENCHMARK(BM_NativeSumKernel<512>);
BENCHMARK(BM_NativeSumKernel<1024>);
BENCHMARK(BM_ControlDivergenceKernel<128>);
BENCHMARK(BM_ControlDivergenceKernel<256>);
BENCHMARK(BM_ControlDivergenceKernel<512>);
BENCHMARK(BM_ControlDivergenceKernel<1024>);
BENCHMARK(BM_SharedMemoryReductionKernel<128>);
BENCHMARK(BM_SharedMemoryReductionKernel<256>);
BENCHMARK(BM_SharedMemoryReductionKernel<512>);
BENCHMARK(BM_SharedMemoryReductionKernel<1024>);
BENCHMARK(BM_ThreadCoarseningKernel<128, 4>);
BENCHMARK(BM_ThreadCoarseningKernel<128, 8>);
BENCHMARK(BM_ThreadCoarseningKernel<128, 16>);
BENCHMARK(BM_ThreadCoarseningKernel<256, 4>);
BENCHMARK(BM_ThreadCoarseningKernel<256, 8>);
BENCHMARK(BM_ThreadCoarseningKernel<256, 16>);
BENCHMARK(BM_ThreadCoarseningKernel<512, 4>);
BENCHMARK(BM_ThreadCoarseningKernel<512, 8>);
BENCHMARK(BM_ThreadCoarseningKernel<512, 16>);
BENCHMARK(BM_ThreadCoarseningKernel<1024, 4>);
BENCHMARK(BM_ThreadCoarseningKernel<1024, 8>);
BENCHMARK(BM_ThreadCoarseningKernel<1024, 16>);
BENCHMARK(BM_WarpReductionKernel<32>);
BENCHMARK(BM_WarpReductionKernel<64>);
BENCHMARK(BM_WarpReductionKernel<128>);
BENCHMARK(BM_WarpReductionKernel<256>);
BENCHMARK(BM_WarpReductionKernel<512>);
BENCHMARK(BM_WarpReductionKernel<1024>);

BENCHMARK_MAIN();
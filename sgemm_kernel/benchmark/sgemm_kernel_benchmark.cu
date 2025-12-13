#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "kernel/native.cuh"
#include "kernel/shared_mem_blocking.cuh"
#include "kernel/shared_mem_blocking_tile.cuh"
#include "kernel/shared_mem_blocking_tile_vec.cuh"
#include "kernel/shared_mem_blocking_tile_strict_vec.cuh"

#ifdef HPC_USE_CUTLASS
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#endif

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

#ifdef HPC_USE_CUTLASS
static void BM_CUTLASS_SGEMM(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);

  using Layout = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, Layout, float, Layout, float,
                                           Layout>;

  Gemm gemm_op;

  // CUTLASS GEMM uses (m, n, k): A[m, k] * B[k, n] = C[m, n]
  cutlass::gemm::GemmCoord problem_size(M, K, N);

  const int lda = N;
  const int ldb = K;
  const int ldc = K;

  typename Gemm::Arguments arguments{
      problem_size,
      {thrust::raw_pointer_cast(d_a.data()), lda},
      {thrust::raw_pointer_cast(d_b.data()), ldb},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {1.0f, 0.0f}};

  if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
    state.SkipWithError("CUTLASS GEMM cannot implement this problem");
    return;
  }

  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  thrust::device_vector<uint8_t> workspace(workspace_size);
  uint8_t *workspace_ptr =
      workspace_size > 0 ? thrust::raw_pointer_cast(workspace.data()) : nullptr;

  if (gemm_op.initialize(arguments, workspace_ptr) !=
      cutlass::Status::kSuccess) {
    state.SkipWithError("CUTLASS GEMM initialize failed");
    return;
  }

  for (auto _ : state) {
    auto status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      state.SkipWithError("CUTLASS GEMM launch failed");
      return;
    }
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(err));
      return;
    }
    benchmark::DoNotOptimize(status);
  }
}
#endif

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

#ifdef HPC_USE_CUTLASS
BENCHMARK(BM_CUTLASS_SGEMM);
#endif

BENCHMARK_MAIN();

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "kernel/native.cuh"
#include "kernel/wrap_reduce.cuh"

#ifdef HPC_USE_CUTLASS
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#endif

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

template <int BLOCK_SIZE>
static void BM_SGEMVWrapReduceXShared(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(M);

  for (auto _ : state) {
    auto ret = launch_sgemv_wrap_reduce_x_shared<BLOCK_SIZE>(
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
static void BM_SGEMVWrapReduceVec4(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(M);

  for (auto _ : state) {
    auto ret = launch_sgemv_wrap_reduce_vec4<BLOCK_SIZE>(
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

template <int BLOCK_SIZE, int WARPS_PER_ROW>
static void BM_SGEMVWarpGroupReduce(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(M);

  for (auto _ : state) {
    auto ret = launch_sgemv_warp_group_reduce<BLOCK_SIZE, WARPS_PER_ROW>(
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

BENCHMARK(BM_SGEMVWrapReduceXShared<256>);
BENCHMARK(BM_SGEMVWrapReduceVec4<256>);
BENCHMARK(BM_SGEMVWarpGroupReduce<256, 2>);
BENCHMARK(BM_SGEMVWarpGroupReduce<256, 4>);

#ifdef HPC_USE_CUTLASS
static void BM_CUTLASS_GEMV(benchmark::State &state) {
  cudaSetDevice(0);

  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(M);

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, LayoutA, float, LayoutB,
                                           float, LayoutC>;

  Gemm gemm_op;

  // CUTLASS GEMM uses (m, n, k): A[m, k] * B[k, n] = C[m, n]
  // GEMV: A[M, N] * x[N] => y[M] == A[M, N] * B[N, 1] where B is a vector.
  cutlass::gemm::GemmCoord problem_size(M, /*n=*/1, /*k=*/N);

  const int lda = N;
  const int ldb = N; // ColumnMajor (k, n) => leading dim is k
  const int ldc = M; // ColumnMajor (m, n) => leading dim is m

  typename Gemm::Arguments arguments{
      problem_size,
      {thrust::raw_pointer_cast(d_a.data()), lda},
      {thrust::raw_pointer_cast(d_x.data()), ldb},
      {thrust::raw_pointer_cast(d_y.data()), ldc},
      {thrust::raw_pointer_cast(d_y.data()), ldc},
      {1.0f, 0.0f}};

  auto can_status = gemm_op.can_implement(arguments);
  if (can_status != cutlass::Status::kSuccess) {
    state.SkipWithError(cutlass::cutlassGetStatusString(can_status));
    return;
  }

  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  thrust::device_vector<uint8_t> workspace(workspace_size);
  uint8_t *workspace_ptr =
      workspace_size > 0 ? thrust::raw_pointer_cast(workspace.data()) : nullptr;

  auto init_status = gemm_op.initialize(arguments, workspace_ptr);
  if (init_status != cutlass::Status::kSuccess) {
    state.SkipWithError(cutlass::cutlassGetStatusString(init_status));
    return;
  }

  for (auto _ : state) {
    auto status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      state.SkipWithError(cutlass::cutlassGetStatusString(status));
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

BENCHMARK(BM_CUTLASS_GEMV);
#endif

BENCHMARK_MAIN();

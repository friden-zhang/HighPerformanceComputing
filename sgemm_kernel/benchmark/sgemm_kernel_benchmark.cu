#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "kernel/native.cuh"
#include "kernel/shared_mem_blocking.cuh"
#include "kernel/shared_mem_blocking_tile.cuh"
#include "kernel/shared_mem_blocking_tile_mdspan_reg_block.cuh"
#include "kernel/shared_mem_blocking_tile_mdspan_reg_block_cp_async.cuh"
#include "kernel/shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice.cuh"
#include "kernel/shared_mem_blocking_tile_mdspan_vec.cuh"
#include "kernel/shared_mem_blocking_tile_raw_scalar.cuh"
#include "kernel/shared_mem_blocking_tile_raw_vec.cuh"
#include "kernel/shared_mem_blocking_tile_raw_vec_no_restrict.cuh"
#include "kernel/shared_mem_blocking_tile_vec.cuh"
#include "kernel/shared_mem_blocking_tile_strict_vec.cuh"
#include "kernel/shared_mem_blocking_tile_strict_vec_shared_float4.cuh"

#ifdef HPC_USE_CUTLASS
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/tfloat32.h>
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

template <int BLOCK_SIZE, int TILE_SIZE>
static void BM_SGEMMSharedMemBlockingTileRawVec(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret = launch_sgemm_shared_mem_blocking_tile_raw_vec<BLOCK_SIZE, TILE_SIZE>(
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
static void BM_SGEMMSharedMemBlockingTileMdspanVec(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret =
        launch_sgemm_shared_mem_blocking_tile_mdspan_vec<BLOCK_SIZE, TILE_SIZE>(
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
static void BM_SGEMMSharedMemBlockingTileRawScalar(benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret =
        launch_sgemm_shared_mem_blocking_tile_raw_scalar<BLOCK_SIZE, TILE_SIZE>(
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
static void BM_SGEMMSharedMemBlockingTileRawVecNoRestrict(
    benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret = launch_sgemm_shared_mem_blocking_tile_raw_vec_no_restrict<
        BLOCK_SIZE, TILE_SIZE>(thrust::raw_pointer_cast(d_a.data()),
                               thrust::raw_pointer_cast(d_b.data()),
                               thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f,
                               M, N, K, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

template <int BLOCK_SIZE, int TILE_SIZE>
static void BM_SGEMMSharedMemBlockingTileStrictVecSharedFloat4(
    benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret = launch_sgemm_shared_mem_blocking_tile_strict_vec_shared_float4<
        BLOCK_SIZE, TILE_SIZE>(thrust::raw_pointer_cast(d_a.data()),
                               thrust::raw_pointer_cast(d_b.data()),
                               thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f,
                               M, N, K, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

template <int BLOCK_SIZE, int ROW_TILE, int COL_TILE>
static void BM_SGEMMSharedMemBlockingTileMdspanRegBlock(
    benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret = launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block<
        BLOCK_SIZE, ROW_TILE, COL_TILE>(thrust::raw_pointer_cast(d_a.data()),
                                        thrust::raw_pointer_cast(d_b.data()),
                                        thrust::raw_pointer_cast(d_c.data()),
                                        1.0f, 0.0f, M, N, K, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

template <int BLOCK_SIZE, int ROW_TILE, int COL_TILE>
static void BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsync(
    benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret =
        launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async<
            BLOCK_SIZE, ROW_TILE, COL_TILE>(
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

template <int BLOCK_SIZE, int ROW_TILE, int COL_TILE, int KSLICE>
static void BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice(
    benchmark::State &state) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_a(M * N);
  thrust::device_vector<float> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);
  for (auto _ : state) {
    auto ret = launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
        BLOCK_SIZE, ROW_TILE, COL_TILE, KSLICE>(
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

static void BM_CUTLASS_TF32_TENSOROP_SGEMM(benchmark::State &state) {
  cudaSetDevice(0);

  thrust::device_vector<cutlass::tfloat32_t> d_a(M * N);
  thrust::device_vector<cutlass::tfloat32_t> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = cutlass::tfloat32_t;
  using ElementInputB = cutlass::tfloat32_t;
  using ElementOutput = float;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue>;

  constexpr int NumStages = 4;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages>;

  Gemm gemm_op;

  // CUTLASS GEMM uses (m, n, k): A[m, k] * B[k, n] = C[m, n]
  cutlass::gemm::GemmCoord problem_size(M, K, N);

  const int lda = N;
  const int ldb = N;  // ColumnMajor (k, n) => leading dim is k
  const int ldc = K;

  typename Gemm::Arguments arguments{
      problem_size,
      {thrust::raw_pointer_cast(d_a.data()), lda},
      {thrust::raw_pointer_cast(d_b.data()), ldb},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
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

static void BM_CUTLASS_FP16_TENSOROP_GEMM(benchmark::State &state) {
  cudaSetDevice(0);

  thrust::device_vector<cutlass::half_t> d_a(M * N);
  thrust::device_vector<cutlass::half_t> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = float;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue>;

  constexpr int NumStages = 4;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages,
      AlignmentA, AlignmentB>;

  Gemm gemm_op;

  // CUTLASS GEMM uses (m, n, k): A[m, k] * B[k, n] = C[m, n]
  cutlass::gemm::GemmCoord problem_size(M, K, N);

  const int lda = N;
  const int ldb = N;  // ColumnMajor (k, n) => leading dim is k
  const int ldc = K;

  typename Gemm::Arguments arguments{
      problem_size,
      {thrust::raw_pointer_cast(d_a.data()), lda},
      {thrust::raw_pointer_cast(d_b.data()), ldb},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
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

static void BM_CUTLASS_BF16_TENSOROP_GEMM(benchmark::State &state) {
  cudaSetDevice(0);

  thrust::device_vector<cutlass::bfloat16_t> d_a(M * N);
  thrust::device_vector<cutlass::bfloat16_t> d_b(N * K);
  thrust::device_vector<float> d_c(M * K);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::bfloat16_t;
  using ElementOutput = float;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue>;

  constexpr int NumStages = 4;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages,
      AlignmentA, AlignmentB>;

  Gemm gemm_op;

  // CUTLASS GEMM uses (m, n, k): A[m, k] * B[k, n] = C[m, n]
  cutlass::gemm::GemmCoord problem_size(M, K, N);

  const int lda = N;
  const int ldb = N;  // ColumnMajor (k, n) => leading dim is k
  const int ldc = K;

  typename Gemm::Arguments arguments{
      problem_size,
      {thrust::raw_pointer_cast(d_a.data()), lda},
      {thrust::raw_pointer_cast(d_b.data()), ldb},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
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

BENCHMARK(BM_SGEMMSharedMemBlockingTileRawVec<32, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileRawVec<32, 8>);

BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanVec<32, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanVec<32, 8>);

BENCHMARK(BM_SGEMMSharedMemBlockingTileRawScalar<32, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileRawScalar<32, 8>);

BENCHMARK(BM_SGEMMSharedMemBlockingTileRawVecNoRestrict<32, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileRawVecNoRestrict<32, 8>);

BENCHMARK(BM_SGEMMSharedMemBlockingTileStrictVecSharedFloat4<32, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileStrictVecSharedFloat4<32, 8>);

BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanRegBlock<32, 4, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanRegBlock<32, 8, 4>);

BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsync<32, 4, 4>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsync<32, 8, 4>);

BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<32, 4, 4, 8>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<32, 8, 4, 8>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<32, 4, 4, 16>);
BENCHMARK(BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<32, 8, 4, 16>);

#ifdef HPC_USE_CUTLASS
BENCHMARK(BM_CUTLASS_SGEMM);
BENCHMARK(BM_CUTLASS_TF32_TENSOROP_SGEMM);
BENCHMARK(BM_CUTLASS_FP16_TENSOROP_GEMM);
BENCHMARK(BM_CUTLASS_BF16_TENSOROP_GEMM);
#endif

BENCHMARK_MAIN();

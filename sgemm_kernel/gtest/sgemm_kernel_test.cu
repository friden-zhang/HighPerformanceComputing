#include <gtest/gtest.h>

#include "kernel/native.cuh"
#include "kernel/shared_mem_blocking.cuh"
#include "kernel/shared_mem_blocking_tile.cuh"
#include <kernel/shared_mem_blocking_tile_mdspan_vec.cuh>
#include <kernel/shared_mem_blocking_tile_mdspan_reg_block.cuh>
#include <kernel/shared_mem_blocking_tile_mdspan_reg_block_cp_async.cuh>
#include <kernel/shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice.cuh>
#include <kernel/shared_mem_blocking_tile_raw_vec.cuh>
#include <kernel/shared_mem_blocking_tile_raw_scalar.cuh>
#include <kernel/shared_mem_blocking_tile_raw_vec_no_restrict.cuh>
#include <kernel/shared_mem_blocking_tile_strict_vec_shared_float4.cuh>
#include <kernel/shared_mem_blocking_tile_vec.cuh>
#include <kernel/shared_mem_blocking_tile_strict_vec.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifdef HPC_USE_CUTLASS
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/tfloat32.h>
#endif

static void cpu_sgemm(float *A, float *B, float *C, float alpha, float beta,
                      int M, int N, int K) {
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < K; col++) {
      float sum = 0.0f;
      for (int index = 0; index < N; index++) {
        sum += A[row * N + index] * B[index * K + col];
      }
      C[row * K + col] = alpha * sum + beta * C[row * K + col];
    }
  }
}

#ifdef HPC_USE_CUTLASS
template <typename ElementA, typename ElementB>
static void cpu_sgemm_col_major_b(const ElementA *A, const ElementB *B,
                                  float *C, float alpha, float beta, int M,
                                  int N, int K) {
  // A: RowMajor [M, N]
  // B: ColumnMajor [N, K] (leading dim N)
  // C: RowMajor [M, K]
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < K; col++) {
      float sum = 0.0f;
      for (int index = 0; index < N; index++) {
        sum += static_cast<float>(A[row * N + index]) *
               static_cast<float>(B[index + col * N]);
      }
      C[row * K + col] = alpha * sum + beta * C[row * K + col];
    }
  }
}
#endif

class SGEMMKernelTest : public ::testing::Test {
public:
  void SetUp() override {
    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count == 0) {
      if (device_err == cudaSuccess) {
        GTEST_SKIP() << "No CUDA device available";
      }
      GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(device_err);
    }
    cudaSetDevice(0);

    A = thrust::host_vector<float>(M * N);
    B = thrust::host_vector<float>(N * K);
    // init A
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        A[i * N + j] = rand() % 5 / 5.0f;
      }
    }
    // init B
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        B[i * K + j] = rand() % 5 / 5.0f;
      }
    }
    cpu_c = thrust::host_vector<float>(M * K);
    cpu_sgemm(A.data(), B.data(), cpu_c.data(), 1.0f, 0.0f, M, N, K);
    d_a = A;
    d_b = B;
  }

  int M = 128;
  int N = 512;
  int K = 128;

  thrust::host_vector<float> A;
  thrust::host_vector<float> B;
  thrust::host_vector<float> cpu_c;
  thrust::device_vector<float> d_a;
  thrust::device_vector<float> d_b;
};

TEST_F(SGEMMKernelTest, NativeKernel) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err = launch_sgemm_native<16, 16>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingKernel) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err = launch_sgemm_shared_mem_blocking<32>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernel) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err = launch_sgemm_shared_mem_blocking_tile<32, 4>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelVec) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err = launch_sgemm_shared_mem_blocking_tile_vec<32, 4>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelStrictVec) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err = launch_sgemm_shared_mem_blocking_tile_strict_vec<32, 4>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelRawVec) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err = launch_sgemm_shared_mem_blocking_tile_raw_vec<32, 4>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelRawVecBeta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err = launch_sgemm_shared_mem_blocking_tile_raw_vec<32, 4>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelMdspanVec) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err = launch_sgemm_shared_mem_blocking_tile_mdspan_vec<32, 4>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelMdspanVecBeta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err = launch_sgemm_shared_mem_blocking_tile_mdspan_vec<32, 4>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelRawScalar) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err = launch_sgemm_shared_mem_blocking_tile_raw_scalar<32, 4>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelRawScalarBeta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err = launch_sgemm_shared_mem_blocking_tile_raw_scalar<32, 4>(
      thrust::raw_pointer_cast(d_a.data()),
      thrust::raw_pointer_cast(d_b.data()),
      thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelRawVecNoRestrict) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_raw_vec_no_restrict<32, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelRawVecNoRestrictBeta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_raw_vec_no_restrict<32, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelStrictVecSharedFloat4) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_strict_vec_shared_float4<32, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelStrictVecSharedFloat4Beta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_strict_vec_shared_float4<32, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelMdspanRegBlock4x4) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block<32, 4, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelMdspanRegBlock4x4Beta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block<32, 4, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelMdspanRegBlock8x4) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block<32, 8, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelMdspanRegBlock8x4Beta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block<32, 8, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelMdspanRegBlockCpAsync4x4) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async<32, 4, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsync4x4Beta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async<32, 4, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest, SharedMemBlockingTileKernelMdspanRegBlockCpAsync8x4) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async<32, 8, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsync8x4Beta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async<32, 8, 4>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N, K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsyncKSlice4x4K8) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
          32, 4, 4, 8>(thrust::raw_pointer_cast(d_a.data()),
                       thrust::raw_pointer_cast(d_b.data()),
                       thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N,
                       K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsyncKSlice4x4K8Beta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
          32, 4, 4, 8>(thrust::raw_pointer_cast(d_a.data()),
                       thrust::raw_pointer_cast(d_b.data()),
                       thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N,
                       K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsyncKSlice8x4K8) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
          32, 8, 4, 8>(thrust::raw_pointer_cast(d_a.data()),
                       thrust::raw_pointer_cast(d_b.data()),
                       thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N,
                       K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsyncKSlice8x4K8Beta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
          32, 8, 4, 8>(thrust::raw_pointer_cast(d_a.data()),
                       thrust::raw_pointer_cast(d_b.data()),
                       thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N,
                       K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsyncKSlice4x4K16) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
          32, 4, 4, 16>(thrust::raw_pointer_cast(d_a.data()),
                        thrust::raw_pointer_cast(d_b.data()),
                        thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N,
                        K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsyncKSlice4x4K16Beta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
          32, 4, 4, 16>(thrust::raw_pointer_cast(d_a.data()),
                        thrust::raw_pointer_cast(d_b.data()),
                        thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N,
                        K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsyncKSlice8x4K16) {
  thrust::device_vector<float> d_c(M * K);
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
          32, 8, 4, 16>(thrust::raw_pointer_cast(d_a.data()),
                        thrust::raw_pointer_cast(d_b.data()),
                        thrust::raw_pointer_cast(d_c.data()), 1.0f, 0.0f, M, N,
                        K, nullptr);
  ASSERT_EQ(err, cudaSuccess);
  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_c[i], 1e-2);
  }
}

TEST_F(SGEMMKernelTest,
       SharedMemBlockingTileKernelMdspanRegBlockCpAsyncKSlice8x4K16Beta) {
  const float alpha = 0.7f;
  const float beta = 0.3f;

  thrust::host_vector<float> init_c(M * K);
  for (int i = 0; i < M * K; ++i) {
    init_c[i] = rand() % 5 / 5.0f;
  }

  thrust::host_vector<float> cpu_ref = init_c;
  cpu_sgemm(A.data(), B.data(), cpu_ref.data(), alpha, beta, M, N, K);

  thrust::device_vector<float> d_c = init_c;
  cudaError_t err =
      launch_sgemm_shared_mem_blocking_tile_mdspan_reg_block_cp_async_k_slice<
          32, 8, 4, 16>(thrust::raw_pointer_cast(d_a.data()),
                        thrust::raw_pointer_cast(d_b.data()),
                        thrust::raw_pointer_cast(d_c.data()), alpha, beta, M, N,
                        K, nullptr);
  ASSERT_EQ(err, cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; i++) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-2);
  }
}

#ifdef HPC_USE_CUTLASS
TEST_F(SGEMMKernelTest, CutlassTf32TensorOp) {
  thrust::host_vector<float> b_col(N * K);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      b_col[n + k * N] = B[n * K + k];
    }
  }

  thrust::host_vector<cutlass::tfloat32_t> a_tf(M * N);
  thrust::host_vector<cutlass::tfloat32_t> b_tf(N * K);
  for (int i = 0; i < M * N; ++i) {
    a_tf[i] = cutlass::tfloat32_t(A[i]);
  }
  for (int i = 0; i < N * K; ++i) {
    b_tf[i] = cutlass::tfloat32_t(b_col[i]);
  }

  thrust::host_vector<float> cpu_ref(M * K, 0.0f);
  cpu_sgemm_col_major_b(a_tf.data(), b_tf.data(), cpu_ref.data(), 1.0f, 0.0f, M,
                        N, K);

  thrust::device_vector<cutlass::tfloat32_t> d_a = a_tf;
  thrust::device_vector<cutlass::tfloat32_t> d_b = b_tf;
  thrust::device_vector<float> d_c(M * K, 0.0f);

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

  cutlass::gemm::GemmCoord problem_size(M, K, N);
  const int lda = N;
  const int ldb = N;
  const int ldc = K;

  typename Gemm::Arguments arguments{
      problem_size,
      {thrust::raw_pointer_cast(d_a.data()), lda},
      {thrust::raw_pointer_cast(d_b.data()), ldb},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {1.0f, 0.0f}};

  ASSERT_EQ(gemm_op.can_implement(arguments), cutlass::Status::kSuccess);

  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  thrust::device_vector<uint8_t> workspace(workspace_size);
  uint8_t *workspace_ptr =
      workspace_size > 0 ? thrust::raw_pointer_cast(workspace.data()) : nullptr;

  ASSERT_EQ(gemm_op.initialize(arguments, workspace_ptr),
            cutlass::Status::kSuccess);
  ASSERT_EQ(gemm_op(), cutlass::Status::kSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; ++i) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-1);
  }
}

TEST_F(SGEMMKernelTest, CutlassFp16TensorOp) {
  thrust::host_vector<float> b_col(N * K);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      b_col[n + k * N] = B[n * K + k];
    }
  }

  thrust::host_vector<cutlass::half_t> a_h(M * N);
  thrust::host_vector<cutlass::half_t> b_h(N * K);
  for (int i = 0; i < M * N; ++i) {
    a_h[i] = cutlass::half_t(A[i]);
  }
  for (int i = 0; i < N * K; ++i) {
    b_h[i] = cutlass::half_t(b_col[i]);
  }

  thrust::host_vector<float> cpu_ref(M * K, 0.0f);
  cpu_sgemm_col_major_b(a_h.data(), b_h.data(), cpu_ref.data(), 1.0f, 0.0f, M,
                        N, K);

  thrust::device_vector<cutlass::half_t> d_a = a_h;
  thrust::device_vector<cutlass::half_t> d_b = b_h;
  thrust::device_vector<float> d_c(M * K, 0.0f);

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

  cutlass::gemm::GemmCoord problem_size(M, K, N);
  const int lda = N;
  const int ldb = N;
  const int ldc = K;

  typename Gemm::Arguments arguments{
      problem_size,
      {thrust::raw_pointer_cast(d_a.data()), lda},
      {thrust::raw_pointer_cast(d_b.data()), ldb},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {1.0f, 0.0f}};

  ASSERT_EQ(gemm_op.can_implement(arguments), cutlass::Status::kSuccess);

  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  thrust::device_vector<uint8_t> workspace(workspace_size);
  uint8_t *workspace_ptr =
      workspace_size > 0 ? thrust::raw_pointer_cast(workspace.data()) : nullptr;

  ASSERT_EQ(gemm_op.initialize(arguments, workspace_ptr),
            cutlass::Status::kSuccess);
  ASSERT_EQ(gemm_op(), cutlass::Status::kSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; ++i) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-1);
  }
}

TEST_F(SGEMMKernelTest, CutlassBf16TensorOp) {
  thrust::host_vector<float> b_col(N * K);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      b_col[n + k * N] = B[n * K + k];
    }
  }

  thrust::host_vector<cutlass::bfloat16_t> a_bf(M * N);
  thrust::host_vector<cutlass::bfloat16_t> b_bf(N * K);
  for (int i = 0; i < M * N; ++i) {
    a_bf[i] = cutlass::bfloat16_t(A[i]);
  }
  for (int i = 0; i < N * K; ++i) {
    b_bf[i] = cutlass::bfloat16_t(b_col[i]);
  }

  thrust::host_vector<float> cpu_ref(M * K, 0.0f);
  cpu_sgemm_col_major_b(a_bf.data(), b_bf.data(), cpu_ref.data(), 1.0f, 0.0f, M,
                        N, K);

  thrust::device_vector<cutlass::bfloat16_t> d_a = a_bf;
  thrust::device_vector<cutlass::bfloat16_t> d_b = b_bf;
  thrust::device_vector<float> d_c(M * K, 0.0f);

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

  cutlass::gemm::GemmCoord problem_size(M, K, N);
  const int lda = N;
  const int ldb = N;
  const int ldc = K;

  typename Gemm::Arguments arguments{
      problem_size,
      {thrust::raw_pointer_cast(d_a.data()), lda},
      {thrust::raw_pointer_cast(d_b.data()), ldb},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {thrust::raw_pointer_cast(d_c.data()), ldc},
      {1.0f, 0.0f}};

  ASSERT_EQ(gemm_op.can_implement(arguments), cutlass::Status::kSuccess);

  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  thrust::device_vector<uint8_t> workspace(workspace_size);
  uint8_t *workspace_ptr =
      workspace_size > 0 ? thrust::raw_pointer_cast(workspace.data()) : nullptr;

  ASSERT_EQ(gemm_op.initialize(arguments, workspace_ptr),
            cutlass::Status::kSuccess);
  ASSERT_EQ(gemm_op(), cutlass::Status::kSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<float> h_c(M * K);
  thrust::copy(d_c.begin(), d_c.end(), h_c.begin());
  for (int i = 0; i < M * K; ++i) {
    ASSERT_NEAR(h_c[i], cpu_ref[i], 1e-1);
  }
}
#endif

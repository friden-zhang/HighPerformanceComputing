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

class SGEMMKernelTest : public ::testing::Test {
public:
  void SetUp() override {
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

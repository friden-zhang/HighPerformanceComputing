#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "kernel/native.cuh"

int N = 4;
int C = 3;
int D = 16;
int H = 256;
int W = 256;
int K = 8;
int KD = 3;
int KH = 3;
int KW = 3;

template <int BLOCK_X, int BLOCK_Y, int BLOCK_Z>
static void BM_Conv3DNative(benchmark::State &state) {
  cudaSetDevice(0);

  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  thrust::device_vector<float> d_input(N * C * D * H * W);
  thrust::device_vector<float> d_kernel(K * C * KD * KH * KW);
  thrust::device_vector<float> d_output(N * K * OD * OH * OW);

  for (auto _ : state) {
    auto ret = launch_conv3d_native<BLOCK_X, BLOCK_Y, BLOCK_Z>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_kernel.data()),
        thrust::raw_pointer_cast(d_output.data()), N, C, D, H, W, K, KD, KH,
        KW, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

BENCHMARK(BM_Conv3DNative<4, 4, 4>);
BENCHMARK(BM_Conv3DNative<8, 4, 4>);
BENCHMARK(BM_Conv3DNative<8, 8, 4>);

BENCHMARK_MAIN();

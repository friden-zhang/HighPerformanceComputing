#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <string>

#include "kernel/mdspan_linear_nk_parallel.cuh"
#include "kernel/mdspan_spatial_nk_serial.cuh"

struct Conv3DShape {
  int N;
  int C;
  int D;
  int H;
  int W;
  int K;
  int KD;
  int KH;
  int KW;
};

static std::string ShapePrefix(const Conv3DShape &shape) {
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "Shape/N%d_C%d_D%d_H%d_W%d_K%d_KD%d_KH%d_KW%d",
                shape.N, shape.C, shape.D, shape.H, shape.W, shape.K, shape.KD,
                shape.KH, shape.KW);
  return std::string(buffer);
}

template <int BLOCK_X, int BLOCK_Y, int BLOCK_Z>
static void BM_Conv3DMdspanSpatialNKSerial(benchmark::State &state,
                                           const Conv3DShape &shape) {
  cudaSetDevice(0);

  const int OD = shape.D - shape.KD + 1;
  const int OH = shape.H - shape.KH + 1;
  const int OW = shape.W - shape.KW + 1;

  if (OD <= 0 || OH <= 0 || OW <= 0) {
    state.SkipWithError("invalid output size");
    return;
  }

  thrust::device_vector<float> d_input(static_cast<size_t>(shape.N) * shape.C *
                                       shape.D * shape.H * shape.W);
  thrust::device_vector<float> d_kernel(static_cast<size_t>(shape.K) * shape.C *
                                        shape.KD * shape.KH * shape.KW);
  thrust::device_vector<float> d_output(static_cast<size_t>(shape.N) * shape.K *
                                        OD * OH * OW);

  for (auto _ : state) {
    auto ret = launch_conv3d_mdspan_spatial_nk_serial<BLOCK_X, BLOCK_Y, BLOCK_Z>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_kernel.data()),
        thrust::raw_pointer_cast(d_output.data()), shape.N, shape.C, shape.D,
        shape.H, shape.W, shape.K, shape.KD, shape.KH, shape.KW, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

template <int BLOCK_SIZE>
static void BM_Conv3DMdspanLinearNKParallel(benchmark::State &state,
                                            const Conv3DShape &shape) {
  cudaSetDevice(0);

  const int OD = shape.D - shape.KD + 1;
  const int OH = shape.H - shape.KH + 1;
  const int OW = shape.W - shape.KW + 1;

  if (OD <= 0 || OH <= 0 || OW <= 0) {
    state.SkipWithError("invalid output size");
    return;
  }

  thrust::device_vector<float> d_input(static_cast<size_t>(shape.N) * shape.C *
                                       shape.D * shape.H * shape.W);
  thrust::device_vector<float> d_kernel(static_cast<size_t>(shape.K) * shape.C *
                                        shape.KD * shape.KH * shape.KW);
  thrust::device_vector<float> d_output(static_cast<size_t>(shape.N) * shape.K *
                                        OD * OH * OW);

  for (auto _ : state) {
    auto ret = launch_conv3d_mdspan_linear_nk_parallel<BLOCK_SIZE>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_kernel.data()),
        thrust::raw_pointer_cast(d_output.data()), shape.N, shape.C, shape.D,
        shape.H, shape.W, shape.K, shape.KD, shape.KH, shape.KW, nullptr);
    if (ret != cudaSuccess) {
      state.SkipWithError(cudaGetErrorString(ret));
      return;
    }
    benchmark::DoNotOptimize(ret);
  }
}

static void RegisterConv3DBenchmarks() {
  const Conv3DShape shapes[] = {
      {1, 3, 16, 64, 64, 8, 3, 3, 3},
      {1, 8, 32, 128, 128, 16, 3, 3, 3},
      {4, 3, 16, 256, 256, 8, 3, 3, 3},
  };

  for (const auto &shape : shapes) {
    const std::string prefix = ShapePrefix(shape);

    benchmark::RegisterBenchmark(
        (prefix + "/MdspanSpatialNKSerial_B4x4x4").c_str(),
        [=](benchmark::State &state) {
          BM_Conv3DMdspanSpatialNKSerial<4, 4, 4>(state, shape);
        });
    benchmark::RegisterBenchmark(
        (prefix + "/MdspanSpatialNKSerial_B8x4x4").c_str(),
        [=](benchmark::State &state) {
          BM_Conv3DMdspanSpatialNKSerial<8, 4, 4>(state, shape);
        });
    benchmark::RegisterBenchmark(
        (prefix + "/MdspanSpatialNKSerial_B8x8x4").c_str(),
        [=](benchmark::State &state) {
          BM_Conv3DMdspanSpatialNKSerial<8, 8, 4>(state, shape);
        });

    benchmark::RegisterBenchmark(
        (prefix + "/MdspanLinearNKParallel_B128").c_str(),
        [=](benchmark::State &state) {
          BM_Conv3DMdspanLinearNKParallel<128>(state, shape);
        });
    benchmark::RegisterBenchmark(
        (prefix + "/MdspanLinearNKParallel_B256").c_str(),
        [=](benchmark::State &state) {
          BM_Conv3DMdspanLinearNKParallel<256>(state, shape);
        });
    benchmark::RegisterBenchmark(
        (prefix + "/MdspanLinearNKParallel_B512").c_str(),
        [=](benchmark::State &state) {
          BM_Conv3DMdspanLinearNKParallel<512>(state, shape);
        });
  }
}

namespace {
struct Conv3DBenchmarkRegister {
  Conv3DBenchmarkRegister() { RegisterConv3DBenchmarks(); }
};
static Conv3DBenchmarkRegister conv3d_benchmark_register;
} // namespace

BENCHMARK_MAIN();

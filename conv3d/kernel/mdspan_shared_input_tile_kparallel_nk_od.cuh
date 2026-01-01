#pragma once

#include <cuda/std/mdspan>
#include <cuda_runtime.h>

// Conv3D: mdspan indexing, block fixed (n, k-block, od), shared input tile for
// (oh, ow). K dimension is parallelized with threadIdx.z.
template <int BLOCK_X, int BLOCK_Y, int K_TILE>
__global__ void conv3d_mdspan_shared_input_tile_kparallel_nk_od(
    const float *__restrict__ input, const float *__restrict__ kernel,
    float *__restrict__ output, int N, int C, int D, int H, int W, int K,
    int KD, int KH, int KW) {

  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  const int tile_ow = blockIdx.x;
  const int tile_oh = blockIdx.y;

  const int K_BLOCKS = (K + K_TILE - 1) / K_TILE;
  int z = blockIdx.z;
  const int k_block = z % K_BLOCKS;
  z /= K_BLOCKS;
  const int od = z % OD;
  const int n = z / OD;

  const int tile_base_ow = tile_ow * blockDim.x;
  const int tile_base_oh = tile_oh * blockDim.y;

  const int sh_w = blockDim.x + KW - 1;
  const int sh_h = blockDim.y + KH - 1;
  const int sh_plane = sh_w * sh_h;
  const int total = C * KD * sh_plane;

  extern __shared__ float sh_input[];

  const int tid =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int num_threads = blockDim.x * blockDim.y * blockDim.z;

  constexpr auto dyn = cuda::std::dynamic_extent;
  using ext_t = cuda::std::extents<int, dyn, dyn, dyn, dyn, dyn>;
  using mdspan_input_t = cuda::std::mdspan<const float, ext_t>;
  using mdspan_kernel_t = cuda::std::mdspan<const float, ext_t>;
  using mdspan_output_t = cuda::std::mdspan<float, ext_t>;

  ext_t ext_input(N, C, D, H, W);
  ext_t ext_kernel(K, C, KD, KH, KW);
  ext_t ext_output(N, K, OD, OH, OW);

  mdspan_input_t mdspan_input{input, ext_input};
  mdspan_kernel_t mdspan_kernel{kernel, ext_kernel};
  mdspan_output_t mdspan_output{output, ext_output};

  for (int idx = tid; idx < total; idx += num_threads) {
    int tmp = idx;
    const int sh_idx = tmp % sh_plane;
    tmp /= sh_plane;
    const int kd = tmp % KD;
    const int c = tmp / KD;

    const int sh_y = sh_idx / sh_w;
    const int sh_x = sh_idx % sh_w;

    const int in_h = tile_base_oh + sh_y;
    const int in_w = tile_base_ow + sh_x;
    const int in_d = od + kd;

    float val = 0.0f;
    if (in_h < H && in_w < W) {
      val = mdspan_input(n, c, in_d, in_h, in_w);
    }
    sh_input[idx] = val;
  }

  __syncthreads();

  const int oh = tile_base_oh + threadIdx.y;
  const int ow = tile_base_ow + threadIdx.x;
  if (oh >= OH || ow >= OW)
    return;

  const int k = k_block * K_TILE + threadIdx.z;
  if (k >= K)
    return;

  float sum = 0.0f;
  for (int c = 0; c < C; ++c) {
    for (int kd = 0; kd < KD; ++kd) {
      const int base = (c * KD + kd) * sh_plane;
      for (int kh = 0; kh < KH; ++kh) {
        const int sh_y = threadIdx.y + kh;
        const int row = base + sh_y * sh_w;
        for (int kw = 0; kw < KW; ++kw) {
          const int sh_x = threadIdx.x + kw;
          sum += sh_input[row + sh_x] * mdspan_kernel(k, c, kd, kh, kw);
        }
      }
    }
  }

  mdspan_output(n, k, od, oh, ow) = sum;
}

template <int BLOCK_X, int BLOCK_Y, int K_TILE>
cudaError_t launch_conv3d_mdspan_shared_input_tile_kparallel_nk_od(
    const float *__restrict__ input, const float *__restrict__ kernel,
    float *__restrict__ output, int N, int C, int D, int H, int W, int K,
    int KD, int KH, int KW, cudaStream_t stream = nullptr) {
  const int OD = D - KD + 1;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;

  if (OD <= 0 || OH <= 0 || OW <= 0 || K_TILE <= 0) {
    return cudaErrorInvalidValue;
  }

  const int sh_w = BLOCK_X + KW - 1;
  const int sh_h = BLOCK_Y + KH - 1;
  const size_t shared_bytes =
      static_cast<size_t>(C) * KD * sh_w * sh_h * sizeof(float);

  dim3 block(BLOCK_X, BLOCK_Y, K_TILE);
  const int k_blocks = (K + K_TILE - 1) / K_TILE;
  const size_t grid_z = static_cast<size_t>(N) * OD * k_blocks;
  dim3 grid((OW + BLOCK_X - 1) / BLOCK_X, (OH + BLOCK_Y - 1) / BLOCK_Y,
            static_cast<unsigned int>(grid_z));

  if (stream) {
    conv3d_mdspan_shared_input_tile_kparallel_nk_od<BLOCK_X, BLOCK_Y, K_TILE>
        <<<grid, block, shared_bytes, stream>>>(input, kernel, output, N, C, D,
                                                H, W, K, KD, KH, KW);
  } else {
    conv3d_mdspan_shared_input_tile_kparallel_nk_od<BLOCK_X, BLOCK_Y, K_TILE>
        <<<grid, block, shared_bytes>>>(input, kernel, output, N, C, D, H, W, K,
                                        KD, KH, KW);
    cudaDeviceSynchronize();
  }

  return cudaGetLastError();
}

# Conv3D Kernels

This directory contains several Conv3D kernel variants used to compare
different parallelization and memory strategies on CUDA.

## Kernel Variants
- `conv3d/kernel/mdspan_spatial_nk_serial.cuh`:
  thread maps to `(od, oh, ow)`, loops serially over `n` and `k`.
- `conv3d/kernel/mdspan_linear_nk_parallel.cuh`:
  linearizes `(n, k, od, oh, ow)` so each thread computes one output.
- `conv3d/kernel/mdspan_shared_input_tile_nk_od.cuh`:
  block fixed `(n, k, od)` with shared input tile for `(oh, ow)`.
- `conv3d/kernel/mdspan_shared_input_tile_kblock_nk_od.cuh`:
  block fixed `(n, k_block, od)`, shared input tile, each thread computes
  `K_TILE` outputs serially.
- `conv3d/kernel/mdspan_shared_input_tile_kparallel_nk_od.cuh`:
  same as above but `k` is parallelized via `threadIdx.z`.
- `conv3d/kernel/mdspan_shared_input_kernel_kblock_nk_od.cuh`:
  shared input tile plus shared kernel weights for `K_TILE`.
- `conv3d/kernel/mdspan_const_kernel_linear_nk_parallel.cuh`:
  linear NK parallelism with kernel weights in `__constant__` memory when
  `K*C*KD*KH*KW <= 16384` floats (64KB), otherwise falls back to linear.

## Benchmark Setup
- GPU: NVIDIA GeForce RTX 3050 (SM 8.6)
- Driver/CUDA: 580.95.05 / CUDA 13.0
- Build: Debug (timings are noisy; prefer Release for real numbers)
- Command: `./build/conv3d/conv3d_benchmark --benchmark_sort=none`

Shapes used:
```
N=1 C=3 D=16 H=64  W=64  K=8  KD=3 KH=3 KW=3
N=1 C=8 D=32 H=128 W=128 K=16 KD=3 KH=3 KW=3
N=4 C=3 D=16 H=256 W=256 K=8  KD=3 KH=3 KW=3
```

## Results Summary (best observed)
- Shape (1,3,16,64,64,8): const-kernel linear (`MdspanConstKernelLinearNKParallel_B128`)
  at ~123 us, about 6-7% faster than linear NK parallel.
- Shape (1,8,32,128,128,16): const-kernel linear (`MdspanConstKernelLinearNKParallel_B128`)
  at ~5.59 ms, about 7-8% faster than linear NK parallel.
- Shape (4,3,16,256,256,8): shared input + shared kernel
  (`MdspanSharedInputKernelKBlockNKOD_B16x16_K4`) at ~8.75 ms,
  about 10% faster than linear NK parallel.

General observations:
- Shared input tiles alone are typically slower for small shapes due to
  load and sync overhead.
- K-parallel (`threadIdx.z`) helped some larger shapes but did not beat the
  const-kernel linear variant.
- Sharing kernel weights helps when the output grid is large enough to amortize
  shared memory overhead.

## Running Tests
```
ctest --test-dir build -R conv3d_test
```

# SGEMM Kernels (CUDA)

This directory implements and compares several CUDA SGEMM kernels
(`C[M,K] = alpha * A[M,N] * B[N,K] + beta * C[M,K]`), evolving from a naive
kernel up to:

- `float4` vectorized global load/store, `__restrict__`, alignment assumptions
- register blocking (each thread computes a 4x4 / 8x4 sub-tile)
- SM80+ `cp.async` experiments (full-tile double buffering and K-slice pipeline)
- CUTLASS baselines (SIMT / TF32 Tensor Core / FP16 Tensor Core / BF16 Tensor Core)

Performance numbers below are from the default benchmark sizes in
`sgemm_kernel_benchmark` (`M=1024, N=2048, K=3072`).

---

## Layout

- `kernel/`: kernel implementations and `launch_*` wrappers
- `gtest/`: correctness tests (vs CPU reference)
- `benchmark/`: Google Benchmark microbenchmarks

---

## Build & Run

### Requirements

- CUDA Toolkit
- CMake >= 3.22
- `gtest` and `benchmark` (this repo uses `find_package(GTest REQUIRED)` /
  `find_package(benchmark REQUIRED)`)

### Build (recommended: Release)

From the repo root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tip: the repo sets `CMAKE_CUDA_ARCHITECTURES=86 89` in the root
`CMakeLists.txt`. If your GPU is different, edit it or override with
`-DCMAKE_CUDA_ARCHITECTURES=<sm>`.

Enable CUTLASS (auto-fetched via FetchContent if not found locally):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_CUTLASS=ON
cmake --build build -j
```

CUTLASS default fetch location (CMake FetchContent): `build/_deps/hpc_cutlass-src`.

Optional flags:

- `-DCUTLASS_DIR=/path/to/cutlass`: point to a local CUTLASS checkout (expects
  `include/cutlass/...`)
- `-DHPC_FETCH_CUTLASS=ON/OFF`: whether to auto-fetch when not found
- `-DHPC_CUTLASS_GIT_TAG=v3.5.0`: switch CUTLASS version/tag

### Run tests

```bash
./build/sgemm_kernel/sgemm_kernel_test
```

### Run benchmarks

```bash
./build/sgemm_kernel/sgemm_kernel_benchmark
```

Run only CUTLASS Tensor Core entries:

```bash
./build/sgemm_kernel/sgemm_kernel_benchmark --benchmark_filter='BM_CUTLASS_.*TENSOROP.*'
```

---

## Data Layout & Conventions

### Hand-written kernels (most kernels in this directory)

- `A`: RowMajor `[M, N]`
- `B`: RowMajor `[N, K]`
- `C`: RowMajor `[M, K]`

### CUTLASS TensorOp baselines (TF32/FP16/BF16 Tensor Core)

To use CUTLASS canonical tensor-op configurations, the TensorOp benchmarks use:

- `A`: RowMajor `[M, N]`
- `B`: **ColumnMajor** `[N, K]` (i.e. `B(n,k)` lives at `B[n + k * N]`, `ldb = N`)
- `C/D`: RowMajor `[M, K]`

If you want apples-to-apples numeric comparisons, `B` must be filled/repacked in
ColumnMajor form (the tests already do this).

---

## Kernel Variants Overview

Names below match the `benchmark` output:

- **Important constraints (to keep the fast path simple):** many vectorized /
  reg-blocked kernels use `assert()` in `launch_*` wrappers. Common assumptions:
  - `M/N/K` are multiples of `BlockSize (=32)`
  - `float4` paths typically require `N % 4 == 0`, `K % 4 == 0`, and 16B-aligned pointers
  - reg-blocking / cp.async variants currently fix `ColTile == 4`
    (see `kernel/shared_mem_blocking_tile_mdspan_reg_block*.cuh`)
  - cp.async K-slice pipeline requires `N % KSlice == 0` and `KSlice % 4 == 0`

- `BM_SGEMMKernel<...>`: naive kernel (heavy global traffic, slowest)
- `BM_SGEMMSharedMemBlocking<...>`: shared-memory blocking
- `BM_SGEMMSharedMemBlockingTile<...>`: finer-grained tiling
- `BM_SGEMMSharedMemBlockingTileVec<...>`: vectorized version (fixes the old bug
  that wrote back `C` inside the N-tile loop; also has a `beta==0` fast path)
- `BM_SGEMMSharedMemBlockingTileStrictVec<...>`: stricter vectorization
- `BM_SGEMMSharedMemBlockingTileRawVec<...>`: raw pointers + `__restrict__` + `float4` global load/store
- `BM_SGEMMSharedMemBlockingTileMdspanVec<...>`: uses `mdspan` for indexing but keeps `float4` vectorization (similar perf to raw pointers)
- `BM_SGEMMSharedMemBlockingTileRawScalar<...>`: raw + restrict, but scalar load/store (to isolate vectorization gains)
- `BM_SGEMMSharedMemBlockingTileRawVecNoRestrict<...>`: keeps `float4`, drops `__restrict__` (to isolate restrict gains)
- `BM_SGEMMSharedMemBlockingTileStrictVecSharedFloat4<...>`: strict shared layout + `float4` shared writes (needs alignment handling)
- `BM_SGEMMSharedMemBlockingTileMdspanRegBlock<32, ROW_TILE, 4>`: register blocking (each thread computes a `ROW_TILE x 4` sub-tile)
- `BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsync<...>`: SM80+ `cp.async` full-tile double buffer
- `BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<..., KSLICE>`: SM80+ `cp.async` K-slice pipeline (finer overlap)
- `BM_CUTLASS_SGEMM`: CUTLASS default `device::Gemm<float,...>` (typically SIMT)
- `BM_CUTLASS_TF32_TENSOROP_SGEMM`: CUTLASS TF32 Tensor Core (Sm80 config, float output)
- `BM_CUTLASS_FP16_TENSOROP_GEMM`: CUTLASS FP16 Tensor Core (Sm80 config, float output)
- `BM_CUTLASS_BF16_TENSOROP_GEMM`: CUTLASS BF16 Tensor Core (Sm80 config, float output)

---

## Benchmark Results (M=1024, N=2048, K=3072)

> Note: this output is from one real run. Units are `ns`. Google Benchmark prints
> both `Time` and `CPU` columns; we keep the raw output for easy comparison.

### Key takeaways (selected entries)

```text
BM_SGEMMSharedMemBlockingTileMdspanRegBlock<32, 8, 4>                     3333633 ns
BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<32, 8, 4, 8>     3321059 ns
BM_CUTLASS_SGEMM                                                          1961581 ns
BM_CUTLASS_TF32_TENSOROP_SGEMM                                            1454433 ns
BM_CUTLASS_FP16_TENSOROP_GEMM                                              754870 ns
BM_CUTLASS_BF16_TENSOROP_GEMM                                              755096 ns
```

Throughput computed as `2*M*N*K` (here ~`12.885e9` FLOPs), roughly:

- `MdspanRegBlock<32,8,4>`：~3.87 TFLOPs
- `CUTLASS_SGEMM`：~6.57 TFLOPs
- `CUTLASS_TF32_TENSOROP_SGEMM`：~8.86 TFLOPs
- `CUTLASS_FP16/BF16_TENSOROP_GEMM`：~17.06 TFLOPs

The conclusion is straightforward: **Tensor Cores (TF32/FP16/BF16) deliver the
largest speedups**. Hand-written FP32 kernels improve significantly with
vectorization, register blocking, and `cp.async`, but the ceiling is still well
below the Tensor Core path.

### Full benchmark output (one sample run)

```text
-----------------------------------------------------------------------------------------------------------------
Benchmark                                                                       Time             CPU   Iterations
-----------------------------------------------------------------------------------------------------------------
BM_SGEMMKernel<16, 8>                                                    24929970 ns     24929840 ns           28
BM_SGEMMKernel<16, 16>                                                   23346974 ns     23346828 ns           30
BM_SGEMMKernel<32, 16>                                                   22930634 ns     22930524 ns           31
BM_SGEMMKernel<32, 32>                                                   28168084 ns     28167404 ns           25
BM_SGEMMSharedMemBlocking<16>                                            24088912 ns     24088784 ns           29
BM_SGEMMSharedMemBlocking<32>                                            26653076 ns     26652324 ns           26
BM_SGEMMSharedMemBlockingTile<16, 2>                                     19434748 ns     19434478 ns           36
BM_SGEMMSharedMemBlockingTile<16, 4>                                     18559196 ns     18559042 ns           38
BM_SGEMMSharedMemBlockingTile<16, 8>                                     21203505 ns     21203504 ns           33
BM_SGEMMSharedMemBlockingTile<32, 2>                                     17354934 ns     17354549 ns           40
BM_SGEMMSharedMemBlockingTile<32, 4>                                     15178251 ns     15178032 ns           46
BM_SGEMMSharedMemBlockingTile<32, 8>                                     14773321 ns     14773185 ns           47
BM_SGEMMSharedMemBlockingTileVec<16, 2>                                  19281381 ns     19281213 ns           36
BM_SGEMMSharedMemBlockingTileVec<16, 4>                                  15301167 ns     15300935 ns           46
BM_SGEMMSharedMemBlockingTileVec<16, 8>                                  16786599 ns     16786146 ns           42
BM_SGEMMSharedMemBlockingTileVec<32, 2>                                  18132738 ns     18132318 ns           39
BM_SGEMMSharedMemBlockingTileVec<32, 4>                                  14140945 ns     14140814 ns           50
BM_SGEMMSharedMemBlockingTileVec<32, 8>                                  14770608 ns     14770512 ns           47
BM_SGEMMSharedMemBlockingTileStrictVec<16, 4>                            15349568 ns     15349231 ns           46
BM_SGEMMSharedMemBlockingTileStrictVec<16, 8>                            15035947 ns     15035865 ns           47
BM_SGEMMSharedMemBlockingTileStrictVec<32, 4>                            14120872 ns     14120726 ns           50
BM_SGEMMSharedMemBlockingTileStrictVec<32, 8>                            13090432 ns     13090123 ns           53
BM_SGEMMSharedMemBlockingTileRawVec<32, 4>                                7430730 ns      7430544 ns           94
BM_SGEMMSharedMemBlockingTileRawVec<32, 8>                                8114888 ns      8114789 ns           86
BM_SGEMMSharedMemBlockingTileMdspanVec<32, 4>                             7430897 ns      7430667 ns           94
BM_SGEMMSharedMemBlockingTileMdspanVec<32, 8>                             8122059 ns      8121969 ns           86
BM_SGEMMSharedMemBlockingTileRawScalar<32, 4>                            16463818 ns     16463360 ns           43
BM_SGEMMSharedMemBlockingTileRawScalar<32, 8>                            19338132 ns     19337577 ns           36
BM_SGEMMSharedMemBlockingTileRawVecNoRestrict<32, 4>                      7430115 ns      7429882 ns           94
BM_SGEMMSharedMemBlockingTileRawVecNoRestrict<32, 8>                      8113922 ns      8113920 ns           86
BM_SGEMMSharedMemBlockingTileStrictVecSharedFloat4<32, 4>                 9172291 ns      9172095 ns           76
BM_SGEMMSharedMemBlockingTileStrictVecSharedFloat4<32, 8>                 7921945 ns      7921904 ns           88
BM_SGEMMSharedMemBlockingTileMdspanRegBlock<32, 4, 4>                     4183243 ns      4183151 ns          167
BM_SGEMMSharedMemBlockingTileMdspanRegBlock<32, 8, 4>                     3333633 ns      3333576 ns          210
BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsync<32, 4, 4>              3884070 ns      3884046 ns          181
BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsync<32, 8, 4>              3985040 ns      3985013 ns          176
BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<32, 4, 4, 8>     3845647 ns      3845509 ns          182
BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<32, 8, 4, 8>     3321059 ns      3321016 ns          202
BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<32, 4, 4, 16>    3915227 ns      3915132 ns          178
BM_SGEMMSharedMemBlockingTileMdspanRegBlockCpAsyncKSlice<32, 8, 4, 16>    3890250 ns      3890221 ns          180
BM_CUTLASS_SGEMM                                                          1961581 ns      1961540 ns          357
BM_CUTLASS_TF32_TENSOROP_SGEMM                                            1454433 ns      1454401 ns          481
BM_CUTLASS_FP16_TENSOROP_GEMM                                              754870 ns       754858 ns          927
BM_CUTLASS_BF16_TENSOROP_GEMM                                              755096 ns       755080 ns          926
```

---

## FAQ

### 1) Where is CUTLASS auto-fetched?

By default: `build/_deps/hpc_cutlass-src`.

### 2) Why does `cp.async` sometimes help only a little?

Common reasons (not exhaustive):

- higher register/shared usage reduces occupancy
- overlap granularity is not a great match (full-tile buffering can underperform depending on tile size/access patterns)
- the kernel is already near a bandwidth/compute bottleneck, or the bottleneck is not global->shared copies

### 3) FP16/BF16 accuracy

FP16/BF16 Tensor Core reduces input precision. For strict comparisons, use
looser tolerances or compare against FP32/TF32 paths instead.

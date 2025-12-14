# GEMV Kernels (CUDA)

This directory implements and compares several CUDA SGEMV kernels
(`y[M] = alpha * A[M,N] * x[N] + beta * y[M]`), including a CUTLASS baseline.

The current focus is on **memory-bound** GEMV variants: one warp (or multiple
warps) computes one output row, reducing partial sums within a warp.

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

### Enable CUTLASS baselines

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_CUTLASS=ON
cmake --build build -j
```

### Run tests

```bash
./build/gemv/gemv_test
```

### Run benchmarks

```bash
./build/gemv/gemv_benchmark
```

Filter examples:

```bash
./build/gemv/gemv_benchmark --benchmark_filter='BM_SGEMVWrapReduce.*'
./build/gemv/gemv_benchmark --benchmark_filter='BM_CUTLASS_.*'
```

---

## Kernel Variants Overview

Names below match the `benchmark` output in `gemv_benchmark`:

- `BM_SGEMVNative<...>`: one thread computes one row (baseline, slowest)
- `BM_SGEMVWrapReduce<...>`: one warp computes one row, strided columns + warp
  reduction
- `BM_SGEMVWrapReduceXShared<...>`: stage `x` into shared memory per block then
  compute (useful only if `x` is not already cached well)
- `BM_SGEMVWrapReduceVec4<...>`: `float4` vectorized loads for `A/x` (requires
  `N % 4 == 0`; otherwise the launcher falls back to `BM_SGEMVWrapReduce`)
- `BM_SGEMVWarpGroupReduce<BLOCK_SIZE, WARPS_PER_ROW>`: multiple warps
  collaborate on one row (e.g. 2 or 4 warps per row), then reduce across warps
- `BM_CUTLASS_GEMV`: CUTLASS `device::Gemm` baseline by treating GEMV as
  GEMM with `n = 1` (`A[M,N] * x[N,1] = y[M,1]`)

---

## Benchmark Results (one sample run)

Default sizes in `gemv_benchmark`: `M=10240, N=2048`. Units are `ns`.

> Note: the sample run below was from a **DEBUG** build (Google Benchmark
> prints a warning). Use `-DCMAKE_BUILD_TYPE=Release` for stable numbers.

```text
Benchmark                                Time             CPU   Iterations
BM_SGEMVNative<128>                1016054 ns      1016033 ns          685
BM_SGEMVNative<256>                1034669 ns      1034644 ns          678
BM_SGEMVNative<512>                1081244 ns      1081234 ns          647
BM_SGEMVWrapReduce<128>             400922 ns       400903 ns         1746
BM_SGEMVWrapReduce<256>             400937 ns       400936 ns         1746
BM_SGEMVWrapReduce<512>             401057 ns       401043 ns         1745
BM_SGEMVWrapReduceXShared<256>      404361 ns       404357 ns         1731
BM_SGEMVWrapReduceVec4<256>         401704 ns       401702 ns         1743
BM_SGEMVWarpGroupReduce<256, 2>     401168 ns       401164 ns         1745
BM_SGEMVWarpGroupReduce<256, 4>     401250 ns       401239 ns         1745
BM_CUTLASS_GEMV                     814422 ns       814259 ns          862
```

### Takeaways (for this shape)

- `WrapReduce` (~0.401 ms) is ~2.5x faster than the naive `Native` (~1.02 ms).
- For `N=2048` and small `x` (~8 KiB), explicitly staging `x` in shared memory
  and `float4` vectorization do **not** help much (extra overhead dominates).
- `2/4 warps per row` does not improve over `1 warp per row` here (extra shared
  + sync overhead).
- CUTLASS GEMM(`n=1`) is slower than the best hand-written GEMV kernels for
  this configuration (GEMM mainloop + epilogue overhead is less amortized when
  `n=1`).


# HighPerformanceComputing

A collection of CUDA/C++ high-performance computing exercises and benchmarks
(vector add, reduction, GEMV, SGEMM, etc.).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Enable CUTLASS (auto-fetched into `build/_deps/hpc_cutlass-src` if not found
locally):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHPC_ENABLE_CUTLASS=ON
cmake --build build -j
```

## Subprojects

- `sgemm_kernel/`: CUDA SGEMM optimizations + CUTLASS Tensor Core baselines
  (see [sgemm_kernel/README.md](sgemm_kernel/README.md))
- `gemv/`: CUDA GEMV kernels (warp-level reduction variants + CUTLASS baseline)
  (see [gemv/README.md](gemv/README.md))

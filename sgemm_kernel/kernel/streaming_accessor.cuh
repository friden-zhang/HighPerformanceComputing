#pragma once

template <class T> struct streaming_accessor {
  using element_type = T;
  struct reference {
    T *p;
    __host__ __device__ operator T() const noexcept {
#ifdef __CUDA_ARCH__
      return __ldcs(p);
#else
      return *p;
#endif
    }
    __host__ __device__ reference &operator=(T v) noexcept {
#ifdef __CUDA_ARCH__
      __stcs(p, v);
#else
      *p = v;
#endif
      return *this;
    }
  };
  using reference = reference;
  using data_handle_type = T *;
  using offset_policy = streaming_accessor;

  __host__ __device__ constexpr reference access(data_handle_type ptr,
                                                 size_t i) const noexcept {
    return reference{ptr + i};
  }

  __host__ __device__ constexpr data_handle_type
  offset(data_handle_type ptr, size_t i) const noexcept {
    return ptr + i;
  }
};
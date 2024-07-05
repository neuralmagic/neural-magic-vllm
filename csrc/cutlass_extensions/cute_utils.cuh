#pragma once

#include <cute/tensor.hpp>

template <typename ElementT, typename Layout>
using gTensor =
    cute::Tensor<cute::ViewEngine<cute::gmem_ptr<ElementT*>>, Layout>;

template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<IntT, cute::Int<1>> make_cute_packed_stride(
    cute::Stride<IntT, cute::Int<1>> s, cute::Shape<int, int, int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
                "Stride must have an integral type so it can be set "
                "dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<IntT>(cute::get<1>(shape_MKL));
  return s_copy;
}

template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, IntT> make_cute_packed_stride(
    cute::Stride<cute::Int<1>, IntT> s, cute::Shape<int, int, int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
                "Stride must have an integral type so it can be set "
                "dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MKL));
  return s_copy;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Strides with batch mode

template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<IntT, cute::Int<1>, int64_t>
make_cute_packed_stride(cute::Stride<IntT, cute::Int<1>, int64_t> s,
                        cute::Shape<int, int, int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
                "Stride must have an integral type so it can be set "
                "dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<IntT>(cute::get<1>(shape_MKL));
  int batch_count = cute::get<2>(shape_MKL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) =
        static_cast<IntT>(cute::get<0>(shape_MKL) * cute::get<1>(shape_MKL));
  } else {
    cute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}

template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, IntT, int64_t>
make_cute_packed_stride(cute::Stride<cute::Int<1>, IntT, int64_t> s,
                        cute::Shape<int, int, int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
                "Stride must have an integral type so it can be set "
                "dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MKL));
  int batch_count = cute::get<2>(shape_MKL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) =
        static_cast<IntT>(cute::get<0>(shape_MKL) * cute::get<1>(shape_MKL));
  } else {
    cute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class PointerType>
static constexpr auto get_logical_ptr(PointerType* ptr) {
  if constexpr (cute::sizeof_bits_v<PointerType> < 8) {
    return cute::subbyte_iterator<PointerType>(ptr);
  } else {
    return ptr;
  }
}
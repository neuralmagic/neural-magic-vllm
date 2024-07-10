#pragma once

#include <torch/all.h>

#include "cutlass/cutlass.h"

using ColumnMajor = typename cutlass::layout::ColumnMajor;
using RowMajor = typename cutlass::layout::RowMajor;

static inline bool is_row_major(torch::Tensor const tensor) {
  TORCH_CHECK(tensor.dim() == 2);
  return tensor.is_contiguous();
}

static inline bool is_column_major(torch::Tensor const tensor) {
  TORCH_CHECK(tensor.dim() == 2);
  return tensor.stride(0) == 1 && tensor.stride(1) == tensor.size(0);
}

template <typename T, typename Layout = RowMajor>
T* maybe_data_ptr(c10::optional<torch::Tensor const> maybe_tensor,
                  char const* name) {
  if constexpr (std::is_same_v<Layout, RowMajor>) {
    TORCH_CHECK(!maybe_tensor || is_row_major(*maybe_tensor), "Expected ", name,
                " to be RowMajor");
  } else if constexpr (std::is_same_v<Layout, ColumnMajor>) {
    TORCH_CHECK(!maybe_tensor || is_column_major(*maybe_tensor), "Expected ",
                name, " to be ColumnMajor");
  } else {
    TORCH_CHECK(false, "Unknown Layout");
  }

  return (maybe_tensor == at::nullopt)
             ? nullptr
             : reinterpret_cast<T*>(maybe_tensor->data_ptr());
}

template <typename T, typename Layout = RowMajor>
T* data_ptr(torch::Tensor const tensor, char const* name) {
  if constexpr (std::is_same_v<Layout, RowMajor>) {
    TORCH_CHECK(is_row_major(tensor), "Expected ", name, " to be RowMajor");
  } else if constexpr (std::is_same_v<Layout, ColumnMajor>) {
    TORCH_CHECK(is_column_major(tensor), "Expected ", name,
                " to be ColumnMajor");
  } else {
    TORCH_CHECK(false, "Unknown Layout");
  }

  return reinterpret_cast<T*>(tensor.data_ptr());
}
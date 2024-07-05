#pragma once

#include <torch/all.h>

#include "cutlass/cutlass.h"

using ColumnMajor = typename cutlass::layout::ColumnMajor;
using RowMajor = typename cutlass::layout::RowMajor;

template <typename T, typename Layout = RowMajor>
T* maybe_data_ptr(c10::optional<torch::Tensor const> maybe_tensor,
                  char const* name) {
  if constexpr (std::is_same_v<Layout, RowMajor>) {
    TORCH_CHECK(!maybe_tensor || maybe_tensor->is_contiguous(), "Expected ",
                name, " to be RowMajor");
  } else if constexpr (std::is_same_v<Layout, ColumnMajor>) {
    TORCH_CHECK(
        !maybe_tensor || (maybe_tensor->stride(0) == 1 &&
                          maybe_tensor->stride(1) == maybe_tensor->size(0)),
        "Expected ", name, " to be ColumnMajor");
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
    TORCH_CHECK(tensor.is_contiguous(), "Expected ", name, " to be RowMajor");
  } else if constexpr (std::is_same_v<Layout, ColumnMajor>) {
    TORCH_CHECK(tensor.stride(0) == 1 && tensor.stride(1) == tensor.size(0),
                "Expected ", name, " to be ColumnMajor");
  } else {
    TORCH_CHECK(false, "Unknown Layout");
  }

  return reinterpret_cast<T*>(tensor.data_ptr());
}
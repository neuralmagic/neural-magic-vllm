#pragma once

#include "marlinv2_mm_kernel.cuh"
#include "cutlass_extensions/cute_utils.cuh"
#include "cutlass_extensions/torch_utils.hpp"

namespace marlinv2 {

template <typename TileShapeMNL, typename ElementB, typename BInTensor,
          typename BTiledOutTensor>
static __global__ void prepack_B_kernel(BInTensor B_in,
                                        BTiledOutTensor B_tiled_out) {
  auto tB_in = local_tile(B_in, TileShapeMNL{},
                          make_coord(blockIdx.x, blockIdx.y, blockIdx.z));
  auto tB_out = B_tiled_out(make_coord(_, _),
                            make_coord(blockIdx.x, blockIdx.y), blockIdx.z);

  auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, ElementB>{},
                                    Layout<Shape<_4, _32>, Stride<_32, _1>>{},
                                    Layout<Shape<_1, _2>>{});

  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor thr_tile_S = thr_copy.partition_S(tB_in);
  Tensor thr_tile_D = thr_copy.partition_D(tB_out);

  // Construct a register-backed Tensor with the same shape as each thread's
  // partition
  auto fragment = make_fragment_like(thr_tile_D);

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(tiled_copy, thr_tile_S, fragment);
  copy(tiled_copy, fragment, thr_tile_D);
}

template <typename PrepackedLayout, typename InLayout>
static void prepack_B(cudaStream_t stream,
                      typename PrepackedLayout::ElementB const* B_in_ptr,
                      InLayout B_layout,
                      typename PrepackedLayout::ElementB* B_out_ptr) {
  using TileShapeMNL =
      decltype(append(typename PrepackedLayout::PPBlockShape_MK{}, _1{}));

  auto MKL_to_TVbMbK = PrepackedLayout::pp_MKL_to_TVbMbK(shape(B_layout));
  auto TVbMbK_to_offset =
      PrepackedLayout::pp_TVbMbKL_to_offset(shape(B_layout));
  auto MKL_to_offset = coalesce(composition(TVbMbK_to_offset, MKL_to_TVbMbK));

  auto MKbMbKL_to_offset =
      PrepackedLayout::pp_MKbMbKL_to_offset(shape(B_layout));

  TORCH_CHECK(size(B_layout) == size(MKL_to_offset));
  TORCH_CHECK(size<0>(B_layout) % size<0>(TileShapeMNL{}) == 0);
  TORCH_CHECK(size<1>(B_layout) % size<1>(TileShapeMNL{}) == 0);
  TORCH_CHECK(size<2>(B_layout) % size<2>(TileShapeMNL{}) == 0);

  auto M_tiles = size<0>(B_layout) / size<0>(TileShapeMNL{});
  auto N_tiles = size<1>(B_layout) / size<1>(TileShapeMNL{});
  auto L_tiles = size<2>(B_layout) / size<2>(TileShapeMNL{});

  auto B_in = make_tensor(get_logical_ptr(B_in_ptr), B_layout);
  auto B_tiled_out = make_tensor(get_logical_ptr(B_out_ptr), MKbMbKL_to_offset);

  prepack_B_kernel<TileShapeMNL, typename PrepackedLayout::ElementB>
      <<<dim3(M_tiles, N_tiles, L_tiles), 128, 0, stream>>>(B_in, B_tiled_out);
}

};  // namespace marlinv2
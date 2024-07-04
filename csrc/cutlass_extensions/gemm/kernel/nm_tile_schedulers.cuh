// Based off of:
//   cutlass/gemm/kernel/tile_scheduler.hpp
// To support:
//   cutlass_extensions/gemm/kernel/nm_sm90_tile_scheduler_stream_k.cuh

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "csrc/cutlass_extensions/gemm/kernel/nm_sm90_tile_scheduler_stream_k.cuh"

namespace cutlass::gemm {

//
// Tags for custom neural magic tile schedulers
//

struct NMStreamKScheduler {};

}  // namespace cutlass::gemm

namespace cutlass::gemm::kernel::detail {

template <class TileShape, class ClusterShape>
struct TileSchedulerSelector<NMStreamKScheduler, arch::Sm90, TileShape,
                             ClusterShape> {
  using Scheduler =
      NMPersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;
};

}  // namespace cutlass::gemm::kernel::detail

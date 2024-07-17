#pragma once

#include "cutlass_extensions/nm_collective_builder_wrapper.cuh"
#include "marlinv2_mainloop.cuh"

namespace cutlass::gemm::collective {
using namespace cute;

struct MarlinV2KernelTag {};

template <class ElementPairA_, class GmemLayoutA_, int AlignmentA,
          class ElementPairB_, class GmemLayoutB_, int AlignmentB,
          class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK,
          class StageCountType, class KernelScheduleType>
struct NMCollectiveBuilder<
    MarlinV2KernelTag, arch::Sm90, arch::OpClassTensorOp, ElementPairA_,
    GmemLayoutA_, AlignmentA, ElementPairB_, GmemLayoutB_, AlignmentB,
    ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType,
    KernelScheduleType,
    cute::enable_if_t<(
        cute::is_same_v<KernelScheduleType,
                        KernelTmaWarpSpecializedMixedInput> ||
        cute::is_same_v<KernelScheduleType,
                        KernelTmaWarpSpecializedPingpongMixedInput> ||
        cute::is_same_v<KernelScheduleType,
                        KernelTmaWarpSpecializedCooperativeMixedInput>)>> {
  using CollectiveOp = marlinv2::MarlinV2CollectiveMma<
      ElementPairA_, GmemLayoutA_, AlignmentA, ElementPairB_, GmemLayoutB_,
      AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK,
      StageCountType, KernelScheduleType>;
};

};  // namespace cutlass::gemm::collective
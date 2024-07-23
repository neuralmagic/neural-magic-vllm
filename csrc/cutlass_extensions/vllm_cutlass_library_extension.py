import enum

from cutlass_library import *

#
#   Extend cutlass library with custom types, and missing values
#

class VLLMTileSchedulerType(enum.Enum):
    StreamK = enum_auto()


TileSchedulerTag.update(
    {VLLMTileSchedulerType.StreamK: "cutlass::gemm::VLLMStreamKScheduler"})


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecializedMixedInput = enum_auto()
    TmaWarpSpecializedPingpongMixedInput = enum_auto()
    TmaWarpSpecializedCooperativeMixedInput = enum_auto()


KernelScheduleTag.update({
    MixedInputKernelScheduleType.TmaWarpSpecializedMixedInput:
    "cutlass::gemm::KernelTmaWarpSpecializedMixedInput",
    MixedInputKernelScheduleType.TmaWarpSpecializedPingpongMixedInput:
    "cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput",
    MixedInputKernelScheduleType.TmaWarpSpecializedCooperativeMixedInput:
    "cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput",
})
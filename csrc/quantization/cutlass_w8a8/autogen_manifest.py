from dataclasses import dataclass
from typing import List, Tuple
import copy

@dataclass
class Cutlass2xArgs:
    arch: int
    tile_shape: Tuple[int, int, int]
    warp_shape: Tuple[int, int, int]
    instruction_shape: Tuple[int, int, int]
    thread_block_swizzle: str
    gemm_mode: str
    main_loop_stages: int

    def with_tile_shape(self, ts):
        clone = copy.deepcopy(self)
        clone.tile_shape = ts 
        return clone

    def with_warp_shape(self, ws):
        clone = copy.deepcopy(self)
        clone.warp_shape = ws
        return clone

    def with_instruction_shape(self, inst):
        clone = copy.deepcopy(self)
        clone.instruction_shape = inst
        return clone

    def with_gemm_mode(self, gemm_mode):
        clone = copy.deepcopy(self)
        clone.gemm_mode = gemm_mode 
        return clone

    def with_main_loop_stages(self, mls):
        clone = copy.deepcopy(self)
        clone.main_loop_stages = mls 
        return clone

    def with_thread_block_swizzle(self, tbs):
        clone = copy.deepcopy(self)
        clone.thread_block_swizzle = tbs
        return clone

## Cutlass2xArgsList 
DefaultCutlass2xArgs = Cutlass2xArgs(89,
        (128, 128, 64),
        (64, 64, 64),
        (16, 8, 32),
        "cutlass::gemm::threadblock::ThreadblockSwizzleStreamK",
        "cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel",
        5)

## TODO (varun) : We get a "Invalid argument" during kernel launch error when we include kGemm with the kGemmSplitKParallel kernels.
# However, running the kGemm kernels on their own works fine.
#Cutlass2xArgsList = Cutlass2xArgsList + list(map(lambda x: x.with_gemm_mode("cutlass::gemm::GemmUniversalMode::kGemm"), Cutlass2xArgsList))

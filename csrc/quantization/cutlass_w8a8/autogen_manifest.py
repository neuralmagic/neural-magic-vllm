from dataclasses import dataclass
from typing import List, Tuple
from itertools import product
import copy

def min_shape(a, b):
    return (min(a[0], b[0]),
            min(a[1], b[1]),
            min(a[2], b[2]))
def shape_is_less(a, b):
    return a[0] <= b[0] and \
           a[1] <= b[1] and \
           a[2] <= b[2]

@dataclass
class Cutlass2xArgs:
    arch: int
    dtype_str: str
    tile_shape: Tuple[int, int, int]
    warp_shape: Tuple[int, int, int]
    instruction_shape: Tuple[int, int, int]
    thread_block_swizzle: str
    gemm_mode: str
    main_loop_stages: int
    fp8_math_operator: str

    def with_dtype_str(self, ds):
        clone = copy.deepcopy(self)
        clone.dtype_str = ds
        return clone

    def with_fp8_math_operator(self, mo):
        clone = copy.deepcopy(self)
        clone.fp8_math_operator = mo
        return clone

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

##
# Gemm Mode
#  kGemm,
#  kGemmSplitKParallel,
#  kBatched,
#  kArray,
#  kGrouped,
#  kInvalid

##
# swizzles 
# "cutlass::gemm::threadblock::ThreadblockSwizzleStreamK",
# cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>
# cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>
# cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>


## Cutlass2xArgsList 
DefaultFP8Cutlass2xArgs = Cutlass2xArgs(89,
        "fp8",
        (128, 128, 64),
        (64, 64, 64),
        (16, 8, 32),
        "cutlass::gemm::threadblock::ThreadblockSwizzleStreamK",
        "cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel",
        5,
        "cutlass::arch::OpMultiplyAdd")

warp_m = [16, 32, 64, 128]
warp_n = [64]
warp_k = [64]
warps = list(product(warp_m, warp_n, warp_k))

tile_m = [16, 32, 64, 128, 256]
tile_n = [32, 64, 128]
tile_k = [64, 128]
tiles = list(product(tile_m, tile_n, tile_k))

stages = [1, 2, 3, 4, 5]
swizzles = ["cutlass::gemm::threadblock::ThreadblockSwizzleStreamK"]
modes = ['cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel']
fp8_operators = ["cutlass::arch::OpMultiplyAddFastAccum"]

#modes = ['cutlass::gemm::GemmUniversalMode::kGemm',
#         'cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel']
#fp8_operators = ["cutlass::arch::OpMultiplyAdd",
#                 "cutlass::arch::OpMultiplyAddFastAccum"]

def is_bad_arg(arg: Cutlass2xArgs):

    if arg.tile_shape[0] == 256 and \
       arg.warp_shape[0] == 16:
        return True

    if arg.tile_shape[0] / arg.warp_shape[0] > 4  or \
       arg.tile_shape[1] / arg.warp_shape[1] > 4 or \
       arg.tile_shape[2] / arg.warp_shape[2] > 4:
           return True

    return False


FP8Cutlass2xArgsList = []
for ts in tiles:
    # use all warp shapes < tile shape
    warp_shapes = []
    for ws in warps:
        if shape_is_less(ws, ts):
            warp_shapes.append(ws)

    configs = list(product([ts], warp_shapes, stages, modes, fp8_operators, swizzles))
    args = list(map(lambda x: DefaultFP8Cutlass2xArgs.with_tile_shape(x[0]) \
                                .with_warp_shape(x[1]) \
                                .with_main_loop_stages(x[2]) \
                                .with_gemm_mode(x[3]) \
                                .with_fp8_math_operator(x[4]) \
                                .with_thread_block_swizzle(x[5]), configs))
    FP8Cutlass2xArgsList.extend(args)

FP8Cutlass2xArgsList = list(filter(lambda x: not is_bad_arg(x), FP8Cutlass2xArgsList))

I8Cutlass2xArgsList = list(map(lambda x: x.with_dtype_str('i8'), FP8Cutlass2xArgsList))

DefaultFP8Cutlass2xArgs = Cutlass2xArgs(89,
        "fp8",
        (128, 128, 64),
        (64, 64, 64),
        (16, 8, 32),
        "cutlass::gemm::threadblock::ThreadblockSwizzleStreamK",
        "cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel",
        5,
        "cutlass::arch::OpMultiplyAdd")

FP8Cutlass2xArgsList = [DefaultFP8Cutlass2xArgs]

FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((16, 128, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAdd'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((16, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAdd'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((16, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(4)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAdd'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((16, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(5)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAdd'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((32, 128, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAdd'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 128, 128))
                        .with_warp_shape((32, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAdd'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAdd'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 64, 128))
                        .with_warp_shape((32, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAdd'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((128, 128, 128))
                        .with_warp_shape((64, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((128, 128, 64))
                        .with_warp_shape((64, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((128, 64, 128))
                        .with_warp_shape((32, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((128, 64, 128))
                        .with_warp_shape((64, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((128, 64, 64))
                        .with_warp_shape((32, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((16, 128, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((16, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((16, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(3)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((16, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(5)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((256, 128, 64))
                        .with_warp_shape((64, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((32, 128, 128))
                        .with_warp_shape((32, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((16, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(4)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((32, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((32, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(4)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((32, 64, 128))
                        .with_warp_shape((16, 64, 64))
                        .with_main_loop_stages(5)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((32, 64, 128))
                        .with_warp_shape((32, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 128, 128))
                        .with_warp_shape((32, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 128, 128))
                        .with_warp_shape((64, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 128, 128))
                        .with_warp_shape((64, 64, 64))
                        .with_main_loop_stages(3)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 128, 64))
                        .with_warp_shape((32, 64, 64))
                        .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 128, 64))
                        .with_warp_shape((32, 64, 64))
                        .with_main_loop_stages(4)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 128, 64))
                        .with_warp_shape((32, 64, 64))
                            .with_main_loop_stages(5)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 64, 128))
                        .with_warp_shape((16, 64, 64))
                            .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
FP8Cutlass2xArgsList.append(
        DefaultFP8Cutlass2xArgs.with_tile_shape((64, 64, 128))
                        .with_warp_shape((32, 64, 64))
                            .with_main_loop_stages(1)
                        .with_fp8_math_operator('cutlass::arch::OpMultiplyAddFastAccum'))
## TODO (varun) : We get a "Invalid argument" during kernel launch error when we include kGemm with the kGemmSplitKParallel kernels.
# However, running the kGemm kernels on their own works fine.
#Cutlass2xArgsList = Cutlass2xArgsList + list(map(lambda x: x.with_gemm_mode("cutlass::gemm::GemmUniversalMode::kGemm"), Cutlass2xArgsList))

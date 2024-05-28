from dataclasses import dataclass
from typing import List, Tuple
from itertools import product
import copy

@dataclass
class Cutlass2xArgs:
    archs: List[int]
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

@dataclass
class Cutlass3xArgs:
    dtype_str: str
    arch: int
    tile_shape: Tuple[int, int, int]
    cluster_shape: Tuple[int, int, int]
    kernel_schedule: str
    epilogue_schedule: str
    tile_schedule: str
    gemm_mode: str

    def with_tile_shape(self, ts):
        clone = copy.deepcopy(self)
        clone.tile_shape = ts 
        return clone

    def with_cluster_shape(self, cs):
        clone = copy.deepcopy(self)
        clone.cluster_shape = cs 
        return clone

    def with_tile_schedule(self, ts):
        clone = copy.deepcopy(self)
        clone.tile_schedule = ts 
        return clone

    def with_kernel_schedule(self, ks):
        clone = copy.deepcopy(self)
        clone.kernel_schedule = ks 
        return clone

    def with_epilogue_schedule(self, es):
        clone = copy.deepcopy(self)
        clone.epilogue_schedule = es 
        return clone

    def with_gemm_mode(self, gm):
        clone = copy.deepcopy(self)
        clone.gemm_mode = gm 
        return clone

    def with_dtype_str(self, dtype_str):
        clone = copy.deepcopy(self)
        clone.dtype_str = dtype_str 
        return clone

## Cutlass2xArgsList 

DefaultCutlass2xArgs = Cutlass2xArgs([80],
        (128, 128, 64),
        (64, 64, 64),
        (16, 8, 32),
        "cutlass::gemm::threadblock::ThreadblockSwizzleStreamK",
        "cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel",
        5)

Cutlass2xArgsList = [
        DefaultCutlass2xArgs,
        DefaultCutlass2xArgs.with_main_loop_stages(4),
        DefaultCutlass2xArgs.with_main_loop_stages(3),
        DefaultCutlass2xArgs.with_tile_shape((128, 64, 64)),
        DefaultCutlass2xArgs.with_tile_shape((128, 64, 64)).with_main_loop_stages(4),
        DefaultCutlass2xArgs.with_tile_shape((128, 64, 64)).with_main_loop_stages(3)]
## TODO (varun) : We get a "Invalid argument" during kernel launch error when we include kGemm with the kGemmSplitKParallel kernels.
# However, running the kGemm kernels on their own works fine.
#Cutlass2xArgsList = Cutlass2xArgsList + list(map(lambda x: x.with_gemm_mode("cutlass::gemm::GemmUniversalMode::kGemm"), Cutlass2xArgsList))

## Cutlass3xArgsList

DefaultCutlass3xArgsI8 = Cutlass3xArgs(
        "int8",
        90,
        (128, 128, 128),
        (1, 2, 1),
        "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
        "cutlass::epilogue::TmaWarpSpecialized",
        "cutlass::gemm::PersistentScheduler",
        "cutlass::gemm::GemmUniversalMode::kGemm")

DefaultCutlass3xArgsFP8 = Cutlass3xArgs(
        "fp8",
        90,
        (128, 128, 128),
        (1, 2, 1),
        "cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative",
        "cutlass::epilogue::TmaWarpSpecializedCooperative",
        "cutlass::gemm::PersistentScheduler",
        "cutlass::gemm::GemmUniversalMode::kGemm")

## Kernel Schedules
## All 
# struct KernelMultistage { };
# struct KernelCpAsyncWarpSpecialized { };
# struct KernelCpAsyncWarpSpecializedPingpong { };
# struct KernelCpAsyncWarpSpecializedCooperative { };
# struct KernelTma { };
# struct KernelTmaWarpSpecialized { };
# struct KernelTmaWarpSpecializedPingpong { };
# struct KernelTmaWarpSpecializedCooperative { };
# struct KernelPtrArrayTmaWarpSpecializedCooperative { };
## FP8
# struct KernelTmaWarpSpecializedFP8FastAccum : KernelTmaWarpSpecialized { };
# struct KernelTmaWarpSpecializedPingpongFP8FastAccum : KernelTmaWarpSpecializedPingpong { };
# struct KernelTmaWarpSpecializedCooperativeFP8FastAccum: KernelTmaWarpSpecializedCooperative { };
# struct KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum : KernelPtrArrayTmaWarpSpecializedCooperative { };

## Epilogue policies
# struct NoSmemWarpSpecialized {};
# struct PtrArrayNoSmemWarpSpecialized {};
# struct TmaWarpSpecialized {};
# struct TmaWarpSpecializedCooperative {};

## Tile scheduler
# struct PersistentScheduler { };
# struct StreamKScheduler { };

## Kgemms
# kGemm
# kGemmSplitKParallel,
# kBatched,
# kArray,
# kGrouped,
# kInvalid

Cutlass3xArgsListI8 = [

    Cutlass3xArgs(
            "int8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecialized",
            "cutlass::epilogue::TmaWarpSpecialized",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),
    Cutlass3xArgs(
            "int8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
            "cutlass::epilogue::TmaWarpSpecialized",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),
    Cutlass3xArgs(
            "int8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
            "cutlass::epilogue::TmaWarpSpecializedCooperative",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),
    Cutlass3xArgs(
            "int8",
            90,
            (128, 128, 128),
            (1, 1, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
            "cutlass::epilogue::TmaWarpSpecializedCooperative",
            "cutlass::gemm::StreamKScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),
        ]

Cutlass3xArgsList=Cutlass3xArgsListI8

def bad_3x_arg(arg):

    bad_tiles = [(256,256,256)]
    if arg.tile_shape in bad_tiles:
        return True

    if "Cooperative" in arg.kernel_schedule and arg.tile_shape[0] < 128:
        return True

    return False

#cluster_shapes = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
#                  (2, 2, 1), (2, 1, 2), (1, 2, 2),
#                  (4, 1, 1), (1, 4, 1), (1, 1, 4)]
tile_shapes_m = [64, 128, 256]
tile_shapes_n = [64, 128, 256]
tile_shapes_k = [32, 64, 128, 256]
tile_shapes = list(product(tile_shapes_m, tile_shapes_n, tile_shapes_k))

tile_cluster_shapes = list(product(tile_shapes, [(1,1,1)]))

Cutlass3xArgsTileList = []
for arg in Cutlass3xArgsList:
    for tile_cluster in tile_cluster_shapes:
        tile, cluster = tile_cluster
        Cutlass3xArgsTileList.append(
                arg.with_tile_shape(tile).with_cluster_shape(cluster))

Cutlass3xArgsTileListI8 = list(filter(lambda x: not bad_3x_arg(x) ,Cutlass3xArgsTileList))

cluster_shapes = [(1, 1, 1), (2, 1, 1), (1, 2, 1), 
                  (2, 2, 1), (4, 1, 1), (1, 4, 1),
                  (8, 1, 1), (1, 8, 1), (4, 4, 1)]

Cutlass3xArgsClusterList = []
for arg in Cutlass3xArgsList:
    for cluster in cluster_shapes:
        Cutlass3xArgsClusterList.append(
                arg.with_cluster_shape(cluster))

Cutlass3xArgsClusterListI8 = list(filter(lambda x: not bad_3x_arg(x) ,Cutlass3xArgsClusterList))

Cutlass3xArgsListFP8 = [
    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecialized",
            "cutlass::epilogue::TmaWarpSpecialized",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),
    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
            "cutlass::epilogue::TmaWarpSpecializedCooperative",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel"),
    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
            "cutlass::epilogue::TmaWarpSpecialized",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),
    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
            "cutlass::epilogue::TmaWarpSpecializedCooperative",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),

    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 64),
            (1, 1, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
            "cutlass::epilogue::TmaWarpSpecializedCooperative",
            "cutlass::gemm::StreamKScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),

    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 64),
            (1, 1, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
            "cutlass::epilogue::TmaWarpSpecializedCooperative",
            "cutlass::gemm::StreamKScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel")
    ]

Cutlass3xArgsListFP8FastAccum = [
    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum",
            "cutlass::epilogue::TmaWarpSpecialized",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),
    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum",
            "cutlass::epilogue::TmaWarpSpecialized",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),
    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 128),
            (1, 2, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum",
            "cutlass::epilogue::TmaWarpSpecializedCooperative",
            "cutlass::gemm::PersistentScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),

    Cutlass3xArgs(
            "fp8",
            90,
            (128, 128, 64),
            (1, 1, 1),
            "cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum",
            "cutlass::epilogue::TmaWarpSpecializedCooperative",
            "cutlass::gemm::StreamKScheduler",
            "cutlass::gemm::GemmUniversalMode::kGemm"),
        ]

clusters_fp8 = [(1, 2, 1), (1, 8, 1), (1, 4, 1), (4, 4, 1), (2, 1, 1) ]

tile_shapes_fp8 =  [(128, 128, 128), (128, 64, 128), (64, 64, 256),
        (64, 64, 128), (64, 128, 256), (64, 128, 128)]

tiles_clusters_fp8 = list(product(tile_shapes_fp8, clusters_fp8))

Cutlass3xArgsListFP8FastAccumTC = []
for arg in Cutlass3xArgsListFP8FastAccum:
    for tile_cluster in tiles_clusters_fp8:
        tile, cluster = tile_cluster
        Cutlass3xArgsListFP8FastAccumTC.append(arg.with_tile_shape(tile).with_cluster_shape(cluster))

Cutlass3xArgsListFP8FastAccumTC = list(filter(
    lambda x: not bad_3x_arg(x), Cutlass3xArgsListFP8FastAccumTC))

import argparse
import copy
import itertools
import pickle as pkl
import time
from typing import Callable, Iterable, List, Tuple, Optional

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]

# helpers


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:

    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    if dtype == torch.int8:
        return to_int8(a), to_int8(b)
    if dtype == torch.float8_e4m3fn:
        return to_fp8(a), to_fp8(b)

    raise ValueError("unsupported dtype")


# impl


def pytorch_mm_impl(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
                    scale_b: torch.Tensor,
                    out_dtype: torch.dtype,
                    impl_fn=None) -> torch.Tensor:
    return torch.mm(a, b)


def pytorch_fp8_impl(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
                     scale_b: torch.Tensor,
                     out_dtype: torch.dtype,
                     impl_fn= None) -> torch.Tensor:
    return torch._scaled_mm(a,
                            b,
                            scale_a=scale_a,
                            scale_b=scale_b,
                            out_dtype=out_dtype)


def pytorch_fp8_impl_fast_accum(a: torch.Tensor, b: torch.Tensor,
                                scale_a: torch.Tensor, scale_b: torch.Tensor,
                                out_dtype: torch.dtype,
                                impl_fn = None) -> torch.Tensor:
    return torch._scaled_mm(a,
                            b,
                            scale_a=scale_a,
                            scale_b=scale_b,
                            out_dtype=out_dtype,
                            use_fast_accum=True)

def cutlass_impl(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
                 scale_b: torch.Tensor,
                 out_dtype: torch.dtype,
                 impl_fn=None) -> torch.Tensor:
    return ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype)

def autogen_cutlass2x_wrapper(a, b, scale_a, scale_b, out_dtype, impl_fn):
    m = a.shape[0]
    n = b.shape[1]
    out = torch.zeros((m, n), dtype=out_dtype, device=a.device)
    bias = None
    return impl_fn(out, a, b, scale_a, scale_b, bias)

def get_autogen_cutlass2x_impls():
    import re
    registration = None
    with open('./csrc/quantization/cutlass_w8a8/autogen_cutlass2x_ops.h') as f:
        registration = f.read()
    assert registration is not None

    bad_kernels = [
            'autogen_cutlass2x_scaled_mm_sm89_128x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_1_OpMultiplyAddFastAccum_fp8',
            'autogen_cutlass2x_scaled_mm_sm89_256x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_1_OpMultiplyAddFastAccum_fp8',
            'autogen_cutlass2x_scaled_mm_sm89_128x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_1_OpMultiplyAdd_fp8',
            'autogen_cutlass2x_scaled_mm_sm89_256x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_1_OpMultiplyAdd_fp8',
            'autogen_cutlass2x_scaled_mm_sm89_128x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_1_OpMultiplyAdd_fp8',
            'autogen_cutlass2x_scaled_mm_sm89_128x128x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_1_OpMultiplyAddFastAccum_fp8',
            'autogen_cutlass2x_scaled_mm_sm89_64x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4_OpMultiplyAdd_fp8',
            'autogen_cutlass2x_scaled_mm_sm89_64x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_4_OpMultiplyAddFastAccum_fp8',
            'autogen_cutlass2x_scaled_mm_sm89_64x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemm_1_OpMultiplyAdd_fp8',
            ]

    must_have_kernels = [
            'autogen_cutlass2x_scaled_mm_sm89_128x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_128x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_128x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_128x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_16x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_256x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_256x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_256x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_32x128x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_32x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_32x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_1_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x128x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_1_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x128x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x128x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_2_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x128x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x64x128_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x64x128_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x64x128_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_3_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x64x64_16x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_4_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x64x64_32x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_1_OpMultiplyAddFastAccum_i8',
'autogen_cutlass2x_scaled_mm_sm89_64x64x64_64x64x64_16x8x32_ThreadblockSwizzleStreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_i8',
            ]

    regex = re.compile('autogen_cutlass2x[A-Za-z_x0-9]*_i8')
    attrs = re.findall(regex, registration)
    attrs = list(filter(lambda x: x not in bad_kernels, attrs))
    assert all([ k in attrs for k in must_have_kernels])
    attrs = must_have_kernels
    attrs = list(set(attrs))

    impls = {}
    for attr in attrs:
        impls[attr] = getattr(torch.ops._C, attr)
        assert impls[attr] is not None

    return impls

# bench
def bench_fn(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
             scale_b: torch.Tensor, out_dtype: torch.dtype, label: str,
             sub_label: str, fn: Callable, description: str,
             impl_fn : Optional[Callable] = None) -> TMeasurement:

    min_run_time = 1

    globals = {
        "a": a,
        "b": b,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "out_dtype": out_dtype,
        "fn": fn,
        "impl_fn": impl_fn,
    }
    return TBenchmark.Timer(
        stmt="fn(a, b, scale_a, scale_b, out_dtype, impl_fn)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)

def bench_cutlass_impls(a, b, scale_a, scale_b,
        out_dtype, label, sub_label):

    autogen_impls = get_autogen_cutlass2x_impls()

    # Try out the autogen kernels first
    impl_keys = list(autogen_impls.keys())
    for impl_key in impl_keys:
        try:
            print (f"Trying {impl_key}")
            autogen_cutlass2x_wrapper(a, b, scale_a, scale_b,
                                    out_dtype, autogen_impls[impl_key])
        except Exception as e:
            print (f"Cannot execute {impl_key} ...\n")
            del autogen_impls[impl_key]
    
    autogen_timers = []
    for desc, fn in autogen_impls.items():
        print (f"bench autogen kernel {desc} ...")
        timer = bench_fn(a, b, scale_a, scale_b, out_dtype, label, sub_label,
                         autogen_cutlass2x_wrapper, desc, fn)
        autogen_timers.append(timer)

    return autogen_timers

def bench_int8(dtype: torch.dtype, m: int, k: int, n: int, label: str,
               sub_label: str) -> Iterable[TMeasurement]:
    assert dtype == torch.int8
    a, b = make_rand_tensors(torch.int8, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    timers = []
    # pytorch impl
    timers.append(
        bench_fn(a.to(dtype=torch.bfloat16, device="cuda"),
                 b.to(dtype=torch.bfloat16, device="cuda"), scale_a, scale_b,
                 torch.bfloat16, label, sub_label, pytorch_mm_impl,
                 "pytorch_bf16_bf16_bf16_matmul-no-scales"))

    ## cutlass impl
    #timers.append(
    #    bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
    #             cutlass_impl, "cutlass_i8_i8_bf16_scaled_mm"))

    # Autogen kernels
    timers.extend(bench_cutlass_impls(
            a, b, scale_a, scale_b, torch.bfloat16, label, sub_label))

    return timers


def bench_fp8(dtype: torch.dtype, m: int, k: int, n: int, label: str,
              sub_label: str) -> Iterable[TMeasurement]:
    assert dtype == torch.float8_e4m3fn
    a, b = make_rand_tensors(torch.float8_e4m3fn, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    timers = []

    # pytorch impl w. bf16
    print ("running pt bf16")
    timers.append(
        bench_fn(a.to(dtype=torch.bfloat16, device="cuda"),
                 b.to(dtype=torch.bfloat16, device="cuda"), scale_a, scale_b,
                 torch.bfloat16, label, sub_label, pytorch_mm_impl,
                 "pytorch_bf16_bf16_bf16_matmul-no-scales"))

    # pytorch impl: bf16 output, without fp8 fast accum
    print ("running pt fp8 no fast accum")
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
                 pytorch_fp8_impl, "pytorch_fp8_fp8_bf16_scaled_mm"))

    # pytorch impl: bf16 output, with fp8 fast accum
    print ("running pt fp8 fast accum")
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
                 pytorch_fp8_impl_fast_accum,
                 "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum"))

    ## Autogen kernels
    timers.extend(bench_cutlass_impls(a, b, scale_a, scale_b,
                                         torch.bfloat16, label, sub_label))

    ## pytorch impl: fp16 output, without fp8 fast accum
    #timers.append(
    #    bench_fn(a, b, scale_a, scale_b, torch.float16, label, sub_label,
    #             pytorch_fp8_impl, "pytorch_fp8_fp8_fp16_scaled_mm"))

    ## pytorch impl: fp16 output, with fp8 fast accum
    #timers.append(
    #    bench_fn(a, b, scale_a, scale_b, torch.float16, label, sub_label,
    #             pytorch_fp8_impl_fast_accum,
    #             "pytorch_fp8_fp8_fp16_scaled_mm_fast_accum"))

    # cutlass impl: bf16 output
    #timers.append(
    #    bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
    #             cutlass_impl, "cutlass_fp8_fp8_bf16_scaled_mm"))

    ## cutlass impl: fp16 output
    #timers.append(
    #    bench_fn(a, b, scale_a, scale_b, torch.float16, label, sub_label,
    #             cutlass_impl, "cutlass_fp8_fp8_fp16_scaled_mm"))
    return timers


def bench(dtype: torch.dtype, m: int, k: int, n: int, label: str,
          sub_label: str) -> Iterable[TMeasurement]:
    if dtype == torch.int8:
        return bench_int8(dtype, m, k, n, label, sub_label)
    if dtype == torch.float8_e4m3fn:
        return bench_fp8(dtype, m, k, n, label, sub_label)
    raise ValueError("unsupported type")


# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(dtype: torch.dtype,
        MKNs: Iterable[Tuple[int, int, int]]) -> Iterable[TMeasurement]:

    results = []
    for m, k, n in MKNs:
        timers = bench(dtype, m, k, n, f"scaled-{dtype}-gemm",
                       f"MKN=({m}x{k}x{n})")
        print_timers(timers)
        results.extend(timers)

    return results


# output makers
def make_output(data: Iterable[TMeasurement],
                MKNs: Iterable[Tuple[int, int, int]],
                base_description: str,
                timestamp=None):

    print(f"== All Results {base_description} ====")
    print_timers(data)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


# argparse runners


def run_square_bench(args):
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    data = run(args.dtype, MKNs)

    make_output(data, MKNs, f"square_bench-{args.dtype}")


def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    data = run(args.dtype, MKNs)

    make_output(data, MKNs, f"range_bench-{args.dtype}")


def run_model_bench(args):

    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
        KNs = []
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KNs.append(KN)
        return KNs

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))

        data = run(args.dtype, MKNs)
        model_bench_data.append(data)

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {args.dtype} {model}-TP{tp_size} ====")
        print_timers(data)

    timestamp = int(time.time())

    all_data = []
    for d in model_bench_data:
        all_data.extend(d)
    # pickle all data
    with open(f"model_bench-{args.dtype}-{timestamp}.pkl", "wb") as f:
        pkl.dump(all_data, f)


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError("unsupported dtype")

    parser = FlexibleArgumentParser(
        description="""
Benchmark Cutlass GEMM.

    To run square GEMMs:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 square_bench --dim-start 128 --dim-end 512 --dim-increment 64
    
    To run constant N and K and sweep M:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384
    
    To run dimensions from a model:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
    
    Output:
        - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various GEMMs.
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--dtype",
                        type=to_torch_dtype,
                        required=True,
                        help="Available options are ['int8', 'fp8']")
    subparsers = parser.add_subparsers(dest="cmd")

    square_parser = subparsers.add_parser("square_bench")
    square_parser.add_argument("--dim-start", type=int, required=True)
    square_parser.add_argument("--dim-end", type=int, required=True)
    square_parser.add_argument("--dim-increment", type=int, required=True)
    square_parser.set_defaults(func=run_square_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--dim-start", type=int, required=True)
    range_parser.add_argument("--dim-end", type=int, required=True)
    range_parser.add_argument("--dim-increment", type=int, required=True)
    range_parser.add_argument("--m-constant", type=int, default=None)
    range_parser.add_argument("--n-constant", type=int, default=None)
    range_parser.add_argument("--k-constant", type=int, default=None)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument("--models",
                              nargs="+",
                              type=str,
                              default=DEFAULT_MODELS,
                              choices=WEIGHT_SHAPES.keys())
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--batch-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_BATCH_SIZES)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)

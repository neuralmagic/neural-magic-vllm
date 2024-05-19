import argparse
import torch
import torch.utils.benchmark as benchmark
import time
import pickle as pkl

from weight_shapes import WEIGHT_SHAPES
from vllm import _custom_ops as ops
from bench_plot import plot_measurements, plot_model_measurements

DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())[1:]
DEFAULT_BATCH_SIZES = [1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]

# helpers 

def to_int8(tensor):
    return torch.round(torch.clamp(tensor, -128, 127)).to(dtype=torch.int8)

def to_fp8(tensor):
    # Assuming input tensor is float32
    # Scale tensor to range of FP8 E4M3 by clamping exponent and truncating mantissa
    max_exp = 2**4 - 1  # Maximum exponent for E4M3
    max_mantissa = 2**3 - 1  # Maximum mantissa for E4M3
    base = 2**max_exp
    # Scale the mantissa
    scaled = torch.clamp(tensor, -base, base)
    # Quantize the mantissa
    quantized = torch.round(scaled * max_mantissa) / max_mantissa
    return quantized.to(dtype=torch.float8_e4m3fn)

def make_rand_tensors(dtype, m, n, k):

    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    if dtype == torch.int8:
        return to_int8(a), to_int8(b)
    if dtype == torch.float8_e4m3fn:
        return to_fp8(a), to_fp8(b)

    raise ValueError("unsupported dtype")


# impl

def pytorch_i8_impl(a, b, scale_a, scale_b, out_dtype, impl_fn = None):
    return torch.mm(a, b)

def pytorch_fp8_impl(a, b, scale_a, scale_b, out_dtype, impl_fn = None):
    return torch._scaled_mm(a, b, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b)

def cutlass_impl(a, b, scale_a, scale_b, out_dtype, impl_fn = None):
    return ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b, out_dtype = out_dtype)

def autogen_cutlass2x_wrapper(a, b, scale_a, scale_b, out_dtype, impl_fn):
    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    return impl_fn(out, a, b, scale_a, scale_b)

def get_autogen_cutlass2x_impls():
    impls = {}
    try:
        import vllm._cutlass2x as cutlass2x
        attrs = dir(cutlass2x)
        attrs = list(filter(lambda x: x.startswith('cutlass'), attrs))
        for attr in attrs:
            assert impls.get(attr) is None
            impls[attr] = getattr(cutlass2x, attr)
    except Exception as e:
        print ("No cutlass2x autogen kernels found")

    return impls

# bench
def bench_fn(a, b, scale_a, scale_b, out_dtype, label, sub_label, fn, description, impl_fn=None):

    min_run_time = 1

    globals = {
            "a" : a,
            "b" : b,
            "scale_a" : scale_a,
            "scale_b" : scale_b,
            "out_dtype" : out_dtype,
            "fn" : fn,
            "impl_fn" : impl_fn,
            }
    return benchmark.Timer(
                stmt="fn(a, b, scale_a, scale_b, out_dtype, impl_fn)",
                globals=globals,
                label=label,
                sub_label=sub_label,
                description=description,
            ).blocked_autorange(min_run_time=min_run_time)


def bench_cutlass_impls(a, b, scale_a, scale_b, out_dtype, label, sub_label, description):

    autogen_impls = get_autogen_cutlass2x_impls()

    autogen_timers = []
    for desc, fn in autogen_impls.items():
        print (f"trying autogen kernel {desc}")
        timer = bench_fn(a, b, scale_a, scale_b, out_dtype, label, sub_label,
                         autogen_cutlass2x_wrapper, desc, fn)
        autogen_timers.append(timer)

    default_impl_timer = bench_fn(a, b, scale_a, scale_b, out_dtype, label, sub_label, 
                             cutlass_impl, "cutlass_i8_i8_bf16_scaled_mm")

    return autogen_timers + [default_impl_timer]

def bench_int8(dtype, m, k, n, label, sub_label):
    assert dtype == torch.int8
    a, b = make_rand_tensors(torch.int8, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype = torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype = torch.float32)

    py_timer = bench_fn(a.to(dtype=torch.bfloat16, device="cuda"), b.to(dtype=torch.bfloat16, device="cuda"),
                        scale_a, scale_b, torch.bfloat16, label, sub_label, 
                        pytorch_i8_impl, "pytorch_bf16_bf16_bf16_matmul-no-scales")

    cutlass_timers = bench_cutlass_impls(a, b, scale_a.to(device="cpu"), scale_b.to(device="cpu"),
                                         torch.bfloat16, label, sub_label, "cutlass_i8_i8_bf16_scaled_mm")

    return [py_timer] + cutlass_timers

def bench_fp8(dtype, m, k, n, label, sub_label):
    assert dtype == torch.float8_e4m3fn
    a, b = make_rand_tensors(torch.float8_e4m3fn, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype = torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype = torch.float32)

    py_timer = bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label, 
                        pytorch_fp8_impl, "pytorch_fp8_fp8_bf16_scaled_mm")

    cutlass_timers = bench_cutlass_impls(a, b, scale_a.to(device="cpu"), scale_b.to(device="cpu"),
                                        torch.bfloat16, label, sub_label, "cutlass_fp8_fp8_bf16_scaled_mm")

    return [py_timer] + cutlass_timers

def bench(dtype, m, k, n, label, sublabel):
    if dtype == torch.int8:
        return bench_int8(dtype, m, k, n, label, sublabel)
    if dtype == torch.float8_e4m3fn:
        return bench_fp8(dtype, m, k, n, label, sublabel)
    raise ValueError("unsupported type")

# runner
def print_timers(timers):
    compare = benchmark.Compare(timers)
    compare.print()

def run(dtype, MKNs):

    results = []
    for m, k, n in MKNs:
        timers = bench(dtype, m, k, n, f"scaled-{dtype}-gemm", f"MKN=({m}x{k}x{n})")
        print_timers(timers)
        results.extend(timers)
    
    return results

# output makers 
def make_title(base_description, MKNs):
    decon = list(zip(*MKNs))
    Ms = decon[0]
    Ks = decon[0]
    Ns = decon[0]

    assert len(Ms) == len(Ns)
    assert len(Ns) == len(Ks)

    if len(Ms) == 1:
        return (f"{base_description} \n"
                f"  Ms - {Ms[0]} \n"
                f"  Ks - {Ks[0]} \n"
                f"  Ns - {Ns[0]} \n")

    return (f"{base_description} \n"
            f"  Ms - {Ms[0]}, {Ms[1]} ... {Ms[-1]} \n"
            f"  Ks - {Ks[0]}, {Ks[1]} ... {Ks[-1]} \n"
            f"  Ns - {Ns[0]}, {Ns[1]} ... {Ns[-1]} \n")

def make_output(data, MKNs, base_description, timestamp = None):

    print (f"== All Results {base_description} ====")
    print_timers(data)

    timestamp = int(time.time()) if timestamp is None else timestamp

    # pickle all the results 
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)

    # plot add data
    plot_measurements(data, make_title(base_description, MKNs), f"{base_description}-{timestamp}.png")
 
# argparse runners

def run_square_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end + 1, args.dim_increment))
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

    model_bench_data = []
    models = args.models
    for model in models:
        Ms = args.batch_sizes
        KNs = []
        for layer in WEIGHT_SHAPES[model]:
            KNs.append((layer[0], layer[1]))

        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))

        data = run(args.dtype, MKNs)
        model_bench_data.append(data)
        #make_output(data, MKNs, f"model_bench-{args.dtype}-{model}")

    # Print all results
    for data, model in zip(model_bench_data, models):
        print (f"== Results {args.dtype} {model} ====")
        print_timers(data)

    timestamp = int(time.time())

    all_data = []
    for d in model_bench_data:
        all_data.extend(d)
    # pickle all data
    with open(f"model_bench-{args.dtype}-{timestamp}.pkl", "wb") as f:
        pkl.dump(all_data, f)

    ## plot add data
    for data, model in zip(model_bench_data, models):
        model_fname = model.replace("/", "_").replace(".", "_")
        plot_model_measurements(data, f"{model_fname}-{args.dtype}-{timestamp}")

if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "int8":
            return  torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError("unsupported dtype")

    parser = argparse.ArgumentParser(
            description=
            """
            Benchmark GEMM.

            To run square matrices gemm:
                python3 ./benchmarks/cutlass_benchmarks.py --dtype fp8 square_bench --dim-start 128 --dim-end 512 --dim-increment 64
            To run constant N and K and sweep M:
                python3 ./benchmarks/cutlass_benchmarks.py --dtype fp8 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384
            To run a model dimensions:
                python3 ./benchmarks/cutlass_benchmarks.py --dtype fp8 model_bench --models meta-llama/Llama-2-7b-hf/TP1 --batch-sizes 16

            The outputs are:
                - a .png file, plotting the tflops for the various gemms for the pytorch and cutlass implementaions.
                - a .pkl file, that is the raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various gemms.
            """)
    parser.add_argument("--dtype", type=to_torch_dtype, required=True)
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
    model_parser.add_argument("--models", nargs="+", type=str, default=DEFAULT_MODELS, choices=WEIGHT_SHAPES.keys())
    model_parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)

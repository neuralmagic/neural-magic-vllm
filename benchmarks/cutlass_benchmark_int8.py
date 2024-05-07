import argparse
import torch
import torch.utils.benchmark as benchmark
import time
import bitsandbytes as bnb

from weight_shapes import WEIGHT_SHAPES
from vllm import _custom_ops as ops
from bench_plot import plot_measurements

DEFAULT_MODELS = ["meta-llama/Llama-2-7b-hf/TP1"]
#DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]
DEFAULT_BATCH_SIZES = [16]

# helpers 

def to_int8(tensor):
    return torch.round(torch.clamp(tensor, -128, 127)).to(dtype=torch.int8)

def make_rand_tensors(m, n, k):
    a = to_int8(torch.randn((m, k), device='cuda') * 5)
    b = to_int8(torch.randn((n, k), device='cuda').t() * 5)
    return a, b

# impl

def pytorch_impl(a, b, scale_a, scale_b):
    return torch.mm(a, b)

def cutlass_impl(a, b, scale_a, scale_b):
    return ops.cutlass_scaled_mm_dq(a, b, scale_a, scale_b)

def bnb_impl(a, b, scale_a, scale_b):
    return bnb.matmul(a, b)

# bench
def bench_fn(a, b, scale_a, scale_b, label, sub_label, fn, description):

    min_run_time = 1

    globals = {
            "a" : a,
            "b" : b,
            "scale_a" : scale_a,
            "scale_b" : scale_b,
            "fn" : fn,
            }
    return benchmark.Timer(
                stmt="fn(a, b, scale_a, scale_b)",
                globals=globals,
                label=label,
                sub_label=sub_label,
                description=description,
            ).blocked_autorange(min_run_time=min_run_time)

def test_correctness(a, b, scale_a, scale_b):
    # Correctness compare implementations

    # Pytorch doesn't have a cuda int8 matmul. Upcast to bf16
    py_c = pytorch_impl(scale_a * a.to(dtype=torch.float32),
                        scale_b * b.to(dtype=torch.float32),
                        scale_a, scale_b).to(dtype=torch.bfloat16)

    # Our cutlass impl. passes the scale by value if the numel is 1.
    # It is better to have the scales in the CPU to avoid data-transfer.
    cutlass_c = cutlass_impl(a, b, scale_a.to(device="cpu"), scale_b.to(device="cpu"))

    assert torch.allclose(py_c, cutlass_c)

def bench(m, k, n, label, sub_label):
    a, b = make_rand_tensors(m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype = torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype = torch.float32)

    test_correctness(a, b, scale_a, scale_b)

    py_timer = bench_fn(a.to(dtype=torch.bfloat16, device="cuda"), b.to(dtype=torch.bfloat16, device="cuda"),
                        scale_a, scale_b, label, sub_label, 
                        pytorch_impl, "pytorch_bf16_bf16_bf16_matmul-no-scales")

    cutlass_timer = bench_fn(a, b, scale_a.to(device="cpu"), scale_b.to(device="cpu"),
                             label, sub_label, 
                             cutlass_impl, "cutlass_i8_i8_bf16_scaled_mm")

    return py_timer, cutlass_timer

# runner
def run(MKNs):
    def print_timers(timers):
        compare = benchmark.Compare(timers)
        compare.print()

    results = []
    for m, k, n in MKNs:
        timers = bench(m, k, n, "scaled-int8-gemm", f"MKN=({m}x{k}x{n})")
        print_timers(timers)
        results.extend(timers)
    
    print ("== All Results ====")
    print_timers(results)
    return results

# argparse runners
def make_title(base_description, MKNs):
    decon = list(zip(*MKNs))
    Ms = decon[0]
    Ks = decon[0]
    Ns = decon[0]

    return (f"{base_description} \n"
            f"  Ms - {Ms[0]}, {Ms[1]} ... {Ms[-1]} \n"
            f"  Ks - {Ks[0]}, {Ks[1]} ... {Ks[-1]} \n"
            f"  Ns - {Ns[0]}, {Ns[1]} ... {Ns[-1]} \n")
 
def run_square_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    data = run(MKNs)

    timestamp = int(time.time())
    plot_measurements(data, make_title("square_bench", MKNs), f"square_bench-{timestamp}.png")

def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes 
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes 
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes 
    MKNs = list(zip(Ms, Ks, Ns))
    data = run(MKNs)

    timestamp = int(time.time())
    plot_measurements(data, make_title("range_bench", MKNs), f"range_bench-{timestamp}.png")

def run_model_bench(args): 

    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    assert len(args.models) == 1
    model = args.models[0]

    Ms = args.batch_sizes

    KNs = []
    for layer in WEIGHT_SHAPES[model]:
        KNs.append((layer[0], layer[1]))

    MKNs = []
    for m in Ms:
        for k, n in KNs:
            MKNs.append((m, k, n))

    data = run(MKNs)

    timestamp = int(time.time())
    plot_measurements(data, make_title("model_bench", MKNs), f"model_bench-{timestamp}.png")

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="Benchmark int8 gemm")
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

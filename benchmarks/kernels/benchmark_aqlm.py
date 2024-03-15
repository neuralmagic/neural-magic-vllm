import os
import sys
from typing import Optional

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from vllm.model_executor.layers.quantization.aqlm import (
    dequantize_partioned_gemm, dequantize_weight)
from vllm._C import ops

import torch
import torch.nn.functional as F


def torch_mult(
        input: torch.Tensor,  #  [..., in_features]
        weights: torch.Tensor,
        scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
) -> torch.Tensor:
    output = F.linear(input, weights)
    return output


def dequant_torch_mult(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    output_partition_sizes: torch.IntTensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:

    #print("input shape:", input.shape, "codes", codes.shape, "scale", scales.shape, "parts", output_partition_sizes)

    weights = ops.aqlm_dequant(codes, codebooks, scales,
                               output_partition_sizes)

    if False: #bias is None:
        output = F.linear(input, weights, bias)
        orig_shape = output.shape
   #     print("ouutput is ", output.shape)
  #      print("scales shape is ", scales.shape)
        flattened_output = output.view(-1, output.size(-1))
 #       print("flate output ", flattened_output.shape)
        f_scales = scales.view(-1, scales.shape[0]) 
        b_scales = f_scales.expand(flattened_output.shape[0], -1)
#        print("b scales ", b_scales.shape)
                
        flattened_output *= b_scales
        return flattened_output.view(orig_shape)
    else:
        #b_scales = scales.view(scales.shape[:-3] + (-1,)).expand(-1, weights.shape[1])
        #weights *= b_scales
        return F.linear(input, weights, bias)


# Compare my kernel against the gold standard.
def dequant_test(k: int, parts: torch.tensor) -> float:

    n = parts.sum().item()

    device = torch.device('cuda:0')

    codes = torch.randint(-32768,
                          32768,
                          size=(n, k // 8, 1),
                          dtype=torch.int16,
                          device=device)

    codebooks = torch.randn(size=(parts.shape[0], 65536, 1, 8),
                            dtype=torch.float16,
                            device=device)

    # ones.
    scales = torch.randn(size=(n, 1, 1, 1), dtype=torch.float16, device=device)

    weights = dequantize_weight(codes, codebooks, scales)
    print("weights are:", weights)

    weights2 = ops.aqlm_dequant(codes, codebooks, scales, parts)
    print("weights2 shape:", weights2.shape)
    print("weights2 are:", weights2)

    flattened_scales = scales.view(scales.shape[:-3] + (-1,))
    print("f scales", flattened_scales.shape)    
    broadcast_scales  = flattened_scales.expand(-1, weights2.shape[1])
    print("b scales", broadcast_scales.shape)

    weights2 *= broadcast_scales
    print("weights2 scaled are:", weights2)


def main():

    timing = run_timing(100, 16, 4096, torch.tensor((4096, )),
                        dequant_torch_mult)
    print("timing was ", timing * 1000, "us")

    return
    methods = [
        dequantize_partioned_gemm, ops.aqlm_gemm, torch_mult,
        dequant_torch_mult
    ]

    filename = "./benchmark.csv"
    print(f"writing benchmarks to file {filename}")
    with open(filename, "a") as f:
        sys.stdout = f

        print('m | k | n | n parts', end='')
        for method in methods:
            print(f' | {method.__name__}', end='')
        print('')

        # These are reasonable prefill sizes.
        ksandpartions = ((4096, (4096, 4096, 4096)), (4096, (4096, )),
                         (4096, (11008, 11008)), (11008, (4096, )))

        # reasonable ranges for m.
        for m in [
                1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
                2048, 3072, 4096
        ]:
            print(f'{m}', file=sys.__stdout__)
            for ksp in ksandpartions:
                run_grid(m, ksp[0], torch.tensor(ksp[1]), methods)

        sys.stdout = sys.__stdout__


def run_grid(m: int, k: int, parts: torch.tensor, methods):

    num_warmup_trials = 1
    num_trials = 1

    num_calls = 100

    # warmup.
    for method in methods:
        for _ in range(num_warmup_trials):
            run_timing(
                num_calls=num_calls,
                m=m,
                k=k,
                parts=parts,
                method=method,
            )

    n = parts.sum().item()
    print(f'{m} | {k} | {n} | {parts.tolist()}', end='')

    for method in methods:
        best_time_us = 1e20
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                m=m,
                k=k,
                parts=parts,
                method=method,
            )

            kernel_dur_us = 1000 * kernel_dur_ms

            if kernel_dur_us < best_time_us:
                best_time_us = kernel_dur_us

        print(f' | {kernel_dur_us:.0f}', end='')

    print('')


def run_timing(num_calls: int, m: int, k: int, parts: torch.tensor,
               method) -> float:

    n = parts.sum().item()

    device = torch.device('cuda:0')

    input = torch.randn((1, m, k), dtype=torch.float16, device=device)

    codes = torch.randint(-32768,
                          32768,
                          size=(n, k // 8, 1),
                          dtype=torch.int16,
                          device=device)

    codebooks = torch.randn(size=(parts.shape[0], 65536, 1, 8),
                            dtype=torch.float16,
                            device=device)

    scales = torch.randn(size=(n, 1, 1, 1), dtype=torch.float16, device=device)

    weights = torch.randn((n, k), dtype=torch.float16, device=device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    if method is torch_mult:
        for i in range(num_calls):
            output = torch_mult(input, weights, scales)
    else:
        for i in range(num_calls):
            output = method(input, codes, codebooks, scales, parts, None)

    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    sys.exit(main())

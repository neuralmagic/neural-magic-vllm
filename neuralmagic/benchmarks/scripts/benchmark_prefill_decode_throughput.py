import argparse
import torch
import time 

from collections import namedtuple

from typing import List
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import get_tokenizer
from common import generate_synthetic_requests


BenchmarkResults = namedtuple("BenchmarkResults", ['outputs', 'time'] )

def print_benchmark_io(results: List[RequestOutput]):
    for result in results:
        output = result.outputs[0]
        print (f"\n\n inputs({len(result.prompt_token_ids)}): {result.prompt}\n output({len(output.token_ids)}): {output.text}")

def run_benchmark_througput(model_id:str,
                            batch_size:int,
                            input_tokens_len:int,
                            output_tokens_len:int,
                            bench_iterations:int,
                            warmup_iterations:int = 3) -> BenchmarkResults:

    model = LLM(model=args.model_id, dtype="float16")
    sampling_params = SamplingParams(max_tokens=output_tokens_len, ignore_eos=True, temperature=0)

    tokenizer = get_tokenizer(model_id)
    prompts = generate_synthetic_requests(input_tokens_len, output_tokens_len, batch_size, tokenizer)
    prompts_txt = list(map(lambda p_tuple: p_tuple[0], prompts))

    print("warming up...")
    for _ in range(warmup_iterations):
        outputs = model.generate(prompts_txt, sampling_params)
    torch.cuda.synchronize()

    print("benchmarking...")
    start = time.perf_counter()
    for _ in range(bench_iterations):
        outputs = model.generate(prompts_txt, sampling_params)
    torch.cuda.synchronize()
    end = time.perf_counter()
    total_time = end - start

    return BenchmarkResults(outputs, total_time)

def run_benchmark_decode_throughput(model_id:str,
                             batch_size:int,
                             input_tokens_len:int,
                             output_tokens_len:int,
                             bench_iterations:int = 10,
                             log_model_io:bool = False) -> None:

    results = run_benchmark_througput(model_id, batch_size, input_tokens_len, output_tokens_len, bench_iterations)

    if log_model_io:
        print_benchmark_io(results.outputs)

    total_output_tokens = 0
    for output in results.outputs:
        total_output_tokens += len(output.outputs[0].token_ids)
    
    total_output_tokens *= bench_iterations
    tput = total_output_tokens / results.time
    
    print(f"----- Batch Size: {batch_size} -----")
    print(f"Total time: {results.time:0.2f}s")
    print(f"Total tokens: {total_output_tokens} tokens")
    print(f"Tput: {tput:0.2f} tokens/sec\n")

    return tput

def run_benchmark_prefill_throughput(model_id:str,
                             batch_size:int,
                             input_tokens_len:int,
                             output_tokens_len:int,
                             bench_iterations:int = 10,
                             log_model_io:bool = False) -> None:

    results = run_benchmark_througput(model_id, batch_size, input_tokens_len, output_tokens_len, bench_iterations)
    if log_model_io:
        print_benchmark_io(results.outputs)

    total_prompt_tokens = 0
    for output in results.outputs:
        total_prompt_tokens += len(output.prompt_token_ids)
    total_prompt_tokens *= bench_iterations
    
    tput = total_prompt_tokens / results.time
    
    print(f"----- Batch Size: {batch_size} -----")
    print(f"Total time: {results.time:0.2f}s")
    print(f"Total prompt tokens: {total_prompt_tokens} tokens")
    print(f"Tput: {tput:0.2f} tokens/sec\n")

    return tput

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prompt-len", type=int, default=None)
    parser.add_argument("--log-model-io", action="store_true")
    arg_group = parser.add_mutually_exclusive_group() 
    arg_group.add_argument("--benchmark-prefill", action="store_true")
    arg_group.add_argument("--benchmark-decode", action="store_true")

    args = parser.parse_args()

    print (f"{args}")

    if args.benchmark_prefill:
        run_benchmark_prefill_throughput(model_id = args.model_id,
                                        batch_size = args.batch_size,
                                        input_tokens_len = args.prompt_len,
                                        output_tokens_len = 1,
                                        log_model_io = args.log_model_io)
    else:
        assert args.benchmark_decode
        assert args.prompt_len is None

        run_benchmark_decode_throughput(model_id = args.model_id,
                                        batch_size = args.batch_size,
                                        input_tokens_len = 2,
                                        output_tokens_len = 10,
                                        log_model_io = args.log_model_io)
"""
Benchmark offline inference throughput.

NOTE: This script is copied from upstream vllm repo (february 13th, 2024).
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import AutoTokenizer

from common import get_bench_environment, sample_requests, generate_synthetic_requests

def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.perf_counter()
    # FIXME(woosuk): Do not use internal method.
    llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    return end - start

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        requests = generate_synthetic_requests(args.input_len, args.output_len,
                                               args.num_prompts, tokenizer)
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)

    elapsed_time = run_vllm(requests, args.model, args.tokenizer,
                            args.quantization, args.tensor_parallel_size,
                            args.seed, args.n, args.use_beam_search,
                            args.trust_remote_code, args.dtype,
                            args.max_model_len, args.enforce_eager)

    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)

    request_throughput = len(requests) / elapsed_time
    token_throughput = total_num_tokens / elapsed_time
    print(f"Throughput: {request_throughput:.2f} requests/s, "
          f"{token_throughput:.2f} tokens/s")

    # Save config and results to json
    save_result = args.save_directory is not None
    if save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["bench_env"] = get_bench_environment()
        result_json.update(vars(args))
        result_json["request_throughput (per second)"] = request_throughput
        result_json["token_throughput (per second)"] = token_throughput

        model_id = args.model.replace('/', '_')
        # Save to file
        file_name = Path(args.save_directory
                         ) / f"{args.backend}-{model_id}-{current_dt}.json"
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument("--save-directory",
                        type=str,
                        default=None,
                        help="Output directory to store result file")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    main(args)

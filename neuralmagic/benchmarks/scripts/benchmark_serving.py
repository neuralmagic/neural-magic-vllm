"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>

NOTE: This script is copied from upstream vllm repo (february 13th, 2024).
"""
import argparse
import asyncio
import json
import random
import time
import random
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Tuple, Optional

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
from common import get_bench_environment

from neuralmagic.tools.call_cmd import call_cmd

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    median_request_latency: float
    p90_request_latency: float
    p99_request_latency: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float

def generate_synthetic_requests(num_input_tokens : int,
                                num_output_tokens : int,
                                num_requests : int,
                                tokenizer : PreTrainedTokenizerBase) -> List[Tuple[str, int, int]]:

    share_gpt_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    if not share_gpt_path.exists():
        share_gpt_download_str =  \
    "wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json";
        share_gpt_download_list = share_gpt_download_str.split(" ") 
        call_cmd(share_gpt_download_list, stdout=None, stderr=None)
    assert share_gpt_path.exists()

    dataset = None
    with open(share_gpt_path) as f:
        dataset = json.load(f)
    assert dataset

    def ids_to_prompt(prompt_ids: list[int]) -> list[int]:
        # remove special tokens from prompt ids
        prompt_ids = list(filter(lambda id: id not in tokenizer.all_special_ids, prompt_ids))
        return tokenizer.decode(prompt_ids)

    sampled_requests = []
    while len(sampled_requests) != num_requests:
        # get a random sample.
        convo = random.choice(dataset)

        # build prompt until we fill as many words as num_input_tokens.
        # We would be over-sampling, but that is fine as we truncate below.
        prompt = ""
        for turn in convo["conversations"]:
            prompt = prompt + " " + turn["value"]
            if len(prompt) >= num_input_tokens:
                break

        prompt_ids = tokenizer(prompt).input_ids 

        if len(prompt_ids) < num_input_tokens:
            continue

        prompt_ids = prompt_ids[:num_input_tokens]
        prompt = ids_to_prompt(prompt_ids)

        sampled_requests.append((prompt, num_input_tokens, num_output_tokens))

    assert len(sampled_requests) == num_requests
    return sampled_requests

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    assert dataset_path
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # some of these will be filtered out, so sample more than we need
    sampled_indices = random.sample(range(len(dataset)),
                                    int(num_requests * 1.2))
    dataset = [dataset[i] for i in sampled_indices]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> BenchmarkMetrics:
    total_output = 0
    total_input = 0
    completed = 0
    latencies = []
    per_token_latencies = []
    ttfts = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(tokenizer.encode(outputs[i].generated_text))
            total_output += output_len
            total_input += input_requests[i][1]
            latencies.append(outputs[i].latency)
            per_token_latencies.append(outputs[i].latency / output_len)
            ttfts.append(outputs[i].ttft)
            completed += 1

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=total_output / dur_s,
        median_request_latency=np.median(latencies) * 1000,
        p90_request_latency=np.percentile(latencies, 90) * 1000,
        p99_request_latency=np.percentile(latencies, 99) * 1000,
        mean_ttft_ms=np.mean(ttfts) * 1000,
        median_ttft_ms=np.median(ttfts) * 1000,
        p90_ttft_ms=np.percentile(ttfts, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts, 99) * 1000,
        mean_tpot_ms=np.mean(per_token_latencies) * 1000,
        median_tpot_ms=np.median(per_token_latencies) * 1000,
        p90_tpot_ms=np.percentile(per_token_latencies, 90) * 1000,
        p99_tpot_ms=np.percentile(per_token_latencies, 99) * 1000,
    )

    return metrics


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    log_model_io: bool
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    print(f"Traffic request rate: {request_rate}")

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # Dump model i/o
    if log_model_io: 
        for i, o in zip(input_requests, outputs):
            prompt = i[0]
            generated_txt = o.generated_text
            input_tokens = tokenizer(prompt).input_ids
            output_tokens = tokenizer(generated_txt).input_ids
            print(f"\n\n\n inputs ({len(input_tokens)}) : {prompt} \n outputs ({len(output_tokens)}) : {generated_txt}")

    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print(f"Successful requests: {metrics.completed}")
    print(f"Benchmark duration: {benchmark_duration:2f} s")
    print(f"Total input tokens: {metrics.total_input}")
    print(f"Total generated tokens: {metrics.total_output}")
    print(f"Request throughput: {metrics.request_throughput:.2f} requests/s")
    print(f"Input token throughput: {metrics.input_throughput:.2f} tokens/s")
    print(f"Output token throughput: {metrics.output_throughput:.2f} tokens/s")
    print(f"Median request latency: {metrics.median_request_latency:.2f} ms")
    print(f"P90 request latency: {metrics.p90_request_latency:.2f} ms")
    print(f"P99 request latency: {metrics.p99_request_latency:.2f} ms")
    print(f"Mean TTFT: {metrics.mean_ttft_ms:.2f} ms")
    print(f"Median TTFT: {metrics.median_ttft_ms:.2f} ms")
    print(f"P90 TTFT: {metrics.p90_ttft_ms:.2f} ms")
    print(f"P99 TTFT: {metrics.p99_ttft_ms:.2f} ms")
    print(f"Mean TPOT: {metrics.mean_tpot_ms:.2f} ms")
    print(f"Median TPOT: {metrics.median_tpot_ms:.2f} ms")
    print(f"P90 TPOT: {metrics.p90_tpot_ms:.2f} ms")
    print(f"P99 TPOT: {metrics.p99_tpot_ms:.2f} ms")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_inthroughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "median_request_latency": metrics.median_request_latency,
        "p90_request_latency": metrics.p90_request_latency,
        "p99_request_latency": metrics.p99_request_latency,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p90_ttft_ms": metrics.p90_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
    }
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    input_requests = None
    if args.dataset:
        input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)
    else:
        input_requests = generate_synthetic_requests(args.num_input_tokens, args.num_output_tokens, args.num_prompts, tokenizer)

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            log_model_io=args.log_model_io
        ))

    # Save config and results to json
    save_result = args.save_directory is not None
    if save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["bench_env"] = get_bench_environment()
        result_json["backend"] = backend
        result_json["version"] = args.version
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = (
            Path(args.save_directory) /
            f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        )
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Benchmark the online serving throughput.''')
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="N/A",
        help="Version of the serving backend/engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/generate",
        help="API endpoint.",
    )
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--num-input-tokens",
                        type=int,
                        default=None,
                        help="Number of tokens in the input prompt")
    parser.add_argument("--num-output-tokens",
                        type=int,
                        default=None,
                        help="Number of generated tokens per prompt")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default model tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--log-model-io", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disbale tqdm progress bar.",
    )
    parser.add_argument("--save-directory",
                        type=str,
                        default=None,
                        help="Output directory to store result file")


    args = parser.parse_args()

    # Sanity check
    if args.dataset is not None:
        assert args.num_input_tokens is None and args.num_output_tokens is None
    else:
        assert args.num_input_tokens is not None and args.num_output_tokens is not None

    main(args)

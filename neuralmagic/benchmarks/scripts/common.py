"""
Common functions used in all benchmarking scripts
"""
import json
import random
from typing import List, Tuple
from pathlib import Path
from transformers import PreTrainedTokenizerBase

from vllm.outputs import RequestOutput
from neuralmagic.tools.call_cmd import call_cmd
from neuralmagic.benchmarks.datasets_registry import SHAREGPT_PATH, SHAREGPT_DOWNLOAD_STR


def get_bench_environment() -> dict:
    """
    Return the current python version, pytorch version and CUDA version as a dict
    """
    import sys
    import torch
    return {
        "python_version": f"{sys.version}",
        "torch_version": f"{torch.__version__}",
        "torch_cuda_version": f"{torch.version.cuda}",
        "cuda_device(0)": f"{torch.cuda.get_device_properties(0)}"
    }


def generate_synthetic_requests(
        num_input_tokens: int, num_output_tokens: int, num_requests: int,
        tokenizer: PreTrainedTokenizerBase) -> List[Tuple[str, int, int]]:

    share_gpt_path = Path(SHAREGPT_PATH)
    if not share_gpt_path.exists():
        share_gpt_download_list = SHAREGPT_DOWNLOAD_STR.split(" ")
        call_cmd(share_gpt_download_list, stdout=None, stderr=None)
    assert share_gpt_path.exists()

    dataset = None
    with open(share_gpt_path) as f:
        dataset = json.load(f)
    assert dataset

    def ids_to_prompt(prompt_ids: list[int]) -> list[int]:
        # remove special tokens from prompt ids
        prompt_ids = list(
            filter(lambda id: id not in tokenizer.all_special_ids, prompt_ids))
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


def print_benchmark_io(results: List[RequestOutput]) -> None:
    for result in results:
        output = result.outputs[0]
        print(
            f"\n\n inputs({len(result.prompt_token_ids)}): {result.prompt}\n output({len(output.token_ids)}): {output.text}"
        )

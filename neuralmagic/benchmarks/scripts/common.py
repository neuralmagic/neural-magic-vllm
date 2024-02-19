"""
Common functions used in all benchmarking scripts
"""
import json
import random
from typing import List, Tuple, Optional
from pathlib import Path
from transformers import PreTrainedTokenizerBase

from vllm.outputs import RequestOutput
from neuralmagic.tools.call_cmd import call_cmd



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

    share_gpt_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    if not share_gpt_path.exists():
        share_gpt_download_str =  \
    "wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
        share_gpt_download_list = share_gpt_download_str.split(" ")
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


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

def print_benchmark_io(results: List[RequestOutput]) -> None:
    for result in results:
        output = result.outputs[0]
        print(
            f"\n\n inputs({len(result.prompt_token_ids)}): {result.prompt}\n output({len(output.token_ids)}): {output.text}"
        )

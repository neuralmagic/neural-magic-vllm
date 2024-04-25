import asyncio
from os import getenv
from typing import List

import openai
import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from tests.conftest import HfRunnerNM
from tests.models.compare_utils import check_logprobs_close
from tests.utils.logging import make_logger
from tests.utils.server import ServerContext


@pytest.fixture(scope="session")
def client():
    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    yield client


@pytest.fixture
def hf_runner_nm():
    return HfRunnerNM


async def my_chat(
    client,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    num_logprobs: int,
):
    """ submit a single prompt chat and collect results. """
    return await client.chat.completions.create(model=model,
                                                messages=[{
                                                    "role": "user",
                                                    "content": prompt
                                                }],
                                                max_tokens=max_tokens,
                                                temperature=temperature,
                                                logprobs=True,
                                                top_logprobs=num_logprobs)


async def concurrent_chats(
    client,
    model: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    num_logprobs: int,
):
    chats = [
        my_chat(client, model, prompt, max_tokens, temperature, num_logprobs)
        for prompt in prompts
    ]
    results = await asyncio.gather(*chats)
    return results


@pytest.mark.parametrize(
    "model, max_model_len, sparsity",
    [
        ("mistralai/Mistral-7B-Instruct-v0.2", 4096, None),
        # ("mistralai/Mixtral-8x7B-Instruct-v0.1", 4096, None),
        # ("neuralmagic/zephyr-7b-beta-marlin", 4096, None),
        # ("neuralmagic/OpenHermes-2.5-Mistral-7B-pruned50",
        #  4096, "sparse_w16a16"),
        # ("NousResearch/Llama-2-7b-chat-hf", 4096, None),
        # pytest.param(
        #     "neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin",
        #     None,
        #     None,
        #     marks=pytest.mark.skip(
        #         "skipped because the HFRunner will need the "
        #         "'optimum' package")
        # ),
        # ("neuralmagic/Llama-2-7b-pruned70-retrained-ultrachat",
        #  4096, "sparse_w16a16"),
        # ("HuggingFaceH4/zephyr-7b-gemma-v0.1", 4096, None),
        # ("Qwen/Qwen1.5-7B-Chat", 4096, None),
        # ("microsoft/phi-2", 2048, None),
        # pytest.param(
        #     "neuralmagic/phi-2-super-marlin",
        #     2048,
        #     None,
        #     marks=pytest.mark.skip(
        #         "skipped because the HFRunner will need the "
        #         "'optimum' package")
        # ),
        # ("neuralmagic/phi-2-pruned50", 2048, "sparse_w16a16"),
        # ("mistralai/Mixtral-8x7B-Instruct-v0.1", 4096, None),
        # pytest.param(
        #     "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        #     4096,
        #     None,
        #     marks=pytest.mark.skip(
        #         "ValueError: The checkpoint you are trying to load has model"
        #         "type `qwen2_moe` but Transformers does not recognize this "
        #         "architecture. This could be because of an issue with the "
        #         "checkpoint, or because your version of Transformers is "
        #         "out of date.")),
        # ("casperhansen/gemma-7b-it-awq", 4096, None),
        # pytest.param(
        #     "TheBloke/Llama-2-7B-Chat-GPTQ",
        #     4096,
        #     None,
        #     marks=pytest.mark.skip(
        #         "skipped because the HFRunner will need the "
        #         "'optimum' package")
        # ),
    ])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
# TODO: add the tensor_parallel_size = 2 param arg value
# @pytest.mark.parametrize("tensor_parallel_size", [None, 2])
@pytest.mark.parametrize("tensor_parallel_size", [None])
# note: repeating the test for 2 values of tensor_parallel_size
#  increases the overall execution time by unnecessarily
#  collecting the HuggingFace runner data twice.
#  Consider refactoring to eliminate that repeat.
def test_models_on_server(
    hf_runner_nm,
    client,
    model: str,
    max_model_len: int,
    sparsity: str,
    tensor_parallel_size: int,
    max_tokens: int,
    num_logprobs: int,
) -> None:

    hf_token = getenv("HF_TOKEN", None)
    ds = load_dataset("nm-testing/qa-chat-prompts", split="train_sft")
    # example_prompts = [m[0]["content"] for m in ds["messages"]]
    num_chat_turns = 3
    tokenizer = AutoTokenizer.from_pretrained(model)
    messages_list = [row["messages"][:num_chat_turns] for row in ds]
    chat_prompts = [
        tokenizer.apply_chat_template(messages,
                                      tokenize=False,
                                      add_generation_prompt=True)
        for messages in messages_list
    ]
    hf_model = hf_runner_nm(model, access_token=hf_token)
    hf_outputs = hf_model.generate_greedy_logprobs_nm_use_tokens(
        chat_prompts, max_tokens, num_logprobs)

    del hf_model

    api_server_args = {
        "--model": model,
        "--max-model-len": max_model_len,
        "--disable-log-requests": None,
    }
    if sparsity:
        api_server_args["--sparsity"] = sparsity
    if tensor_parallel_size:
        api_server_args["--tensor-parallel-size"] = tensor_parallel_size

    logger = make_logger("vllm_server")
    asyncio_event_loop = asyncio.get_event_loop()
    temperature = 0.0
    with ServerContext(api_server_args, logger=logger) as _:
        # submit an asynchronous request to the server for each prompt
        chats = [
            my_chat(client, model, prompt, max_tokens, temperature,
                    num_logprobs) for prompt in chat_prompts
        ]
        # await for all the requests to return, and gather their results
        # in one place
        results = asyncio_event_loop.run_until_complete(asyncio.gather(*chats))

    vllm_outputs = list()
    for task_result in results:
        for req_output in task_result.choices:
            output_str = req_output.message.content
            output_tokens = req_output.logprobs.model_extra["tokens"]
            output_logprobs = req_output.logprobs.model_extra["top_logprobs"]
            vllm_outputs.append((output_tokens, output_str, output_logprobs))

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )

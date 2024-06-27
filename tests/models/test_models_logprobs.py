"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models_logprobs.py --forked`.
"""
import pytest

from tests.models.utils import check_logprobs_close
from tests.nm_utils.utils_skip import should_skip_test_group

if should_skip_test_group(group_name="TEST_MODELS"):
    pytest.skip("TEST_MODELS=DISABLE, skipping model test group",
                allow_module_level=True)

MODEL_MAX_LEN = 1024

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "Deci/DeciLM-7b",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-1b",
    "bigscience/bloom-1b1",
    "mosaicml/mpt-7b",
    "microsoft/phi-2",
    "stabilityai/stablelm-3b-4e1t",
    "allenai/OLMo-1B",
    "bigcode/starcoder2-3b",
    "Qwen/Qwen1.5-0.5B",
]

SKIPPED_MODELS_ACC = [
    "mosaicml/mpt-7b",
    "allenai/OLMo-1B",
    "bigcode/starcoder2-3b",
]

SKIPPED_MODELS_CI = [
    "EleutherAI/gpt-j-6b",  # OOM on CPU RAM
    "tiiuae/falcon-7b",  # Fails in vllm if trust_remote_code=True
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
def test_models(
    vllm_runner,
    hf_runner_nm,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    if model in SKIPPED_MODELS_ACC:
        pytest.skip(reason="Low priority models not currently passing. "
                    "We need to re-enable these.")
    if model in SKIPPED_MODELS_CI:
        pytest.skip(reason="These models cause some CI issue unrelated "
                    "to the correctness of the implementation.")

    hf_model = hf_runner_nm(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy_logprobs_nm(example_prompts,
                                                      max_tokens, num_logprobs)

    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype, max_model_len=MODEL_MAX_LEN)
    vllm_outputs = vllm_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)

    del vllm_model

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )

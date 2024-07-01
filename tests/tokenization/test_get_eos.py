"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
:meth:`vllm.LLMEngine._get_eos_token_id`.
"""
import pytest

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.tokenizer import get_tokenizer

if should_skip_test_group(group_name="TEST_TOKENIZATION"):
    pytest.skip("TEST_TOKENIZATION=DISABLE, skipping tokenization test group",
                allow_module_level=True)


def test_get_llama3_eos_token():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name,
                                                  trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name,
                                                  trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118

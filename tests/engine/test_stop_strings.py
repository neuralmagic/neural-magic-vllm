from typing import Any, List, Optional

import pytest

from vllm import CompletionOutput, LLMEngine, SamplingParams

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_TOKENS = 200


@pytest.fixture(scope="session")
def vllm_model(vllm_runner):
    return vllm_runner(MODEL)


@pytest.mark.skip_global_cleanup
def test_stop_basic(vllm_model):
    _test_stopping(vllm_model.model.llm_engine,
                   stop=["in"],
                   include_in_output=False,
                   expected_output="\nVLLM is a company that specializes ",
                   expected_reason="in")

    _test_stopping(vllm_model.model.llm_engine,
                   stop=["in"],
                   include_in_output=True,
                   expected_output="\nVLLM is a company that specializes in",
                   expected_reason="in")


@pytest.mark.skip_global_cleanup
def test_stop_multi_tokens(vllm_model):
    _test_stopping(vllm_model.model.llm_engine,
                   stop=["providing virtual", "short"],
                   include_in_output=False,
                   expected_output="\nVLLM is a company that specializes in ",
                   expected_reason="providing virtual")

    _test_stopping(vllm_model.model.llm_engine,
                   stop=["providing virtual", "short"],
                   include_in_output=True,
                   expected_output=
                   "\nVLLM is a company that specializes in providing virtual",
                   expected_reason="providing virtual")


@pytest.mark.skip_global_cleanup
def test_stop_partial_token(vllm_model):
    _test_stopping(vllm_model.model.llm_engine,
                   stop=["izes"],
                   include_in_output=False,
                   expected_output="\nVLLM is a company that special",
                   expected_reason="izes")

    _test_stopping(vllm_model.model.llm_engine,
                   stop=["izes"],
                   include_in_output=True,
                   expected_output="\nVLLM is a company that specializes",
                   expected_reason="izes")


@pytest.mark.skip_global_cleanup
def test_stop_token_id(vllm_model):
    # token id 6901 => "virtual"

    _test_stopping(
        vllm_model.model.llm_engine,
        stop_token_ids=[6901],
        include_in_output=False,
        expected_output="\nVLLM is a company that specializes in providing",
        expected_reason=6901)

    _test_stopping(vllm_model.model.llm_engine,
                   stop_token_ids=[6901],
                   include_in_output=True,
                   expected_output=
                   "\nVLLM is a company that specializes in providing virtual",
                   expected_reason=6901)


def _test_stopping(llm_engine: LLMEngine,
                   expected_output: str,
                   expected_reason: Any,
                   stop: Optional[List[str]] = None,
                   stop_token_ids: Optional[List[int]] = None,
                   include_in_output: bool = False) -> None:
    llm_engine.add_request(
        "id", "A story about vLLM:\n",
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            stop=stop,
            stop_token_ids=stop_token_ids,
            include_stop_str_in_output=include_in_output,
        ), None)

    output: Optional[CompletionOutput] = None
    output_text = ""
    stop_reason = None
    while llm_engine.has_unfinished_requests():
        (request_output, ) = llm_engine.step()
        (output, ) = request_output.outputs

        # Ensure we don't backtrack
        assert output.text.startswith(output_text)
        output_text = output.text
        stop_reason = output.stop_reason

    assert output is not None
    assert output_text == expected_output
    assert stop_reason == expected_reason

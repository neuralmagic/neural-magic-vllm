"""Tests whether Marlin models can be loaded from the autogptq config.

Run `pytest tests/quantization/test_autogptq_marlin_configs.py --forked`.
"""

from dataclasses import dataclass

import pytest

from vllm.config import ModelConfig


@dataclass
class ModelPair:
    model_marlin: str
    model_gptq: str


# Model Id // Expected Kernel
MODELS_GPTQ_MARLIN_QUANT_TYPE = [
    ("TheBloke/Llama-2-7B-Chat-GPTQ", "gptq_marlin"),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", "gptq_marlin")
]

# Model Id // Expected Kernel
MODELS_MARLIN_QUANT_TYPE = [
    # compat: autogptq <=0.7.1 is_marlin_format: bool
    ("neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin", "marlin"),
    # compat: autogptq >=0.8.0 use checkpoint_format: str
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit", "marlin"),
]


@pytest.mark.parametrize("model_quant_type", MODELS_MARLIN_QUANT_TYPE)
def test_marlin_config(model_quant_type: str, ) -> None:
    model_path, quant_type = model_quant_type

    model_config_no_quant_arg = ModelConfig(
        model_path,
        model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        download_dir=None,
        load_format="dummy",
        seed=0,
        dtype="float16",
        revision=None,
        quantization=None  # case 1: Will choose marlin
    )

    model_config_quant_arg = ModelConfig(
        model_path,
        model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        download_dir=None,
        load_format="dummy",
        seed=0,
        dtype="float16",
        revision=None,
        quantization="gptq"  # case 2: Will replace gptq with marlin
    )

    assert model_config_no_quant_arg.quantization == quant_type, (
        f"Expected quant_type == {quant_type} for {model_path}, "
        f"but found {model_config_no_quant_arg.quantization} "
        "for no --quantization None case")

    assert model_config_quant_arg.quantization == quant_type, (
        f"Expected quant_type == {quant_type} for {model_path}, "
        f"but found {model_config_quant_arg.quantization} "
        "for --quantization gptq case")


@pytest.mark.parametrize("model_quant_type", MODELS_GPTQ_MARLIN_QUANT_TYPE)
def test_gptq_marlin_config(model_quant_type: str, ) -> None:
    model_path, quant_type = model_quant_type

    model_config_no_quant_arg = ModelConfig(
        model_path,
        model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        download_dir=None,
        load_format="dummy",
        seed=0,
        dtype="float16",
        revision=None,
        quantization=None  # case 1: Will default to gptq_marlin
    )

    model_config_quant_arg = ModelConfig(
        model_path,
        model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        download_dir=None,
        load_format="dummy",
        seed=0,
        dtype="float16",
        revision=None,
        quantization="gptq"  # case 2: Will force gptq (and not gptq_marlin)
    )

    assert model_config_no_quant_arg.quantization == quant_type, (
        f"Expected quant_type == {quant_type} for {model_path}, "
        f"but found {model_config_no_quant_arg.quantization} "
        "for no --quantization None case")

    assert model_config_quant_arg.quantization == "gptq", (
        f"Expected quant_type == gptq for {model_path}, "
        f"but found {model_config_quant_arg.quantization} "
        "for --quantization gptq case")

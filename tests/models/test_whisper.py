import pytest
import torch
from datasets import load_dataset
from vllm.config import AudioFeaturesConfig

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

@pytest.fixture()
def model_id():
    return "openai/whisper-tiny"


@pytest.fixture()
def audio_features_config():
    return AudioFeaturesConfig()


def sample_from_librispeech():
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                           "clean",
                           split="validation")
    return dataset[0]


audio_sample = sample_from_librispeech()["audio"]


@pytest.mark.parametrize("dtype", ["float"])  # TODO fix that
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("prompts, audio_samples", [([""], [audio_sample])])
def test_text_to_audio_scenario(hf_runner, vllm_runner, model_id, prompts,
                                audio_samples, dtype: str,
                                max_tokens: int) -> None:
    
    hf_model = hf_runner(model_id, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(prompts=prompts,
                                          audio_samples=audio_samples,
                                          max_tokens=max_tokens)
    del hf_model

    # Truly cleans up GPU memory.
    torch.cuda.empty_cache()

    vllm_model = vllm_runner(model_id,
                             dtype=dtype,
                             enforce_eager=True,
                             tensor_parallel_size=1,
                             gpu_memory_utilization=0.5)

    vllm_outputs = vllm_model.generate_greedy(prompts,
                                              max_tokens,
                                              audio_samples=audio_samples)
    del vllm_model
    # Truly cleans up GPU memory.
    torch.cuda.empty_cache()

    for i in range(len(prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        print(f"hf_output_str: {hf_output_str}")
        print(f"first 10 tokens: {hf_output_ids[:10]}")
        print(f"vllm_output_str: {vllm_output_str}")
        print(f"first 10 tokens: {vllm_output_ids[:10]}")


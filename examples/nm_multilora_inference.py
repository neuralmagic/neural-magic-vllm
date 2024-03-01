"""
This example shows how to use the multi-LoRA functionality for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import Optional, List, Tuple

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest

from enum import Enum
import os 


class LoraEnum(Enum):
    ARC = "PEFT-arc-easy"
    LAW = "PEFT-law"
    SCI = "PEFT-science-middle"
    
    


# MODEL_PATH = "/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-base/training"
# LORA_PATH = "/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-base/PEFT-law"

MODEL_PATH = "/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50/training"
# LORA_PATH = "/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50/PEFT-arc-easy"
LORA_PATH = "/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50"




# MODEL_PATH = "/network/abhinav/llama/models/Llama-2-7b-hf"
# LORA_PATH = "/network/abhinav/hackathon/models/PEFT-law"

def create_test_prompts(
        lora_path: str
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.
    
    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
 
    return [
        # ("In Massachusetts, as in the rest of the U.S., Miranda rights come into play when",
        #  SamplingParams(n=3,
        #                 best_of=3,
        #                 use_beam_search=True,
        #                 temperature=0,
        #                 max_tokens=128,
        #                 stop_token_ids=[32003]),
        #  LoRARequest("law-lora", 1, os.path.join(LORA_PATH, LoraEnum.LAW.value))),
        # ("If a plant is not watered, what will happen?",
        #  SamplingParams(n=3,
        #                 best_of=3,
        #                 use_beam_search=True,
        #                 temperature=0,
        #                 max_tokens=128,
        #                 stop_token_ids=[32003]),
        #  LoRARequest("arc-lora", 2, os.path.join(LORA_PATH, LoraEnum.ARC.value))),
        ("Describe the structure and function of a cell membrane.",
         SamplingParams(n=3,
                        best_of=3,
                        use_beam_search=True,
                        temperature=0,
                        max_tokens=128,
                        stop_token_ids=[32003]),
         LoRARequest("sci-lora", 3, os.path.join(LORA_PATH, LoraEnum.SCI.value))),
         ("Describe the structure and function of a cell membrane.",
         SamplingParams(n=3,
                        best_of=3,
                        use_beam_search=True,
                        temperature=0,
                        max_tokens=128,
                        stop_token_ids=[32003]),
         LoRARequest("arc-lora", 4, os.path.join(LORA_PATH, LoraEnum.ARC.value))),
          ("Describe the structure and function of a cell membrane.",
         SamplingParams(n=3,
                        best_of=3,
                        use_beam_search=True,
                        temperature=0,
                        max_tokens=128,
                        stop_token_ids=[32003]),
         LoRARequest("law-lora", 4, os.path.join(LORA_PATH, LoraEnum.LAW.value))),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.

    engine_args = EngineArgs(
        model=MODEL_PATH,
                             enable_lora=True,
                             max_loras=3,
                             max_lora_rank=8,
                             max_cpu_loras=4,
                             max_num_seqs=256)
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    lora_path = LORA_PATH
    
    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    main()



"""
CUDA_VISIBLE_DEVICES=4 python3 examples/nm_multilora_inference.py 

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "arc-easy",
        "messages": [
            {"role": "user", "content": "[Question]:\n{Who won the world series in 2020?}\n\n[Response]:"}
        ],
        "stream": true
    }'

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "arc-easy",
        "messages": [
            {"role": "user", "content": "[Question]:\n{Name me cold-blooded animals}\n\n[Response]:"}
        ],
        "stream": true
    }'
    
curl http://localhost:8000/v1/chat/completions     -H "Content-Type: application/json"     -d '{
        "model": "arc-easy",
        "messages": [
            {"role": "user", "content": "Name me some common animals"}
        ],
        "stream": true,
        "max_tokens": 256
    }'
    
    
"""
# [Question]:\n{Who won the world series in 2020?}\n\n[Response]:""".format(example[0])
"""
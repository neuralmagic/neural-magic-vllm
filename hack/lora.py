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

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class LoraEnum(Enum):
    # ARC = "PEFT-arc-easy"
    # LAW = "PEFT-law"
    # SCI = "PEFT-science-middle"
    
    ARC_EASY="PEFT-mmlu_arc_easy"
    ARC_HARD="PEFT-mmlu_arc_hard"
    LAW="PEFT-mmlu_law"
    MC="PEFT-mmlu_mc"
    OBQA="PEFT-mmlu_obqa"
    RACE="PEFT-mmlu_race"
    SCI_ELE="PEFT-mmlu_science_elementary"
    SCI_MIDDLE="PEFT-mmlu_science_middle"



MODEL_PATH = "/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/training"
LORA_PATH = "/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned70/"

llm = LLM(model=MODEL_PATH, enable_lora=True)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    stop=["[/assistant]"]
)

question = "Name me all cold-blooded animals"
prompts = [
    #  "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
    #  "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    # "What is endomediated cytosis?",
    # "What is clathrin? What how is it in involed in transport inside a cell?",
   
    f"""[Question]:\n{question}\n\n[Response]:"""
    
]

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("lora", 1, os.path.join(LORA_PATH, LoraEnum.LAW.value))
)


print(outputs)


"""
CUDA_VISIBLE_DEVICES=4 python3 hack/lora.py


/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50/PEFT-arc-easy
/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50/PEFT-law
/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50/PEFT-science-middle

python -m vllm.entrypoints.openai.api_server \
    --model /network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50/training \
    --enable-lora \
    --lora-modules arc=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50/PEFT-arc-easy law=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50/PEFT-law sci=/network/abhinav/hackathon/models/mistral-7b-open_platypus_orca_mistral_pretrain-pruned50/PEFT-science-middle

"""


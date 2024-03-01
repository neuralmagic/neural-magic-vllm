import os 

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from enum import Enum
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

enum_values = list(LoraEnum)

for val in enum_values:
    print(
        os.path.join(LORA_PATH, val.value)
    )
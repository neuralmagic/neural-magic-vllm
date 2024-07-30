from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams

VLLM_GENERATE_RPC_PATH  = "tcp://localhost:5570"
VLLM_GET_DATA_RPC_PATH  = "tcp://localhost:5571"
VLLM_IS_READY_RPC_PATH  = "tcp://localhost:5572"
VLLM_ABORT_RPC_PATH     = "tcp://localhost:5573"
VLLM_LOG_STATS_RPC_PATH = "tcp://localhost:5574"

VLLM_ABORT_RESPONSE_STR = "ABORTED"
VLLM_READY_RESPONSE_STR = "READY"

@dataclass
class GenerateRequest:
    inputs: PromptInputs
    sampling_params: SamplingParams
    request_id: str
    lora_request: Optional[LoRARequest] = None
    trace_headers: Optional[Mapping[str, str]] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None


@dataclass
class AbortRequest:
    request_id: str

class RPCRequestType(Enum):
    MODEL_CONFIG = 1
    DO_LOG_STATS = 2

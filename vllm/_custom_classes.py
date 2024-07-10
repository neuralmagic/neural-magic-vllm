import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import vllm._C
except ImportError as e:
    logger.warning("Failed to import from vllm._C with %r", e)


VLLMType = torch.classes._C.VLLMType

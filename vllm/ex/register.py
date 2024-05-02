import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

SUPPORTED = set()
FUSABLE = dict()


# TODO: make sure this is usable as a decorator
def register_supported(op_name: str):
    SUPPORTED.add(op_name)


# TODO: make sure this is usable as a decorator
def register_fusable(op_name: str, is_compute: bool = False):
    assert op_name not in FUSABLE or FUSABLE[op_name] == is_compute
    FUSABLE[op_name] = is_compute


def register_defaults():
    logger.info("REGISTER DEFAULTS")
    register_supported('_operator.add')
    register_supported('_operator.mul')
    register_supported('_operator.getitem')
    register_supported('torch.matmul')
    register_supported('torch.relu')
    register_supported('torch.nn.functional.silu')
    register_supported('torch._C._nn.linear')
    register_supported('torch.ops.vllm.silu_and_mul')

    register_fusable('_operator.add')
    register_fusable('_operator.mul')
    register_fusable('_operator.getitem')
    register_fusable('torch.relu')
    register_fusable('torch.nn.functional.silu')
    register_fusable('torch.ops.vllm.silu_and_mul')
    register_fusable('torch.matmul', True)
    register_fusable('torch._C._nn.linear', True)


register_defaults()

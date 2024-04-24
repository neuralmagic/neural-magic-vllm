from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
import torch
from typing import Dict, List
from torch.nn import Parameter
from vllm.model_executor.utils import set_weight_attrs
import torch.nn.functional as F

__all__ = ["CompressedTensorsUnquantized"]


class CompressedTensorsUnquantized(CompressedTensorsScheme):
    """
    Implements the scheme for all layers which are ignored in the CompressedTensors 
    config. The input and loaded weight are used in a linear transformation.
    """

    def create_weights(self, output_sizes_per_partition: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, **kwargs):

        weight = Parameter(torch.empty(sum(output_sizes_per_partition),
                                       input_size_per_partition,
                                       device="cuda",
                                       dtype=params_dtype),
                           requires_grad=False)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        return {"weight": weight}

    def apply_weights(self, weights: Dict, x: torch.Tensor):
        weight = weights.get("weight")
        return F.linear(x, weight)

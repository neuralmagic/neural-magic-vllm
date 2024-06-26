from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.parameter import vLLMParameter

__all__ = ["CompressedTensorsUnquantized"]


class CompressedTensorsUnquantized(CompressedTensorsScheme):
    """
    Implements the scheme for all layers which are ignored 
    in the CompressedTensors config. The input and loaded weight are used 
    in a linear transformation.
    """

    def get_min_capability(self) -> int:
        # volta and up
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        weight = vLLMParameter(data=torch.empty(sum(output_partition_sizes),
                                                input_size_per_partition,
                                                dtype=params_dtype),
                               input_dim=1,
                               output_dim=0,
                               weight_loader=weight_loader)

        layer.register_parameter("weight", weight)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:

        return F.linear(x, layer.weight, bias)

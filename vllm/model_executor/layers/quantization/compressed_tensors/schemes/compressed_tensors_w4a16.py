from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Parameter

from vllm._C import ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.utils import set_weight_attrs


__all__ = ["CompressedTensorsW4A16"]


class CompressedTensorsW4A16(CompressedTensorsScheme):

    def create_weights(self, layer: torch.nn.Module,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable, layer_name: str,
                    **kwargs):

        pack_factor = 8
        group_size = 128 # 128 things next to each other in memory, 2nd dimension for things next to each other in memory
 
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0, "packed_dim": 1})
        set_weight_attrs(weight, {"weight_loader": weight_loader})

        weight_scale = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // group_size,
                1,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {"weight_loader": weight_loader})
        set_weight_attrs(weight_scale, {"input_dim": 1, "output_dim": 0})

        weight_zero_point = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // group_size,
                1,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight_zero_point", weight_zero_point)
        set_weight_attrs(weight_zero_point, {"weight_loader": weight_loader})
        set_weight_attrs(weight_zero_point, {"input_dim": 1, "output_dim": 0})

        weight_shape = Parameter(torch.empty(2,
                                            device="cuda",
                                            dtype=torch.int64),
                                requires_grad=False)

        layer.register_parameter("weight_shape", weight_shape)
        set_weight_attrs(weight_shape, {"weight_loader": weight_loader})


    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        weight = layer.weight
        weight_scale = layer.weight_scale 
        return
                    

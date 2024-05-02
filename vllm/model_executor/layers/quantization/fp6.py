from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


logger = init_logger(__name__)


class Fp6Config(QuantizationConfig):
    """Config class for FP6 weight-only (W6A16) quantization."""

    @classmethod
    def get_name(cls) -> str:
        return "fp6"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Requires >= Ampere because of async read
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp6Config":
        return cls()

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["Fp6LinearMethod"]:
        if isinstance(layer, LinearBase):
            return Fp6LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp6LinearMethod(LinearMethodBase):
    """Linear method for FP6 weight-only (W6A16) quantization.
    Support loading FP16/BF16 checkpoints and quantizating their weights to
    FP6.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp6Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.process_after_load = True

        # WEIGHT
        weight_dtype = params_dtype
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=weight_dtype),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs,
            "input_dim": 1,
            "output_dim": 0,
        })

    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return

        # Quantize the weights to FP6.
        # qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
        #                                                 scale=None)
        # layer.weight = Parameter(qweight.t(), requires_grad=False)
        return

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.act_scale is None and x_scale computed from x.
        #   If static,  layer.act_scale is scalar and x_scale set to act_scale.
        qinput, x_scale = ops.scaled_fp8_quant(x, layer.act_scale)

        # Fused GEMM_DQ
        output, _ = torch._scaled_mm(
            qinput,
            layer.weight,
            out_dtype=x.dtype,
            scale_a=x_scale,
            scale_b=layer.weight_scale,
            bias=bias,
        )

        return output


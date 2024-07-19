from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp8_linear, create_per_channel_scale_param)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

# Note: this is a hack. We should update each model to register the
# stacked params and get it from there instead in a future PR.
# fused_name: List[shard_name]
_FUSED_LAYER_NAME_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"]
}


class FBGEMMFp8Config(QuantizationConfig):
    """Config class for FBGEMM Fp8."""

    def __init__(self, ignore_list: List[str]):
        self.ignore_list = ignore_list

    @classmethod
    def get_name(cls) -> str:
        return "fbgemm_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FBGEMMFp8Config":
        ignore_list = cls.get_from_keys(config, ["modules_to_not_convert"])
        return cls(ignore_list=ignore_list)

    def _is_layer_skipped(self, prefix: str) -> bool:
        # prefix: model.layers.0.self_attn.q_proj
        # proj_name: q_proj
        proj_name = prefix.split(".")[-1]
        if proj_name in _FUSED_LAYER_NAME_MAPPING:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in _FUSED_LAYER_NAME_MAPPING[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = shard_prefix in self.ignore_list

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision.")
        else:
            is_skipped = prefix in self.ignore_list

        assert is_skipped is not None
        return is_skipped

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self._is_layer_skipped(prefix):
                return UnquantizedLinearMethod()
            return FBGEMMFp8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class FBGEMMFp8LinearMethod(LinearMethodBase):

    def __init__(self, quant_config: FBGEMMFp8Config):
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

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=torch.float8_e4m3fn),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            **extra_weight_attrs,
        })

        # WEIGHT SCALE
        weight_scale = create_per_channel_scale_param(output_partition_sizes,
                                                      **extra_weight_attrs)
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE UPPER BOUND
        input_scale_ub = torch.nn.Parameter(torch.zeros((1),
                                                        dtype=torch.float32),
                                            requires_grad=False)
        layer.register_parameter("input_scale_ub", input_scale_ub)
        set_weight_attrs(input_scale_ub, {
            "ignore_warning": True,
            **extra_weight_attrs
        })

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight
        layer.weight = Parameter(weight.t(), requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        return apply_fp8_linear(input=x,
                                weight=layer.weight,
                                weight_scale=layer.weight_scale,
                                input_scale=None,
                                input_scale_ub=layer.input_scale_ub,
                                bias=bias,
                                cutlass_fp8_supported=True,
                                use_per_token_if_dynamic=True)

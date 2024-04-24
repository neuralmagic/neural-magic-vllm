from typing import Any, Dict, List, Tuple, Type, Optional, Union

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.smoothquant.formats import (
    SmoothQuantFormat,
    SmoothQuantDynamicPerToken,
    SmoothQuantStaticPerTensor,
)
from vllm.model_executor.layers.quantization.smoothquant.cutlass_gemm import (
    cutlass_gemm_dq)

LAYER_KEYS = ["qkv", "out", "fc1", "fc2"]
FORMAT_REGISTRY = {
    "per-token": SmoothQuantDynamicPerToken,
    "per-tensor": SmoothQuantStaticPerTensor,
}


def get_sq_format_cls(format_key: str) -> Type[SmoothQuantFormat]:
    if format_key not in FORMAT_REGISTRY:
        raise ValueError(f"Invalid smoothquant format: {format_key}")
    return FORMAT_REGISTRY[format_key]


class SmoothQuantConfig(QuantizationConfig):
    """Config class for SmoothQuant.

    Reference: https://github.com/mit-han-lab/smoothquant
    """

    def __init__(self, layer_format_map: Dict[str, str]) -> None:
        self.layer_format_map = layer_format_map

        for key, format in self.layer_format_map.items():
            if key not in LAYER_KEYS:
                raise ValueError(
                    f"Found key of {key} in {self.layer_format_map}, "
                    f"but key must be one of {LAYER_KEYS}")
            if format not in FORMAT_REGISTRY:
                raise ValueError(
                    f"Found format of {format} in {self.layer_format_map}, "
                    f"but format must be one of {FORMAT_REGISTRY}")
        for key in LAYER_KEYS:
            if key not in self.layer_format_map:
                raise ValueError(f"Could not find {key} in {layer_format_map}")

    def __repr__(self) -> str:
        return (f"SmoothQuantConfig(layer_format_map={self.layer_format_map})")

    def get_name(self) -> str:
        return "smoothquant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        # TODO: check if we support fp16 / bf16 as well.
        return [torch.float]

    def get_min_capability(self) -> int:
        # TODO: check if this is right.
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        return [
            "quant_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmoothQuantConfig":
        layer_format_map: Dict[str, str] = {}
        for layer_key, format in config.items():
            if format in FORMAT_REGISTRY:
                layer_format_map[layer_key] = format
        return cls(layer_format_map)

    def get_linear_method(self) -> "SmoothQuantLinearMethod":
        return SmoothQuantLinearMethod(self)


class SmoothQuantLinearMethod(LinearMethodBase):

    def __init__(self, sq_config: SmoothQuantConfig) -> None:
        self.sq_config = sq_config
        self.sq_type = None

    def maybe_update_loaded_weight_name(self, name: str) -> str:
        """Convert serialized name k_dequant_scale to dequant_scale.

        This function is called by model_cls.load_weights() during the weight
        loading process to match on disk state dict to vllm state dict.
        """
        if "dequant_scale" in name:
            suffix = name.split('.')[-1]
            name.replace(suffix, "dequant_scale")
        return name

    def shard_id_as_int(self, shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id

        assert isinstance(shard_id, str)
        qkv_idxs = {"q": 0, "k": 1, "v": 2}
        assert shard_id in qkv_idxs
        return qkv_idxs[shard_id]

    def scales_shard_splitter(
            self, param: torch.Tensor, loaded_weight: torch.Tensor,
            shard_id: Union[str, int],
            logical_widths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_id = self.shard_id_as_int(shard_id)
        offset = sum(logical_widths[:shard_id])
        size = logical_widths[shard_id]
        # update loaded weight with copies for broadcast.
        loaded_weight = loaded_weight.repeat(size)
        return param[offset:offset + size], loaded_weight

    def get_layer_format(self, layer_name: str) -> SmoothQuantFormat:
        """
        Gets the SmoothQuantFormat for a specific layer.

        SmoothQuantLinearMethod uses SmoothQuantType to support non-uniform quantization 
        (where each layer has a different format). To determine the SmoothQuantFormat 
        for a layer, we match the layer_name to the layer_keys=["qkv","out","fc1","fc2"] 
        and use layer_format_map to to determine the SQFormat.
        
        Args:
            layer_name: Name of the layer we are creating the LinearMethod for.
        Returns
            sq_linear_method: SmoothQuantLinearMethod with the right SQFormat.
        """
        # Note: AutoSmoothQuant Serialization is not very good yet.
        #
        # It looks like the following (which does not map to layer names in the model):
        # {
        #   "qkv": "per-tensor",
        #   "out": "per-token",
        #   "fc1": "per-tensor",
        #   "fc2": "per-token"
        # }
        #
        # So, this is a hack for llama now. But with the SparseMLConfig, we can make robust,
        # where we actually use the layer_name in the model to look up what the format is
        # based on the config.
        #
        # What it would actually look like:
        # layer_config is None
        # for supported_key in SUPPORTED_LAYER_KEYS:
        #     if supported_key in layer_name:
        #         sq_format = self.layer_mapping[lookup_key]
        #           return get_sq_format_cls(sq_format)()

        HACKED_REMAP_FOR_LLAMA = {
            "qkv": "qkv",
            "o_proj": "out",
            "gate_up": "fc1",
            "down": "fc2",
        }

        for match_key, lookup_key in HACKED_REMAP_FOR_LLAMA.items():
            if match_key in layer_name:
                sq_format = self.sq_config.layer_format_map[lookup_key]
                return get_sq_format_cls(sq_format)()

        raise ValueError

    def create_weights(self, layer_name: str, input_size_per_partition: int,
                       output_sizes_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        del input_size, output_size

        # Statically Quantized Weights.
        weight = Parameter(
            torch.empty(
                sum(output_sizes_per_partition),
                input_size_per_partition,
                device="cuda",
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })

        if len(output_sizes_per_partition) == 1:
            # Single static scale for the entire tensor.
            dequant_scale = Parameter(torch.empty((1),
                                                  device='cuda',
                                                  dtype=params_dtype),
                                      requires_grad=False)
        else:
            # Static scale for each logical weight (e.g. 3 for QKV).
            dequant_scale = Parameter(torch.empty(
                (sum(output_sizes_per_partition)),
                device='cuda',
                dtype=params_dtype),
                                      requires_grad=False)
            set_weight_attrs(
                dequant_scale, {
                    "shard_splitter": self.scales_shard_splitter,
                    "logical_widths": output_sizes_per_partition
                })

        return {
            "weight": weight,
            "dequant_scale": dequant_scale,
            "logical_widths": output_sizes_per_partition,
            "sq_format": self.get_layer_format(layer_name)
        }

    def _quantize(
        self, x: torch.Tensor, sq_format: SmoothQuantFormat
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Quantize activations.

        Args:
            x: Activation at floating point precision.
        Returns:
            x_q: Quantized activation at INT8
            activation_scales: Optional dynamic scales for each token.
        """
        x_q = torch.empty_like(x, dtype=torch.int8, device="cuda")
        x_q, activation_scales = sq_format.quantize_op(x, x_q)
        return x_q, activation_scales

    def apply_weights(self,
                      weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward method. Computes Q --> GEMM --> DQ.

        Args:
            weigths: Dictionary of weights, scales, and metadata.
            x: Input in floating point precision.
            bias: Optional bias.
        Returns:
            a_dq: Dequantized activation at floating point precision.
        """
        if bias is not None:
            raise NotImplementedError
        weight_q = weights["weight"]
        static_scales = weights["dequant_scale"]
        sq_format = weights["sq_format"]

        # Q
        x_q, activation_scales = self._quantize(x, sq_format)

        # GEMM and DQ
        return cutlass_gemm_dq(x_q, weight_q, x.dtype, static_scales,
                               activation_scales)

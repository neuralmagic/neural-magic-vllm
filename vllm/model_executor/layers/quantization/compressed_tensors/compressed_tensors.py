import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

from vllm.model_executor.layers.quantization.compressed_tensors.cutlass_gemm import (
    cutlass_gemm_dq
)

class CompressedTensorsConfig(QuantizationConfig):

    def __init__(self, layer_quant_details: Dict[str, Any], ignore: List[str],
                 fake_quant: bool):
        self.fake_quant = fake_quant
        self.ignore = ignore
        self.layer_quant_details = layer_quant_details
        # per target details; keys are targets, values are
        # activations and weight attributes
        self._config_to_schema_mapping = {}

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    # Needed to v
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float32]

    # Need to figure it out
    def get_min_capability(self) -> int:
        return 60

    # Need a smart way to store the combinations
    @property
    def config_to_schema_mapping(self):
        return self._config_to_schema_mapping

    def get_name(self) -> str:
        return "compressed_tensors"

    # parse through the quantization_config keys
    # store each layer name in targets
    # For each target; get the relevant information
    # (static, num_bits, symmetric, strategy)
    # Remove anything in ignore
    # Go from layer details to the particular scheme method using
    # a separate get_layer_format method
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        layer_quant_details: Dict[str:Any] = dict()
        ignore = config.get("ignore")
        fake_quant = config.get("format") == "fake_quant"

        print(config["config_groups"])

        for key, quant_config in config["config_groups"].items():
            weight_attributes = quant_config.get("weights")
            targets = quant_config.get(
                "targets")  # list of layers to target with this configuration
            input_activations = quant_config.get("input_activations")
            for target in targets:
                layer_quant_details[target] = {}
                layer_quant_details[target]["weight"] = weight_attributes
                layer_quant_details[target]["act"] = input_activations

        return cls(layer_quant_details=layer_quant_details,
                   ignore=ignore,
                   fake_quant=fake_quant)

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["config.json"]

    # Symmetric = no zero point
    def _get_schema(self, weight_quant: Dict, act_quant: Dict):
        return None

    def get_scheme(self, layer: torch.nn.Module):
        if layer in self.ignore:
            return None

        for target in self.layer_quant_details:
            # Probably need something smarter than this?
            if target.lower() in layer.__class__.__name__.lower():
                weight_quant = self.layer_quant_details[target]["weight"]
                act_quant = self.layer_quant_details[target]["act"]

                try:
                    return self._get_schema(weight_quant, act_quant)
                except NotImplementedError as e:
                    raise e


class CompressedTensorsLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config

    # Fetch the appropriate schema based on the layer name
    # Create weights using the scheme
    def create_weights(self, layer: torch.nn.Module, layer_name: str,
                       input_size_per_partition: int,
                       output_sizes_per_partition: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype):

        scheme = self.quantization_config.get_scheme(layer)
        weights = dict()

        if layer_name not in self.quantization_config.ignore:

            is_tensor_partitioned = len(output_sizes_per_partition) != 1
            dim = sum(output_sizes_per_partition) if is_tensor_partitioned else 1

            input_scale = Parameter(torch.empty(dim,
                                                device="cuda",
                                                dtype=torch.float32),
                                    requires_grad=False)
            input_zero_point = Parameter(torch.empty(1,
                                                     device="cuda",
                                                     dtype=torch.int8),
                                         requires_grad=False)

            weight_scale = Parameter(torch.empty(dim,
                                                 device="cuda",
                                                 dtype=torch.float32),
                                     requires_grad=False)
            weight_zero_point = Parameter(torch.empty(1,
                                                      device="cuda",
                                                      dtype=torch.int8),
                                          requires_grad=False)

            if is_tensor_partitioned:
                set_weight_attrs(input_scale,
                                 {"shard_splitter" : self.scales_shard_splitter,
                                  "logical_widths" : output_sizes_per_partition})
                set_weight_attrs(weight_scale,
                                 {"shard_splitter" : self.scales_shard_splitter,
                                  "logical_widths" : output_sizes_per_partition})

            weights["input_scale"] = input_scale
            weights["input_zero_point"] = input_zero_point

            weights["weight_scale"] = weight_scale
            weights["weight_zero_point"] = weight_zero_point

        weight = Parameter(torch.empty(sum(output_sizes_per_partition),
                                       input_size_per_partition,
                                       device="cuda",
                                       dtype=params_dtype),
                           requires_grad=False)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        weights["scheme"] = scheme
        weights["weight"] = weight
        weights["layer_name"] = layer_name
        weights["output_size"] = output_size

        weights["logical_widths"] = output_sizes_per_partition
        return weights

    def shard_id_as_int(self, shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id

        assert isinstance(shard_id, str)
        qkv_idxs = { "q": 0, "k": 1, "v": 2 }
        assert shard_id in qkv_idxs
        return qkv_idxs[shard_id]

    def scales_shard_splitter(self,
                              param: torch.Tensor,
                              loaded_weight: torch.Tensor,
                              shard_id: Union[str, int],
                              logical_widths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_id = self.shard_id_as_int(shard_id)
        offset = sum(logical_widths[:shard_id]) 
        size = logical_widths[shard_id]
        # update loaded weight with copies for broadcast.
        loaded_weight = loaded_weight.repeat(size)
        return param[offset : offset + size], loaded_weight

    def _quantize(self, x: torch.Tensor, scales: torch.Tensor,
                  logical_widths: List[int], split_dim : int = 0) -> torch.Tensor:

        assert (torch.numel(scales) == len(logical_widths))
        x_q = torch.empty_like(x, dtype=torch.int8, device="cuda")
        x_q_split = x_q.split(logical_widths, dim=split_dim)
        x_split = x.split(logical_widths, dim=split_dim)

        for q, dq, scale in zip(x_q_split, x_split, scales):
            ops.quant(q, dq, scale.item())

        return x_q

    # Apply weights using the scheme; schema should be one of the inputs
    # to the function
    def apply_dense(self, weight, input):
        return torch.matmul(input, torch.transpose(weight, 0, 1))

    def apply_weights(self,
                      weights: Dict,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None):

        weight_dq = weights.get("weight")
        weight_scale = weights.get("weight_scale")
        act_scale = weights.get("input_scale")
        logical_widths = weights.get("logical_widths")
        layer_name = weights.get("layer_name")

        if layer_name in self.quantization_config.ignore:
            return self.apply_dense(weight_dq, x)

        # Input quantize
        x_scales = torch.FloatTensor([act_scale[0].item()], device=torch.device("cpu"))
        x_q = self._quantize(x, x_scales, [x.shape[0]])

        # Weight quantize
        # TODO : try not to remove device-to-host copy. i.e. keep the non-duplicated version
        # of scales in the CPU
        w_scales = [weight_scale[sum(logical_widths[:i])].item() for i in range(len(logical_widths))]
        w_scales = torch.FloatTensor(w_scales, device=torch.device("cpu"))
        w_q = self._quantize(weight_dq, w_scales, logical_widths)

        # GEMM and dq
        return cutlass_gemm_dq(x_q, w_q, x.dtype, weight_scale, act_scale)

import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


# Why is this needed?
class Int8GEMM(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        if not hasattr(self, "i8cugemm"):
           # self.i8cugemm = ops.I8CUGEMM()
           pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(Int8GEMM, "_instance"):
            with Int8GEMM._instance_lock:
                if not hasattr(Int8GEMM, "_instance"):
                    Int8GEMM._instance = object.__new__(cls)
        return Int8GEMM._instance

    def get_i8cugemm(self):
        pass
        #return self.i8cugemm


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
        return [torch.float16]

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

        print("INSIDE from_config")
        return cls(layer_quant_details=layer_quant_details,
                   ignore=ignore,
                   fake_quant=fake_quant)

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["config.json"]

    # Symmetric = no zero point
    def _get_schema(self, weight_quant: Dict, act_quant: Dict):
        return None

    # Take the layer name and get the appropriate scheme
    # Have to first map target to linear name
    # Then use attributes to determine the scheme mapping
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

        # What happens to the rest of the layers?
        # Are only linear layers passed in through this loading format?


class CompressedTensorsLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config
        self.i8cugemm = Int8GEMM().get_i8cugemm()

    # Fetch the appropriate schema based on the layer name
    # Create weights using the scheme
    def create_weights(self, layer: torch.nn.Module, layer_name: str,
                       input_size_per_partition: int,
                       output_sizes_per_partition: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype):

        scheme = self.quantization_config.get_scheme(layer)
        weights = dict()

        if layer_name not in self.quantization_config.ignore:
            # From the state dict
            input_scale = Parameter(
                torch.empty(1, device="cuda", dtype=torch.float32), requires_grad=False)
            input_zero_point = Parameter(
                torch.empty(1, device="cuda", dtype=torch.int8), requires_grad=False)

            weight_scale = Parameter(
                torch.empty(1, device="cuda", dtype=torch.float32), requires_grad=False)
            weight_zero_point = Parameter(
                torch.empty(1, device="cuda", dtype=torch.int8), requires_grad=False)
            
            weights["input_scale"] = input_scale
            weights["input_zero_point"] = input_zero_point

            weights["weight_scale"] = weight_scale
            weights["weight_zero_point"] = weight_zero_point

        # Why is it this way (dim1/dim0 swapped?)
        weight = Parameter(torch.empty(sum(output_sizes_per_partition),
                                       input_size_per_partition,
                                       device="cuda",
                                       dtype=params_dtype),
                           requires_grad=False)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        weights["scheme"] = scheme
        weights["weight"] = weight

        weights["logical_widths"] = output_sizes_per_partition

        return weights

    def scales_shard_splitter(
            self, param: torch.Tensor, loaded_weight: torch.Tensor,
            shard_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index into param for for loading.
        
        This function is called by QKVColumnLinear and MergedColumnParallelLinear
        during weight loading to put the scales from disk in the right spot.
        """
        if type(shard_id) == str:
            qkv_idxs = {"q": 0, "k": 1, "v": 2}
            if shard_id not in qkv_idxs:
                raise ValueError(f"Invalid shard_id {shard_id}")
            shard_id = qkv_idxs[shard_id]
        elif type(shard_id) != int:
            raise ValueError(f"Invalid shard id {shard_id}")

        return param[shard_id], loaded_weight

    def _quantize(self, weight: torch.Tensor, weight_scale: torch.Tensor,
                  x: torch.Tensor, act_scale: torch.Tensor):
        weight_q = torch.empty_like(weight, dtype=torch.int8)
        ops.quant(weight_q, weight, weight_scale)

        x_q = torch.empty_like(x, dtype=torch.int8)
        ops.quant(x_q, x, act_scale)
        return weight_q, x_q

    def _dequantize(self, x_q: torch.Tensor, logical_widths: List[int],
                    dtype: torch.dtype, act_weight_scale: torch.Tensor):
                    
        x_q_split = x_q.split(logical_widths, dim=-1)
        x_dq = torch.empty_like(x_q, dtype=dtype)
        x_dq_split = x_dq.split(logical_widths, dim=-1)

        for xdq, xq in zip(x_dq_split, x_q_split):
            ops.dequant(xdq, xq, act_weight_scale)
        return x_dq

    # Apply weights using the scheme; schema should be one of the inputs
    # to the function
    def apply_weights(self,
                      weights: Dict,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None):
        # x: inputs at floating points

        weight_dq = weights.get("weight")
        weight_scale = weights.get("weight_scale")

        act_scale = weights.get("input_scale")

        logical_widths = weights.get("logical_widths")

        weight_q, x_q = self._quantize(weight=weight_dq,
                                       weight_scale=weight_scale,
                                       x=x,
                                       act_scale=act_scale)

        act_weight_scale = weight_scale * act_scale  # A16out = A8Out * (sw * sa)

        x_q = x_q.view(-1, x_q.shape[-1])
        # empty tensor to store the output from the gem
        a_q = torch.empty((x_q.shape[0], weight_q.shape[0]),
                          dtype=torch.int32,
                          device="cuda")
        self.i8cugemm.linear_a8_w8_o32_(x_q, weight_q, a_q)
        a_q = a_q.view(*x_q.shape[:-1], -1)

        return self._dequantize(a_q,
                                logical_widths=logical_widths,
                                dtype=x.dtype,
                                act_weight_scale=act_weight_scale)

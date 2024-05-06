from typing import Any, Dict, Iterable, List, Optional

import torch
#from compressed_tensors.quantization.lifecycle.apply import (
#    find_first_name_or_class_match) # TODO: needed
from compressed_tensors.quantization.quant_args import QuantizationStrategy

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.data import (
    QuantizationFields)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme, CompressedTensorsUnquantized,
    CompressedTensorsW8A8DynamicToken, CompressedTensorsW8A8StaticTensor)


class CompressedTensorsConfig(QuantizationConfig):

    def __init__(self, layer_quant_details: Dict[str, Any], ignore: List[str],
                 fake_quant: bool):
        self.fake_quant = fake_quant
        self.ignore = ignore
        self.layer_quant_details = layer_quant_details

        self.num_bits = QuantizationFields.num_bits.value
        self.strategy = QuantizationFields.strategy.value
        self.symmetric = QuantizationFields.symmetric.value
        self.dynamic = QuantizationFields.dynamic.value

        llama_mapping = {
            "q_proj": "qkv_proj",
            "k_proj": "qkv_proj",
            "v_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj"
        }

        # Update the ignore list: layer with q_proj are replaced to be qkv_proj
        # drop duplicates?
        for layer in self.ignore:
            for k in llama_mapping:
                if k in layer:
                    layer.replace(k, llama_mapping.get(k, k))

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float32, torch.int8]

    # Need to figure it out
    def get_min_capability(self) -> int:
        return 60

    def get_name(self) -> str:
        return "compressed_tensors"

    def get_quant_method(
            self, layer: torch.nn.Module
    ) -> Optional["CompressedTensorsLinearMethod"]:
        if isinstance(layer, LinearBase):
            return CompressedTensorsLinearMethod(self)
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        config = config["compression_config"]["quantization_config"]

        layer_quant_details: Dict[str, Any] = dict()
        ignore: List[str] = config.get("ignore", None)
        fake_quant: bool = config.get("format") == "fakequant"

        for key, quant_config in config["config_groups"].items():
            targets = quant_config.get("targets")
            for target in targets:
                layer_quant_details[target] = {}
                layer_quant_details[target]["weight"] = quant_config.get(
                    "weights")
                layer_quant_details[target]["input"] = quant_config.get(
                    "input_activations")

        return cls(layer_quant_details=layer_quant_details,
                   ignore=ignore,
                   fake_quant=fake_quant)

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_config.json"]

    def _is_static_tensor_w8a8(self, weight_quant: Dict, input_quant: Dict):
        is_8_bits = weight_quant.get(self.num_bits) == input_quant.get(
            self.num_bits) == 8
        is_tensor = weight_quant.get(self.strategy) == input_quant.get(
            self.strategy) == QuantizationStrategy.TENSOR.value
        is_symmetric = weight_quant.get(self.symmetric) and input_quant.get(
            self.symmetric)
        is_static = not weight_quant.get(self.dynamic) and not input_quant.get(
            self.dynamic)

        if is_8_bits and is_tensor and is_symmetric and is_static:
            return True
        return False

    def _is_dynamic_token_w8a8(self, weight_quant: Dict, input_quant: Dict):
        is_8_bits = weight_quant.get(self.num_bits) == input_quant.get(
            self.num_bits) == 8
        is_token_tensor = (weight_quant.get(self.strategy)
                           == QuantizationStrategy.TENSOR.value) and (
                               input_quant.get(self.strategy) == "token"
                           )  # TODO: QuantizationStrategy should have token
        is_symmetric = weight_quant.get(self.symmetric) and input_quant.get(
            self.symmetric)
        is_dynamic = not weight_quant.get(self.dynamic) and input_quant.get(
            self.dynamic)

        if is_8_bits and is_token_tensor and is_symmetric and is_dynamic:
            return True
        return False

    def _get_schema(self, weight_quant: Dict, input_quant: Dict):
        if self._is_static_tensor_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8StaticTensor(
                fake_quant=self.fake_quant)

        elif self._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8DynamicToken(
                fake_quant=self.fake_quant)

        raise NotImplementedError("Scheme not supported.")

    def find_first_name_or_class_match(
            self,
            name: str,
            module: torch.nn.Module,
            targets: Iterable[str],
            check_contains: bool = False) -> Optional[str]:
        # first element of targets that matches the given name
        # if no name matches returns first target that matches the class name
        # returns None otherwise
        return self._find_first_match(name, targets) or self._find_first_match(
            module.__class__.__name__, targets, check_contains)

    def _find_first_match(self,
                          value: str,
                          targets: Iterable[str],
                          check_contains: bool = False) -> Optional[str]:
        # returns first element of target that matches value either
        # exactly or as a regex after 're:'. if check_contains is set to True,
        # additionally checks if the target string is contained with value.
        import re

        for target in targets:
            if target.startswith("re:"):
                pattern = target[3:]
                if re.match(pattern, value):
                    return target
            elif check_contains:
                if target.lower() in value.lower():
                    return target
            elif target == value:
                return target
        return None

    def get_scheme(
            self,
            layer: torch.nn.Module,
            layer_name: Optional[str] = None) -> "CompressedTensorsScheme":

        if layer_name is None:
            raise ValueError(
                "layer_name must be provided for CompressedTensorsConfig")

        if layer_name in self.ignore:
            return CompressedTensorsUnquantized()

        # TODO: update/map layer_name for llama models before
        # using find_first_name_or_class_match?
        layer_type_name = self.find_first_name_or_class_match(
            name=layer_name,
            module=layer,
            targets=self.layer_quant_details.keys(),
            check_contains=True)

        if layer_type_name is None:
            raise ValueError(f"Could not matching target for layer {layer}")

        layer_quant_details: Dict[str, Any] = self.layer_quant_details.get(
            layer_type_name, None)
        if layer_quant_details is None:
            raise ValueError(
                f"Could not find quantization details for {layer_name}.")
        try:
            return self._get_schema(weight_quant=layer_quant_details["weight"],
                                    input_quant=layer_quant_details["input"])
        except NotImplementedError as e:
            raise e


class CompressedTensorsLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config

    def create_weights(self,
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int],
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       layer_name: Optional[str] = None,
                       **extra_weight_attrs):
        """
        Use the CompressedTensorsScheme associated with each layer to create 
        the necessary parameters for the layer.
        """
        weight_loader = extra_weight_attrs.get("weight_loader")

        scheme = self.quantization_config.get_scheme(layer=layer,
                                                     layer_name=layer_name)
        scheme.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader)

        layer.scheme = scheme

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None):
        """
        Use the output of create_weights and the CompressedTensorsScheme 
        associated with the layer to apply the forward pass with the 
        layer input.
        """

        if bias is not None:
            raise ValueError("bias is not supported for this linear method")

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x)

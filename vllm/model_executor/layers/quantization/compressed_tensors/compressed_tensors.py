from typing import Any, Dict, List, Optional

import torch
from compressed_tensors.quantization.lifecycle.apply import (
    find_first_name_or_class_match)
from compressed_tensors.quantization.quant_args import (QuantizationArgs,
                                                        QuantizationStrategy)
from pydantic import BaseModel

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme, CompressedTensorsUnquantized,
    CompressedTensorsW8A8DynamicToken, CompressedTensorsW8A8StaticTensor,
    CompressedTensorsW4A16)


class CompressedTensorsConfig(QuantizationConfig):

    def __init__(self, layer_quant_details: Dict[str, Any], ignore: List[str],
                 fake_quant: bool):
        """
        :param layer_quant_detials: dictionary mapping target layers to
            quantization details
        :param ignore: list of layers that are skipped/unquantized 
        :param fake_quant: if we're running fake quant or not
        """
        self.fake_quant = fake_quant
        self.ignore = ignore
        self.layer_quant_details = layer_quant_details

        llama_mapping = {
            "q_proj": "qkv_proj",
            "k_proj": "qkv_proj",
            "v_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj"
        }

        # Update the ignore list: e.g layers with q_proj are replaced
        # to be qkv_proj to be compatible with vllm
        for layer in self.ignore:
            for k in llama_mapping:
                if k in layer:
                    layer.replace(k, llama_mapping.get(k, k))

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float32, torch.int8, torch.float16]

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
                layer_quant_details[target][
                    "weight"] = QuantizationArgs.parse_obj(
                        quant_config.get("weights"))
                try:
                    layer_quant_details[target][
                        "input"] = QuantizationArgs.parse_obj(
                            quant_config.get("input_activations"))
                except:
                    layer_quant_details[target]["input"] = None

        return cls(layer_quant_details=layer_quant_details,
                   ignore=ignore,
                   fake_quant=fake_quant)

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        # TODO: this is a workaround for an incompatibility between
        # sparseml and vllm
        return ["quant_config.json"]

    def _is_static_tensor_w8a8(self, weight_quant: BaseModel,
                               input_quant: BaseModel) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        is_tensor = (weight_quant.strategy == input_quant.strategy ==
                     QuantizationStrategy.TENSOR.value)
        is_symmetric = weight_quant.symmetric and input_quant.symmetric
        is_static = not weight_quant.dynamic and not input_quant.dynamic

        if is_8_bits and is_tensor and is_symmetric and is_static:
            return True
        return False

    def _is_dynamic_token_w8a8(self, weight_quant: BaseModel,
                               input_quant: BaseModel) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        is_token_tensor = (weight_quant.strategy
                           == QuantizationStrategy.TENSOR.value) and (
                               input_quant.strategy
                               == QuantizationStrategy.TOKEN.value)
        is_symmetric = weight_quant.symmetric and input_quant.symmetric
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        if is_8_bits and is_token_tensor and is_symmetric and is_dynamic:
            return True
        return False

    def _is_w4a16(self, weight_quant: BaseModel,
                  input_quant: BaseModel) -> bool:
        input_quant_none = input_quant is None
        is_4_bits = weight_quant.num_bits == 4
        is_symmetric = weight_quant.symmetric
        is_static = not weight_quant.dynamic

        if is_4_bits and input_quant_none and is_symmetric and is_static:
            return True
        return False

    def _get_schema(self, weight_quant: BaseModel,
                    input_quant: BaseModel) -> "CompressedTensorsScheme":
        if self._is_w4a16(weight_quant, input_quant):
            return CompressedTensorsW4A16(strategy=weight_quant.strategy,
                                          group_size=weight_quant.group_size)

        elif self._is_static_tensor_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8StaticTensor(
                fake_quant=self.fake_quant)

        elif self._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8DynamicToken(
                fake_quant=self.fake_quant)

        raise NotImplementedError("Scheme not supported.")

    def get_scheme(
            self,
            layer: torch.nn.Module,
            layer_name: Optional[str] = None) -> "CompressedTensorsScheme":
        """
        Fetch the appropriate scheme based on the values in the config
        for a given layer. Returns CompressedTensorsUnquantized if the layer 
        is in the ignore list. For all other layers, use 
        find_first_name_or_class_match from compressed_tensors to map the 
        layer/layer name to the list of targets defined in the config. 
        The target is then used to fetch the corresponding 
        CompressedTensorsScheme

        :param layer: torch layer
        :param layer_name: name of the layer
        :return: the CompressedTensorsScheme for the layer 
        """

        if layer_name is None:
            raise ValueError(
                "layer_name must be provided for CompressedTensorsConfig")

        if layer_name in self.ignore:
            return CompressedTensorsUnquantized()

        # TODO: update/map layer_name for llama models before
        # using find_first_name_or_class_match?
        layer_type_name = find_first_name_or_class_match(
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
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")

        scheme = self.quantization_config.get_scheme(layer=layer,
                                                     layer_name=layer_name)
        scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            layer_name=layer_name,
            weight_loader=weight_loader)

        layer.scheme = scheme

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None):
        """
        Use the output of create_weights and the CompressedTensorsScheme 
        associated with the layer to apply the forward pass with the 
        layer input.  See LinearMethodBase for param details
        """

        if bias is not None:
            raise ValueError("bias is not supported for this linear method")

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x)

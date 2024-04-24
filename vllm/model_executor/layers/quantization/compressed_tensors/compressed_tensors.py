from typing import Any, Dict, List, Optional

import torch

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8StaticTensor, CompressedTensorsUnquantized,
    CompressedTensorsScheme)


class CompressedTensorsConfig(QuantizationConfig):

    def __init__(self, layer_quant_details: Dict[str, Any], ignore: List[str],
                 fake_quant: bool):
        self.fake_quant = fake_quant
        self.ignore = ignore
        self.layer_quant_details = layer_quant_details

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float32]

    # Need to figure it out
    def get_min_capability(self) -> int:
        return 60

    def get_name(self) -> str:
        return "compressed_tensors"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        layer_quant_details: Dict[str:Any] = dict()
        ignore = config.get("ignore")
        fake_quant = config.get("format") == "fakequant"

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
        return ["config.json"]

    def _get_schema(self, weight_quant: Dict, input_quant: Dict):
        # TODO: Will static vs dynamic be defined in the config?
        # TODO: Expand conditions/break into separate fxs as other
        # schemes are supported

        weight_bit = weight_quant.get("num_bits")
        input_bit = input_quant.get("num_bits")

        weight_strategy = weight_quant.get("strategy")
        input_strategy = input_quant.get("strategy")

        weight_symmetric = weight_quant.get("symmetric")
        input_symmetric = input_quant.get("symmetric")

        is_8_bits = weight_bit == input_bit == 8
        is_tensor = weight_strategy == input_strategy == "tensor"
        is_symmetric = weight_symmetric and input_symmetric

        if is_8_bits and is_tensor and is_symmetric:
            return CompressedTensorsW8A8StaticTensor(
                fake_quant=self.fake_quant)
        raise NotImplementedError(
            "Scheme not supported. Only 8-bit static symmtetric "
            "per tensor quantization is currently supported")

    def get_scheme(self, layer: torch.nn.Module,
                   layer_name: str) -> "CompressedTensorsScheme":
        # TODO: How are layers which are combined in vllm listed in the ignore list?
        if layer_name in self.ignore:
            return CompressedTensorsUnquantized()

        # TODO: need a better matching function; can adapt shared function with sparseml
        layer_type_name = None
        layer_name_class = type(layer).__name__.lower()
        for target in self.layer_quant_details:
            if target.lower() in layer_name_class:
                layer_type_name = target
                break

        layer_quant_details = self.layer_quant_details.get(layer_type_name)
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

    def create_weights(self, layer: torch.nn.Module, layer_name: str,
                       input_size_per_partition: int,
                       output_sizes_per_partition: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype):
        """
        Use the CompressedTensorsScheme associated with each layer to create the 
        necessary parameters for the layer.
        """

        scheme = self.quantization_config.get_scheme(layer=layer,
                                                     layer_name=layer_name)
        weights = scheme.create_weights(
            input_size_per_partition=input_size_per_partition,
            output_sizes_per_partition=output_sizes_per_partition,
            output_size=output_size,
            params_dtype=params_dtype)
        weights["scheme"] = scheme
        return weights

    def apply_weights(self,
                      weights: Dict,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None):
        """
        Use the output of create_weights and the CompressedTensorsScheme associated with 
        the layer to apply the forward pass with the layer input.
        """

        if bias is not None:
            raise ValueError("bias is not supported for this linear method")

        scheme = weights.get("scheme")
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(weights, x)

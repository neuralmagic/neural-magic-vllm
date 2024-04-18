from vllm.model_executor.layers.quantization.base_config import 
    QuantizationConfig

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors import CompressedTensorsW8A8StaticTensor
from vllm.model_executor.layers.quantization.compressed_tensors.data import Strategy
from typing import Dict, Any, List

class CompressedTensorsConfig(QuantizationConfig):
    def __init__(self, layer_quant_details: Dict[str, Any], ignore: List[str], fake_quant: bool):
        self.fake_quant = fake_quant
        self.ignore = ignore
        self.layer_quant_details = layer_quant_details # per target details; keys are targets, values are activations and weight attributes
        self._config_to_schema_mapping = {}
    
    # Need a smart way to store the combinations
    @property
    def config_to_schema_mapping(self):
        return self._config_to_schema_mapping
    
    @classmethod
    def get_name(cls) -> str:
        return "compressed_tensors"

    # parse through the quantization_config keys 
    # store each layer name in targets
    # For each target; get the relevant information (static, num_bits, symmetric, strategy)
    # Remove anything in ignore
    # Go from layer details to the particular scheme method using a separate get_layer_format method
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> CompressedTensorsConfig:
        layer_quant_details: Dict[str: Any] = dict() 
        ignore = config.get("ignore")
        fake_quant = config.get("format") == "fake_quant"

        for key, quant_config in config.items():
            weight_attributes = quant_config.get("weights")
            targets = quant_config.get("targets") # list of layers to target with this configuration
            input_activations = quant_config.get("input_activations")
            for target in targets:
                layer_quant_details[target] = {}
                layer_quant_details[target]["weight"] = weight_attributes
                layer_quant_details[target]["act"] = input_activations

        return cls(layer_quant_details=layer_quant_details, ignore=ignore, fake_quant=fake_quant)
    
    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["config.json"]


    def _get_schema(self, weight_quant: Dict, act_quant: Dict):
        return CompressedTensorsW8A8StaticTensor

    # Take the layer name and get the appropriate scheme
    # Have to first map target to linear name
    # Then use attributes to determine the scheme mapping
    def get_scheme(self, layer_name: str):
        if layer_name in self.ignore:
            return None
        
        for target in self.layer_quant_details:
            # Probably need something smarter than this?
            if target.lower() in layer_name:
                weight_quant = layer_quant_details[target]["weight"]
                act_quant = layer_quant_details[target]["act"]

                try:
                    return self._get_schema(weight_quant, act_quant)
                except NotImplementedError as e:
                    raise e

        # What happens to the rest of the layers?
        # Are only linear layers passed in through this loading format?
        # Nothing about static vs dynamic from the config?


class CompressedTensorsLinearMethod(LinearMethodBase):
    def __init__(quantization_config: CompressedTensorsConfig):
        self.quantization_config = quant_config

    # Fetch the appropriate schema based on the layer name 
    # Create weights using the scheme
    def create_weights(self, layer_name: str, 
        input_size_per_partition: int, 
        output_size_per_partition: List[int], 
        input_size: int, 
        output_size: int,
        params_dtype):

        pass
    
    # Apply weights using the scheme; schema should be one of the inputs
    # to the function
    def apply_weights(self):
        pass
from abc import ABC, abstractmethod
import torch 
from typing import Dict

__all__ = ["CompressedTensorsScheme"]

class CompressedTensorsScheme(ABC):
    """
    Abstract class used to describe the weight creation and forward pass of different
    quantization schemes supported by CompressedTensors.
    """
    @abstractmethod
    def create_weights(self, *args, **kwargs):
        """
        Weight creation for the particular scheme. Inputs to this function 

        """
        raise NotImplementedError
    

    @abstractmethod
    def apply_weights(self, weights: Dict, x: torch.Tensor):
        """
        Run the forward pass for the particular scheme. This is where scheme-specific
        dequant/quant steps/kernels should be applied.

        :param weights: dictionary of weights and other parameters relevant to the
            particular scheme. Corresponds to the output from create_weights
        :param x: input to the layer

        """
        raise NotImplementedError
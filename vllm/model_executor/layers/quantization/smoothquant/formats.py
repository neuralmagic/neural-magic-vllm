from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from vllm._C import ops


class SmoothQuantFormat(ABC):

    @abstractmethod
    def quantize_op(
            self, x: torch.Tensor,
            x_q: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Quantize the input and (optionally compute dequant scales).

        Args:
            x: Input data in floating point format.
            x_q: Buffer for quantized inputs.
        Returns:
            x_q: Quantized input.
            activation_scales: Optional dynamic scales for the activations.
        """
        raise NotImplementedError


class SmoothQuantDynamicPerToken(SmoothQuantFormat):

    def quantize_op(self, x: torch.Tensor,
                    x_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Notes:
        Returns quantized activaiton and dynamic activation scales.
        """
        activation_scales = torch.empty((x.numel() // x.shape[-1], 1),
                                        dtype=x.dtype,
                                        device=x.device)
        ops.quant(x_q, x, activation_scales)
        return x_q, activation_scales


class SmoothQuantStaticPerTensor(SmoothQuantFormat):

    def quantize_op(self, x: torch.Tensor,
                    x_q: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Notes:
        Returns quantized activaiton and no dynamic scales.
        """
        ops.quant(x_q, x, 1.0)
        return x_q, None

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type

import torch

from vllm._C import ops


class SmoothQuantFormat(ABC):
    @abstractmethod
    def dequantize_op(self,
                      x_qs: List[torch.Tensor],
                      x_dqs: List[torch.Tensor],
                      dynamic_scales: Optional[torch.Tensor],
                      static_scales: torch.Tensor) -> None:
        """Dequantize the activations. x_dq is updated in place.

        Args:
            x_qs: List of N quantized activations.
            x_dqs: List of N buffers to fill with dequantized values.
            dynamic_scales: Optional dynamic scales for dequantization.
            static_scales: Static scales for dequantization. N values.
        """
        raise NotImplementedError
    

    @abstractmethod
    def quantize_op(self,
                    x: torch.Tensor,
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
    def dequantize_op(self,
                      x_qs: List[torch.Tensor],
                      x_dqs: List[torch.Tensor],
                      dynamic_scales: Optional[torch.Tensor],
                      static_scales: torch.Tensor) -> None:
        """Notes:
        dynamic_scales: N scales for N tokens in the activation.
        static_scales: K scales for K logical activations (equals just w_scale).
        """
        if dynamic_scales is None:
            raise ValueError
        
        # Dequantize each logical activation.
        # TODO: test this for case when logical_widths > 1 (may need to reshape)
        for x_dq, x_q, dynamic_scale, static_scale in zip(
            x_dqs, x_qs, dynamic_scales, static_scales):
            
            # Dequantize (updates x_dq in place).
            ops.dequant(x_dq, x_q, dynamic_scale, static_scale)


    def quantize_op(self,
                    x: torch.Tensor,
                    x_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Notes:
        Returns quantized activaiton and dynamic activation scales.
        """
        activation_scales = torch.empty(x.numel() // x.shape[-1], dtype=x.dtype, device=x.device)
        ops.quant(x_q, x, activation_scales)
        return x_q, activation_scales
    

class SmoothQuantStaticPerTensor(SmoothQuantFormat):
    def dequantize_op(self,
                      x_qs: List[torch.Tensor],
                      x_dqs: List[torch.Tensor],
                      dynamic_scales: Optional[torch.Tensor],
                      static_scales: torch.Tensor) -> None:
        """Notes:
        dynamic_scales: None
        static_scales: K scales for K logical activations (equals w_scale * a_scale).
        """
        if dynamic_scales is not None:
            raise ValueError
        
        # Dequantize each logical activation.
        for xdq, xq, static_scale in zip(x_dqs, x_qs, static_scales):
            ops.dequant(xdq, xq, static_scale)

    def quantize_op(self,
                    x: torch.Tensor,
                    x_q: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Notes:
        Returns quantized activaiton and no dynamic scales.
        """
        ops.quant(x_q, x, 1.0)
        return x_q, None

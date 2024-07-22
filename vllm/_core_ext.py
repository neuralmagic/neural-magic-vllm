import importlib.util
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
core_C_available = importlib.util.find_spec('._core_C', 'vllm') is not None

if core_C_available:
    try:
        import vllm._core_C # ruff: noqa: F401
    except ImportError as e:
        logger.warning("Failed to import from vllm._core_C with %r", e)

    ScalarType = torch.classes._core_C.ScalarType

elif not TYPE_CHECKING:
    # On platforms were we cannot use/build the C++ core extension (i.e. namely
    # neuron and tpu), we define the mock ScalarType class here that partially
    # mimics the C++ ScalarType class.

    from dataclasses import dataclass

    # See: _custom_classes.pyi for docstrings
    @dataclass
    class ScalarType:
        exponent: int
        mantissa: int
        bias: int
        signed: bool

        finite_values_only: bool = False
        nan_repr: int = 0

        @classmethod
        def s(cls, size_bits: int, bias: int):
            return cls(size_bits - 1, size_bits, bias, True)

        @classmethod
        def u(cls, size_bits: int, bias: int):
            return cls(size_bits, size_bits, bias, False)

        @classmethod
        def f(cls, exponent: int, mantissa: int):
            return cls(exponent, mantissa, 0, True)

        @classmethod
        def fn(cls, exponent: int, mantissa: int, finite_values_only: bool,
               nan_repr: int):
            return cls(exponent, mantissa, 0, True, finite_values_only,
                       nan_repr)

        def size_bits(self):
            return self.exponent + self.mantissa + int(self.signed)

        def is_floating_point(self):
            return self.exponent != 0

        def is_integer(self):
            return self.exponent == 0

        def has_bias(self):
            return self.bias != 0

        def has_infs(self):
            return not self.finite_values_only

        def has_nans(self):
            return self.nan_repr != 0

        def min(self):
            raise NotImplementedError

        def max(self):
            raise NotImplementedError

        def __str__(self):
            raise NotImplementedError

        def __repr__(self):
            raise NotImplementedError

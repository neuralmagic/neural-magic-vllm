from abc import abstractmethod
from typing import Any, Dict, List, Type

import torch
from magic_wand import CompressedStorageFormat

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationConfig


class SparsityConfig(QuantizationConfig):
    """Base class for sparsity configs."""

    @abstractmethod
    def get_storage_format_cls(self) -> Type[CompressedStorageFormat]:
        """Sparse representation format"""
        raise NotImplementedError

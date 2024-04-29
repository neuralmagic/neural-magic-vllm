import importlib.util
from collections import namedtuple
from typing import Type

is_magic_wand_available = importlib.util.find_spec("magic_wand") is not None
if not is_magic_wand_available:
    raise ValueError(
        "magic_wand is not available and required for sparsity "
        "support. Please install it with `pip install nm-magic-wand`")

from vllm.model_executor.layers.sparsity.base_config import (  # noqa: E402
    SparsityConfig)
from vllm.model_executor.layers.sparsity.semi_structured_sparse_w16a16 import (  # noqa: E402
    SemiStructuredSparseW16A16Config)
from vllm.model_executor.layers.sparsity.sparse_w16a16 import (  # noqa: E402
    SparseW16A16Config)

# UPSTREAM SYNC: where we keep the sparsity configs
sparsity_structure_meta = namedtuple('SparsityStructure', ['name', 'config'])

SparsityStructures = dict(
    sparse_w16a16=sparsity_structure_meta("sparse_w16a16", SparseW16A16Config),
    semi_structured_sparse_w16a16=sparsity_structure_meta(
        "semi_structured_sparse_w16a16", SemiStructuredSparseW16A16Config),
)


# UPSTREAM SYNC: needed for sparsity
def get_sparsity_config(
        model_config: "ModelConfig") -> Type[SparsityConfig]:  # noqa: F821
    # fetch the sparsity config from the model config
    sparsity = model_config.sparsity
    if sparsity not in SparsityStructures:
        raise ValueError(
            f"Invalid sparsity method: {sparsity}. "
            f"Available sparsity methods: {list(SparsityStructures.keys())}")
    sparsity_cls = SparsityStructures[sparsity].config
    return sparsity_cls()


__all__ = [
    "SparsityConfig",
    "get_sparsity_config",
]

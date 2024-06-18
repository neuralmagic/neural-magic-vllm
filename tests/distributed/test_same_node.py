import os
import pytest
import torch

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.distributed.parallel_state import is_in_the_same_node

if should_skip_test_group(group_name="TEST_DISTRIBUTED"):
    pytest.skip("TEST_DISTRIBUTED=DISABLE, skipping distributed test group",
                allow_module_level=True)

torch.distributed.init_process_group(backend="gloo")
test_result = is_in_the_same_node(torch.distributed.group.WORLD)

expected = os.environ.get("VLLM_TEST_SAME_HOST", "1") == "1"
assert test_result == expected, f"Expected {expected}, got {test_result}"

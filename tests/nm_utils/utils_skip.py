"""Checks environment variables to skip various test groups.
The functions here are imported by each test file.
The .github/actions/nm-test-skipping-env-setup sets these 
    variables in the testing automation.
"""

import os


def should_skip_accuracy_test_group():
    TEST_ACCURACY = os.getenv("TEST_ACCURACY", "1")
    return TEST_ACCURACY == "0"


def should_skip_async_engine_test_group():
    TEST_ASYNC_ENGINE = os.getenv("TEST_ASYNC_ENGINE", "1")
    return TEST_ASYNC_ENGINE == "0"


def should_skip_basic_correctness_test_group():
    TEST_BASIC_CORRECTNESS = os.getenv("TEST_BASIC_CORRECTNESS", "1")
    return TEST_BASIC_CORRECTNESS == "0"


def should_skip_core_test_group():
    TEST_CORE = os.getenv("TEST_CORE", "1")
    return TEST_CORE == "0"


def should_skip_distributed_test_group():
    TEST_DISTRIBUTED = os.getenv("TEST_DISTRIBUTED", "1")
    return TEST_DISTRIBUTED == "0"


def should_skip_engine_test_group():
    TEST_ENGINE = os.getenv("TEST_ENGINE", "1")
    return TEST_ENGINE == "0"


def should_skip_entrypoints_test_group():
    TEST_ENTRYPOINTS = os.getenv("TEST_ENTRYPOINTS", "1")
    return TEST_ENTRYPOINTS == "0"


def should_skip_kernels_test_groups():
    TEST_KERNELS = os.getenv("TEST_KERNELS", "1")
    return TEST_KERNELS == "0"


def should_skip_lora_test_group():
    TEST_LORA = os.getenv("TEST_LORA", "1")
    return TEST_LORA == "0"


def should_skip_metrics_test_group():
    TEST_METRICS = os.getenv("TEST_METRICS", "1")
    return TEST_METRICS == "0"


def should_skip_model_executor_test_group():
    TEST_MODEL_EXECUTOR = os.getenv("TEST_MODEL_EXECUTOR", "1")
    return TEST_MODEL_EXECUTOR == "0"


def should_skip_models_test_group():
    TEST_MODELS = os.getenv("TEST_MODELS", "0")
    return TEST_MODELS != "1"


def should_skip_models_core_test_group():
    TEST_MODELS_CORE = os.getenv("TEST_MODELS_CORE", "0")
    return TEST_MODELS_CORE != "1"


def should_skip_prefix_caching_test_group():
    TEST_PREFIX_CACHING = os.getenv("TEST_PREFIX_CACHING", "0")
    return TEST_PREFIX_CACHING != "1"


def should_skip_quantization_test_group():
    TEST_QUANTIZATION = os.getenv("TEST_QUANTIZATION", "0")
    return TEST_QUANTIZATION != "1"


def should_skip_samplers_test_group():
    TEST_SAMPLERS = os.getenv("TEST_SAMPLERS", "0")
    return TEST_SAMPLERS != "1"


def should_skip_spec_decode_test_group():
    TEST_SPEC_DECODE = os.getenv("TEST_SPEC_DECODE", "0")
    return TEST_SPEC_DECODE != "1"


def should_skip_tensorizer_loader_test_group():
    TEST_TENSORIZER_LOADER = os.getenv("TEST_TENSORIZER_LOADER", "0")
    return TEST_TENSORIZER_LOADER != "1"


def should_skip_tokenization_test_group():
    TEST_TOKENIZATION = os.getenv("TEST_TOKENIZATION", "0")
    return TEST_TOKENIZATION != "1"


def should_skip_worker_test_group():
    TEST_WORKER = os.getenv("TEST_WORKER", "0")
    return TEST_WORKER != "1"


MAP = {
    "TEST_ACCURACY": should_skip_accuracy_test_group,
    "TEST_ASYNC_ENGINE": should_skip_async_engine_test_group,
    "TEST_BASIC_CORRECTNESS": should_skip_basic_correctness_test_group,
    "TEST_CORE": should_skip_core_test_group,
    "TEST_DISTRIBUTED": should_skip_distributed_test_group,
    "TEST_ENGINE": should_skip_engine_test_group,
    "TEST_ENTRYPOINTS": should_skip_entrypoints_test_group,
    "TEST_KERNELS": should_skip_kernels_test_groups,
    "TEST_LORA": should_skip_lora_test_group,
    "TEST_METRICS": should_skip_metrics_test_group,
    "TEST_MODELS": should_skip_models_test_group,
    "TEST_MODELS_CORE": should_skip_models_core_test_group,
    "TEST_PREFIX_CACHING": should_skip_prefix_caching_test_group,
    "TEST_QUANTIZATION": should_skip_quantization_test_group,
    "TEST_SAMPLERS": should_skip_samplers_test_group,
    "TEST_SPEC_DECODE": should_skip_spec_decode_test_group,
    "TEST_TENSORIZER_LOADER": should_skip_tensorizer_loader_test_group,
    "TEST_TOKENIZATION": should_skip_tokenization_test_group,
    "TEST_WORKER": should_skip_worker_test_group,
}


def should_skip_test_group(group_name: str) -> bool:
    return MAP[group_name]()

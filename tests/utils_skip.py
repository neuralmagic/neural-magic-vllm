"""Checks environment variables to skip various test groups.

The functions here are imported by each test file.
The .github/actions/nm-test-skipping-env-setup sets these 
    variables in the testing automation.
"""

import os


def should_skip_kernel_test_group():
    TEST_KERNELS = os.getenv("TEST_KERNELS", "0")
    return TEST_KERNELS != "1"


def should_skip_lora_test_group():
    TEST_LORA = os.getenv("TEST_LORA", "0")
    return TEST_LORA != "1"


def should_skip_spec_decode_test_group():
    TEST_SPEC_DECODE = os.getenv("TEST_SPEC_DECODE", "0")
    return TEST_SPEC_DECODE != "1"


def should_skip_models_test_group():
    TEST_ALL_MODELS = os.getenv("TEST_ALL_MODELS", "0")
    return TEST_ALL_MODELS != "1"


def should_skip_lm_eval_test_group():
    TEST_LM_EVAL = os.getenv("TEST_LM_EVAL", "0")
    return TEST_LM_EVAL != "1"


def should_skip_tensorizer_test_group():
    TEST_TENSORIZER = os.getenv("TEST_TENSORIZER", "0")
    return TEST_TENSORIZER != "1"
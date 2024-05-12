import os

def should_skip_kernel_test_group():
    TEST_KERNELS = os.getenv("TEST_KERNELS", "0")
    return TEST_KERNELS != "1"

def should_skip_lora_test_group():
    TEST_LORA = os.getenv("TEST_LORA", "0")
    return TEST_LORA != "1"

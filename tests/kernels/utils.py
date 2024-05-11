import os

def should_skip_test_group():
    TEST_KERNELS = os.getenv("TEST_KERNELS", 0)
    return TEST_KERNELS == 1

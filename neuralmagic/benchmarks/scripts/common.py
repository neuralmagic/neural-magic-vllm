"""
Common functions used in all benchmarking scripts
"""

def get_bench_environment() -> dict:
    """
    Return the current python version, pytorch version and CUDA version as a dict
    """
    import sys
    import torch
    return {
        "python_version" :  f"{sys.version}",
        "torch_version" : f"{torch.__version__}",
        "torch_cuda_version" : f"{torch.version.cuda}",
        "cuda_device(0)": f"{torch.cuda.get_device_properties(0)}"
    }
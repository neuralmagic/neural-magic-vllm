from .benchmark_result import BenchmarkResult

def short_description(result_json:dict) -> None:
    """
    Given a result_json, that is the JSON version for some
    BenchmarkResult object, return a string that captures a few key high
    level information like the user given benchmark description, GPU name etc.
    """

    # TODO (varun) : Hoist "cuda_device_names" to a string
    gpu_names = self.result_dict[
        BenchmarkResult.BENCHMARKING_CONTEXT_KEY_]["cuda_device_names"]
    gpu_name = gpu_names[0]
    num_gpus_used = json_data[BenchmarkResult.TENSOR_PARALLEL_SIZE_KEY_]
    # Make sure all gpus are the same before we report
    assert all(map(lambda x: x == gpu_name, gpu_names[:num_gpus_used]))

    return f"{self.result_dict[self.DESCRIPTION_KEY_]}\n GPU: {gpu_name} x {num_gpus_used}" 

def long_description(self) -> None:
    """
    Given a result_json, that is the JSON version for some
    BenchmarkResult object, eeturn a string that is fully-descriptive of this benchmark run.
    """
    pass (f"{self.result_dict[self.DESCRIPTION_KEY_]}\n" 
          f"context : {self.result_dict[BenchmarkResult.BENCHMARKING_CONTEXT_KEY_]}\n"
          f"script name : {self.result_dict[BenchmarkResult.SCRIPT_NAME_KEY_]}\n"
          f"script args : {self.result_dict[BenchmarkResult.SCRIPT_ARGS_KEY_]}")

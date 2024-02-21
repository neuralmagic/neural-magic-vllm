import argparse

from pathlib import Path
from neuralmagic.benchmarks.common import benchmark_configs
from neuralmagic.benchmarks import (
    run_benchmark_serving_script,
    run_benchmark_throughput_script,
    run_benchmark_prefill_decode_throughput_script
)

def run(config_file_path: Path, output_directory: Path) -> None:

    for config in benchmark_configs(config_file_path):
        if config.script_name == "benchmark_serving":
            run_benchmark_serving_script(config, output_directory)
            continue

        if config.script_name == "benchmark_throughput.py":
            run_benchmark_throughput_script(config, output_directory)
            continue

        if config.script_name == "benchmark_prefill_decode_throughput.py":
            run_benchmark_prefill_decode_throughput_script(
                config, output_directory)
            continue

        raise ValueError(f"Unhandled benchmark script {config.script_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs benchmark-scripts as a subprocess")
    parser.add_argument(
        "-i",
        "--input-config-file",
        required=True,
        type=str,
        help="Path to the input config file describing the benhmarks to run",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        required=True,
        type=str,
        help="Path to a directory that is the output store",
    )

    args = parser.parse_args()
    run(Path(args.input_config_file), Path(args.output_directory))

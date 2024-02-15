import argparse
import json

from argparse import Namespace
from pathlib import Path

from run_benchmark_serving import run_benchmark_serving_script
from run_benchmark_throughput import run_benchmark_throughput_script

def run(config_file_path: Path, output_directory: Path) -> None:
    assert config_file_path.exists()

    configs = None
    with open(config_file_path, "r") as f:
        configs = json.load(f, object_hook=lambda d: Namespace(**d))
    assert configs is not None

    for config in configs.configs:
        if config.script_name == "benchmark_serving.py":
            run_benchmark_serving_script(config, output_directory)
            continue

        if config.script_name == "benchmark_throughput.py":
            run_benchmark_throughput_script(config, output_directory)
            continue

        raise ValueError(f"Unhandled benchmark script f{config.script_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs benchmark-scripts as a subprocess"
    )
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

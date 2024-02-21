import argparse
from pathlib import Path
from typing import NamedTuple, Optional

from neuralmagic.tools.call_cmd import call_cmd
from common import download_datasets, script_args_to_cla, benchmark_configs


def get_this_script_dir() -> Path:
    return Path(__file__).parent.resolve()


def run_benchmark_throughput_script(config: NamedTuple,
                                    output_directory: Optional[Path] = None
                                    ) -> None:

    assert config.script_name == 'benchmark_throughput.py'

    # Process config.download_dataset_cmds
    download_datasets(config)

    script_path = get_this_script_dir() / f"scripts/{config.script_name}"

    for model in config.models:
        for script_args in script_args_to_cla(config):
            bench_cmd = (["python3", f"{script_path}"] + script_args +
                         ["--model", f"{model}"] + ["--tokenizer", f"{model}"])

            if output_directory:
                bench_cmd = bench_cmd + [
                    "--save-directory", f"{output_directory}"
                ]

            call_cmd(bench_cmd, stdout=None, stderr=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Runs the benchmark_serving.py script as a subprocess")
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
        type=str,
        default=None,
        help="Path to a directory that is the output store",
    )

    args = parser.parse_args()

    output_directory = Path(
        args.output_directory) if args.output_directory is not None else None

    for config in benchmark_configs(Path(args.input_config_file)):
        run_benchmark_throughput_script(config, output_directory)

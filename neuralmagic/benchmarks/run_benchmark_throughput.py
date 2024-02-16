from pathlib import Path
from typing import NamedTuple, Optional

from neuralmagic.tools.call_cmd import call_cmd
from common import download_datasets, script_args_to_cla

def get_this_script_dir() -> Path:
    return Path(__file__).parent.resolve()

def run_benchmark_throughput_script(config:NamedTuple, output_directory:Optional[Path] = None) -> None:

    assert config.script_name == 'benchmark_throughput.py'

    # Process config.download_dataset_cmds
    download_datasets(config)

    script_path = get_this_script_dir() / f"scripts/{config.script_name}"

    for model in config.models:
        for script_args in script_args_to_cla(config):
            bench_cmd = (
                ["python3", f"{script_path}"]
                + script_args
                + ["--model", f"{model}"]
                + ["--tokenizer", f"{model}"]
            )

            if output_directory:
                bench_cmd = bench_cmd + ["--save-directory", f"{output_directory}"]

            call_cmd(bench_cmd, stdout=None, stderr=None)
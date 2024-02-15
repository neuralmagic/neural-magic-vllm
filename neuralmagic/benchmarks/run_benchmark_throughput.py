from pathlib import Path
from typing import NamedTuple

from neuralmagic.tools.call_cmd import call_cmd
from common import download_datasets, script_args_to_cla

def get_this_script_dir() -> Path:
    return Path(__file__).parent.resolve()

def run_benchmark_throughput_script(config:NamedTuple, output_directory:Path) -> None:

    # Process config.download_dataset_cmds
    #download_datasets(config)

    script_path = get_this_script_dir() / f"scripts/{config.script_name}"

    for model in config.models:
        for script_args in script_args_to_cla(config):
            bench_cmd = (
                ["python3", f"{script_path}"]
                + script_args
                + ["--save-directory", f"{output_directory}"]
                + ["--model", f"{model}"]
                + ["--tokenizer", f"{model}"]
            )
            call_cmd(bench_cmd, stdout=None, stderr=None)

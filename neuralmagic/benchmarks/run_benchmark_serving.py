import argparse
import subprocess
import requests
import time

from typing import NamedTuple, Optional
from pathlib import Path

from neuralmagic.tools.call_cmd import call_cmd
from neuralmagic.benchmarks.common import download_model, download_datasets, script_args_to_cla, benchmark_configs

BENCH_SERVER_HOST = "localhost"
BENCH_SERVER_PORT = 9000


def get_this_script_dir() -> Path:
    return Path(__file__).parent.resolve()


def is_server_running(host: str, port: int, timeout=60) -> bool:

    def try_connection() -> bool:
        try:
            r = requests.get(f"http://{host}:{port}/health")
            return r.status_code == 200
        except Exception as _:
            return False

    timeout_part = 15  # retry every 15 seconds
    time_waited = 0
    while time_waited <= timeout:
        time.sleep(timeout_part)
        if try_connection():
            return True
        time_waited = time_waited + timeout_part

    return False


def run_benchmark_serving_script(config: NamedTuple,
                                 output_directory: Optional[Path] = None
                                 ) -> None:

    assert config.script_name == 'benchmark_serving'

    def run_bench(server_cmd: str, bench_cmd: list[str]) -> None:
        try:
            # start server
            server_process = subprocess.Popen("exec " + server_cmd, shell=True)
            if not is_server_running(BENCH_SERVER_HOST, BENCH_SERVER_PORT):
                raise ValueError(
                    f"Aborting bench run with : server-cmd {server_cmd} , bench-cmd {bench_cmd}. Reason: Cannot start Server"
                )
            # run bench
            call_cmd(bench_cmd, stdout=None, stderr=None)
        finally:
            # kill the server
            assert server_process is not None
            server_process.kill()

    # Process config.download_dataset_cmds
    # download_datasets(config)

    script_path = f"neuralmagic.benchmarks.scripts.{config.script_name}"

    for model in config.models:
        for sparsity in config.sparsity:


        server_cmd = f"python3 -m vllm.entrypoints.api_server --model {model} --tokenizer {model} --sparsity {sparsity} --host {BENCH_SERVER_HOST} --port {BENCH_SERVER_PORT} --disable-log-requests"
        for script_args in script_args_to_cla(config):
            bench_cmd = (["python3", "-m" f"{script_path}"] + script_args +
                         ["--model", f"{model}"] +
                         ["--tokenizer", f"{model}"] +
                         ["--port", f"{BENCH_SERVER_PORT}"] +
                         ["--host", f"{BENCH_SERVER_HOST}"])

                if output_directory:
                    bench_cmd = bench_cmd + [
                        "--save-directory", f"{output_directory}"
                    ]

                run_bench(server_cmd, bench_cmd)


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
        run_benchmark_serving_script(config, output_directory)

import argparse
import json
import itertools
import subprocess
import time
import socket

from argparse import Namespace
from pathlib import Path
from typing import NamedTuple, Iterable

from neuralmagic.tools.call_cmd import call_cmd

BENCH_SERVER_HOST = "localhost"
BENCH_SERVER_PORT = 9000


def get_this_script_dir() -> Path:
    return Path(__file__).parent.resolve()


def is_server_running(host: str, port: int, timeout=20) -> bool:
    def try_connection() -> bool:
        try:
            sock = socket.create_connection((host, port))
            sock.close()
            return True
        except Exception as e:
            return False

    retries = 5
    timeout_part = timeout / retries
    while retries:
        time.sleep(timeout_part)
        if try_connection():
            return True
        retries = retries - 1

    return False


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


def script_args_to_cla(kv: dict) -> Iterable[list[str]]:
    # Input kv is a dict of lists. The idea is to provide command line args that is a cartesian product of these lists
    arg_lists = kv.values()
    assert all(map(lambda le: isinstance(le, list), arg_lists))

    keys = kv.keys()
    for args in itertools.product(*arg_lists):
        cla = []
        for name, value in zip(keys, args):
            cla.extend([f"--{name}", f"{value}"])
        yield cla


def run_benchmark_serving_script(config: NamedTuple, output_directory: Path) -> None:
    # download all required datasets
    for download_cmd in config.dataset_download_cmds:
        download_cmd_as_list = list(
            filter(lambda x: len(x) != 0, download_cmd.split(" "))
        )
        call_cmd(download_cmd_as_list, stdout=None, stderr=None)

    script_path = get_this_script_dir() / f"scripts/{config.script_name}"
    script_args_kv = vars(config.script_args)

    for model in config.models:
        server_cmd = f"python3 -m vllm.entrypoints.api_server --model {model} --tokenizer {model} --host {BENCH_SERVER_HOST} --port {BENCH_SERVER_PORT}"
        for script_args in script_args_to_cla(script_args_kv):
            bench_cmd = (
                ["python3", f"{script_path}"]
                + script_args
                + ["--save-directory", f"{output_directory}"]
                + ["--model", f"{model}"]
                + ["--tokenizer", f"{model}"]
                + ["--port", f"{BENCH_SERVER_PORT}"]
                + ["--host", f"{BENCH_SERVER_HOST}"]
            )
            run_bench(server_cmd, bench_cmd)


def run(config_file_path: Path, output_directory: Path) -> None:
    assert config_file_path.exists()

    config = None
    with open(config_file_path, "r") as f:
        config = json.load(f, object_hook=lambda d: Namespace(**d))
    assert config is not None

    if config.script_name == "benchmark_serving.py":
        return run_benchmark_serving_script(config, output_directory)

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

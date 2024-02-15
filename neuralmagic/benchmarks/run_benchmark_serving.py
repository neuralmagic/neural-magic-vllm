import subprocess
import time
import socket

from typing import NamedTuple
from pathlib import Path

from neuralmagic.tools.call_cmd import call_cmd
from common import download_datasets, script_args_to_cla

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
        except Exception as _:
            return False

    retries = 5
    timeout_part = timeout / retries
    while retries:
        time.sleep(timeout_part)
        if try_connection():
            return True
        retries = retries - 1

    return False

def run_benchmark_serving_script(config: NamedTuple, output_directory: Path) -> None:

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
    download_datasets(config)

    script_path = get_this_script_dir() / f"scripts/{config.script_name}"

    for model in config.models:
        server_cmd = f"python3 -m vllm.entrypoints.api_server --model {model} --tokenizer {model} --host {BENCH_SERVER_HOST} --port {BENCH_SERVER_PORT} --disable-log-requests"

        for script_args in script_args_to_cla(config):
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

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryFile
from typing import Dict, List, NamedTuple, Optional

import requests

from ...tests.utils.logging import log_banner, make_logger
from ..tools.call_cmd import call_cmd
from .common import (benchmark_configs, download_model,
                     max_model_length_from_model_id, script_args_to_cla)
from .scripts.common import num_available_gpus, warmup_server

BENCH_SERVER_HOST = "localhost"
BENCH_SERVER_PORT = 9000


class Server:

    def __init__(self, args: Dict, max_ready_wait: int = 600):
        self.logger = make_logger("nm-vllm-server")
        self.cmd = [sys.executable, "-m", "vllm.entrypoints.api_server"]
        for k, v in args.items():
            self.cmd.extend([f"--{k}", str(v)])
        self.max_ready_wait = max_ready_wait
        self.proc = None
        self.output_file = TemporaryFile()

    def __enter__(self):
        log_banner(self.logger, "server startup command", shlex.join(self.cmd))
        self.proc = subprocess.Popen(self.cmd,
                                     stderr=subprocess.STDOUT,
                                     stdout=self.output_file.fileno())
        self._wait_for_server_ready()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.proc and self.proc.poll() is None:
            self.logger.info("killing server")
            self.proc.kill()

        if exc_type is None:
            return  # only log if an exception occurred

        self.output_file.seek(0)
        self.output = self.output_file.read()
        self.output_file.close()

        log_banner(self.logger, "server output", self.output)

    def _wait_for_server_ready(self):
        self.logger.info("waiting for server to become ready")
        start = time.time()
        while time.time() - start < self.max_ready_wait:
            try:
                if requests.get(
                        f"http://{BENCH_SERVER_HOST}:{BENCH_SERVER_PORT}/health",
                        timeout=10).status_code == 200:
                    break
            except Exception as e:
                if self.proc and self.proc.poll() is not None:
                    raise RuntimeError("server exited unexpectedly") from e
                time.sleep(0.5)
        else:
            raise RuntimeError("server failed to start in time")


def get_tensor_parallel_size(config: NamedTuple) -> int:

    num_tp_directives = [
        hasattr(config, 'tensor_parallel_size'),
        hasattr(config, 'use_all_available_gpus')
    ].count(True)
    if num_tp_directives == 0:
        # by default - use just one GPU
        return 1

    # must have exactly one directive
    assert num_tp_directives == 1

    tensor_parallel_size = config.tensor_parallel_size if hasattr(
        config, 'tensor_parallel_size') else num_available_gpus()
    assert tensor_parallel_size > 0 and \
           tensor_parallel_size <= num_available_gpus()
    return tensor_parallel_size


def is_server_running(host: str, port: int, timeout=600) -> bool:

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

    def run_bench(server_cmd: str, bench_cmd: List[str], model: str) -> None:
        with Server(server_cmd):
            # server warmup
            warmup_server(server_host=BENCH_SERVER_HOST,
                          server_port=BENCH_SERVER_PORT,
                          model=model,
                          num_prompts=1000)

            # run bench
            call_cmd(bench_cmd, stdout=None, stderr=None)

    tensor_parallel_size = get_tensor_parallel_size(config)

    script_path = f"neuralmagic.benchmarks.scripts.{config.script_name}"

    sparsities = [None] if len(config.sparsity) == 0 else config.sparsity

    for model, sparsity in itertools.product(config.models, sparsities):

        # download model beforehand so the server can start without any holdup
        download_model(model)

        supported_max_model_len = max_model_length_from_model_id(model)

        # If the requested model-len is too big, try running with the
        # maximum supported for this model.
        max_model_lens = set(
            map(lambda v: min(v, supported_max_model_len),
                config.max_model_lens))
        if (config.max_model_lens != list(max_model_lens)):
            print(f"WARNING: max_model_len modified to {max_model_lens} "
                  f"from {config.max_model_lens} for model {model}")

        for max_model_len in max_model_lens:

            server_args = {
                "model": model,
                "tokenizer": model,
                "max-model-len": max_model_len,
                "host": BENCH_SERVER_HOST,
                "port": BENCH_SERVER_PORT,
                "tensor-parallel-size": tensor_parallel_size,
                "disable-log-requests": ""
            }
            if sparsity:
                server_args["sparsity"] = sparsity

            for script_args in script_args_to_cla(config):

                description = (f"{config.description}\n" +
                               f"model - {model}\n" +
                               f"max-model-len - {max_model_len}\n" +
                               f"sparsity - {sparsity}\n" +
                               f"{config.script_name} " +
                               f"{json.dumps(script_args, indent=2)}")

                bench_cmd = (["python3", "-m"
                              f"{script_path}"] +
                             ["--description", f"{description}"] +
                             ["--model", f"{model}"] +
                             ["--tokenizer", f"{model}"] +
                             ["--port", f"{BENCH_SERVER_PORT}"] +
                             ["--host", f"{BENCH_SERVER_HOST}"])
                # Add script args
                for k, v in script_args.items():
                    bench_cmd.append(f"--{k}")
                    if v != "":
                        bench_cmd.append(f"{v}")

                if output_directory:
                    bench_cmd += (["--save-directory", f"{output_directory}"] +
                                  ["--server-args", f"{server_args}"] + [
                                      "--server-tensor-parallel-size",
                                      f"{tensor_parallel_size}"
                                  ])

                run_bench(server_args, bench_cmd, model)


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

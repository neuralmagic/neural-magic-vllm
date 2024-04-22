import logging
import shlex
from typing import Any, Dict, List

import ray

from tests.entrypoints.test_openai_server import ServerRunner
from tests.utils.logging import log_banner


class ServerContext:
    """
    Context manager for the lifecycle of a vLLM server, wrapping `ServerRunner`.
    """

    def __init__(self, args: Dict[str, str], *,
                 logger: logging.Logger) -> None:
        """Initialize a vLLM server

        :param args: dictionary of flags/values to pass to the server command
        :param logger: logging.Logger instance to use for logging
        :param port: port the server is running on
        """
        self._args = self._args_to_list(args)
        self._logger = logger
        self.server_runner = None

    def __enter__(self):
        """Executes the server process and waits for it to become ready."""
        log_banner(
            self._logger,
            "server startup command args",
            shlex.join(self._args),
            logging.DEBUG,
        )

        ray.init(ignore_reinit_error=True)
        self.server_runner = ServerRunner.remote(self._args)
        ray.get(self.server_runner.ready.remote())
        return self.server_runner

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Stops the server if it's still running.
        """
        if self.server_runner is not None:
            del self.server_runner
        ray.shutdown()

    def _args_to_list(self, args: Dict[str, Any]) -> List[str]:
        """
        Convert a dict mapping of CLI args to a list. All values must be
        string-able.

        :param args: `dict` containing CLI flags and their values
        :return: flattened list to pass to a CLI
        """

        arg_list: List[str] = []
        for flag, value in args.items():
            # minimal error-checking: flag names must be strings
            if not isinstance(flag, str):
                error = f"all flags must be strings, got {type(flag)} ({flag})"
                raise ValueError(error)

            arg_list.append(flag)
            if value is not None:
                arg_list.append(str(value))

        return arg_list

import itertools
from typing import NamedTuple, Iterable
from neuralmagic.tools.call_cmd import call_cmd

def download_datasets(config:NamedTuple) -> None:
    "config is a NamedTuple constructed from some JSON in neuralmagic/benchmarks/configs"
    # download all required datasets
    for download_cmd in config.dataset_download_cmds:
        download_cmd_as_list = list(
            filter(lambda x: len(x) != 0, download_cmd.split(" "))
        )
        call_cmd(download_cmd_as_list, stdout=None, stderr=None)

def script_args_to_cla(config:NamedTuple) -> Iterable[list[str]]:
    "config is a NamedTuple constructed from some JSON in neuralmagic/benchmarks/configs"

    kv = vars(config.script_args)
    arg_lists = kv.values()
    assert all(map(lambda le: isinstance(le, list), arg_lists))

    keys = kv.keys()
    for args in itertools.product(*arg_lists):
        cla = []
        for name, value in zip(keys, args):
            cla.extend([f"--{name}", f"{value}"])
        yield cla

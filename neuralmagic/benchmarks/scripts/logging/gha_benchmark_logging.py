"""
Scripts to process GHA benchmarking JSONs produced by BenchmarkResult and
output JSONs that could be consumed by `github-action-benchmark`
Reference : https://github.com/benchmark-action/github-action-benchmark
"""
import argparse
import json
from pathlib import Path
from functools import reduce
from dataclasses import dataclass
from typing import List, Iterable, NamedTuple

from .benchmark_result import GHABenchmarkToolName, BenchmarkResult, MetricTemplate, short_description, long_description

@dataclass
class GHARecord:
    """
    GHARecord is what actually goes into the output JSON.
        - name : Chart name.
        - unit : Y-axis label.
        - value : Value to plot.
        - extra : This information shows up when you hover
                  over a data-point in the chart.
    """
    name: str
    unit: str
    value: float
    extra: str
    short_description: str
    long_description: str

    @staticmethod
    def extra_from_benchmark_result(br: BenchmarkResult) -> str:
        extra_as_dict = {
            BenchmarkResult.BENCHMARKING_CONTEXT_KEY_:
            br[BenchmarkResult.BENCHMARKING_CONTEXT_KEY_],
            BenchmarkResult.SCRIPT_NAME_KEY_:
            br[BenchmarkResult.SCRIPT_NAME_KEY_],
            BenchmarkResult.SCRIPT_ARGS_KEY_:
            br[BenchmarkResult.SCRIPT_ARGS_KEY_],
        }

        return f"{extra_as_dict}"

    @staticmethod
    def from_metric_template(metric_template: MetricTemplate,
                             extra: str = "",
                             short_description: str = "",
                             long_description: str = ""):
        return GHARecord(name=f"{short_description} - {metric_template.key}\n{long_description} - {metric_template.key}",
                         unit=metric_template.unit,
                         value=metric_template.value,
                         extra=extra,
                         short_description = short_description,
                         long_description = long_description)


class Tool_Record_T(NamedTuple):
    tool: GHABenchmarkToolName
    record: GHARecord


def process(json_file_path: Path) -> Iterable[Tool_Record_T]:

    assert json_file_path.exists()

    json_data:dict = None
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    assert json_data is not None

    print(f"processing file : {json_file_path}")

    hover_data = GHARecord.extra_from_benchmark_result(json_data)
    metrics: Iterable[dict] = json_data.get(BenchmarkResult.METRICS_KEY_)
    metrics: Iterable[MetricTemplate] = map(
        lambda md: MetricTemplate.from_dict(md), metrics.values())

    return map(
        lambda metric: Tool_Record_T(
            metric.tool,
            GHARecord.from_metric_template(
                metric, extra=hover_data, short_description = short_description(json_data), long_description = long_description(json_data))),
        metrics)


def main(input_directory: Path, output_directory: Path) -> None:

    BIGGER_IS_BETTER_OUTPUT_JSON_FILE_NAME = "bigger_is_better.json"
    SMALLER_IS_BETTER_OUTPUT_JSON_FILE_NAME = "smaller_is_better.json"

    def dump_to_json(gha_records: List[GHARecord], output_path: Path):
        # Make data JSON serializable
        gha_record_dicts = list(map(lambda x: x.__dict__, gha_records))
        with open(output_path, 'w+') as f:
            json.dump(gha_record_dicts, f, indent=4)

    json_file_paths = input_directory.glob('*.json')
    tool_records: List[Tool_Record_T] = list(
        reduce(lambda whole, part: whole + part,
               (map(lambda json_file_path: list(process(json_file_path)),
                    json_file_paths))))

    bigger_is_better: List[GHARecord] = list(
        map(
            lambda tool_record: tool_record.record,
            filter(
                lambda tool_record: tool_record.tool == GHABenchmarkToolName.
                BiggerIsBetter, tool_records)))

    smaller_is_better: List[GHARecord] = list(
        map(
            lambda tool_record: tool_record.record,
            filter(
                lambda tool_record: tool_record.tool == GHABenchmarkToolName.
                SmallerIsBetter, tool_records)))

    dump_to_json(bigger_is_better,
                 output_directory / BIGGER_IS_BETTER_OUTPUT_JSON_FILE_NAME)
    dump_to_json(smaller_is_better,
                 output_directory / SMALLER_IS_BETTER_OUTPUT_JSON_FILE_NAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Process the benchmark JSONs produced by BenchmarkResult and output JSONs
        that could be consumed by `github-action-benchmark`
        Reference : https://github.com/benchmark-action/github-action-benchmark
        """)

    parser.add_argument("-i",
                        "--input-json-directory",
                        required=True,
                        type=str,
                        help="""
            Path to the directory containing BenchmarkResult jsons.
            This is typically the output directory passed to the benchmark
            runner scripts like neuralmagic/benchmarks/run_benchmarks.py.
        """)
    parser.add_argument("-o",
                        "--output-directory",
                        type=str,
                        default=None,
                        help="""
            Path to the output directory where the JSONs that would be consumed
            `github-action-benchmark` are stored.
        """)

    args = parser.parse_args()

    main(Path(args.input_json_directory), Path(args.output_directory))

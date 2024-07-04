import jinja2
import os
import enum
from enum import auto as enum_auto
from typing import Tuple, List
from dataclasses import dataclass
from collections.abc import Iterable
from cutlass_library import (
    DataType,
    KernelScheduleType,
    EpilogueScheduleType,
    TileSchedulerType,
    KernelScheduleTag,
    EpilogueScheduleTag,
    TileSchedulerTag,
    DataTypeTag,
    DataTypeNames,
)


DISPATCH_TEMPLATE = """
#include "cuda/marlinv2/marlinv2_mm_launcher.cuh"

namespace marlinv2 {
using KernelDispatcher_ = KernelDispatcher<
    {{ DataTypeTag[type_config.element_a] }},  // ElementA
    {{ DataTypeTag[type_config.element_b] }},  // ElementB
    {{ DataTypeTag[type_config.element_d] }},  // ElementD
    {{ DataTypeTag[type_config.accumulator] }}, // Accumulator
    {{ DataTypeTag[type_config.element_b_scale] }}, // Scales
    {{ DataTypeTag[type_config.element_b_zeropoint] }}>; // Zeropoints

{% for _, schedule_name in schedules %}extern torch::Tensor 
impl_{{type_name}}_sch_{{schedule_name}}(PytorchArguments args);
{% endfor %}
template <>
torch::Tensor KernelDispatcher_::dispatch(PytorchArguments args) {
  if (!args.schedule) {
    return impl_{{ type_name }}_sch_{{ schedules[0][1] }}(args);
  }
  {% for _, schedule_name in schedules %}
  if (*args.schedule == "{{ schedule_name }}") {
    return impl_{{ type_name }}_sch_{{ schedule_name }}(args);
  }
  {% endfor %}
  TORCH_CHECK_NOT_IMPLEMENTED(false, "marlinv2_mm(..) is not implemented for "
                                     "schedule = ", *args.schedule);
}
}; // namespace marlinv2
"""

IMPL_TEMPLATE = """
#include "cuda/marlinv2/marlinv2_mm_launcher.cuh"

namespace marlinv2 {
template <typename Config, bool with_C, bool with_scales, bool with_zeropoints>
using Kernel = KernelTemplate<
    {{ DataTypeTag[type_config.element_a] }},  // ElementA
    {{ DataTypeTag[type_config.element_b] }},  // ElementB
    {{ DataTypeTag[type_config.element_d] }},  // ElementD
    {{ DataTypeTag[type_config.accumulator] }}, // Accumulator
    {{ DataTypeTag[type_config.element_b_scale] }}, // Scales
    {{ DataTypeTag[type_config.element_b_zeropoint] }}, // Zeropoints
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput
>::Speacialization<Config, with_C, with_scales, with_zeropoints>;

struct sch_{{ schedule_name }} {
  using TileShapeNM = Shape<
    {{ to_cute_constant(schedule.tile_shape_mn)|join(', ') }}>;
  using ClusterShape = Shape<
    {{ to_cute_constant(schedule.cluster_shape_mnk)|join(', ') }}>;
  // TODO: Reimplement
  // using KernelSchedule   = {{ KernelScheduleTag[schedule.kernel_schedule] }};
  using EpilogueSchedule = {{ EpilogueScheduleTag[schedule.epilogue_schedule]}};
  using TileScheduler    = {{ TileSchedulerTag[schedule.tile_scheduler] }};
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

torch::Tensor 
impl_{{type_name}}_sch_{{schedule_name}}(PytorchArguments args) {
  bool with_C = args.C.has_value(), with_scales = args.scales.has_value(),
       with_zeropoints = args.zeros.has_value();

  if (!with_C && with_scales && with_zeropoints) {
    return run_impl<Kernel<sch_{{schedule_name}}, false, true, true>>(args);
  }

  if (!with_C && with_scales && !with_zeropoints) {
    return run_impl<Kernel<sch_{{schedule_name}}, false, true, false>>(args);
  }

  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "for the sake of compile times marlinv2_mm(..) is not implemented "
      "for with_C=", with_C, ", with_scales=", with_scales, 
      ", with_zeropoints=", with_zeropoints);
}
}; // namespace marlinv2
"""


PREPACK_TEMPLATE = """
#include "cuda/marlinv2/marlinv2_prepack_launcher.cuh"

namespace marlinv2 {
using PrepackDispatcher_ = PrepackDispatcher<
  {{ DataTypeTag[type_config.element_a] }}, // ElementA
  {{ DataTypeTag[type_config.element_b] }}, // ElementB
  {{ DataTypeTag[type_config.element_d] }}, // ElementD
  {{ DataTypeTag[type_config.accumulator] }}, // Accumulator
  {{ DataTypeTag[type_config.element_b_scale] }}, // Scales
  {{ DataTypeTag[type_config.element_b_zeropoint] }}>; // Zeropoints

using PrepackedLayout = PrepackedLayoutTemplate<
  {{ DataTypeTag[type_config.element_a] }}, // ElementA
  {{ DataTypeTag[type_config.element_b] }}, // ElementB
  {{ DataTypeTag[type_config.element_d] }}, // ElementD
  {{ DataTypeTag[type_config.accumulator] }}, // Accumulator
  cutlass::layout::ColumnMajor,
  cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput>;

template <>
torch::Tensor PrepackDispatcher_::dispatch(torch::Tensor B) {
  return prepack_impl<PrepackedLayout>(B);
}
}; // namespace marlinv2
"""

class NMTileSchedulerType(enum.Enum):
  StreamK = enum_auto()
#
TileSchedulerTag.update({
  NMTileSchedulerType.StreamK: 'cutlass::gemm::NMStreamKScheduler'
})

class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecializedMixedInput = enum_auto()
    TmaWarpSpecializedPingpongMixedInput = enum_auto()
    TmaWarpSpecializedCooperativeMixedInput = enum_auto()


KernelScheduleTag.update({
    MixedInputKernelScheduleType.TmaWarpSpecializedMixedInput:
        "cutlass::gemm::KernelTmaWarpSpecializedMixedInput",
    MixedInputKernelScheduleType.TmaWarpSpecializedPingpongMixedInput:
        "cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput",
    MixedInputKernelScheduleType.TmaWarpSpecializedCooperativeMixedInput:
        "cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput",
})


TmaMI = MixedInputKernelScheduleType.TmaWarpSpecializedCooperativeMixedInput
TmaCoop = EpilogueScheduleType.TmaWarpSpecializedCooperative


@dataclass
class ScheduleConfig:
    tile_shape_mn: Tuple[int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    kernel_schedule: KernelScheduleType
    epilogue_schedule: EpilogueScheduleType
    tile_scheduler: TileSchedulerType


@dataclass
class KernelTypeConfig:
    element_a: DataType
    element_b: DataType
    element_b_scale: DataType
    element_b_zeropoint: DataType
    element_d: DataType
    accumulator: DataType


def generate_schedule_name(schedule_config: ScheduleConfig) -> str:
    tile_shape = (
        f"{schedule_config.tile_shape_mn[0]}x{schedule_config.tile_shape_mn[1]}"
    )
    cluster_shape = f"{schedule_config.cluster_shape_mnk[0]}" + \
                    f"x{schedule_config.cluster_shape_mnk[1]}" + \
                    f"x{schedule_config.cluster_shape_mnk[2]}"
    kernel_schedule = KernelScheduleTag[schedule_config.kernel_schedule]\
        .split("::")[-1]
    epilogue_schedule = EpilogueScheduleTag[schedule_config.epilogue_schedule]\
        .split("::")[-1]
    tile_scheduler = TileSchedulerTag[schedule_config.tile_scheduler]\
        .split("::")[-1]

    return f"{tile_shape}_{cluster_shape}_{kernel_schedule}" + \
           f"_{epilogue_schedule}_{tile_scheduler}"


# mostly unique shorter schedule_name
def generate_terse_schedule_name(schedule_config: ScheduleConfig) -> str:
    kernel_terse_names_replace = {
        "KernelTmaWarpSpecializedCooperativeMixedInput_": "TmaMI_",
        "TmaWarpSpecializedCooperative_": "TmaCoop_",
        "NMStreamKScheduler": "NMstreamK",
    }

    schedule_name = generate_schedule_name(schedule_config)
    for orig, terse in kernel_terse_names_replace.items():
        schedule_name = schedule_name.replace(orig, terse)
    return schedule_name


# unique type_name
def generate_kernel_type_name(kernel_type_config: KernelTypeConfig):
    element_a = DataTypeNames[kernel_type_config.element_a]
    element_b = DataTypeNames[kernel_type_config.element_b]
    element_d = DataTypeNames[kernel_type_config.element_d]
    accumulator = DataTypeNames[kernel_type_config.accumulator]
    element_scale = DataTypeNames[kernel_type_config.element_b_scale]
    element_zeropoint = DataTypeNames[kernel_type_config.element_b_zeropoint]

    return f"{element_a}{element_b}{element_d}{accumulator}{element_scale}{element_zeropoint}"


# non-unique shorter type_name
def generate_terse_kernel_type_name(kernel_type_config: KernelTypeConfig):
    element_a = DataTypeNames[kernel_type_config.element_a]
    element_b = DataTypeNames[kernel_type_config.element_b]

    return f"{element_a}{element_b}"


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def to_cute_constant(value: List[int]):
    def _to_cute_constant(value: int):
        if is_power_of_two(value):
            return f"_{value}"
        else:
            return f"Int<{value}>"

    if isinstance(value, Iterable):
        return [_to_cute_constant(value) for value in value]
    else:
        return _to_cute_constant(value)


template_globals = {
    "DataTypeTag": DataTypeTag,
    "KernelScheduleTag": KernelScheduleTag,
    "EpilogueScheduleTag": EpilogueScheduleTag,
    "TileSchedulerTag": TileSchedulerTag,
    "to_cute_constant": to_cute_constant,
}


def create_template(template_str):
    template = jinja2.Template(template_str)
    template.globals.update(template_globals)
    return template


mm_dispatch_template = create_template(DISPATCH_TEMPLATE)
mm_impl_template = create_template(IMPL_TEMPLATE)
prepack_dispatch_template = create_template(PREPACK_TEMPLATE)


def create_sources(type_config, schedule_configs):
    sources = []
    # Render the template with the provided configurations
    schedules_with_names = [
        (schedule, generate_terse_schedule_name(schedule))
        for schedule in schedule_configs
    ]

    type_name = generate_kernel_type_name(type_config)
    terse_type_name = generate_terse_kernel_type_name(type_config)

    sources.append((f"marlinv2_mm_{terse_type_name}",
        mm_dispatch_template.render(
            type_name=type_name,
            type_config=type_config,
            schedules=schedules_with_names,
    )))

    sources.append((f"marlinv2_prepack_{terse_type_name}",
        prepack_dispatch_template.render(
            type_name=type_name,
            type_config=type_config,
    )))

    for schedule in schedule_configs:
        schedule_name = generate_terse_schedule_name(schedule)
        sources.append((f"marlinv2_mm_{terse_type_name}_{schedule_name}",
            mm_impl_template.render(
                type_name=type_name,
                type_config=type_config,
                schedule=schedule,
                schedule_name=schedule_name,
        )))
    return sources


def jit(type_config, schedules):
    for source in create_sources(type_config, schedules):
        filename, code = source
        print(f"Generated {filename}.cu")
        print(code)


def AOT_generate():
    SCRIPT_DIR = os.path.dirname(__file__)

    AOT_schedules = list((
        ScheduleConfig(
            tile_shape_mn=tile_shape_mn,
            cluster_shape_mnk=cluster_shape_mnk,
            kernel_schedule=kernel_schedule,
            epilogue_schedule=epilogue_schedule,
            tile_scheduler=tile_scheduler,
        )
        for tile_shape_mn, cluster_shape_mnk in (
            ((128, 16), (1, 1, 1)),
            ((128, 64), (1, 1, 1)),
            ((128, 128), (1, 1, 1)),
            #((128, 256), (1, 1, 1)),
        )
        for kernel_schedule in (TmaMI,)
        for epilogue_schedule in (TmaCoop,)
        for tile_scheduler in (
            TileSchedulerType.Default,
            NMTileSchedulerType.StreamK)
    ))

    AOT_kernel_type_configs = list((
        KernelTypeConfig(
            element_a=DataType.f16,
            element_b=element_b,
            element_b_scale=DataType.f16,
            element_b_zeropoint=DataType.f16,
            element_d=DataType.f16,
            accumulator=DataType.f32,
        )
        for element_b in (DataType.s4, DataType.u4)
    ))

    output_dir = os.path.join(SCRIPT_DIR, "generated")

    # Delete the "generated" directory if it exists
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    # Create the "generated" directory
    os.makedirs(output_dir)

    # Render each group of configurations into separate files
    for type_config in AOT_kernel_type_configs:
        terse_type_name = generate_terse_kernel_type_name(type_config)

        schedules_filename = f"marlinv2_mm_{terse_type_name}_schedules.txt"
        schedules_filepath = os.path.join(output_dir, schedules_filename)
        with open(schedules_filepath, "w") as schedules_file:
            for schedule in AOT_schedules:
                schedule_name = generate_terse_schedule_name(schedule)
                schedules_file.write(f"{schedule_name}\n")

        for source in create_sources(type_config, AOT_schedules):
            filename, code = source
            filepath = os.path.join(output_dir, f"{filename}.cu")
            with open(filepath, "w") as output_file:
                output_file.write(code)
            print(f"Rendered template to {filepath}")


if __name__ == "__main__":
    AOT_generate()

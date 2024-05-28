from typing import Tuple, List
from enum import Enum
import jinja2
from pathlib import Path
from itertools import product
from abc import ABC, abstractmethod
from dataclasses import dataclass
from autogen_manifest import (Cutlass2xArgs, Cutlass3xArgs, Cutlass2xArgsList,
        Cutlass3xArgsList, Cutlass3xArgsTileList, Cutlass3xArgsListFP8FastAccum,
        Cutlass3xArgsClusterList, Cutlass3xArgsListFP8FastAccumTC)
import os

## Utilities ####

def get_as_cutlass_gemm_shape(shape: Tuple[int, int, int]):
    return f'cutlass::gemm::GemmShape<{shape[0]}, {shape[1]}, {shape[2]}>'

def get_as_cutlass3x_gemm_shape(shape: Tuple[int, int, int]):
    return f'Shape<_{shape[0]}, _{shape[1]}, _{shape[2]}>'

def file_contents_same(filepath, contents):
    if not Path(filepath).exists():
        return

    f_contents = None
    with open(filepath, "r") as f:
        f_contents = f.read()

    return f_contents == contents

## Abstract generator

class Generator(ABC):

    @staticmethod
    def write_ops(pybind_fn_names, ops_fn_defns, ops_macro, filename):
        s = "#pragma once\n"
        s = '#include <torch/extension.h>\n\n'
        s += f"#define {ops_macro}\\\n"
        for fn_name in pybind_fn_names:
            s += f' ops.def("{fn_name}", &{fn_name}, "{fn_name}"); \\\n'
        s += "\n"

        for ops_fn_defn in ops_fn_defns:
            s += f'{ops_fn_defn}\n'

        with open(filename, 'w+') as f:
            f.write(s)

    @staticmethod
    def last_namespace(s):
        return s.split('::')[-1]

    @staticmethod
    def swizzle_short_name(swizzle):
        return Generator.last_namespace(swizzle)

    @staticmethod
    def gemm_mode_short_name(gemm_mode):
        return Generator.last_namespace(gemm_mode)

    @staticmethod
    def generate():
        ...

## Cutlass 2x generator

class Cutlass2xGenerator(Generator):

    SCRIPT_DIR=Path(os.path.dirname(os.path.realpath(__file__)))
    GENERATE_DIR= SCRIPT_DIR / "generated"
    FN_DEFN_JINJA= SCRIPT_DIR / "scaled_mm_dq_c2x.jinja"
    FN_DECL_JINJA= SCRIPT_DIR / "scaled_mm_dq_fnprototype.jinja"
    OPS_FILE= SCRIPT_DIR / "autogen_cutlass2x_ops.h"
    OPS_MACRO = "CUTLASS2X_DEFS"

    def __init__(self):
        pass

    @staticmethod
    def generate_name(
                arch: int,
                tile_shape: Tuple[int, int, int],
                warp_shape:Tuple[int, int, int],
                instruction_shape: Tuple[int, int, int],
                thread_block_swizzle: str,
                gemm_mode: str,
                main_loop_stages: int):

        return 'autogen_cutlass2x_scaled_mm_dq_sm{}_{}x{}x{}_{}x{}x{}_{}x{}x{}_{}_{}_{}'.format(
                arch,
                tile_shape[0], tile_shape[1], tile_shape[2],
                warp_shape[0], warp_shape[1], warp_shape[2],
                instruction_shape[0], instruction_shape[1], instruction_shape[2],
                Generator.swizzle_short_name(thread_block_swizzle),
                Generator.gemm_mode_short_name(gemm_mode),
                main_loop_stages)


    @staticmethod
    def generate_filename(archs: List[int],
                           tile_shape: Tuple[int, int, int],
                           warp_shape: Tuple[int, int, int],
                           instruction_shape: Tuple[int, int, int],
                           thread_block_swizzle: str,
                           gemm_mode: str,
                           main_loop_stages : int):
        f = '{}/autogen_cutlass_scaled_mm_dq_c2x_{}x{}x{}_{}x{}x{}_{}x{}x{}_{}_{}_{}'.format(
                Cutlass2xGenerator.GENERATE_DIR,
                tile_shape[0], tile_shape[1], tile_shape[2],
                warp_shape[0], warp_shape[1], warp_shape[2],
                instruction_shape[0], instruction_shape[1], instruction_shape[2],
                Generator.swizzle_short_name(thread_block_swizzle),
                Generator.gemm_mode_short_name(gemm_mode),
                main_loop_stages)
        for arch in archs:
            f = f + f"_{arch}"
    
        f = f + ".cu"
        return f

    @staticmethod
    def generate_2x_file(
        archs :List[int],
        tile_shape:Tuple[int, int, int],
        warp_shape:Tuple[int, int, int],
        instruction_shape: Tuple[int, int, int],
        thread_block_swizzle: str,
        gemm_mode: str,
        main_loop_stages: int):

        # Make the generate dir
        Cutlass2xGenerator.GENERATE_DIR.mkdir(exist_ok=True)
    
        jenv = jinja2.Environment(loader=jinja2.FileSystemLoader("/"))
        fn_defn_template = jenv.get_template(str(Cutlass2xGenerator.FN_DEFN_JINJA))
        fn_decl_template = jenv.get_template(str(Cutlass2xGenerator.FN_DECL_JINJA))

        pybind_fn_names = []
        ops_fn_decl = []
    
        code = ""
        for arch in archs:
            fn_name = Cutlass2xGenerator.generate_name(arch, tile_shape, warp_shape, instruction_shape, thread_block_swizzle, gemm_mode, main_loop_stages)
            fn_decl = fn_decl_template.render(_name = fn_name)
            code += fn_defn_template.render(_name = fn_name,
                                    _tile_shape = get_as_cutlass_gemm_shape(tile_shape),
                                    _warp_shape = get_as_cutlass_gemm_shape(warp_shape),
                                    _instruction_shape = get_as_cutlass_gemm_shape(instruction_shape),
                                    _main_loop_stages = main_loop_stages,
                                    _thread_block_swizzle = thread_block_swizzle,
                                    _gemm_mode = gemm_mode,
                                    _arch = arch)

            pybind_fn_names.append(fn_name)
            ops_fn_decl.append(fn_decl)
    
        filename = Cutlass2xGenerator.generate_filename(archs, tile_shape, warp_shape, instruction_shape, thread_block_swizzle, gemm_mode, main_loop_stages)
    
        if file_contents_same(filename, code):
            print(f"{filename} exists with the same content - Not re-generating it!")
            return pybind_fn_names, ops_fn_decl
    
        # write code
        with open(filename, "w+") as f:
            f.write(code)

        return pybind_fn_names, ops_fn_decl

    @staticmethod
    def generate(args_list: List[Cutlass2xArgs]):
        pybind_fn_names = []
        ops_fn_decls = []
        for args in args_list:
            pybind_names, ops_decls = Cutlass2xGenerator.generate_2x_file(args.archs, args.tile_shape,
                    args.warp_shape, args.instruction_shape, args.thread_block_swizzle, args.gemm_mode,
                    args.main_loop_stages)
            pybind_fn_names.extend(pybind_names)
            ops_fn_decls.extend(ops_decls)

        # fill-out ops.h
        Generator.write_ops(pybind_fn_names, ops_fn_decls, Cutlass2xGenerator.OPS_MACRO, Cutlass2xGenerator.OPS_FILE)

## Cutlass 3x Generator

class Cutlass3xGenerator(Generator):

    SCRIPT_DIR=Path(os.path.dirname(os.path.realpath(__file__)))
    GENERATE_DIR= SCRIPT_DIR / "generated"
    FN_DEFN_JINJA= SCRIPT_DIR / "scaled_mm_dq_c3x.jinja"
    FN_DECL_JINJA= SCRIPT_DIR / "scaled_mm_dq_fnprototype.jinja"
    OPS_FILE= SCRIPT_DIR / "autogen_cutlass3x_ops.h"
    OPS_MACRO = "CUTLASS3X_DEFS"

    @staticmethod
    def generate_name(
                dtype_str: str,
                arch: int,
                tile_shape: Tuple[int, int, int],
                cluster_shape: Tuple[int, int, int],
                kernel_schedule: str,
                epilogue_schedule: str,
                tile_schedule: str,
                gemm_mode: str):

        return 'autogen_cutlass3x_scaled_mm_dq_sm{}_{}x{}x{}_{}x{}x{}_{}_{}_{}_{}_{}'.format(
                arch,
                tile_shape[0], tile_shape[1], tile_shape[2],
                cluster_shape[0], cluster_shape[1], cluster_shape[2],
                Generator.last_namespace(kernel_schedule), 
                Generator.last_namespace(epilogue_schedule), 
                Generator.last_namespace(tile_schedule), 
                Generator.last_namespace(gemm_mode),
                dtype_str)

    @staticmethod
    def generate_filename(
                dtype_str: str,
                arch: int,
                tile_shape: Tuple[int, int, int],
                cluster_shape: Tuple[int, int, int],
                kernel_schedule: str,
                epilogue_schedule: str,
                tile_schedule: str,
                gemm_mode: str):

        f = '{}/autogen_cutlass_scaled_mm_dq_c3x_{}x{}x{}_{}x{}x{}_{}_{}_{}_{}_{}_{}'.format(
                Cutlass2xGenerator.GENERATE_DIR,
                tile_shape[0], tile_shape[1], tile_shape[2],
                cluster_shape[0], cluster_shape[1], cluster_shape[2],
                Generator.last_namespace(kernel_schedule), 
                Generator.last_namespace(epilogue_schedule), 
                Generator.last_namespace(tile_schedule), 
                Generator.last_namespace(gemm_mode),
                dtype_str,
                arch)
    
        f = f + ".cu"
        return f

    def generate_3x_file(
        dtype_str: str,
        archs :List[int],
        tile_shape:Tuple[int, int, int],
        cluster_shape:Tuple[int, int, int],
        kernel_schedule,
        epilogue_schedule,
        tile_schedule,
        gemm_mode):

        def to_torch_dtype_str(dtype_str):
            if dtype_str == "int8":
                return "torch::kInt8"
            if dtype_str == "fp8":
                return  "torch::kFloat8_e4m3fn"
            raise ValueError("unknown type")

        def to_cutlass_dtype_str(dtype_str):
            if dtype_str == "int8":
                return "int8_t"
            if dtype_str == "fp8":
                return  "cutlass::float_e4m3_t"
            raise ValueError("unknown type")

        # Make the generate dir
        Cutlass3xGenerator.GENERATE_DIR.mkdir(exist_ok=True)
    
        jenv = jinja2.Environment(loader=jinja2.FileSystemLoader("/"))
        fn_defn_template = jenv.get_template(str(Cutlass3xGenerator.FN_DEFN_JINJA))
        fn_decl_template = jenv.get_template(str(Cutlass3xGenerator.FN_DECL_JINJA))

        pybind_fn_names = []
        ops_fn_decl = []
    
        code = ""
        for arch in archs:
            fn_name = Cutlass3xGenerator.generate_name(dtype_str, arch, tile_shape, cluster_shape,
                                                        kernel_schedule, epilogue_schedule, tile_schedule, gemm_mode)
            fn_decl = fn_decl_template.render(_name = fn_name)
            code += fn_defn_template.render(_name = fn_name,
                                    _torch_input_dtype = to_torch_dtype_str(dtype_str),
                                    _cutlass_input_dtype = to_cutlass_dtype_str(dtype_str),
                                    _tile_shape = get_as_cutlass3x_gemm_shape(tile_shape),
                                    _cluster_shape = get_as_cutlass3x_gemm_shape(cluster_shape),
                                    _kernel_schedule = kernel_schedule,
                                    _epilogue_schedule = epilogue_schedule,
                                    _tile_schedule = tile_schedule, 
                                    _gemm_mode = gemm_mode)

            pybind_fn_names.append(fn_name)
            ops_fn_decl.append(fn_decl)
    
        filename = Cutlass3xGenerator.generate_filename(dtype_str, arch, tile_shape, cluster_shape,
                                                        kernel_schedule, epilogue_schedule, tile_schedule, gemm_mode)
    
        if file_contents_same(filename, code):
            print(f"{filename} exists with the same content - Not re-generating it!")
            return pybind_fn_names, ops_fn_decl
    
        # write code
        with open(filename, "w+") as f:
            f.write(code)

        return pybind_fn_names, ops_fn_decl

    @staticmethod
    def generate(args_list: List[Cutlass3xArgs]):
        pybind_fn_names = []
        ops_fn_decls = []
        for args in args_list:
            pybind_names, ops_decls = Cutlass3xGenerator.generate_3x_file(args.dtype_str, [args.arch], args.tile_shape,
                    args.cluster_shape, args.kernel_schedule, args.epilogue_schedule, args.tile_schedule, 
                    args.gemm_mode)
            pybind_fn_names.extend(pybind_names)
            ops_fn_decls.extend(ops_decls)

        # fill-out ops.h
        Generator.write_ops(pybind_fn_names, ops_fn_decls, Cutlass3xGenerator.OPS_MACRO, Cutlass3xGenerator.OPS_FILE)

def generate_cutlass2x_kernels():
    Cutlass2xGenerator.generate(Cutlass2xArgsList)

def generate_cutlass3x_kernels():
    Cutlass3xGenerator.generate(Cutlass3xArgsListFP8FastAccumTC)

def main(args):
    if args.version == "all":
        generate_cutlass2x_kernels()
        generate_cutlass3x_kernels()
    if args.version == "2x":
        generate_cutlass2x_kernels()
    if args.version == "3x":
        generate_cutlass3x_kernels()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description="Autogen cutlass kernels")

    parser.add_argument("--version", required=True, type=str, default="all", choices=["all", "2x", "3x"])

    args = parser.parse_args()
    main(args)

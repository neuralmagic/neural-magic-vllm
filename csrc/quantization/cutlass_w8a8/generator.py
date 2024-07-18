from typing import Tuple, List
from enum import Enum
import jinja2
from pathlib import Path
from itertools import product
from abc import ABC, abstractmethod
from dataclasses import dataclass
from autogen_manifest import Cutlass2xArgs, DefaultCutlass2xArgs
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
        s += f"#define {ops_macro}\\\n"
        for fn_name in pybind_fn_names:
            s += (f' ops.def("{fn_name}(Tensor! out, Tensor a, Tensor b,'
                  f'Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()", &{fn_name}); \\\n'
                  f' ops.impl("{fn_name}", torch::kCUDA, &{fn_name});\\\n\n'
                  )
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
    FN_DEFN_JINJA= SCRIPT_DIR / "scaled_mm_c2x.jinja"
    FN_DECL_JINJA= SCRIPT_DIR / "scaled_mm_fnprototype.jinja"
    OPS_FILE= SCRIPT_DIR / "autogen_cutlass2x_ops.h"
    OPS_MACRO = "CUTLASS2X_DEFS"

    def __init__(self):
        pass

    @staticmethod
    def generate_name(args: Cutlass2xArgs):

        return 'autogen_cutlass2x_scaled_mm_sm{}_{}x{}x{}_{}x{}x{}_{}x{}x{}_{}_{}_{}'.format(
                args.arch,
                args.tile_shape[0], args.tile_shape[1], args.tile_shape[2],
                args.warp_shape[0], args.warp_shape[1], args.warp_shape[2],
                args.instruction_shape[0], args.instruction_shape[1], args.instruction_shape[2],
                Generator.swizzle_short_name(args.thread_block_swizzle),
                Generator.gemm_mode_short_name(args.gemm_mode),
                args.main_loop_stages)


    @staticmethod
    def generate_filename(args: Cutlass2xArgs):
        f = '{}/autogen_cutlass_scaled_mm_c2x_{}x{}x{}_{}x{}x{}_{}x{}x{}_{}_{}_{}_{}.cu'.format(
                Cutlass2xGenerator.GENERATE_DIR,
                args.tile_shape[0], args.tile_shape[1], args.tile_shape[2],
                args.warp_shape[0], args.warp_shape[1], args.warp_shape[2],
                args.instruction_shape[0], args.instruction_shape[1], args.instruction_shape[2],
                Generator.swizzle_short_name(args.thread_block_swizzle),
                Generator.gemm_mode_short_name(args.gemm_mode),
                args.main_loop_stages,
                args.arch)
        return f

    @staticmethod
    def generate_2x_file(args: Cutlass2xArgs):

        # Make the generate dir
        Cutlass2xGenerator.GENERATE_DIR.mkdir(exist_ok=True)
    
        jenv = jinja2.Environment(loader=jinja2.FileSystemLoader("/"))
        fn_defn_template = jenv.get_template(str(Cutlass2xGenerator.FN_DEFN_JINJA))
        fn_decl_template = jenv.get_template(str(Cutlass2xGenerator.FN_DECL_JINJA))

        pybind_fn_names = []
        ops_fn_decl = []
    
        code = ""
        fn_name = Cutlass2xGenerator.generate_name(args)
        fn_decl = fn_decl_template.render(_name = fn_name)
        code += fn_defn_template.render(
                _name = fn_name,
                _tile_shape = get_as_cutlass_gemm_shape(args.tile_shape),
                _warp_shape = get_as_cutlass_gemm_shape(args.warp_shape),
                _instruction_shape = get_as_cutlass_gemm_shape(args.instruction_shape),
                _main_loop_stages = args.main_loop_stages,
                _thread_block_swizzle = args.thread_block_swizzle,
                _gemm_mode = args.gemm_mode,
                _arch = args.arch)

        pybind_fn_names.append(fn_name)
        ops_fn_decl.append(fn_decl)
    
        filename = Cutlass2xGenerator.generate_filename(args)
    
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
            pybind_names, ops_decls = Cutlass2xGenerator.generate_2x_file(args)
            pybind_fn_names.extend(pybind_names)
            ops_fn_decls.extend(ops_decls)

        # fill-out ops.h
        Generator.write_ops(pybind_fn_names, ops_fn_decls, Cutlass2xGenerator.OPS_MACRO, Cutlass2xGenerator.OPS_FILE)

def generate_cutlass2x_kernels():
    Cutlass2xGenerator.generate([DefaultCutlass2xArgs])

def main():
    generate_cutlass2x_kernels()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description="Autogen cutlass kernels")
    main()


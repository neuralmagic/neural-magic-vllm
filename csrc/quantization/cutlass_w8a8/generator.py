from typing import Tuple, List
from enum import Enum
import jinja2
from pathlib import Path
from itertools import product
from abc import ABC, abstractmethod
import os

class CutlassKernelVersion(Enum):
    CUTLASS2 = 1
    CUTLASS3 = 2

## Utilities ####

def get_as_cutlass_gemm_shape(shape: Tuple[int, int, int]):
    return f'cutlass::gemm::GemmShape<{shape[0]}, {shape[1]}, {shape[2]}>'

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
    def write_pybind_cpp(pybind_fn_names, filename):
        s = '#include <torch/extension.h>\n'
        s += '#include "cutlass2x_ops.h"\n\n'
        s += f'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)' 
        s += '{\n'
        for fn_name in pybind_fn_names:
            s += f' m.def("{fn_name}", &{fn_name}, "{fn_name}");\n'
        s += '}\n'

        with open(filename, 'w+') as f:
            f.write(s)

    @staticmethod
    def write_ops_hpp(ops_fn_defns, filename):
        s = "#pragma once\n"
        s += "#include <torch/extension.h>\n\n"

        for ops_fn_defn in ops_fn_defns:
            s += f'{ops_fn_defn}\n'

        with open(filename, 'w+') as f:
            f.write(s)

    @abstractmethod
    def generate(self):
        ...

## Cutlass 2x generator

class Cutlass2xGenerator(Generator):

    SCRIPT_DIR=Path(os.path.dirname(os.path.realpath(__file__)))
    GENERATE_DIR= SCRIPT_DIR / "generated"
    FN_DEFN_JINJA= SCRIPT_DIR / "scaled_mm_dq_c2x.jinja"
    FN_DECL_JINJA= SCRIPT_DIR / "scaled_mm_dq_c2x_fnprototype.jinja"
    PYBIND_FILE=GENERATE_DIR / "cutlass2x_pybind.cpp" 
    OPS_FILE=GENERATE_DIR / "cutlass2x_ops.h"

    def __init__(self,
                 archs: List[int],
                 tile_shapes: List[Tuple[int, int, int]],
                 warp_shapes: List[Tuple[int, int, int]],
                 instruction_shapes: List[Tuple[int, int, int]],
                 main_loop_stages: List[int]):
        self.archs = archs
        self.tile_shapes = tile_shapes
        self.warp_shapes = warp_shapes
        self.instruction_shapes = instruction_shapes
        self.main_loop_stages = main_loop_stages

    @staticmethod
    def generate_name(
                arch: int,
                tile_shape: Tuple[int, int, int],
                warp_shape:Tuple[int, int, int],
                instruction_shape: Tuple[int, int, int],
                main_loop_stages: int):
        return 'autogen_cutlass_scaled_mm_dq_sm{}_{}x{}x{}_{}x{}x{}_{}x{}x{}_{}'.format(
                arch,
                tile_shape[0], tile_shape[1], tile_shape[2],
                warp_shape[0], warp_shape[1], warp_shape[2],
                instruction_shape[0], instruction_shape[1], instruction_shape[2],
                main_loop_stages)


    @staticmethod
    def generate_filename(archs: List[int],
                           tile_shape: Tuple[int, int, int],
                           warp_shape: Tuple[int, int, int],
                           instruction_shape: Tuple[int, int, int],
                           main_loop_stages : int):
        f = '{}/autogen_cutlass_scaled_mm_dq_c2x_{}x{}x{}_{}x{}x{}_{}x{}x{}_{}'.format(
                Cutlass2xGenerator.GENERATE_DIR,
                tile_shape[0], tile_shape[1], tile_shape[2],
                warp_shape[0], warp_shape[1], warp_shape[2],
                instruction_shape[0], instruction_shape[1], instruction_shape[2],
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
            fn_name = Cutlass2xGenerator.generate_name(arch, tile_shape, warp_shape, instruction_shape, main_loop_stages)
            fn_decl = fn_decl_template.render(_name = fn_name)
            code += fn_defn_template.render(_name = fn_name,
                                    _tile_shape = get_as_cutlass_gemm_shape(tile_shape),
                                    _warp_shape = get_as_cutlass_gemm_shape(warp_shape),
                                    _instruction_shape = get_as_cutlass_gemm_shape(instruction_shape),
                                    _main_loop_stages = main_loop_stages,
                                    _arch = arch)

            pybind_fn_names.append(fn_name)
            ops_fn_decl.append(fn_decl)
    
        filename = Cutlass2xGenerator.generate_filename(archs, tile_shape, warp_shape, instruction_shape, main_loop_stages)
    
        if file_contents_same(filename, code):
            print(f"{filename} exists with the same content - Not re-generating it!")
            return pybind_fn_names, ops_fn_decl
    
        # write code
        with open(filename, "w+") as f:
            f.write(code)

        return pybind_fn_names, ops_fn_decl

    def generate(self):
        pybind_fn_names = []
        ops_fn_decls = []
        for ts, ws, inst_shape, ml_stage in product(self.tile_shapes,
                self.warp_shapes, self.instruction_shapes,
                self.main_loop_stages):
            pybind_names, ops_decls = self.generate_2x_file(self.archs, ts, ws, inst_shape, ml_stage)
            pybind_fn_names.extend(pybind_names)
            ops_fn_decls.extend(ops_decls)

        # Write out the pybind and ops
        self.write_pybind_cpp(pybind_fn_names, Cutlass2xGenerator.PYBIND_FILE)
        self.write_ops_hpp(ops_fn_decls, Cutlass2xGenerator.OPS_FILE)

def generate_cutlass2x_kernels():

    archs = [80]
    tile_shapes = [(128, 128, 64), (128, 64, 64)]
    warp_shapes = [(64, 64, 64)]
    instruction_shapes = [(16, 8, 32)]
    main_loop_stages = [5]

    generator = Cutlass2xGenerator(archs,
                                   tile_shapes,
                                   warp_shapes,
                                   instruction_shapes,
                                   main_loop_stages)
    generator.generate()

if __name__ == "__main__":
    generate_cutlass2x_kernels()

import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

setup(
    name='matrix_utils',
    version='1.0',
    description='matrix_utils',
    long_description='matrix_utils',
    ext_modules=[
        CUDAExtension(
            name='matrix_utils',
            sources=['matrix_utils.cpp', 'matrix_utils_kernel.cu'],
            include_dirs=include_dirs
            # extra_compile_args={'cxx': ['-O2'],
            #                     'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
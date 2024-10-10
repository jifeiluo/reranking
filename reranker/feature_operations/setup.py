import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

setup(
    name='feature_operations',
    version='1.0',
    description='feature_operations',
    long_description='feature_operations',
    ext_modules=[
        CUDAExtension(
            name='feature_operations',
            sources=['feature_operations.cpp', 'feature_operations_kernel.cu'],
            include_dirs=include_dirs
            # extra_compile_args={'cxx': ['-O2'],
            #                     'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
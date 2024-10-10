import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

setup(
    name='sparse_divergence',
    version='1.0',
    description='sparse_divergence',
    long_description='sparse_divergence',
    ext_modules=[
        CUDAExtension(
            name='sparse_divergence',
            sources=['sparse_divergence.cpp', 'sparse_divergence_kernel.cu'],
            include_dirs=include_dirs
            # extra_compile_args={'cxx': ['-O2'],
            #                     'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
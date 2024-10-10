import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

setup(
    name='geodesic_distance',
    version='1.0',
    description='geodesic_distance',
    long_description='geodesic_distance',
    ext_modules=[
        CUDAExtension(
            name='geodesic_distance',
            sources=['geodesic_distance.cpp', 'geodesic_distance_kernel.cu'],
            include_dirs=include_dirs
            # extra_compile_args={'cxx': ['-O2'],
            #                     'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
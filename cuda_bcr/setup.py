from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# CUDA paths
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda-12.8')

ext_modules = [
    Pybind11Extension(
        "_cuda_bcr",
        ["src/python_bindings.cpp"],
        include_dirs=[
            "include",
            f"{cuda_home}/include",
        ],
        library_dirs=[f"{cuda_home}/lib64"],
        libraries=["cuda_bcr_lib", "cudart", "cublas"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="cuda_bcr",
    version="0.1.0",
    author="Your Name",
    description="CUDA-accelerated Block Cyclic Reduction for LQR",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pytest>=6.0",
    ],
)
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import glob
import shutil

class PrebuiltExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class PrebuiltBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # Look for pre-built module
        built_modules = glob.glob('python/_cuda_bcr*.so')
        if not built_modules:
            built_modules = glob.glob('build/_cuda_bcr*.so')
        
        if built_modules:
            print(f"Found pre-built module: {built_modules[0]}")
            # Copy to package directory
            dest_dir = os.path.join('python', 'cuda_bcr')
            os.makedirs(dest_dir, exist_ok=True)
            dest_file = os.path.join(dest_dir, os.path.basename(built_modules[0]))
            shutil.copy(built_modules[0], dest_file)
            print(f"Copied to: {dest_file}")
        else:
            print("\n" + "="*60)
            print("ERROR: Pre-built CUDA module not found!")
            print("="*60)
            print("\nPlease build the CUDA extension first:")
            print("  cd build")
            print("  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90")
            print("  make -j$(nproc)")
            print("  cd ..")
            print("\nOr run: ./build.sh")
            print("="*60 + "\n")
            sys.exit(1)

setup(
    name="cuda_bcr",
    version="0.1.0",
    author="Xiaoming Zhao",
    description="CUDA-accelerated Block Cyclic Reduction for LQR",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[PrebuiltExtension("cuda_bcr._cuda_bcr")],
    cmdclass={"build_ext": PrebuiltBuild},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "matplotlib>=3.0"],
    },
    include_package_data=True,
    zip_safe=False,
)
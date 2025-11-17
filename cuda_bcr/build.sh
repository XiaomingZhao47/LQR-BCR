#!/bin/bash
set -e

echo "Building CUDA BCR..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90

# Build
make -j$(nproc)

# Install Python package
cd ..
pip install -e .

echo "Build complete!"
echo "Run tests with: python tests/test_cuda_bcr.py"=
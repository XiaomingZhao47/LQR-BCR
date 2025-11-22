#!/bin/bash
set -e

echo "Building CUDA BCR..."
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Not in a virtual environment. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install numpy pytest matplotlib
    echo ""
fi

# Verify CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA is installed and in PATH."
    exit 1
fi

echo "CUDA version:"
nvcc --version | grep release
echo ""

# Create build directory
rm -rf build
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo ""
echo "Building..."
make -j$(nproc)

# Copy the built module to package directory
echo ""
echo "Installing Python module..."
cd ..
mkdir -p python/cuda_bcr
cp build/_cuda_bcr*.so python/cuda_bcr/

# Install Python package in development mode
pip install -e . --no-build-isolation

echo ""
echo "======================================"
echo "Build complete!"
echo "======================================"
echo ""
echo "Test with:"
echo "  python -c 'import cuda_bcr; print(cuda_bcr.is_cuda_available())'"
echo ""
echo "Run tests:"
echo "  python tests/test_cuda_bcr.py"
echo "  python tests/benchmark_vs_cpu.py"
echo ""
```

---

## Create MANIFEST.in

### `cuda_bcr/MANIFEST.in`
```
include README.md
include LICENSE
recursive-include include *.h *.cuh
recursive-include src *.cu *.cpp
recursive-include python/cuda_bcr *.so
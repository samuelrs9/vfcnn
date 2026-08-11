#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Voxel Fluid Net - Complete Setup"
echo "=========================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. Install system dependencies
echo "Step 1/5: Installing system dependencies..."
echo "Installing cmake, build-essential, gcc-10, g++-10..."
sudo apt-get update
sudo apt-get install -y cmake build-essential gcc-10 g++-10

# 2. Create and configure conda environment
echo ""
echo "Step 2/5: Setting up conda environment..."

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^vfnet "; then
    echo "Conda environment 'vfnet' already exists."
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n vfnet -y
    else
        echo "Using existing environment."
    fi
fi

if ! conda env list | grep -q "^vfnet "; then
    echo "Creating conda environment from vfnet_env_before_tf22_cuda101.yml..."
    conda env create -f vfnet_env_before_tf22_cuda101.yml
fi

# Activate environment
echo "Activating vfnet environment..."
eval "$(conda shell.bash hook)"
conda activate vfnet

# 3. Install Python dependencies
echo ""
echo "Step 3/5: Installing Python dependencies..."
echo "Installing packages from requirements.txt..."
export LDFLAGS="$LDFLAGS -Wl,--sysroot=$CONDA_BUILD_SYSROOT"
pip install --no-build-isolation -r requirements.txt

# 4. Build tf_kdtree library
echo ""
echo "Step 4/5: Building tf_kdtree library..."

cd tf_kdtree/src

# Modify CMakeLists.txt to add -allow-unsupported-compiler flag
if ! grep -q "set(CMAKE_CUDA_FLAGS.*-allow-unsupported-compiler" CMakeLists.txt; then
    echo "Patching CMakeLists.txt for CUDA compatibility..."
    sed -i '1a set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")' CMakeLists.txt
fi

# Build the library
echo "Configuring CMake..."
mkdir -p build
cd build
rm -rf *

CC=gcc-10 CXX=g++-10 CMAKE_CUDA_COMPILER=/usr/bin/nvcc cmake .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DPython_EXECUTABLE=$(which python) \
    -DCMAKE_CUDA_HOST_COMPILER=g++-10

echo "Compiling tf_kdtree (this may take a few minutes)..."
make -j$(nproc)

# Copy library to main directory
echo "Installing library..."
cp libtf_nndistance.so ../..

cd ../../..

# 5. Configure environment
echo ""
echo "Step 5/5: Configuring environment..."

# Set Python path
export PYTHONPATH="$SCRIPT_DIR/tf_kdtree:$SCRIPT_DIR/voxelizer:$SCRIPT_DIR/voxel-cloud-net:${PYTHONPATH}"

# Test import
echo "Testing library import..."
python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/tf_kdtree')
from tf_kdtree import nn_distance
print('✓ tf_kdtree imported successfully')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use this environment, run:"
echo "  conda activate vfnet"
echo "  export PYTHONPATH=$SCRIPT_DIR/tf_kdtree:$SCRIPT_DIR/voxelizer:$SCRIPT_DIR/voxel-cloud-net:\${PYTHONPATH}"
echo ""
echo "Or simply source this script:"
echo "  source setup.sh"
echo ""
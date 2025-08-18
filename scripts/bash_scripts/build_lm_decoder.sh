#!/bin/bash

# Build script for lm_decoder module
# This script builds the language model decoder C++ extension for Python
# Run this script if you get "ModuleNotFoundError: No module named 'lm_decoder'"

set -e  # Exit on any error

echo "=========================================="
echo "Building lm_decoder module..."
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ] && [ ! -d "src" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Navigate to the language model decoder directory
LM_DECODER_DIR="src/LanguageModelDecoder/runtime/server/x86"

if [ ! -d "$LM_DECODER_DIR" ]; then
    echo "Error: Language model decoder directory not found at $LM_DECODER_DIR"
    exit 1
fi

echo "Navigating to $LM_DECODER_DIR..."
cd "$LM_DECODER_DIR"

# Remove conflicting system packages that cause build issues
echo "Removing conflicting system packages..."
apt-get remove -y libgflags-dev libgoogle-glog-dev 2>/dev/null || true

# Clean previous build artifacts
echo "Cleaning previous build artifacts..."
rm -rf build fc_base
mkdir build

# Navigate to build directory
cd build

echo "Running cmake configuration..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Building with make (this may take several minutes)..."
make -j$(nproc)

# Check if the shared library was built successfully
if [ ! -f "lm_decoder.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "Error: Failed to build lm_decoder.cpython-310-x86_64-linux-gnu.so"
    exit 1
fi

echo "Installing lm_decoder module to Python path..."
cp lm_decoder.cpython-310-x86_64-linux-gnu.so /usr/local/lib/python3.10/dist-packages/

# Test the installation
echo "Testing lm_decoder import..."
cd /code
python -c "import lm_decoder; print('✓ lm_decoder module imported successfully!')" || {
    echo "Error: Failed to import lm_decoder module"
    exit 1
}

echo "=========================================="
echo "lm_decoder module built and installed successfully!"
echo "You can now run the inference script:"
echo "  bash scripts/bash_scripts/inference.sh"
echo "=========================================="

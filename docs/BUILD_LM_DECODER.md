# Building the Language Model Decoder

## Problem
When running inference scripts, you may encounter:
```
ModuleNotFoundError: No module named 'lm_decoder'
```

## Solution
The `lm_decoder` module is a C++ extension that needs to be compiled before use. This module provides language model decoding functionality for the neural sequence decoder.

## Quick Fix
Run the build script:
```bash
bash scripts/bash_scripts/build_lm_decoder.sh
```

## Manual Build Process
If the script doesn't work, you can build manually:

1. **Remove conflicting packages:**
   ```bash
   apt-get remove -y libgflags-dev libgoogle-glog-dev
   ```

2. **Navigate to the build directory:**
   ```bash
   cd src/LanguageModelDecoder/runtime/server/x86
   rm -rf build fc_base
   mkdir build && cd build
   ```

3. **Configure and build:**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

4. **Install the module:**
   ```bash
   cp lm_decoder.cpython-310-x86_64-linux-gnu.so /usr/local/lib/python3.10/dist-packages/
   ```

5. **Test the installation:**
   ```bash
   python -c "import lm_decoder; print('Success!')"
   ```

## Dependencies
The build process will automatically download and compile:
- gflags
- glog  
- googletest
- boost
- cnpy
- libtorch (PyTorch C++ API)
- OpenFST
- Kaldi components

## Notes
- The build process takes several minutes
- Requires CMake >= 3.14 and gcc >= 10.1
- The 3-gram language model should be present in `data/models/three_gram_lm/` before running inference
- This only needs to be done once per environment setup

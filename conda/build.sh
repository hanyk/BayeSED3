#!/bin/bash
# Conda build script for BayeSED3
# This script is called by conda-build during package creation

set -e

# setup.py will auto-detect the platform and install only appropriate binaries
# - Linux builds: only bin/linux/* installed
# - macOS builds: only bin/mac/* installed
# - Windows builds: blocked by meta.yaml (skip: True  # [win])

echo "Building BayeSED3 for $(uname) - setup.py will auto-detect platform"

# Install Python package with pip
# This installs:
# - Python package (bayesed/) via setuptools
# - Data files (bin/, models/, nets/, data/, etc.) via setup.py's data_files
# - Dependencies from pyproject.toml
$PYTHON -m pip install . --no-deps --ignore-installed -vv

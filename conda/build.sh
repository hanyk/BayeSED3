#!/bin/bash
# Conda build script for BayeSED3
# This script is called by conda-build during package creation

set -e

# Install Python package with pip
# This installs:
# - Python package (bayesed/) via setuptools
# - Data files (bin/, models/, nets/, data/, etc.) via setup.py's data_files
# - Dependencies from pyproject.toml
$PYTHON -m pip install . --no-deps --ignore-installed -vv

# Conda Package Files

This directory contains files needed to build BayeSED3 conda packages.

## Quick Start

```bash
# Build package
conda build conda/

# Install from local build
conda install --use-local bayesed3
```

## Essential Files

These files are **required** for conda builds:

- **`meta.yaml`** - Conda recipe (package metadata, dependencies, build config)
- **`build.sh`** - Build script (copies binaries and data files)
- **`setup.py`** - Python package setup (in repo root)

## Helper Scripts

- **`quick_update.sh`** - Quick rebuild after source changes
- **`update_package.sh`** - Interactive update script
- **`cleanup.sh`** - Clean build cache
- **`prepare_conda_forge.sh`** - Prepare conda-forge submission

## What Gets Included

The conda package automatically includes:
- All Python code (`bayesed/` package)
- Binaries (`bin/linux/` or `bin/mac/`)
- Data files (`models/`, `nets/`, `data/`)
- Git-tracked files (documentation, examples, tools)

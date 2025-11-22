#!/bin/bash
# Cleanup script for conda build artifacts and caches
# Usage: bash conda/cleanup.sh [--dry-run]

set -e

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
  DRY_RUN=true
  echo "DRY RUN MODE - No files will be deleted"
fi

echo "Conda Build and Install Cleanup"
echo "================================"
echo ""

# Check conda-bld directory
CONDA_BLD="$CONDA_PREFIX/conda-bld"
if [ -z "$CONDA_PREFIX" ]; then
  # Try common locations
  if [ -d "$HOME/miniforge3/conda-bld" ]; then
    CONDA_BLD="$HOME/miniforge3/conda-bld"
  elif [ -d "$HOME/miniconda3/conda-bld" ]; then
    CONDA_BLD="$HOME/miniconda3/conda-bld"
  elif [ -d "$HOME/anaconda3/conda-bld" ]; then
    CONDA_BLD="$HOME/anaconda3/conda-bld"
  fi
fi

if [ -d "$CONDA_BLD" ]; then
  SIZE=$(du -sh "$CONDA_BLD" 2>/dev/null | cut -f1)
  echo "Found conda-bld directory: $CONDA_BLD"
  echo "Size: $SIZE"
  echo ""
  
  if [ "$DRY_RUN" = false ]; then
    echo "Cleaning conda build cache..."
    if command -v conda >/dev/null 2>&1; then
      conda build purge 2>&1 || echo "  Note: conda build purge may have failed (this is OK)"
    fi
    
    echo "Removing conda-bld directory..."
    rm -rf "$CONDA_BLD"
    echo "  ✓ Removed $CONDA_BLD"
  else
    echo "  [DRY RUN] Would remove: $CONDA_BLD"
  fi
else
  echo "conda-bld directory not found (may already be clean)"
fi

echo ""

# Clean conda package cache
if command -v conda >/dev/null 2>&1; then
  echo "Cleaning conda package cache..."
  if [ "$DRY_RUN" = false ]; then
    conda clean --all --yes 2>&1 | grep -E "(Cache|Removed|Total)" || echo "  Cache cleaned"
  else
    echo "  [DRY RUN] Would run: conda clean --all --yes"
    conda clean --all --dry-run 2>&1 | head -10 || true
  fi
else
  echo "conda command not found, skipping cache cleanup"
fi

echo ""
echo "Cleanup complete!"


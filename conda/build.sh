#!/bin/bash
# Conda build script for BayeSED3
# This script is called by conda-build during package creation
#
# Strategy: Automatically include most git-tracked files, with sensible exclusions

set -e

# Install Python package
$PYTHON -m pip install . --no-deps --ignore-installed -vv

# Create share directory
mkdir -p $PREFIX/share/bayesed3

# Copy binaries to share/bayesed3/bin/
# IMPORTANT: Only copy git-tracked binaries to avoid including untracked files
mkdir -p $PREFIX/share/bayesed3/bin
if [ "$(uname)" == "Linux" ]; then
  mkdir -p $PREFIX/share/bayesed3/bin/linux
  git ls-files bin/linux/ | while IFS= read -r file; do
    [ -z "$file" ] || [ ! -f "$file" ] && continue
    dest_file="$PREFIX/share/bayesed3/$file"
    mkdir -p "$(dirname "$dest_file")"
    cp "$file" "$dest_file"
  done
elif [ "$(uname)" == "Darwin" ]; then
  mkdir -p $PREFIX/share/bayesed3/bin/mac
  git ls-files bin/mac/ | while IFS= read -r file; do
    [ -z "$file" ] || [ ! -f "$file" ] && continue
    dest_file="$PREFIX/share/bayesed3/$file"
    mkdir -p "$(dirname "$dest_file")"
    cp "$file" "$dest_file"
  done
fi

# Copy models, nets, and data directories (core data)
# IMPORTANT: Only copy git-tracked files to avoid including untracked files
if [ -d "$SRC_DIR/models" ]; then
  mkdir -p $PREFIX/share/bayesed3/models
  git ls-files models/ | while IFS= read -r file; do
    [ -z "$file" ] || [ ! -f "$file" ] && continue
    dest_file="$PREFIX/share/bayesed3/$file"
    mkdir -p "$(dirname "$dest_file")"
    cp "$file" "$dest_file"
  done
fi
if [ -d "$SRC_DIR/nets" ]; then
  mkdir -p $PREFIX/share/bayesed3/nets
  git ls-files nets/ | while IFS= read -r file; do
    [ -z "$file" ] || [ ! -f "$file" ] && continue
    dest_file="$PREFIX/share/bayesed3/$file"
    mkdir -p "$(dirname "$dest_file")"
    cp "$file" "$dest_file"
  done
fi
if [ -d "$SRC_DIR/data" ]; then
  mkdir -p $PREFIX/share/bayesed3/data
  git ls-files data/ | while IFS= read -r file; do
    [ -z "$file" ] || [ ! -f "$file" ] && continue
    dest_file="$PREFIX/share/bayesed3/$file"
    mkdir -p "$(dirname "$dest_file")"
    cp "$file" "$dest_file"
  done
fi

# Automatically copy most git-tracked files, excluding build artifacts and unnecessary files
cd $SRC_DIR

# Get list of git-tracked files
if command -v git >/dev/null 2>&1 && [ -d .git ]; then
  echo "Copying git-tracked files to conda package..."
  
  # Patterns to exclude (files/directories that shouldn't be in package)
  EXCLUDE_PATTERNS=(
    "^\.git/"
    "^__pycache__/"
    "^build/"
    "^dist/"
    "^\.eggs/"
    "^.*\.egg-info/"
    "^conda/"
    "^openmpi/"
    "^output/"
    "^output1/"
    "^test[0-9]/"
    "^log/"
    "^\.idea/"
    "^\.vscode/"
    "^\.pytest_cache/"
    "^\.mypy_cache/"
    "^\.coverage"
    "^htmlcov/"
    "^\.tox/"
    "^\.nox/"
    "^\.env"
    "^\.venv/"
    "^venv/"
    "^env/"
    "^ENV/"
    "^\.swp$"
    "^\.swo$"
    "^~$"
    "^\.DS_Store$"
    "^Thumbs\.db$"
  )
  
  # Files already handled by pip install (don't copy these)
  SKIP_FILES=(
    "^bayesed/.*\.py$"  # Python package files installed via pip
    "^bayesed_gui\.py$"  # GUI module installed via pip
  )
  
  # Get all git-tracked files and process them
  # IMPORTANT: git ls-files only returns files tracked by git
  git ls-files | while IFS= read -r file; do
    # Skip empty lines
    [ -z "$file" ] && continue
    
    # Skip if file doesn't exist
    [ ! -e "$file" ] && continue
    
    # Skip directories (git ls-files lists files, but be safe)
    [ -d "$file" ] && continue
    
    # Safety check: Verify file is actually tracked by git
    # (git ls-files should already ensure this, but double-check)
    if ! git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
      echo "  Warning: Skipping untracked file: $file"
      continue
    fi
    
    # Check exclude patterns
    exclude=false
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
      if echo "$file" | grep -qE "$pattern"; then
        exclude=true
        break
      fi
    done
    [ "$exclude" = true ] && continue
    
    # Check skip files (already handled by pip)
    skip=false
    for pattern in "${SKIP_FILES[@]}"; do
      if echo "$file" | grep -qE "$pattern"; then
        skip=true
        break
      fi
    done
    [ "$skip" = true ] && continue
    
    # Skip binaries and core data (already copied above)
    case "$file" in
      bin/linux/*|bin/mac/*|models/*|nets/*|data/*)
        continue
        ;;
    esac
    
    # Copy file maintaining directory structure
    dest_file="$PREFIX/share/bayesed3/$file"
    dest_dir=$(dirname "$dest_file")
    mkdir -p "$dest_dir"
    cp "$file" "$dest_file"
    echo "  Copied: $file"
  done
  
  echo "Finished copying git-tracked files."
else
  echo "Warning: git not available or not in a git repository. Skipping automatic file copying."
  echo "Only core files (binaries, models, nets, data) will be included."
fi

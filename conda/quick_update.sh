#!/bin/bash
# Quick update script - minimal prompts
# Usage: bash conda/quick_update.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Get version from bayesed/__init__.py
VERSION=$(grep -E "^__version__\s*=" bayesed/__init__.py | sed "s/.*['\"]\(.*\)['\"].*/\1/" || echo "3.0.0")

echo "Building BayeSED3 conda package v$VERSION..."

# Update meta.yaml version to match
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/^\(\s*version:\s*\).*/\1$VERSION/" conda/meta.yaml
else
    sed -i "s/^\(\s*version:\s*\).*/\1$VERSION/" conda/meta.yaml
fi

# Build
conda build conda/ --no-test

# Install
PACKAGE_FILE=$(conda build conda/ --output)
conda install --offline "$PACKAGE_FILE" --force-reinstall -y

echo "✓ Done! Package updated and installed."


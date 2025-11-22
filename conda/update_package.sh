#!/bin/bash
# Automated script to update conda package after source code changes
# Usage: bash conda/update_package.sh [version]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Get version from command line or detect from bayesed/__init__.py
if [ -n "$1" ]; then
    NEW_VERSION="$1"
else
    # Extract version from bayesed/__init__.py
    NEW_VERSION=$(grep -E "^__version__\s*=" bayesed/__init__.py | sed "s/.*['\"]\(.*\)['\"].*/\1/" || echo "")
    if [ -z "$NEW_VERSION" ]; then
        echo "Error: Could not detect version from bayesed/__init__.py"
        echo "Usage: $0 [version]"
        exit 1
    fi
fi

echo "=========================================="
echo "Updating Conda Package for BayeSED3"
echo "=========================================="
echo "Version: $NEW_VERSION"
echo ""

# Step 1: Verify version in bayesed/__init__.py matches
CURRENT_VERSION=$(grep -E "^__version__\s*=" bayesed/__init__.py | sed "s/.*['\"]\(.*\)['\"].*/\1/" || echo "")
if [ "$CURRENT_VERSION" != "$NEW_VERSION" ]; then
    echo "⚠️  Warning: Version mismatch!"
    echo "   bayesed/__init__.py: $CURRENT_VERSION"
    echo "   Requested: $NEW_VERSION"
    read -p "Update bayesed/__init__.py to $NEW_VERSION? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/^__version__\s*=.*/__version__ = \"$NEW_VERSION\"/" bayesed/__init__.py
        else
            sed -i "s/^__version__\s*=.*/__version__ = \"$NEW_VERSION\"/" bayesed/__init__.py
        fi
        echo "✓ Updated bayesed/__init__.py"
    fi
fi

# Step 2: Update conda/meta.yaml version
META_VERSION=$(grep -E "^\s*version:" conda/meta.yaml | sed "s/.*:\s*\(.*\)/\1/" | tr -d ' ' || echo "")
if [ "$META_VERSION" != "$NEW_VERSION" ]; then
    echo "Updating conda/meta.yaml version: $META_VERSION -> $NEW_VERSION"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/^\(\s*version:\s*\).*/\1$NEW_VERSION/" conda/meta.yaml
    else
        sed -i "s/^\(\s*version:\s*\).*/\1$NEW_VERSION/" conda/meta.yaml
    fi
    echo "✓ Updated conda/meta.yaml"
else
    echo "✓ conda/meta.yaml version already correct"
fi

# Step 3: Check for uncommitted changes
if ! git diff --quiet; then
    echo ""
    echo "⚠️  Warning: You have uncommitted changes!"
    echo "   Consider committing changes before building package"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 4: Clean old builds (optional)
read -p "Clean old conda builds? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning old builds..."
    conda build purge 2>&1 | grep -v "WARNING" || true
    echo "✓ Cleaned old builds"
fi

# Step 5: Build package
echo ""
echo "Building conda package..."
echo "This may take several minutes..."
conda build conda/ --no-test

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    
    # Step 6: Offer to install
    read -p "Install updated package? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing package..."
        PACKAGE_FILE=$(conda build conda/ --output)
        conda install --offline "$PACKAGE_FILE" --force-reinstall -y
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "=========================================="
            echo "✓ Package updated and installed!"
            echo "=========================================="
            echo ""
            echo "Verify installation:"
            echo "  python -c \"import bayesed; print(bayesed.__version__)\""
            echo "  BayeSED3_GUI --help"
        else
            echo "⚠️  Installation failed. You can install manually:"
            echo "  conda install --use-local bayesed3 --force-reinstall -y"
        fi
    else
        echo ""
        echo "Package built successfully!"
        echo "Install with: conda install --use-local bayesed3 --force-reinstall -y"
    fi
else
    echo ""
    echo "❌ Build failed. Check errors above."
    exit 1
fi


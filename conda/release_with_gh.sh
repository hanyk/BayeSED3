#!/bin/bash
# Fully automated release workflow using GitHub CLI (gh)
# Requires: gh CLI installed and authenticated

set -e

echo "=========================================="
echo "BayeSED3 Automated Release (gh CLI)"
echo "=========================================="
echo ""

# Check if gh is installed
if ! command -v gh >/dev/null 2>&1; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo ""
    echo "Install it with:"
    echo "  macOS:   brew install gh"
    echo "  Linux:   See https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
    echo "  Windows: See https://github.com/cli/cli#windows"
    echo ""
    exit 1
fi

# Check if gh is authenticated
if ! gh auth status >/dev/null 2>&1; then
    echo "Error: GitHub CLI is not authenticated"
    echo ""
    echo "Run: gh auth login"
    echo ""
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "bayesed/__init__.py" ]; then
    echo "Error: Must run from BayeSED3 root directory"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(grep -oP "__version__ = ['\"]([^'\"]+)['\"]" bayesed/__init__.py | grep -oP "[0-9]+\.[0-9]+\.[0-9]+")
echo "Current version: $CURRENT_VERSION"
echo ""

# Ask for new version
read -p "Enter new version (e.g., 3.0.1): " NEW_VERSION

if [ -z "$NEW_VERSION" ]; then
    echo "Error: Version cannot be empty"
    exit 1
fi

# Validate version format
if ! echo "$NEW_VERSION" | grep -qE "^[0-9]+\.[0-9]+\.[0-9]+$"; then
    echo "Error: Version must be in format X.Y.Z (e.g., 3.0.1)"
    exit 1
fi

echo ""
echo "Preparing release $NEW_VERSION..."
echo ""

# Update version in __init__.py
echo "1. Updating version in bayesed/__init__.py..."
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" bayesed/__init__.py
rm bayesed/__init__.py.bak
echo "   ✓ Updated to $NEW_VERSION"

# Update version in conda/meta.yaml
echo "2. Updating version in conda/meta.yaml..."
sed -i.bak "s/{% set version = \".*\" %}/{% set version = \"$NEW_VERSION\" %}/" conda/meta.yaml
rm conda/meta.yaml.bak
echo "   ✓ Updated to $NEW_VERSION"

# Show changes
echo ""
echo "3. Changes to commit:"
git diff bayesed/__init__.py conda/meta.yaml

echo ""
read -p "Commit these changes? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo "Aborted. Reverting changes..."
    git checkout bayesed/__init__.py conda/meta.yaml
    exit 1
fi

# Commit changes
echo ""
echo "4. Committing changes..."
git add bayesed/__init__.py conda/meta.yaml
git commit -m "Bump version to $NEW_VERSION"
echo "   ✓ Committed"

# Create git tag
echo ""
echo "5. Creating git tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Release $NEW_VERSION"
echo "   ✓ Tag created"

# Push changes
echo ""
echo "6. Pushing to GitHub..."
git push origin main
git push origin "v$NEW_VERSION"
echo "   ✓ Pushed to GitHub"

# Wait a moment for GitHub to process
sleep 2

# Ask for release notes
echo ""
echo "7. Creating GitHub release..."
echo ""
echo "Enter release notes (press Ctrl+D when done, or leave empty for default):"
echo "---"

# Read multi-line input
RELEASE_NOTES_INPUT=$(cat)

if [ -z "$RELEASE_NOTES_INPUT" ]; then
    # Default release notes
    RELEASE_NOTES="## What's New in $NEW_VERSION

### Installation

\`\`\`bash
conda install -c conda-forge bayesed3
\`\`\`

### Changes

- Bug fixes and improvements

### Documentation

See the [README](https://github.com/hanyk/BayeSED3) for usage examples.
"
else
    RELEASE_NOTES="$RELEASE_NOTES_INPUT"
fi

# Create release
echo ""
echo "Creating release on GitHub..."
echo "$RELEASE_NOTES" | gh release create "v$NEW_VERSION" \
    --title "Release $NEW_VERSION" \
    --notes-file -

echo ""
echo "=========================================="
echo "✓ Release $NEW_VERSION published!"
echo "=========================================="
echo ""
echo "View release: https://github.com/hanyk/BayeSED3/releases/tag/v$NEW_VERSION"
echo ""
echo "Next steps:"
echo ""
echo "1. conda-forge bot will detect the release (within a few hours)"
echo "2. Bot will create a PR to conda-forge/bayesed3-feedstock"
echo "3. Review and merge the PR"
echo "4. Package will be built and published automatically"
echo ""
echo "Monitor feedstock PRs:"
echo "  https://github.com/conda-forge/bayesed3-feedstock/pulls"
echo ""
echo "=========================================="

# Calculate and show SHA256
echo ""
echo "Calculating SHA256 for reference..."
TARBALL_URL="https://github.com/hanyk/BayeSED3/archive/refs/tags/v${NEW_VERSION}.tar.gz"
SHA256=$(curl -sL "$TARBALL_URL" | shasum -a 256 | cut -d' ' -f1)
echo "SHA256: $SHA256"
echo ""
echo "(conda-forge bot will calculate this automatically)"
echo "=========================================="

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
CURRENT_VERSION=$(grep "__version__" bayesed/__init__.py | sed -E 's/.*"([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
echo "Current version: $CURRENT_VERSION"
echo ""

# Generate date-based default version (YYYY.MM.DD format)
DEFAULT_VERSION=$(date +"%Y.%m.%d")

# Ask for new version with date-based default
read -p "Enter new version (default: $DEFAULT_VERSION): " NEW_VERSION

# Use default if empty
if [ -z "$NEW_VERSION" ]; then
    NEW_VERSION="$DEFAULT_VERSION"
    echo "Using default version: $NEW_VERSION"
fi

# Validate version format (allow YYYY.MM.DD or X.Y.Z)
if ! echo "$NEW_VERSION" | grep -qE "^[0-9]+\.[0-9]+\.[0-9]+$"; then
    echo "Error: Version must be in format X.Y.Z or YYYY.MM.DD (e.g., 2026.01.30)"
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

# Check if there are changes
echo ""
echo "3. Checking for changes..."
if git diff --quiet bayesed/__init__.py conda/meta.yaml; then
    echo "   ℹ No changes detected (version already set to $NEW_VERSION)"
    SKIP_COMMIT=true
else
    echo "   Changes to commit:"
    git diff bayesed/__init__.py conda/meta.yaml
    
    echo ""
    read -p "Commit these changes? (y/n): " CONFIRM
    
    if [ "$CONFIRM" != "y" ]; then
        echo "Aborted. Reverting changes..."
        git checkout bayesed/__init__.py conda/meta.yaml
        exit 1
    fi
    SKIP_COMMIT=false
fi

# Commit changes if needed
if [ "$SKIP_COMMIT" = false ]; then
    echo ""
    echo "4. Committing changes..."
    git add bayesed/__init__.py conda/meta.yaml
    git commit -m "Bump version to $NEW_VERSION"
    echo "   ✓ Committed"
else
    echo ""
    echo "4. Skipping commit (no changes)"
fi

# Check if tag already exists
echo ""
echo "5. Checking git tag v$NEW_VERSION..."
if git rev-parse "v$NEW_VERSION" >/dev/null 2>&1; then
    echo "   ⚠ Tag v$NEW_VERSION already exists locally"
    read -p "Delete and recreate tag? (y/n): " RECREATE_TAG
    
    if [ "$RECREATE_TAG" = "y" ]; then
        git tag -d "v$NEW_VERSION"
        echo "   ✓ Deleted old tag"
        git tag -a "v$NEW_VERSION" -m "Release $NEW_VERSION"
        echo "   ✓ Tag recreated"
    else
        echo "   Using existing tag"
    fi
else
    git tag -a "v$NEW_VERSION" -m "Release $NEW_VERSION"
    echo "   ✓ Tag created"
fi

# Push changes
echo ""
echo "6. Pushing to GitHub..."
if [ "$SKIP_COMMIT" = false ]; then
    git push origin main
    echo "   ✓ Pushed commits to main"
else
    echo "   Skipping push of main (no new commits)"
fi

# Push tag (with force if recreated)
if git ls-remote --tags origin | grep -q "refs/tags/v$NEW_VERSION"; then
    echo "   ⚠ Tag v$NEW_VERSION already exists on GitHub"
    read -p "Force push tag? (y/n): " FORCE_PUSH
    
    if [ "$FORCE_PUSH" = "y" ]; then
        git push --force origin "v$NEW_VERSION"
        echo "   ✓ Force pushed tag to GitHub"
    else
        echo "   Using existing remote tag"
    fi
else
    git push origin "v$NEW_VERSION"
    echo "   ✓ Pushed tag to GitHub"
fi

# Wait a moment for GitHub to process
sleep 2

# Check if release already exists FIRST
echo ""
echo "7. Checking GitHub release..."
if gh release view "v$NEW_VERSION" >/dev/null 2>&1; then
    echo "   ⚠ Release v$NEW_VERSION already exists on GitHub"
    echo ""
    read -p "Update release notes? (y/n): " UPDATE_RELEASE
    
    if [ "$UPDATE_RELEASE" != "y" ]; then
        echo "   Keeping existing release unchanged"
        echo ""
        echo "=========================================="
        echo "✓ Release $NEW_VERSION already published!"
        echo "=========================================="
        echo ""
        echo "View release: https://github.com/hanyk/BayeSED3/releases/tag/v$NEW_VERSION"
        exit 0
    fi
    
    echo ""
    echo "Enter new release notes (press Ctrl+D when done, or leave empty to keep existing):"
    echo "---"
    RELEASE_NOTES_INPUT=$(cat)
    
    if [ -z "$RELEASE_NOTES_INPUT" ]; then
        echo "   Keeping existing release notes"
        exit 0
    fi
    
    RELEASE_NOTES="$RELEASE_NOTES_INPUT"
    
    echo ""
    echo "Updating release notes on GitHub..."
    echo "$RELEASE_NOTES" | gh release edit "v$NEW_VERSION" --notes-file -
    echo "   ✓ Release notes updated"
else
    # New release - ask for notes
    echo "   Release does not exist, creating new release..."
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
    
    echo ""
    echo "Creating release on GitHub..."
    echo "$RELEASE_NOTES" | gh release create "v$NEW_VERSION" \
        --title "Release $NEW_VERSION" \
        --notes-file -
    echo "   ✓ Release created"
fi

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

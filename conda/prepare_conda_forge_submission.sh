#!/bin/bash
# Script to prepare BayeSED3 for conda-forge submission
# This helps you create the initial conda-forge recipe
# Usage: bash conda/prepare_conda_forge_submission.sh [version]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "BayeSED3 conda-forge Submission Helper"
echo "=========================================="
echo ""

# Get version from argument or detect from code
# Use sed for better cross-platform compatibility (works on both Linux and macOS)
if [ -n "$1" ]; then
    VERSION="$1"
else
    VERSION=$(sed -n 's/^__version__ = ['\''\"]\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)['\''\"]/\1/p' bayesed/__init__.py)
fi

if [ -z "$VERSION" ]; then
    echo "Error: Could not detect version"
    echo "Usage: $0 [version]"
    exit 1
fi

echo "Preparing conda-forge recipe for version: $VERSION"
echo ""

# Check if git repo is clean
if ! git diff --quiet 2>/dev/null; then
    echo "⚠️  Warning: You have uncommitted changes!"
    echo "   Consider committing before creating release"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if this version has a git tag
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo "✓ Git tag v$VERSION exists"
else
    echo "⚠️  Git tag v$VERSION does NOT exist"
    echo ""
    read -p "Create tag v$VERSION now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -a "v$VERSION" -m "Release version $VERSION"
        echo "✓ Created tag v$VERSION"
        echo "  Remember to push: git push origin v$VERSION"
        echo ""
    else
        echo ""
        echo "You need to create a git tag and GitHub release first:"
        echo "  git tag -a v$VERSION -m 'Release $VERSION'"
        echo "  git push origin v$VERSION"
        echo ""
        echo "Then create a GitHub release at:"
        echo "  https://github.com/hanyk/BayeSED3/releases/new"
        echo ""
        exit 1
    fi
fi

# Calculate SHA256 of the release tarball
echo ""
echo "Calculating SHA256 hash of release tarball..."
TARBALL_URL="https://github.com/hanyk/BayeSED3/archive/refs/tags/v${VERSION}.tar.gz"
SHA256=$(curl -sL "$TARBALL_URL" | shasum -a 256 | cut -d' ' -f1)

if [ -z "$SHA256" ]; then
    echo "✗ Failed to download or calculate SHA256"
    echo "  Make sure the GitHub release exists at:"
    echo "  $TARBALL_URL"
    echo ""
    echo "  You can calculate it manually after creating the release:"
    echo "  curl -sL $TARBALL_URL | shasum -a 256"
    SHA256="PLACEHOLDER_SHA256"
else
    echo "✓ SHA256: $SHA256"
fi
echo ""

# Create conda-forge recipe directory
RECIPE_DIR="conda_forge_recipe"
rm -rf "$RECIPE_DIR"
mkdir -p "$RECIPE_DIR"

echo "Creating conda-forge recipe in $RECIPE_DIR/..."

# Generate meta.yaml for conda-forge
# Note: Dependencies are read from pyproject.toml via pip
cat > "$RECIPE_DIR/meta.yaml" << EOF
{% set name = "bayesed3" %}
{% set version = "$VERSION" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://github.com/hanyk/BayeSED3/archive/refs/tags/v{{ version }}.tar.gz
  sha256: $SHA256

build:
  number: 0
  skip: true  # [win]

requirements:
  build:
    - {{ compiler('c') }}
    - {{ compiler('fortran') }}
    - {{ stdlib("c") }}
  host:
    - python
    - pip
    - setuptools
  run:
    - openmpi 4.1.6
    - python
    # Python dependencies from pyproject.toml
    - numpy >=1.20.0
    - h5py >=3.1.0
    - astropy-base >=4.2
    - matplotlib-base >=3.3.0
    - getdist >=1.3.0
    - requests >=2.25.0
    - tqdm >=4.60.0

test:
  imports:
    - bayesed
  commands:
    - python -c "import bayesed; print(bayesed.__version__)"
    - python -c "from bayesed import BayeSEDInterface; print('BayeSEDInterface imported successfully')"

about:
  home: https://github.com/hanyk/BayeSED3
  license: MIT
  license_file: LICENSE
  summary: Bayesian SED synthesis and analysis tool for galaxies and AGNs with full posterior inference and model comparison
  description: |
    BayeSED3 is a sophisticated Bayesian tool for interpreting spectral energy distributions (SEDs) 
    of galaxies and Active Galactic Nuclei (AGNs). It performs rigorous Bayesian parameter estimation 
    using posterior probability distributions and model comparison via Bayesian evidence calculation.
    
    Key Capabilities:
    - Multi-component SED modeling: stellar populations, dust attenuation/emission, AGN components
    - Flexible star formation histories (exponential, delayed, non-parametric)
    - Comprehensive AGN modeling (accretion disk, BLR/NLR, torus, FeII emission)
    - Handles photometric and spectroscopic data (individual or combined)
    - Machine learning-based SED emulation (FANN, AKNN) for computational efficiency
    - MPI-based parallel processing for large datasets
    - Nested sampling via MultiNest for robust Bayesian inference
    - Python API with high-level interface and comprehensive result analysis
    - GetDist integration for advanced posterior visualization
    - Optional GUI for interactive model configuration
    
    Scientific Applications:
    - Galaxy stellar mass, age, and star formation rate estimation
    - AGN-host galaxy decomposition
    - Dust properties and attenuation curves
    - Model comparison using Bayesian evidence
    - Mock survey analysis and forecasting
    
    Platform Support: Linux x86_64, macOS x86_64/ARM64 (via Rosetta 2), Windows (via WSL)
    
    Citation: Han & Han 2012 (ApJ 749, 123); Han & Han 2014 (ApJS 215, 2); 
    Han & Han 2019 (ApJS 240, 3); Han et al. 2023 (ApJS 269, 39)
  doc_url: https://github.com/hanyk/BayeSED3
  dev_url: https://github.com/hanyk/BayeSED3

extra:
  recipe-maintainers:
    - hanyk
EOF

# Create a simplified build script for conda-forge
# Note: conda-forge builds from GitHub release tarballs (created via git archive)
# which only contain git-tracked files, so cp -r is safe here
cat > "$RECIPE_DIR/build.sh" << 'EOF'
#!/bin/bash
set -e

# Install Python package WITH dependencies
# Dependencies are read from pyproject.toml
$PYTHON -m pip install . --ignore-installed -vv

# Create share directory for data files
mkdir -p $PREFIX/share/bayesed3

# Copy binaries
# Note: Source is a GitHub release tarball which only contains git-tracked files
if [ "$(uname)" == "Linux" ]; then
  if [ -d "bin/linux" ]; then
    mkdir -p $PREFIX/share/bayesed3/bin/linux
    cp -r bin/linux/* $PREFIX/share/bayesed3/bin/linux/
  fi
elif [ "$(uname)" == "Darwin" ]; then
  if [ -d "bin/mac" ]; then
    mkdir -p $PREFIX/share/bayesed3/bin/mac
    cp -r bin/mac/* $PREFIX/share/bayesed3/bin/mac/
  fi
fi

# Copy data directories
# Note: Only git-tracked files are in the tarball, so this is safe
for dir in models nets data filters observation papers docs plot tools; do
  if [ -d "$dir" ]; then
    cp -r "$dir" $PREFIX/share/bayesed3/
  fi
done
EOF

chmod +x "$RECIPE_DIR/build.sh"

# Create detailed instructions
cat > "$RECIPE_DIR/INSTRUCTIONS.md" << EOF
# Conda-Forge Submission Instructions for BayeSED3 v$VERSION

## Files Prepared

- \`meta.yaml\` - Conda recipe with version $VERSION and SHA256
- \`build.sh\` - Build script for conda-forge

## Prerequisites

1. **GitHub Release Must Exist**
   - Tag: v$VERSION
   - URL: https://github.com/hanyk/BayeSED3/releases/tag/v$VERSION
   - If not created yet, run: \`./release_with_gh.sh\`

2. **Fork staged-recipes**
   - Go to: https://github.com/conda-forge/staged-recipes
   - Click "Fork" button

## Step-by-Step Submission

### 1. Clone Your Fork

\`\`\`bash
git clone https://github.com/YOUR_USERNAME/staged-recipes.git
cd staged-recipes
git checkout -b bayesed3
\`\`\`

### 2. Copy Recipe Files

\`\`\`bash
mkdir -p recipes/bayesed3
cp $(pwd)/../$RECIPE_DIR/meta.yaml recipes/bayesed3/
cp $(pwd)/../$RECIPE_DIR/build.sh recipes/bayesed3/
\`\`\`

### 3. Verify SHA256 (if needed)

If SHA256 was PLACEHOLDER_SHA256, calculate it:

\`\`\`bash
curl -sL https://github.com/hanyk/BayeSED3/archive/refs/tags/v$VERSION.tar.gz | shasum -a 256
\`\`\`

Then update \`recipes/bayesed3/meta.yaml\` with the correct hash.

### 4. Test Build Locally (Optional but Recommended)

\`\`\`bash
conda build recipes/bayesed3
\`\`\`

### 5. Commit and Push

\`\`\`bash
git add recipes/bayesed3/
git commit -m "Add bayesed3 recipe"
git push origin bayesed3
\`\`\`

### 6. Create Pull Request

1. Go to: https://github.com/conda-forge/staged-recipes
2. Click "New Pull Request"
3. Select your fork and the \`bayesed3\` branch
4. Fill out the PR template:
   - Describe the package
   - Confirm you've read the guidelines
   - List yourself as maintainer
5. Submit the PR

### 7. Wait for Review

- conda-forge bots will run automated checks
- Maintainers will review (usually 1-7 days)
- Address any feedback or requested changes
- Once approved and merged, your package will be on conda-forge!

## After Acceptance

Once your PR is merged:

1. **Feedstock Created**
   - A new repo will be created: \`conda-forge/bayesed3-feedstock\`
   - You'll be added as a maintainer

2. **Future Updates Are Automatic**
   - Just create GitHub releases with \`./release_with_gh.sh\`
   - conda-forge bot detects releases and creates PRs automatically
   - You review and merge the bot's PRs
   - Package updates automatically

## Troubleshooting

**Q: Build fails with "sha256 mismatch"**

A: Recalculate SHA256:
\`\`\`bash
curl -sL https://github.com/hanyk/BayeSED3/archive/refs/tags/v$VERSION.tar.gz | shasum -a 256
\`\`\`

**Q: Tests fail**

A: Check the CI logs for details. Common issues:
- Missing dependencies
- Import errors
- Platform-specific problems

**Q: Linter errors**

A: conda-forge has strict formatting requirements. Follow the error messages to fix.

## Resources

- **conda-forge docs:** https://conda-forge.org/docs/
- **Staged recipes:** https://github.com/conda-forge/staged-recipes
- **Guidelines:** https://conda-forge.org/docs/maintainer/adding_pkgs.html

## Current Recipe Details

- **Version:** $VERSION
- **SHA256:** $SHA256
- **Source URL:** https://github.com/hanyk/BayeSED3/archive/refs/tags/v$VERSION.tar.gz
EOF

echo "✓ Recipe created in $RECIPE_DIR/"
echo ""

echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Fork conda-forge/staged-recipes:"
echo "   https://github.com/conda-forge/staged-recipes"
echo ""
echo "2. Clone your fork and create a branch:"
echo "   git clone https://github.com/YOUR_USERNAME/staged-recipes.git"
echo "   cd staged-recipes"
echo "   git checkout -b bayesed3"
echo ""
echo "3. Copy the recipe:"
echo "   cp -r $(pwd)/$RECIPE_DIR recipes/bayesed3"
echo ""
echo "4. Review and edit recipes/bayesed3/meta.yaml:"
echo "   - Verify all dependencies"
echo "   - Add co-maintainers if any"
echo "   - Check license file path"
echo ""
echo "5. Test the recipe locally (optional but recommended):"
echo "   cd staged-recipes"
echo "   conda build recipes/bayesed3"
echo ""
echo "6. Commit and push:"
echo "   git add recipes/bayesed3"
echo "   git commit -m 'Add bayesed3 recipe'"
echo "   git push origin bayesed3"
echo ""
echo "7. Create Pull Request:"
echo "   Go to your fork on GitHub and create a PR to conda-forge/staged-recipes"
echo ""
echo "8. Wait for review (usually 1-7 days)"
echo "   - conda-forge bots will run tests"
echo "   - Maintainers will review your recipe"
echo "   - Address any feedback"
echo ""
echo "9. After merge:"
echo "   - Your package will be on conda-forge!"
echo "   - A feedstock repo will be created: conda-forge/bayesed3-feedstock"
echo "   - Future updates will be automatic via bot PRs"
echo ""
echo "=========================================="
echo "For more details, see: conda/README_CONDA_FORGE.md"
echo "=========================================="

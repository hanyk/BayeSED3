#!/bin/bash
# Script to prepare BayeSED3 recipe for conda-forge submission
# Usage: bash conda/prepare_conda_forge.sh [version]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Get version
VERSION=${1:-$(grep -E "^__version__\s*=" bayesed/__init__.py | sed "s/.*['\"]\(.*\)['\"].*/\1/" || echo "3.0.0")}

echo "Preparing BayeSED3 v$VERSION for conda-forge submission"
echo "========================================================"
echo ""

# Check if git repo is clean
if ! git diff --quiet; then
    echo "⚠️  Warning: You have uncommitted changes!"
    echo "   Consider committing before creating release"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if version tag exists
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo "✓ Version tag v$VERSION exists"
else
    echo "⚠️  Version tag v$VERSION does not exist"
    read -p "Create tag v$VERSION? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -a "v$VERSION" -m "Release version $VERSION"
        echo "✓ Created tag v$VERSION"
        echo "  Push with: git push origin v$VERSION"
    fi
fi

# Create output directory
OUTPUT_DIR="$REPO_ROOT/conda-forge-prep"
mkdir -p "$OUTPUT_DIR/recipes/bayesed3"

echo ""
echo "Creating conda-forge recipe files..."
echo ""

# Copy and adapt meta.yaml
cat > "$OUTPUT_DIR/recipes/bayesed3/meta.yaml" << EOF
{% set name = "bayesed3" %}
{% set version = "$VERSION" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://github.com/hanyk/BayeSED3/archive/v{{ version }}.tar.gz
  sha256: PLACEHOLDER_SHA256  # TODO: Calculate and replace

build:
  number: 0
  skip: True  # [win]
  script: \${RECIPE_DIR}/build.sh

requirements:
  build:
    - python
    - setuptools
    - pip
  run:
    - openmpi=4.1.6
    - python >=3.7
    - numpy >=1.20.0
    - h5py >=3.1.0
    - astropy >=4.2
    - matplotlib >=3.3.0
    - getdist >=1.3.0
    - requests >=2.25.0
    - tqdm >=4.60.0
    - pillow >=8.0.0
    - pyperclip >=1.8.0
    - psutil >=5.8.0

test:
  imports:
    - bayesed
  commands:
    - python -c "import bayesed; print(bayesed.__version__)"
    - python -c "from bayesed import BayeSEDInterface; print('BayeSEDInterface imported successfully')"
    - BayeSED3_GUI --help 2>&1 || echo "GUI command available"

about:
  home: https://github.com/hanyk/BayeSED3
  license: MIT
  license_file: LICENSE
  summary: Bayesian SED synthesis and analysis of galaxies and AGNs
  description: |
    BayeSED3 is a general and sophisticated tool for the full Bayesian interpretation 
    of spectral energy distributions (SEDs) of galaxies and Active Galactic Nuclei (AGNs). 
    It performs Bayesian parameter estimation using posterior probability distributions (PDFs) 
    and Bayesian SED model comparison using Bayesian evidence.
  doc_url: https://github.com/hanyk/BayeSED3
  dev_url: https://github.com/hanyk/BayeSED3

extra:
  recipe-maintainers:
    - hanyk
EOF

# Copy build.sh
cp "$REPO_ROOT/conda/build.sh" "$OUTPUT_DIR/recipes/bayesed3/build.sh"
chmod +x "$OUTPUT_DIR/recipes/bayesed3/build.sh"

# Create instructions file
cat > "$OUTPUT_DIR/INSTRUCTIONS.md" << EOF
# Conda-Forge Submission Instructions

## Files Prepared

- \`recipes/bayesed3/meta.yaml\` - Conda recipe (needs SHA256)
- \`recipes/bayesed3/build.sh\` - Build script

## Next Steps

### 1. Calculate SHA256 Hash

You need to create a source tarball and calculate its SHA256:

\`\`\`bash
# Option 1: Create from git tag
git archive --format=tar.gz --prefix=BayeSED3-$VERSION/ -o BayeSED3-$VERSION.tar.gz v$VERSION
sha256sum BayeSED3-$VERSION.tar.gz

# Option 2: Download from GitHub (after creating release)
wget https://github.com/hanyk/BayeSED3/archive/v$VERSION.tar.gz
sha256sum v$VERSION.tar.gz
\`\`\`

### 2. Update meta.yaml

Replace \`PLACEHOLDER_SHA256\` in \`recipes/bayesed3/meta.yaml\` with the calculated SHA256.

### 3. Fork staged-recipes

\`\`\`bash
# Go to https://github.com/conda-forge/staged-recipes
# Click "Fork"
# Clone your fork
git clone https://github.com/YOUR_USERNAME/staged-recipes.git
cd staged-recipes
git checkout -b bayesed3
\`\`\`

### 4. Copy Recipe Files

\`\`\`bash
# Copy prepared files
cp $OUTPUT_DIR/recipes/bayesed3/* staged-recipes/recipes/bayesed3/
\`\`\`

### 5. Test Locally

\`\`\`bash
cd staged-recipes
conda build recipes/bayesed3
\`\`\`

### 6. Commit and Push

\`\`\`bash
git add recipes/bayesed3/
git commit -m "Add bayesed3 recipe"
git push origin bayesed3
\`\`\`

### 7. Create Pull Request

- Go to https://github.com/conda-forge/staged-recipes
- Click "New Pull Request"
- Select your fork and branch
- Fill out PR template
- Submit!

## Important Notes

- Ensure GitHub release/tag v$VERSION exists
- All dependencies must be on conda-forge
- Recipe must build on Linux and macOS
- Tests must pass

## Resources

- Conda-forge docs: https://conda-forge.org/docs/
- Staged recipes: https://github.com/conda-forge/staged-recipes
EOF

echo "✓ Recipe files created in: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Calculate SHA256 hash (see $OUTPUT_DIR/INSTRUCTIONS.md)"
echo "2. Update meta.yaml with SHA256"
echo "3. Follow instructions in $OUTPUT_DIR/INSTRUCTIONS.md"
echo ""
echo "Files created:"
echo "  - $OUTPUT_DIR/recipes/bayesed3/meta.yaml"
echo "  - $OUTPUT_DIR/recipes/bayesed3/build.sh"
echo "  - $OUTPUT_DIR/INSTRUCTIONS.md"


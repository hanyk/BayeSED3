# Conda Package for BayeSED3

Complete guide for building, testing, and publishing BayeSED3 to conda-forge.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Release Workflow](#release-workflow)
- [Development Workflow](#development-workflow)
- [conda-forge Setup](#conda-forge-setup)
- [Scripts Reference](#scripts-reference)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### For Users (Installing)

```bash
conda install -c conda-forge bayesed3
```

### For Maintainers (Releasing)

```bash
cd conda
./release_with_gh.sh
```

That's it! The script handles everything and conda-forge auto-updates.

---

## Installation

### GitHub CLI (Required for Releases)

**macOS:**
```bash
brew install gh
gh auth login
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
gh auth login
```

**Other platforms:** https://github.com/cli/cli#installation

---

## Release Workflow

### Automated Release (Recommended)

```bash
cd conda
./release_with_gh.sh
```

**What it does:**
1. Updates version in code and conda recipe
2. Commits changes
3. Creates git tag
4. Pushes to GitHub
5. Creates GitHub release
6. Triggers conda-forge auto-update

**Time:** ~30 seconds

### Manual Release (Using gh CLI)

```bash
# 1. Update version
vim bayesed/__init__.py  # Change __version__ = "3.0.1"
vim conda/meta.yaml      # Change {% set version = "3.0.1" %}

# 2. Commit and tag
git add bayesed/__init__.py conda/meta.yaml
git commit -m "Bump version to 3.0.1"
git tag -a v3.0.1 -m "Release 3.0.1"
git push origin main
git push origin v3.0.1

# 3. Create release
gh release create v3.0.1 --title "Release 3.0.1" --notes "Release notes here"
```

### After Release

1. **conda-forge bot detects release** (within a few hours)
2. **Bot creates PR** to `conda-forge/bayesed3-feedstock`
3. **You review and merge** the PR
4. **Package builds automatically** for all platforms
5. **Users can install:** `conda install -c conda-forge bayesed3`

---

---

## How to Update Dependencies

Dependencies are maintained in **ONE place**: `pyproject.toml`

Both pip and conda installations read from `pyproject.toml`, ensuring perfect synchronization.

### Update Process

1. **Edit pyproject.toml:**
   ```bash
   vim pyproject.toml  # Update dependencies array
   ```

2. **Test locally:**
   ```bash
   # Test with pip
   pip install -e .
   
   # Test with conda
   cd conda
   ./quick_update.sh
   ```

3. **Release:**
   ```bash
   cd conda
   ./release_with_gh.sh  # Auto-generates MANIFEST.in
   ```

### Example: Adding scipy

```bash
# 1. Add to pyproject.toml
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.9.0",  # Add this
    # ... other deps
]

# 2. Test and release
cd conda
./quick_update.sh
./release_with_gh.sh
```

**That's it!** Both pip and conda will automatically use the dependencies from `pyproject.toml`.

### Special Note: OpenMPI

OpenMPI is a system-level dependency (not a Python package) and is specified separately in `conda/meta.yaml`. Users installing via pip must install OpenMPI separately:

```bash
# macOS
brew install openmpi

# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev

# conda (any platform)
conda install openmpi=4.1.6
```

---

## How to Update Data Files

Data files (binaries, models, nets, etc.) are automatically included based on `git ls-files`.

### How It Works

Both pip and conda installations use the **same mechanism** via `setup.py`:

1. **`setup.py`** uses `git ls-files` to discover all git-tracked files
2. Filters out Python package files and build artifacts
3. Installs data files to `$PREFIX/share/bayesed3/`

**`conda/build.sh`** is now extremely simple:
```bash
$PYTHON -m pip install . --no-deps --ignore-installed -vv
```

That's it! The `setup.py` handles all file installation for both pip and conda.

### Adding New Data Files

1. **Add files to git:**
   ```bash
   git add models/new_model.txt
   git commit -m "Add new model"
   ```

2. **Release:**
   ```bash
   cd conda
   ./release_with_gh.sh
   ```

All distribution methods (conda, pip) will automatically include the new files via `setup.py`'s `git ls-files` logic.

### Why This Works

- **For pip installs:** `setup.py` runs directly from the git repository
- **For conda builds:** `setup.py` runs from the source directory (which has `.git/`)
- **For conda-forge:** The feedstock clones the git repository, so `.git/` is available

No manual file copying needed!

---

## Development Workflow

### Quick Testing (Fast Iteration)

```bash
cd conda
./quick_update.sh
```

**Use when:** Making frequent code changes and need fast feedback

**What it does:**
- Builds conda package locally
- Installs immediately
- No prompts, very fast

### Interactive Testing (Careful Validation)

```bash
cd conda
./update_package.sh [version]
```

**Use when:** Want more control over the build process

**What it does:**
- Prompts for version updates
- Checks for uncommitted changes
- Offers to clean old builds
- Builds and optionally installs

### Clean Build Cache

```bash
cd conda
./cleanup.sh          # Actually clean
./cleanup.sh --dry-run  # See what would be deleted
```

**Use when:** conda-bld directory gets too large or you want a fresh start

### Example Development Workflow

```bash
# 1. Make code changes
vim bayesed/model.py

# 2. Quick test
cd conda
./quick_update.sh

# 3. Test your changes
python tests/quick_start.py

# 4. Repeat steps 1-3 as needed

# 5. When ready to release
./release_with_gh.sh
```

---

## conda-forge Setup

### Initial Submission (One-Time Only)

#### Step 1: Prepare Recipe

```bash
cd conda
./prepare_conda_forge_submission.sh
```

This creates `conda_forge_recipe/` with:
- `meta.yaml` - Recipe with version and SHA256
- `build.sh` - Build script
- `INSTRUCTIONS.md` - Detailed submission guide

#### Step 2: Fork staged-recipes

1. Go to: https://github.com/conda-forge/staged-recipes
2. Click "Fork"

#### Step 3: Submit Recipe

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/staged-recipes.git
cd staged-recipes
git checkout -b bayesed3

# Copy recipe
mkdir -p recipes/bayesed3
cp /path/to/BayeSED3/conda/conda_forge_recipe/* recipes/bayesed3/

# Commit and push
git add recipes/bayesed3/
git commit -m "Add bayesed3 recipe"
git push origin bayesed3
```

#### Step 4: Create Pull Request

1. Go to: https://github.com/conda-forge/staged-recipes
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out PR template
5. Submit

#### Step 5: Wait for Review

- conda-forge team reviews (1-7 days)
- Address any feedback
- After merge, you're done!

### After Acceptance

Once merged:
- A feedstock repo is created: `conda-forge/bayesed3-feedstock`
- You're added as a maintainer
- Future updates are automatic via bot PRs

---

## Scripts Reference

### Production Scripts

#### `release_with_gh.sh` ⭐ PRIMARY TOOL

**Purpose:** Automated release to GitHub and conda-forge

**Usage:**
```bash
./release_with_gh.sh
```

**Requirements:** GitHub CLI (`gh`) installed and authenticated

**What it does:**
- Updates version in code and conda recipe
- Commits and creates git tag
- Pushes to GitHub
- Creates GitHub release
- Triggers conda-forge auto-update

#### `prepare_conda_forge_submission.sh`

**Purpose:** One-time setup for conda-forge

**Usage:**
```bash
./prepare_conda_forge_submission.sh [version]
```

**What it does:**
- Creates conda-forge compatible recipe
- Calculates SHA256 hash
- Generates submission instructions

### Development Scripts

#### `quick_update.sh`

**Purpose:** Fast rebuild and reinstall for testing

**Usage:**
```bash
./quick_update.sh
```

**When to use:** During development when making frequent code changes

**Features:**
- No prompts, very fast
- Auto-detects version from code
- Builds and installs immediately

#### `update_package.sh`

**Purpose:** Interactive rebuild with version management

**Usage:**
```bash
./update_package.sh [version]
```

**When to use:** When you want more control over the build process

**Features:**
- Prompts for version updates
- Checks for uncommitted changes
- Offers to clean old builds
- Interactive installation

#### `cleanup.sh`

**Purpose:** Clean conda build artifacts and cache

**Usage:**
```bash
./cleanup.sh          # Actually clean
./cleanup.sh --dry-run  # Preview what would be deleted
```

**When to use:** When conda-bld directory gets too large

**What it cleans:**
- conda-bld directory
- conda package cache
- Frees up disk space

### Script Comparison

| Script | Purpose | Speed | Prompts | Use Case |
|--------|---------|-------|---------|----------|
| `release_with_gh.sh` | Release | Fast | Few | Production release |
| `quick_update.sh` | Test | Very Fast | None | Rapid development |
| `update_package.sh` | Test | Medium | Many | Careful testing |
| `cleanup.sh` | Clean | Fast | Optional | Disk space |
| `prepare_conda_forge_submission.sh` | Setup | Slow | Few | One-time only |

---

## Troubleshooting

### GitHub CLI Issues

**Q: gh: command not found**

A: Install GitHub CLI:
```bash
brew install gh  # macOS
# or see https://github.com/cli/cli#installation
```

**Q: gh: authentication required**

A: Run:
```bash
gh auth login
```

**Q: gh: not authorized**

A: Refresh token:
```bash
gh auth refresh
```

### conda-forge Issues

**Q: Bot didn't create a PR after 24 hours**

A: Manually update the feedstock:
```bash
git clone https://github.com/conda-forge/bayesed3-feedstock
cd bayesed3-feedstock
git checkout -b update-3.0.1
# Edit recipe/meta.yaml: update version and sha256
git commit -am "Update to 3.0.1"
git push origin update-3.0.1
# Create PR on GitHub
```

**Q: How to get SHA256 hash?**

A: Run:
```bash
curl -sL https://github.com/hanyk/BayeSED3/archive/refs/tags/v3.0.1.tar.gz | shasum -a 256
```

**Q: Build fails in CI**

A: Check the logs in the feedstock PR. Common issues:
- Missing dependencies
- Test failures
- Platform-specific problems

### Build Issues

**Q: quick_update.sh fails with "package not found"**

A: Run cleanup first:
```bash
./cleanup.sh
./quick_update.sh
```

**Q: Version mismatch error**

A: Your `__init__.py` version doesn't match `meta.yaml`. Let the script update it automatically.

**Q: Disk space error**

A: Clean build cache:
```bash
./cleanup.sh
```

### Testing Issues

**Q: Want to test without installing**

A: Build only:
```bash
conda build conda/ --no-test
```

**Q: Install specific build**

A: Get package path and install:
```bash
PACKAGE=$(conda build conda/ --output)
conda install $PACKAGE --force-reinstall
```

---

## Advanced Usage

### Build for Different Python Versions

```bash
conda build conda/ --python=3.9
conda build conda/ --python=3.10
conda build conda/ --python=3.11
```

### Test Recipe Without Building

```bash
conda render conda/
```

### Build Locally (For Testing)

```bash
# Build package
conda build conda/

# Install from local build
conda install --use-local bayesed3

# Test installation
python -c "import bayesed; print(bayesed.__version__)"
```

---

## Files in This Directory

### Essential Files
- `meta.yaml` - Current conda recipe (for local builds)
- `meta_conda_forge.yaml` - Template for conda-forge submission
- `build.sh` - Build script for conda package

### Scripts
- `release_with_gh.sh` - Automated release script (main tool)
- `prepare_conda_forge_submission.sh` - One-time conda-forge setup
- `quick_update.sh` - Fast local rebuild
- `update_package.sh` - Interactive local rebuild
- `cleanup.sh` - Clean build artifacts

---

## Benefits of This Approach

✅ **Fully automated** - One command does everything  
✅ **Fast** - Takes ~30 seconds to release  
✅ **Consistent** - Same process every time  
✅ **No manual steps** - No website interaction needed  
✅ **conda-forge integration** - Bot handles package updates automatically  
✅ **Multi-platform** - Builds for Linux, macOS automatically  
✅ **Free infrastructure** - No cost for building and hosting  

---

## Resources

- **conda-forge docs:** https://conda-forge.org/docs/
- **GitHub CLI docs:** https://cli.github.com/manual/
- **Your feedstock:** https://github.com/conda-forge/bayesed3-feedstock (after acceptance)
- **Staged recipes:** https://github.com/conda-forge/staged-recipes

---

## Quick Reference

### I want to...

**Release a new version**
```bash
./release_with_gh.sh
```

**Test my changes quickly**
```bash
./quick_update.sh
```

**Clean up disk space**
```bash
./cleanup.sh
```

**Submit to conda-forge (first time)**
```bash
./prepare_conda_forge_submission.sh
```

**Build locally**
```bash
conda build conda/
```

**Install local build**
```bash
conda install --use-local bayesed3
```

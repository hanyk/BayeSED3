# Conda Package for BayeSED3

## Quick Start

### For Users
```bash
conda install -c conda-forge bayesed3
```

### For Maintainers

**Prerequisites:** `conda install conda-build`

**Development workflow:**
```bash
# 1. Make changes
vim bayesed/model.py

# 2. Test locally
conda/quick_update.sh

# 3. Verify
python -c "import bayesed; print(bayesed.__version__)"
python tests/quick_start.py

# 4. Release
conda/release_with_gh.sh
```

**Initial conda-forge submission (one-time):**
```bash
conda/prepare_conda_forge_submission.sh
# Follow instructions in conda_forge_recipe/INSTRUCTIONS.md
```

---

## Scripts

### `release_with_gh.sh`
Create GitHub release. Triggers conda-forge auto-update after initial submission.

### `prepare_conda_forge_submission.sh`
Generate conda-forge recipe for initial submission. Run after `release_with_gh.sh`.

### `quick_update.sh`
Fast local rebuild for testing.

### `update_package.sh`
Interactive local rebuild with prompts.

### `cleanup.sh`
Clean build cache.

---

## Dependencies

Dependencies are in `pyproject.toml`. Both pip and conda read from there automatically.

**Exception:** OpenMPI (system dependency) is specified in `conda/meta.yaml`.

---

## Troubleshooting

**Build fails:** Run `./cleanup.sh` then retry

**Version mismatch:** Scripts auto-sync versions between files

**conda-forge bot didn't create PR:** Wait 24 hours or manually update feedstock

**SHA256 needed:**
```bash
curl -sL https://github.com/hanyk/BayeSED3/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256
```

---

## Resources

- conda-forge docs: https://conda-forge.org/docs/
- GitHub CLI: https://cli.github.com/manual/
- Your feedstock (after acceptance): https://github.com/conda-forge/bayesed3-feedstock

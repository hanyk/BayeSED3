"""Session guard against a stale non-editable bayesed install.

A ``pip install .`` snapshot in site-packages hides every repository edit
from the ``bayesed3`` entry point and from scripts run outside the repo
root, while pytest from the repo root still passes (cwd shadowing puts the
repo copy on ``sys.path`` first). Tests stay green while the CLI runs stale
code — the Oct 2026 debugging session that cost two rounds of "the edit
didn't take effect". This check turns that state into an immediate,
attributable error instead.
"""

import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERRIDE_ENV = 'BAYESED_ALLOW_INSTALLED_BAYESED'
FIX_HINT = 'pip install -e . --no-deps'


def _loaded_package_path():
    """Absolute path of the bayesed package directory the test run uses."""
    spec = importlib.util.find_spec('bayesed')
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0]).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    return None


def _stale_copy_path():
    """Path of a copied (non-editable) bayesed package in site-packages.

    The dist-info ``RECORD`` of a ``pip install .`` snapshot claims
    ``bayesed/__init__.py`` inside site-packages; an editable install does
    not ship the package files there. Distributions scoped inside the repo
    (a ``bayesed3.egg-info/`` build artifact at the root also registers as
    a distribution) are skipped so only real installed copies count.
    """
    try:
        dists = importlib.metadata.distributions()
    except importlib.metadata.PackageNotFoundError:
        return None
    for dist in dists:
        if (dist.metadata.get('Name') or '').lower() != 'bayesed3':
            continue
        candidate = Path(dist.locate_file('bayesed/__init__.py'))
        if not candidate.exists():
            continue
        candidate = candidate.resolve()
        if REPO_ROOT not in candidate.parents:
            return candidate
    return None


def pytest_configure(config):
    if os.environ.get(OVERRIDE_ENV):
        return

    loaded = _loaded_package_path()
    repo_package = REPO_ROOT / 'bayesed'
    if loaded is not None and loaded != repo_package:
        raise pytest.UsageError(
            f"\nImporting 'bayesed' resolves to {loaded}, not the repository "
            f"copy at {repo_package}.\n"
            f"A non-editable install is shadowing the repo; run:\n"
            f"    {FIX_HINT}\n"
            f"Set {OVERRIDE_ENV}=1 to bypass this check intentionally.")

    stale_copy = _stale_copy_path()
    if stale_copy is not None:
        # Tests pass via cwd shadowing, but the bayesed3 CLI and scripts
        # run outside the repo root still pick up the stale snapshot.
        import warnings
        warnings.warn(
            f"\nA non-editable bayesed3 install coexists with this checkout: "
            f"{stale_copy}\n"
            f"'bayesed3' and out-of-repo scripts will run the stale copy.\n"
            f"Run {FIX_HINT} to replace it with this checkout.",
            stacklevel=1)

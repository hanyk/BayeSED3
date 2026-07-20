#!/usr/bin/env python3
"""Unit tests for PriorEditor seams (no binary / OpenMPI).

Seams (ADR-0006):
  A — Priors facade raises when editor not bound (pre-priors_init)
  B — PriorEditor.set_prior(..., confirm=False) changes nbin (memory + disk)
  C — print_priors stdout includes Hyperparameters column
"""

import os
import shutil
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from bayesed.prior_editor import PriorEditor
from bayesed.prior_manager import PriorManager
from bayesed.priors import Priors

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAMPLE_IPRIOR = os.path.join(_REPO_ROOT, 'observation', 'test', '2dal8.iprior')


def _editor_with_temp_iprior():
    """PriorEditor backed by a temp copy of observation/test/2dal8.iprior."""
    tmpdir = tempfile.mkdtemp(prefix='bayesed_prior_editor_')
    dst = os.path.join(tmpdir, '2dal8.iprior')
    shutil.copy(_SAMPLE_IPRIOR, dst)
    manager = PriorManager(base_directory=tmpdir)
    manager.load_prior_file('2dal8.iprior')
    return PriorEditor(manager), manager, tmpdir


# --- Seam A -----------------------------------------------------------------


def test_set_prior_requires_priors_init():
    priors = Priors()
    try:
        priors.set_prior('Av_2', nbin=50, confirm=False)
    except ValueError as e:
        assert 'priors_init' in str(e)
    else:
        raise AssertionError('expected ValueError when _editor is None')


def test_validate_priors_requires_priors_init():
    priors = Priors()
    try:
        priors.validate_priors()
    except ValueError as e:
        assert 'priors_init' in str(e)
    else:
        raise AssertionError('expected ValueError when _editor is None')


def test_failed_reinit_clears_editor_before_early_raises():
    """ADR-0006: early re-init failure must not leave a stale PriorEditor."""
    priors = Priors()
    priors._editor = PriorEditor(PriorManager())
    try:
        priors.priors_init(SimpleNamespace(input_file='', outdir='result'))
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for empty input_file')

    try:
        priors.set_prior('Av_2', nbin=50, confirm=False)
    except ValueError as e:
        assert 'priors_init' in str(e)
    else:
        raise AssertionError('expected ValueError; stale editor still bound')


# --- Seam B -----------------------------------------------------------------


def test_set_prior_confirm_false_updates_nbin_memory_and_disk():
    editor, manager, tmpdir = _editor_with_temp_iprior()
    try:
        assert manager.get_prior('Av_2').nbin == 40

        with redirect_stdout(StringIO()):
            editor.set_prior('Av_2', nbin=99, confirm=False)

        assert manager.get_prior('Av_2').nbin == 99

        reloaded = PriorManager(base_directory=tmpdir)
        reloaded.load_prior_file('2dal8.iprior')
        assert reloaded.get_prior('Av_2').nbin == 99
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Seam C -----------------------------------------------------------------


def test_print_priors_stdout_includes_hyperparameters():
    editor, _manager, tmpdir = _editor_with_temp_iprior()
    try:
        buf = StringIO()
        with redirect_stdout(buf):
            editor.print_priors(iprior_file='2dal8.iprior')
        out = buf.getvalue()
        assert 'Hyperparameters' in out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    test_set_prior_requires_priors_init()
    test_validate_priors_requires_priors_init()
    test_failed_reinit_clears_editor_before_early_raises()
    test_set_prior_confirm_false_updates_nbin_memory_and_disk()
    test_print_priors_stdout_includes_hyperparameters()
    print('test_prior_editor: OK')

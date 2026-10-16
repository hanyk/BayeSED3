"""Launcher cold start: overview must not pull Results/specutils (ADR-0022)."""

import subprocess
import sys


def test_help_does_not_import_specutils_or_inspect_cli():
    """Fresh process: import launcher + --help leave heavy modules unloaded."""
    script = (
        'import sys\n'
        'from bayesed.launcher import run_launcher\n'
        'assert "specutils" not in sys.modules\n'
        'assert "bayesed.results" not in sys.modules\n'
        'assert "bayesed.emission_lines" not in sys.modules\n'
        'assert "bayesed.inspect_cli" not in sys.modules\n'
        'code = run_launcher(["--help"], prog="bayesed3")\n'
        'assert code == 0\n'
        'assert "specutils" not in sys.modules\n'
        'assert "bayesed.results" not in sys.modules\n'
        'assert "bayesed.emission_lines" not in sys.modules\n'
        'assert "bayesed.inspect_cli" not in sys.modules\n'
    )
    completed = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stdout + '\n' + completed.stderr
    )


def test_package_getattr_still_exports_results():
    script = (
        'from bayesed import BayeSEDResults\n'
        'assert BayeSEDResults is not None\n'
    )
    completed = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stdout + '\n' + completed.stderr
    )


def _assert_inspect_usage_skips_results(argv):
    """Fresh process: Inspect usage/--help must not load Results/specutils."""
    script = (
        'import sys\n'
        'from bayesed.launcher import run_launcher\n'
        f'code = run_launcher({argv!r}, prog="bayesed3")\n'
        'assert code in (0, 1)\n'
        'assert "specutils" not in sys.modules\n'
        'assert "bayesed.results" not in sys.modules\n'
        'assert "bayesed.emission_lines" not in sys.modules\n'
    )
    completed = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stdout + '\n' + completed.stderr
    )


def test_plot_help_does_not_import_results():
    _assert_inspect_usage_skips_results(['plot', '--help'])


def test_results_incomplete_does_not_import_results():
    _assert_inspect_usage_skips_results(['results'])


def test_priors_help_does_not_import_results():
    _assert_inspect_usage_skips_results(['priors', '--help'])

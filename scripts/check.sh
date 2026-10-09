#!/usr/bin/env bash
# Fast guardrail for local-only testing (CI workflows were removed in
# 29fa0ce): ruff F-rules on Python files changed vs HEAD, plus the fast
# test subset covering plotting / emission lines (~5 s). MPI-backed and
# full-suite tests stay out on purpose.
set -euo pipefail
cd "$(dirname "$0")/.."

changed=$({
    git diff --name-only HEAD -- '*.py'
    git diff --cached --name-only -- '*.py'
} | sort -u)
if [ -n "$changed" ]; then
    ruff check --select F -- $changed
fi

python -m pytest tests/test_plot_bestfit_residuals.py \
                  tests/test_plot_bestfit_emission_lines.py \
                  tests/test_emission_lines.py \
                  tests/test_emission_line_detection.py \
                  tests/test_fit_quality.py -q

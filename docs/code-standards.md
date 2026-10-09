# Coding standards

Review-time judgement calls for this repo. Mechanical violations (unused
imports/variables, f-strings without placeholders, undefined names, …) are
delegated to ruff `F` rules wired into `scripts/check.sh` — they are not
restated here.

## One number, one semantic

A single threshold must not serve two statistical definitions. State which
quantity a parameter measures — per-pixel sigma, integrated SNR, velocity
window, axis fraction — at its definition site, and when two consumers need
different semantics, add a second parameter instead of bending one. Session
evidence (2026-10): the emission-line label layout's 2% axis fraction did
double duty as blend-grouping criterion and collision slot, and the
detection threshold drifted between per-pixel and integrated-SNR meanings —
both produced days of threshold-chasing fixed by splitting the semantics
(see docs/adr/0021-emission-line-detection-engine.md).

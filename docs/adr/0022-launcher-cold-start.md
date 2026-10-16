# Launcher cold start without eager Results

`bayesed3` overview was paying seconds on import because `bayesed/__init__.py` eagerly loaded Results (→ emission_lines → specutils) and `launcher` imported `inspect_cli` at module level. Package `__init__` keeps the usual public surface eager, but resolves `BayeSEDResults` and `plot_bestfit` lazily. Launcher defers `inspect_cli` until Inspect runs; overview verb tokens live on `launcher`. `runtime` stays a normal top-level import: eager `Priors` already pulls `core`/`runtime` via package `__init__`, so deferring it in `launcher` alone does not change overview cost.

Inspect usage / `--help` / incomplete-argv paths also must not load Results: `inspect_cli` keeps usage strings and argv parsing light, and imports `BayeSEDResults` / `results.utils` only on product paths (HDF5 discovery, load, plot). `SEDInference` / Prior editor imports stay module-level (out of scope for this acceptance).

Acceptance is “Results/specutils stay unloaded on overview and on Inspect usage-only paths,” not a wall-clock gate.

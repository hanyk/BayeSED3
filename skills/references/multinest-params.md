---
# MultiNest slot meanings
---

Open only when the user asks what a MultiNest / `--multinest` slot *is*, or wants to change a field beyond the `SKILL.md` FAQ quick test (`Ntest=1`, `nlive=40`). Not on ordinary fits. Not on every Prefer `run` assembly. When to *change* `nlive` / `efr`: `advanced-run-modes.md`. Tuple *order*: live `bayesed3 run --help`, indexed in `binary-cli.md`.

BayeSED order is **not** `README_multinest.txt` `nestRun`: there `tol` precedes `efr`; here `nlive,efr,tol`. Do not copy README field order. `ndims` / `nPar` / `pWrap` / `initMPI` are binary-internal, not slots. `acpt_min` is BayeSED-only.

Default when the flag / `params.multinest` is omitted: `1,0,0,100,0.1,0.5,1000,-1e90,1,0,0,0,-1e90,100000,0.01`.

| # | Name | Meaning |
|---|------|---------|
| 1 | `IS` / `INS` | Importance Nested Sampling (1 on). More accurate evidence at given `nlive`; incompatible with `mmodal=1`. |
| 2 | `mmodal` | Separate posterior modes (0 off). |
| 3 | `ceff` | Constant-efficiency mode (0 off). Tunes ellipsoid enlargement toward `efr`; vanilla-NS evidence is then inaccurate (INS can still estimate it). |
| 4 | `nlive` | Live points = nested-sampling resolution. Cost scales with this. |
| 5 | `efr` | Target sampling efficiency (accepted / proposed). Lower is usually slower and stricter. |
| 6 | `tol` | Evidence tolerance; one nested-sampling stopping criterion. |
| 7 | `updInt` | Iterations between MultiNest output/checkpoint writes. |
| 8 | `Ztol` | With `mmodal=1`, only modes with local log-evidence above this are reported. |
| 9 | `seed` | RNG seed; ≥0 is reproducible; negative uses the clock. |
| 10 | `fb` | MultiNest console feedback level. |
| 11 | `resume` | Resume from MultiNest checkpoint (0 = fresh). |
| 12 | `outfile` | Write MultiNest text posterior files (HDF5 Results do not depend on this). |
| 13 | `logZero` | Discard points with log-likelihood below this. |
| 14 | `maxiter` | Max MultiNest iterations (non-positive = no cap besides `tol` / `acpt_min`). |
| 15 | `acpt_min` | BayeSED extra nested-sampling stop: halt if the acceptance rate falls below this (alongside `tol` and `maxiter`). |

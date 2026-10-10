---
# Launcher Inspect — results / plot / priors
---

Present existing **Results** or **Priors** via the Launcher (`bayesed3` / `python -m bayesed`). Not a fit path. For assembling **Run argv**, use live `bayesed3 run --help` plus `binary-cli.md` after a Prefer `run` gate (`SKILL.md`, ADR-0014). Decision record: ADR-0009.

Prefer the Python Results API in `results-analysis.md` when you already hold a `BayeSEDResults` in-process. Use Inspect when working from an `outdir` on the shell (humans or agents).

## Three first-token paths

| First token | Role |
|-------------|------|
| `version` / `--version` | Package version + Binary banner (no fit) |
| `results` / `plot` / `priors` | **Inspect** — read existing products |
| `run` | Forward **Run argv** to the compiled binary (`binary-cli.md` after a Prefer `run` gate) |

Empty argv and top-level `--help` / `-h` print a short overview (Version / Inspect / Run sketches)—enough to discover the next verb. Full Inspect flag lists appear on incomplete subcommand usage, not on the overview.

Optional Launcher-global `--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL` applies to Inspect only (default `WARNING`). Do not put `--log-level` before `run` / `version`.

## Discover

```bash
bayesed3 --help
bayesed3 results <outdir> catalogs [--json]
bayesed3 results <outdir> model_configs [--cat PATTERN] [--json]
```

`--cat` / `--cfg` / `--obj` / `--par` use **Inspect loose match** (exact → contains → regex). Resolve to exact names before loading where a single choice is required.

## results (one verb per invocation)

All verbs except the provenance pair support optional `--json`. Load is Results-catalog-scoped (`--cat` / `--cfg`); do **not** pass `object_id` into the Results constructor—filter with `--obj` after load where the verb allows it.

| Verb | `--json` shape | Notes |
|------|----------------|-------|
| `summary` | object (orientation card) | Ignores `--obj`. Includes `n_objects`; `objects` list truncated when long |
| `par_names` | object `{free, derived}` | Ignores `--obj` |
| `evidence` | array of row objects | `--json`: HDF5 column names (`ID`, present `INSlogZ` / `logZ` / errs). Human print (no `--json`): Python-style `log_evidence: … ± …` |
| `objects` | array of ids | Optional `--obj` filter |
| `par_values` | array of row objects | Requires `--par`; optional `--obj`. Human-only `--wide` / `--transpose` (ignored with `--json`) |
| `catalogs` / `model_configs` | array of names | Discovery; no HDF5 load |
| `run_arguments` / `run_version` | none (plain text only) | Whole-file provenance: the `run_arguments` / `version` attributes the binary writes on the `parameters` dataset. No `--obj`, no `--json` |
| `fit_quality` | array of row objects | Per-object phot/spec fit quality; optional `--obj`. Semantics (χ²_red, `frac_outside`, `line_residual_outliers` flags, `--band`/`--top`/`--sigma` defaults): `CONTEXT.md` "Fit quality" entry |

```bash
bayesed3 results <outdir> summary [--cat PATTERN] [--cfg PATTERN] [--json]
bayesed3 results <outdir> evidence [--cat PATTERN] [--cfg PATTERN] [--obj PATTERN] [--json]
bayesed3 results <outdir> par_names [--cat PATTERN] [--cfg PATTERN] [--json]
bayesed3 results <outdir> objects [--cat PATTERN] [--cfg PATTERN] [--obj PATTERN] [--json]
bayesed3 results <outdir> par_values --par PATTERN [--cat PATTERN] [--cfg PATTERN] [--obj PATTERN] [--wide] [--transpose] [--json]
bayesed3 results <outdir> run_arguments [--cat PATTERN] [--cfg PATTERN]
bayesed3 results <outdir> run_version [--cat PATTERN] [--cfg PATTERN]
bayesed3 results <outdir> fit_quality [--cat PATTERN] [--cfg PATTERN] [--obj PATTERN] [--band EDGES] [--top N] [--sigma T] [--json]
```

### `par_values` human table

Default human print uses Astropy terminal-width truncation (middle columns may show as `...`). Narrow with a tighter `--par` / `--obj`, or:

- `--wide` — `pprint(max_width=-1, max_lines=-1)` (no column or row elision); does not change JSON
- `--transpose` — pass `transpose=True` to `get_parameter_values` (rows = stats/params, columns = objects); independent of `--wide`; no object-count guard

With `--json`, both flags are ignored: payload stays an array of row objects (one object per row).

### `--json` and large object counts (N)

Inspect `--json` does **not** export posterior sample chains (use `plot` or Python `get_posterior_samples`). Scale risk is **Results catalog object count N** (and wide `par_values` tables), not MultiNest chain length.

There is no stream, page, or hard cap on full `evidence` / `objects` / `par_values` dumps without `--obj`. For agents and large Results catalogs:

1. Orient: `summary --json` (or `catalogs` / `model_configs` / `par_names`).
2. Narrow: `--obj PATTERN` (loose match; several hits allowed on table verbs).
3. Pull: `evidence --json` / `par_values --json` for the matched subset.

Do not default to unfiltered `objects --json` or `evidence --json` when `n_objects` is large.

## plot

```bash
bayesed3 plot <outdir> bestfit [--cat PATTERN] [--cfg PATTERN] [--obj PATTERN] [--observed-frame] [--flux fnu|nufnu|flambda] [--log x|y|both] [--linear] [--no-lines] [--no-total] [--no-components] [--full-range] [--show-residuals] [--save]
bayesed3 plot <outdir> posterior free|derived [--max N] ... [--save]
bayesed3 plot <outdir> posterior --par PATTERN [--lab TEXT] ... [--save]
```

One object when the set has a single id; several without `--obj` error (no auto-pick). Several `--obj` hits: plot each in turn. Boolean `--save` (no path): write one image per object beside that object's Results products; omit it to display only (Inspect passes `save=False`, so it does not inherit Python `plot_bestfit` / `plot_posterior*` default sidecar PNGs). Not Run argv `--save_*`. Bestfit appearance flags (`--observed-frame`, `--flux`, `--log`, `--linear`, `--no-lines`, `--no-total`, `--no-components`, `--full-range`) apply only to `bestfit`. A present flag is passed through to `plot_bestfit`; an omitted flag is not passed, so the Python default stands. `--log` takes `x`, `y`, or `both`. `--log` together with `--linear` is an error. These flags on `posterior` are unknown options.

## priors

```bash
bayesed3 priors list [dir]
```

One `list [dir]` form, two meanings: no directory prints the Prior type table; a directory loads `.iprior` files there and prints their Priors. Presentation only—not `priors_init` / not fit. Prior edit lifecycle stays on `SEDInference` / `prior-management.md`.

## When to open the other refs

| Need | Open |
|------|------|
| In-process stats / GetDist / Python plots | `results-analysis.md` |
| `bayesed3 run …` Run argv | `binary-cli.md` (Prefer `run` gate in `SKILL.md`) |
| Why Inspect grammar is this way | ADR-0009 |

---
name: bayesed3
description: Use for BayeSED3 galaxy / AGN SED synthesis/analysis, priors, Results analysis, evidence comparison, advanced run modes, Launcher Inspect (`results` / `plot` / `priors`), and assembling Launcher Run argv. Trigger on vague asks like "how do I fit this galaxy?"
---

# BayeSED3 Bayesian SED Analysis Workflow

Agent workflow router (ADR-0001, ADR-0013, ADR-0014, ADR-0015). Canonical copy: repo `skills/` (sync to `~/.agents/skills/bayesed3/` after edits). Full API = package docstrings; `references/*` = branch workflow only.

**Happy path:** Observation → Input catalog → `SEDModel` → `SEDInference.priors_init` → (optional `set_prior`) → `inference.run` → **Results** → analyze / compare.

**Default ref rule:** ordinary new fit → follow Skeleton + FAQ; open **zero** `references/*` unless a gate below fires.

## Interfaces

| Entry | Use when |
|-------|----------|
| `SEDInference` | Default fit path — returns **Results** |
| `BayeSEDInterface` | Batch / diagnostics — returns **Execution**; load Results from `outdir` afterward |
| `bayesed3 results` / `plot` / `priors` | Inspect existing products from an `outdir` on the shell (`--json` for agents) |
| `bayesed3 run` | Assemble **Run argv** when a Prefer `run` gate holds (ADR-0014). Same installed Python package as `SEDInference`. |

**Prefer when surfaces overlap:**
- **Ordinary new fit** — `SEDInference` + Skeleton + FAQ.
- **Results already in-process** — Skeleton (`print_summary` / `get_evidence` / listed `plot_*`); open `results-analysis.md` only beyond that.
- **Only an `outdir` on the shell** — `bayesed3 results` / `plot` / `priors` (`launcher-inspect.md`).
- **Prefer `run` (open `binary-cli.md`)** — any one: (1) the user wants a copy-paste `bayesed3 run` command, or delivers `--import` / a full argv string; (2) the deliverable is a shell command (job script / one-liner), not a Python snippet; (3) `bayesed3 run --help` lists a flag that is not a `BayeSEDParams` field and the user rejects `import_files`. The skill’s own mention of `bayesed3 run` is not a gate. Params looking harder or slower is not a gate.

Model assembly (`SEDModel` / `.galaxy()` / `*Params`) is not a separate entry — see decision tree step 2.

## Decision Tree

```
0. Entry?
   → Only outdir on shell → 8a (launcher-inspect.md)
   → Prefer `run` gate (the three Prefer clauses) → 8b
   → Else ordinary fit → Skeleton + FAQ (no refs yet)
1. Need an input catalog?
   → Existing .txt, or data-preparation.md (arrays → catalog)
2. Build the SED model
   → Default: SEDModel.create_* + Skeleton (no ref)
   → Dust / torus / mixed / non-default components → custom-models.md
   → Factory .galaxy() / .agn() or SSP/SFH tables → galaxy-fitting.md / agn-fitting.md
   → Fine control: *Params → same refs
3. Beyond FAQ sampling / input mode?
   → AB mag, phot/spec-only, NNLM/RDF/SNR, or when to change nlive/efr beyond FAQ → advanced-run-modes.md
   → What a MultiNest / --multinest slot means, or change a field other than FAQ quick-test → multinest-params.md
   → Else Ntest / nlive from FAQ only
4. priors_init(params); optional set_prior
   → set_prior / print_priors / validate_priors → prior-management.md
   → priors_init alone → no ref
5. inference.run(params) → Results
   → Batch/diagnostics: Interface.run → Execution, then BayeSEDResults(outdir)
6. Analyze
   → summary / get_evidence / Skeleton plot_* → no ref
   → GetDist / corr / custom stats → results-analysis.md
7. Compare model configurations → model-comparison.md
8a. Launcher Inspect → launcher-inspect.md
8b. Assemble Run argv → live `bayesed3 run --help` + binary-cli.md; then Inspect
```

## Reference Files

| File | Open when |
|------|-----------|
| `references/data-preparation.md` | Building catalog from arrays / filters / nondetections / spectroscopy |
| `references/custom-models.md` | Dust, torus, mixed, or other non-default `SEDModel` assembly |
| `references/galaxy-fitting.md` | Factory `.galaxy()` or SSP/SFH/DAL tables |
| `references/agn-fitting.md` | Factory `.agn()` or AGN component tables |
| `references/prior-management.md` | `set_prior` / `print_priors` / `validate_priors` (not bare `priors_init`) |
| `references/advanced-run-modes.md` | AB mag, phot/spec-only, NNLM/RDF/SNR, or when to change `nlive` / `efr` beyond FAQ |
| `references/multinest-params.md` | What a MultiNest / `--multinest` slot means, or changing a field other than FAQ quick-test (ADR-0015) |
| `references/results-analysis.md` | Beyond Skeleton summary / evidence / basic `plot_*` |
| `references/model-comparison.md` | Evidence ranking / posterior comparison across configurations |
| `references/launcher-inspect.md` | Shell Inspect: `results` / `plot` / `priors` / `--json` |
| `references/binary-cli.md` | Prefer `run` gate (the three clauses in Prefer, ADR-0014); assembly rules after live `--help` |

## Happy-path steps

Each step ends when its **Done** criterion holds. Follow in order; open a reference only when its **Open when** gate fires.

1. **Obtain an input catalog** — path to a BayeSED `.txt` catalog (build via `data-preparation.md` if needed).  
   **Done:** `input_file` points at an existing catalog file.

2. **Build the SED model** — default `SEDModel.create_galaxy` / `create_agn` + `params.add_*` (Skeleton; `custom-models.md` only if non-default).  
   **Done:** `params` has components attached and `save_sample_par=True` when posteriors or comparison are required.

3. **Initialize priors** — `SEDInference().priors_init(params)`; optional `set_prior` (`prior-management.md` only if editing priors).  
   **Done:** `priors_init` has run for this `params` (and every `set_prior` finished before the run).

4. **Run inference** — `results = inference.run(params)` → **Results**. Batch path: `Interface.run` → **Execution**, then `BayeSEDResults(outdir, ...)`.  
   **Done:** a `BayeSEDResults` instance is held (from `inference.run` or explicit load).

5. **Analyze Results** — Skeleton: `print_summary()`, `get_evidence()`, optional `plot_bestfit` / `plot_posterior_free`. Open `results-analysis.md` only beyond that.  
   **Done:** `print_summary()` (or equivalent) succeeded and `get_evidence()` returned `log_evidence` / `log_evidence_error` when evidence was requested.

6. **Compare model configurations** (optional) — separate `outdir` per model configuration (`model-comparison.md`).  
   **Done:** each configuration has Results; evidence or posterior comparison produced if asked.

### Skeleton

```python
from bayesed import BayeSEDParams, SEDInference
from bayesed.model import SEDModel

input_file = 'observation/test/gal.txt'  # step 1

galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti',
)
params = BayeSEDParams(
    input_type=0,
    input_file=input_file,
    outdir='tests/output_skill_happy_path',
    save_sample_par=True,
)
params.add_galaxy(galaxy)  # step 2
# Optional quick test: params.Ntest = 1; params.multinest = MultiNestParams(nlive=40)
# (import MultiNestParams from bayesed; do not call configure_multinest(nlive=40))

inference = SEDInference()
inference.priors_init(params)  # step 3
# Optional: inference.set_prior('log(age/yr)', min_val=8.5, max_val=9.8, confirm=False)

results = inference.run(params)  # step 4 → Results

results.print_summary()  # step 5
results.plot_bestfit()
results.plot_posterior_free()
evidence = results.get_evidence()  # log_evidence, log_evidence_error
# step 6 optional → model-comparison.md
```

### Branches

Happy-path exceptions (assembly / batch). Launcher Inspect stays 8a; assemble-run stays 8b.

- **Factory:** `.galaxy()` / `.agn()` then same steps 3–5 → galaxy/agn refs.
- **Mixed / dust / torus:** `custom-models.md`.
- **Batch / diagnostics:** `execution = BayeSEDInterface(...).run(params)` then load Results.

### Assemble Run argv (only if a Prefer `run` gate holds)

Each step ends when its **Done** criterion holds. Flag *shapes* come from live `bayesed3 run --help` in this session — not from memory and not by pasting help into notes. Open `binary-cli.md` for rules `--help` does not give (`igroup`/`id`, `iscalable`, SSP fields, default `.iprior`).

1. **Input catalog** — existing BayeSED `.txt` (same as happy-path step 1). QSO spectroscopic also needs the line-list files named on `-ls1` (paths on the QSO worked line).  
   **Done:** catalog path exists (and those line-list files exist when `-ls1` is used).

2. **Help** — run `bayesed3 run --help` (Launcher-owned `--mpi` / `--np` are not in that help).  
   **Done:** this session contains the help text.

3. **Assemble** — `bayesed3 run` + Run argv from help + `binary-cli.md`. Omit `--multinest` unless the user stated a sampling intent (binary defaults). Quick test: copy the live `--help` default `--multinest` tuple, set the `nlive` slot to 40, append `--Ntest 1`.  
   **Done:** the command starts with `bayesed3 run` and includes `-i` (or `--input`), `--outdir`, every SED component the user asked for (galaxy spectroscopic: `--ssp`, `--sfh`, `--dal`; QSO spectroscopic: those plus BBB, BLR, FeII, NLR as on the QSO worked line), `--Ntest` when limiting objects or quick-testing, `--multinest` only when sampling was asked for, and `--save_sample_par` when posteriors or comparison are required.

4. **Fit** — `bayesed3 run` with the Input catalog. The binary writes any missing `.iprior` files and continues the fit with those defaults. Custom bounds → leave this branch; use Python `set_prior`. There is no `bayesed3 priors init` or `set`.  
   **Done:** process exit code 0.

5. **Inspect** — `bayesed3 results <outdir> summary` (and `evidence` when evidence was asked). Open `launcher-inspect.md` for Inspect grammar. Optional: `BayeSEDResults(outdir)` if this session already uses Python or needs GetDist (`results-analysis.md`).  
   **Done:** Inspect `summary` succeeded (and `evidence` returned rows when requested).

## Imports (happy path)

```python
from bayesed import BayeSEDParams, SEDInference, BayeSEDResults
from bayesed.model import SEDModel
```

Low-level `*Params`, `SEDObservation`, `BayeSEDExecution`, `standardize_parameter_names`, `list_catalog_names`: see the reference for that branch.

## Output conventions

- Outdirs: `tests/output_<name>/` or `observation/<testname>/output/`
- Posteriors: HDF5; best-fit SED: FITS
- Parameter names: `param_name[igroup,id]` (e.g. `log(age/yr)[0,1]`) — offsets in `custom-models.md` when that ref is open

## FAQ

**Execution vs Results?**  
`inference.run` returns **Results**. `BayeSEDInterface.run` returns **Execution** (exit code, paths, timing); load Results from `outdir` afterward. Analyze posteriors only via Results.

**Why `priors_init` every run?**  
Happy-path convention: load/generate `.iprior` before fitting. Call it before `set_prior` / `print_priors` / `validate_priors`.

**Quick test?**  
`params.Ntest = 1` and `params.multinest = MultiNestParams(nlive=40)` (`from bayesed import MultiNestParams`). `run(..., Ntest=)` overrides `params.Ntest`. Do not call `configure_multinest(nlive=40)` — that method overwrites `efr` / `fb` / `outfile`. Assemble-run: live `--help` default `--multinest` tuple with `nlive` (4th slot) = 40, plus `--Ntest 1`. Not `binary-cli.md` unless a Prefer `run` gate already holds. Not `multinest-params.md`.

**Where are `nlive` / `seed` / `efr` / `Ntest`?**  
Unattended fit: leave defaults (`nlive=100`, `efr=0.1`). Quick test: this FAQ. Faster parameter estimation: set `efr=0.8` on `MultiNestParams` (`advanced-run-modes.md`). Slot meanings: `multinest-params.md` only if asking what a field is or changing one beyond this FAQ. Assemble-run: `--Ntest` / `--multinest` from live `--help`; patch slots using the `binary-cli.md` index after a Prefer `run` gate.

**Evidence keys?**  
Prefer `log_evidence` / `log_evidence_error` from `get_evidence()`. Raw `INSlogZ` may also appear in the dict when present in HDF5. Inspect `--json` uses HDF5 column names (`INSlogZ` / `logZ` / …).

**Custom priors on assemble-run?**  
See assemble-run step 4: defaults come from the fit itself; custom bounds → Python `set_prior`.

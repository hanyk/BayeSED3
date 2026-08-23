---
name: bayesed3
description: Use for BayeSED3 SED work — galaxy or AGN fitting, prior setup, Results and posterior plots, Bayesian evidence or model-configuration comparison, MultiNest or advanced run modes (AB mag, phot/spec-only). Trigger immediately on vague asks like "how do I fit this galaxy?"
---

# BayeSED3 Bayesian SED Analysis Workflow

Agent workflow router. Canonical copy: repo `skills/` (sync to `~/.agents/skills/bayesed3/` after edits).

**Happy path:** Observation → Input catalog → `SEDModel` → `SEDInference.priors_init` → (optional `set_prior`) → `inference.run` → **Results** → analyze / compare.

## Decision Tree

```
1. Need an input catalog?
   → Existing .txt, or data-preparation.md (SEDObservation / filters)
2. Build the SED model
   → Default: SEDModel → custom-models.md
   → Shortcut: .galaxy() / .agn() → galaxy-fitting.md / agn-fitting.md
   → Fine control: *Params → same refs + run_test.py
3. Advanced run mode? (AB mag, phot/spec-only, NNLM/RDF/SNR, MultiNest)
   → advanced-run-modes.md
4. priors_init(params); optional set_prior → prior-management.md
5. inference.run(params) → Results
   → Batch/diagnostics: Interface.run → Execution, then BayeSEDResults(outdir)
6. Analyze → results-analysis.md
7. Compare model configurations → model-comparison.md
8. CLI / Launcher (`bayesed3`) / --import → binary-cli.md
```

## Reference Files

| File | Open when |
|------|-----------|
| `references/data-preparation.md` | Arrays → catalog, filters, nondetections, spectroscopy |
| `references/custom-models.md` | Canonical `SEDModel`, dust, torus, mixed components |
| `references/galaxy-fitting.md` | Factory + low-level galaxy; SSP/SFH/DAL tables |
| `references/agn-fitting.md` | Factory + low-level AGN |
| `references/prior-management.md` | `set_prior`, prior types, `.iprior` |
| `references/advanced-run-modes.md` | AB mag, phot/spec-only, NNLM/RDF, MultiNest |
| `references/results-analysis.md` | Load Results, stats, plots, GetDist |
| `references/model-comparison.md` | Evidence ranking, posterior comparison |
| `references/binary-cli.md` | Launcher (`bayesed3`) and binary flags / `--import` |

## Happy-path steps

Each step ends when its **Done** criterion holds. Follow in order; open a reference only when the decision tree points there.

1. **Input catalog** — path to a BayeSED `.txt` catalog (build via `data-preparation.md` if needed).  
   **Done:** `input_file` points at an existing catalog file.

2. **SED model** — default `SEDModel.create_galaxy` / `create_agn` + `params.add_*` (`custom-models.md`).  
   **Done:** `params` has components attached and `save_sample_par=True` when posteriors or comparison are required.

3. **Priors** — `SEDInference().priors_init(params)`; optional `set_prior` (`prior-management.md`).  
   **Done:** `priors_init` has run for this `params` (and every `set_prior` finished before the run).

4. **Run** — `results = inference.run(params)` → **Results**. Batch path: `Interface.run` → **Execution**, then `BayeSEDResults(outdir, ...)`.  
   **Done:** you hold a `BayeSEDResults` instance (from `inference.run` or explicit load).

5. **Analyze** — summary, best-fit, posteriors, evidence (`results-analysis.md`).  
   **Done:** `print_summary()` (or equivalent) succeeded and `get_evidence()` returned `log_evidence` / `log_evidence_error` when evidence was requested.

6. **Compare** (optional) — separate `outdir` per model configuration (`model-comparison.md`).  
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

- **Factory:** `.galaxy()` / `.agn()` then same steps 3–5 → galaxy/agn refs.
- **Mixed / dust / torus:** `custom-models.md`.
- **Batch / diagnostics:** `execution = BayeSEDInterface(...).run(params)` then load Results.
- **Advanced modes / CLI:** `advanced-run-modes.md` / `binary-cli.md` (Launcher: `bayesed3`).

## Imports (happy path)

```python
from bayesed import BayeSEDParams, SEDInference, BayeSEDResults
from bayesed.model import SEDModel
```

Low-level `*Params`, `SEDObservation`, `BayeSEDExecution`, `standardize_parameter_names`, `list_catalog_names`: see the reference for that branch.

## Output conventions

- Outdirs: `tests/output_<name>/` or `observation/<testname>/output/`
- Posteriors: HDF5; best-fit SED: FITS
- Parameter names: `param_name[igroup,id]` (e.g. `log(age/yr)[0,1]`) — offsets in `custom-models.md`

## FAQ

**Execution vs Results?**  
`inference.run` returns **Results**. `BayeSEDInterface.run` returns **Execution** (exit code, paths, timing); load Results from `outdir` afterward. Analyze posteriors only via Results.

**Why `priors_init` every run?**  
Happy-path convention: load/generate `.iprior` before fitting. Call it before `set_prior` / `print_priors` / `validate_priors`.

**Quick test?**  
`inference.run(params, Ntest=2)`, or `Ntest=2` on `BayeSEDInterface`, or `params.configure_multinest(nlive=40)`.

**Evidence keys?**  
Prefer `log_evidence` / `log_evidence_error` from `get_evidence()`. Raw `INSlogZ` may also appear in the dict when present in HDF5.

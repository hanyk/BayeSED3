---
# Prior Management — Detailed Guide (SEDInference API)
---

> **Happy-path order: `priors_init` → (optional `set_prior`) → `inference.run()`**
>
> Always call `priors_init(params)` before a run on the skill happy path.
> `set_prior` / `print_priors` / `validate_priors` **require** `priors_init` first.
> Changing priors after a run does not affect that completed analysis.

Call these methods on `SEDInference` (skill happy path). Lifecycle implementation lives in
`bayesed.priors.Priors`; `PriorManager` is an internal `.iprior` I/O adapter (ADR-0002).

Working `.iprior` copies sit beside the Input catalog (`dirname(input_file)`). Factory
defaults for this Model configuration live in `{outdir}/priors_defaults/`. `priors_init`
copies only names the current configuration still lacks; it does not overwrite existing
working files. `inference.run` does not call `priors_init`.

Do **not** pass `clean_input_dir=True` when the Input catalog directory is shared
(several configurations, e.g. `observation/test/gal.txt` and `qso.txt`). That flag
deletes every valid `.iprior` in the directory. Use it only on a private catalog
directory. Restore a parameter with `set_prior(..., reset_to_default=True)`.

`inference.run(params)` returns **Results** (`BayeSEDResults`).
`BayeSEDInterface.run(params)` returns **Execution** — use only for batch/diagnostics, then load Results separately.

## Basic Usage

```python
from bayesed import SEDInference, BayeSEDParams
from bayesed.model import SEDModel

galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti',
)
params = BayeSEDParams(
    input_file='observation/test/gal.txt',
    outdir='tests/output_prior_test',
    save_sample_par=True,
)
params.add_galaxy(galaxy)

inference = SEDInference()
inference.priors_init(params)
inference.print_priors()
inference.list_prior_types()
# Prints the table of names accepted by set_prior (Uniform, Gaussian, …, StudentT, Weibull, …)
```

## Modifying Prior Ranges (Uniform Prior)

Pass `confirm=False` in agent / scripted runs — the default `confirm=True` prompts on stdin.

```python
inference.set_prior('log(age/yr)', min_val=8.5, max_val=9.8, nbin=60, confirm=False)
inference.set_prior('Av_2', min_val=0.0, max_val=3.0, nbin=50, confirm=False)
inference.set_prior('z', min_val=0.0, max_val=1.0, nbin=100, confirm=False)
```

## Setting Non-Uniform Priors

```python
inference.set_prior('log(age/yr)',
                   prior_type='Gaussian',
                   hyperparameters=[9.0, 0.3],
                   confirm=False)

inference.set_prior('Av_2',
                   prior_type='Gamma',
                   hyperparameters=[2.0, 0.5],
                   confirm=False)

inference.set_prior('log(Z/Zsun)',
                   prior_type='Beta',
                   hyperparameters=[2.0, 2.0],
                   min_val=-2.0, max_val=0.5,
                   confirm=False)
```

## Batch Matching (Regular Expressions)

```python
inference.set_prior(r'log\(age.*\)', min_val=8.0, max_val=10.0, confirm=False)
inference.set_prior('age', min_val=8.0, max_val=10.0, confirm=False)  # substring match
# Query without modifying:
inference.set_prior('age')  # shows matching parameter info
inference.print_priors()    # all loaded .iprior files
# inference.print_priors(iprior_file='2dal8.iprior')
# inference.print_priors(parameter_names=['log(age/yr)', 'Av_2'])
```

## Resetting Priors

Reset with `set_prior(..., reset_to_default=True)` — that flag lives on `set_prior`, not as a separate method:

```python
inference.set_prior('log(age/yr)', reset_to_default=True, confirm=False)
inference.set_prior(r'Av_.*', reset_to_default=True, confirm=False)
inference.set_prior('.*', reset_to_default=True, confirm=False)  # all parameters
```

## Prior Type Reference

Names must match `Prior.type_name_to_int` / `list_prior_types()` exactly. Prefix with `Log10_` (or use a negative integer type) for log10 space.

| Type | `prior_type` string | Hyperparameters |
|------|---------------------|------------------|
| Mirror | `'Mirror'` | `[model_id, par_type, par_id]` |
| Uniform | `'Uniform'` | None (use min/max) |
| Linear increasing | `'Linear-Inc'` | None |
| Linear decreasing | `'Linear-Dec'` | None |
| Truncated Gaussian | `'TruncGaussian'` | `[mu, sigma]` |
| Gaussian | `'Gaussian'` | `[mu, sigma]` |
| Gamma | `'Gamma'` | `[alpha, beta]` |
| Student's t | `'StudentT'` | `[mu, sigma, nu]` |
| Beta | `'Beta'` | `[a, b]` |
| Weibull | `'Weibull'` | `[alpha, beta]` |

Call `inference.validate_priors()` after edits if you need a dict of per-parameter error lists.

## Prior File (.iprior)

Working copies sit beside the Input catalog. Factory defaults are in
`{outdir}/priors_defaults/`. Prefer the `SEDInference` API over hand-editing.

## Sampler Helpers on SEDInference

Optional; applied when you call `inference.run`. Full MultiNest / NNLM recipes: `advanced-run-modes.md`.

```python
inference.multinest(nlive=100, efr=0.1, INS=1, fb=2)
# Also: gsl, nnlm, ndumper, set_logzero, set_unweighted_samples, set_nsample, set_niteration
```

## Complete Workflow Example

```python
from bayesed import BayeSEDParams, SEDInference
from bayesed.model import SEDModel

galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti',
)
params = BayeSEDParams(
    input_file='observation/test/gal.txt',
    outdir='tests/output_custom_prior',
    save_sample_par=True,
)
params.add_galaxy(galaxy)

inference = SEDInference()
inference.priors_init(params)
inference.set_prior('log(age/yr)', min_val=8.0, max_val=10.1, confirm=False)
inference.set_prior('log(tau/yr)', min_val=7.0, max_val=10.5, confirm=False)
inference.set_prior('Av_2', min_val=0.0, max_val=4.0, confirm=False)
inference.set_prior('log(Z/Zsun)', min_val=-2.0, max_val=0.5, confirm=False)

results = inference.run(params)  # BayeSEDResults
```

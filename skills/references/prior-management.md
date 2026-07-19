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
# Output: Uniform, Gaussian, Gamma, Beta, Student's t, Weibull, ...
```

## Modifying Prior Ranges (Uniform Prior)

```python
inference.set_prior('log(age/yr)', min_val=8.5, max_val=9.8, nbin=60)
inference.set_prior('Av_2', min_val=0.0, max_val=3.0, nbin=50)
inference.set_prior('z', min_val=0.0, max_val=1.0, nbin=100)
```

## Setting Non-Uniform Priors

```python
inference.set_prior('log(age/yr)',
                   prior_type='Gaussian',
                   hyperparameters=[9.0, 0.3])

inference.set_prior('Av_2',
                   prior_type='Gamma',
                   hyperparameters=[2.0, 0.5])

inference.set_prior('log(Z/Zsun)',
                   prior_type='Beta',
                   hyperparameters=[2.0, 2.0],
                   min_val=-2.0, max_val=0.5)
```

## Batch Matching (Regular Expressions)

```python
inference.set_prior(r'log\(age.*\)', min_val=8.0, max_val=10.0)  # may prompt for confirmation
inference.set_prior('age', min_val=8.0, max_val=10.0)            # substring match
# Query only:
inference.print_priors(pattern='age')
```

## Resetting Priors

```python
inference.reset_to_default('log(age/yr)')
inference.reset_to_default(r'Av_.*')
inference.reset_to_default('.*')  # all parameters
```

## Prior Type Reference

| Type | `prior_type` string | Hyperparameters |
|------|---------------------|------------------|
| Uniform | `'Uniform'` | None (use min/max) |
| Gaussian | `'Gaussian'` | `[mean μ, std σ]` |
| Gamma | `'Gamma'` | `[shape k, scale θ]` |
| Beta | `'Beta'` | `[α, β]` |
| Student's t | `'Student'` | `[dof ν, mean μ, scale σ]` |
| Weibull | `'Weibull'` | `[shape k, scale λ]` |

## Prior File (.iprior)

Prior configurations are saved as `.iprior` files under `outdir`.  
Format: `param_name prior_type min max nbin [hyperparameters...]`  
Prefer the `SEDInference` API over hand-editing.

## Sampler Helpers on SEDInference

Optional; applied when you call `inference.run`. Full MultiNest / NNLM recipes: `advanced-run-modes.md`.

```python
inference.multinest(nlive=100, efr=0.1, INS=1, fb=2)
# Also: inference.gsl(...), inference.nnlm(...), inference.ndumper(...), inference.set_logzero(...)
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
inference.set_prior('log(age/yr)', min_val=8.0, max_val=10.1)
inference.set_prior('log(tau/yr)', min_val=7.0, max_val=10.5)
inference.set_prior('Av_2', min_val=0.0, max_val=4.0)
inference.set_prior('log(Z/Zsun)', min_val=-2.0, max_val=0.5)

results = inference.run(params)  # BayeSEDResults
```

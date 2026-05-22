---
# Prior Management — Detailed Guide (SEDInference API)
---

> **Critical order: priors_init → set_prior → run()**
> `priors_init()` and `set_prior()` MUST be called BEFORE `bayesed.run()` or `inference.run()`.
> Modifying priors after running has no effect on the current analysis.

## Basic Usage

```python
from bayesed import SEDInference, BayeSEDParams

# Create parameter configuration first
params = BayeSEDParams.galaxy(
    input_file='observation/test/gal.txt',
    outdir='tests/output_prior_test'
)

# Initialize prior manager
inference = SEDInference()
inference.priors_init(params)  # load default priors from params

# View all priors
inference.print_priors()

# List available prior types
inference.list_prior_types()
# Output: Uniform, Gaussian, Gamma, Beta, Student's t, Weibull, ...
```

## Modifying Prior Ranges (Uniform Prior)

```python
# Exact parameter name match
inference.set_prior('log(age/yr)', min_val=8.5, max_val=9.8, nbin=60)

# Modify extinction parameter
inference.set_prior('Av_2', min_val=0.0, max_val=3.0, nbin=50)

# Modify redshift prior
inference.set_prior('z', min_val=0.0, max_val=1.0, nbin=100)
```

## Setting Non-Uniform Priors

```python
# Gaussian prior: hyperparameters=[mean, std]
inference.set_prior('log(age/yr)',
                   prior_type='Gaussian',
                   min_val=8.0, max_val=12.0,
                   hyperparameters=[10.0, 1.0])

# Gamma prior: hyperparameters=[shape k, scale θ]
inference.set_prior('Av_2',
                   prior_type='Gamma',
                   min_val=0.0, max_val=5.0,
                   hyperparameters=[2.0, 0.5])

# Beta prior: hyperparameters=[α, β]
inference.set_prior('log(Z/Zsun)',
                   prior_type='Beta',
                   min_val=-2.0, max_val=0.5,
                   hyperparameters=[2.0, 5.0])
```

## Batch Matching (Regular Expressions)

```python
# Regex match (shows matches and requests confirmation)
inference.set_prior('^Av_.*',
                   prior_type='Gaussian',
                   hyperparameters=[1.0, 0.3])

# Partial string match (matches all parameters containing 'age')
inference.set_prior('age', min_val=8.0, max_val=10.0)

# Query only, no modification (shows all matching parameters)
inference.set_prior('age')
inference.set_prior('Av')
```

## Resetting Priors

```python
# Reset a single parameter to default prior
inference.set_prior('log(age/yr)', reset_to_default=True)

# Batch reset (regex)
inference.set_prior('Av_.*', reset_to_default=True)

# Reset all priors
inference.reset_all_priors()
```

## Prior Type Reference

| Type | `prior_type` | `hyperparameters` |
|------|-------------|-------------------|
| Uniform | `'Uniform'` | None (use min/max) |
| Gaussian | `'Gaussian'` | `[mean μ, std σ]` |
| Gamma | `'Gamma'` | `[shape k, scale θ]` |
| Beta | `'Beta'` | `[α, β]` |
| Student's t | `'Student'` | `[dof ν, mean μ, scale σ]` |
| Weibull | `'Weibull'` | `[shape k, scale λ]` |

## Prior File (.iprior)

Prior configurations are automatically saved as `.iprior` files in the `outdir` directory.
Format: `param_name prior_type min max nbin [hyperparameters...]`

Manual editing of `.iprior` files is possible, but using the `SEDInference` API is recommended.

## Complete Workflow Example

```python
from bayesed import BayeSEDInterface, BayeSEDParams, SEDInference

bayesed = BayeSEDInterface(mpi_mode='auto')

# 1. Create parameters
params = BayeSEDParams.galaxy(
    input_file='observation/test/gal.txt',
    outdir='tests/output_custom_prior',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti'
)

# 2. Customize priors (BEFORE run)
inference = SEDInference()
inference.priors_init(params)

# Set priors based on physical constraints
inference.set_prior('log(age/yr)', min_val=8.0, max_val=10.1)  # 0.1Gyr - 13Gyr
inference.set_prior('log(tau/yr)', min_val=7.0, max_val=10.5)  # SFH timescale
inference.set_prior('Av_2', min_val=0.0, max_val=4.0)          # extinction
inference.set_prior('log(Z/Zsun)', min_val=-2.0, max_val=0.5)  # metallicity

# 3. Run (priors are now active)
result = bayesed.run(params)
```

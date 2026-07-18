---
# Multi-Model Bayesian Comparison — Detailed Guide
---

## Basic Workflow

The core idea: run multiple different SED models on the same observational data, then use Bayesian Evidence to determine which model is preferred.

```python
from bayesed import BayeSEDInterface, BayeSEDParams, BayeSEDResults
from bayesed.results import standardize_parameter_names, plot_posterior_comparison

bayesed = BayeSEDInterface(mpi_mode='auto')
input_file = 'observation/test/gal.txt'

# Model 1: exponential SFH + Calzetti extinction
params1 = BayeSEDParams.galaxy(
    input_file=input_file,
    outdir='tests/output_model1_exp_calzetti',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti',
    save_sample_par=True  # required for posterior comparison plots
)
bayesed.run(params1)

# Model 2: delayed SFH + SMC extinction
params2 = BayeSEDParams.galaxy(
    input_file=input_file,
    outdir='tests/output_model2_delayed_smc',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='delayed',
    dal_law='smc',
    save_sample_par=True  # required for posterior comparison plots
)
bayesed.run(params2)
```

## Bayesian Evidence Comparison

```python
results1 = BayeSEDResults('tests/output_model1_exp_calzetti', catalog_name='gal')
results2 = BayeSEDResults('tests/output_model2_delayed_smc', catalog_name='gal')

# Get Bayesian evidence
evidence1 = results1.get_evidence()
evidence2 = results2.get_evidence()

# Compute Bayes factor
delta_logZ = evidence1['INSlogZ'] - evidence2['INSlogZ']
delta_logZ_err = (evidence1['INSlogZerr']**2 + evidence2['INSlogZerr']**2)**0.5

print(f"Δln(Z) = {delta_logZ:.2f} ± {delta_logZ_err:.2f}")
# Jeffreys scale:
# |Δln(Z)| < 1.0:  inconclusive
# 1.0 < |Δln(Z)| < 2.5:  weak evidence
# 2.5 < |Δln(Z)| < 5.0:  moderate evidence
# |Δln(Z)| > 5.0:  strong evidence
```

## get_evidence() Return Format

```python
# Default: returns dict
evidence = results.get_evidence()
# {'INSlogZ': -123.45, 'INSlogZerr': 0.12, ...}

# Return astropy Table
evidence_table = results.get_evidence(return_format='table')

# Specific object
evidence_obj = results.get_evidence(object_ids='spec-0285-51930-0184_GALAXY_STARFORMING')

# Multiple objects
evidence_multi = results.get_evidence(object_ids=['obj1', 'obj2'])
```

## Posterior Distribution Comparison

```python
# Standardize parameter names (required for cross-model comparison — names contain component IDs)
standardize_parameter_names([results1, results2])

# Plot posterior comparison
plot_posterior_comparison(
    [results1, results2],
    labels=['Exp+Calzetti', 'Delayed+SMC'],
    output_file='tests/model_comparison.png'
)
```

## GetDist Advanced Comparison

```python
from getdist import plots
import matplotlib.pyplot as plt

# Get GetDist sample objects
samples1 = results1.get_getdist_samples(object_id=objects[0])
samples2 = results2.get_getdist_samples(object_id=objects[0])
samples1.label = 'Exp+Calzetti'
samples2.label = 'Delayed+SMC'

# Triangle plot comparison
g = plots.get_subplot_plotter()
g.triangle_plot(
    [samples1, samples2],
    ['log(age/yr)', 'log(Mstar)', 'Av'],
    filled=True,
    legend_labels=['Exp+Calzetti', 'Delayed+SMC']
)
plt.savefig('tests/triangle_comparison.png', dpi=150, bbox_inches='tight')
```

## Systematic Multi-Model Comparison

```python
# Loop over multiple model configurations
model_configs = [
    {'sfh_type': 'exponential', 'dal_law': 'calzetti', 'label': 'Exp+Cal'},
    {'sfh_type': 'delayed',     'dal_law': 'calzetti', 'label': 'Del+Cal'},
    {'sfh_type': 'exponential', 'dal_law': 'smc',      'label': 'Exp+SMC'},
    {'sfh_type': 'delayed',     'dal_law': 'smc',      'label': 'Del+SMC'},
]

results_list = []
for cfg in model_configs:
    outdir = f"tests/output_{cfg['sfh_type']}_{cfg['dal_law']}"
    params = BayeSEDParams.galaxy(
        input_file=input_file,
        outdir=outdir,
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type=cfg['sfh_type'],
        dal_law=cfg['dal_law'],
        save_sample_par=True
    )
    bayesed.run(params)
    results_list.append(BayeSEDResults(outdir, catalog_name='gal'))

# Standardize and compare
standardize_parameter_names(results_list)
plot_posterior_comparison(
    results_list,
    labels=[cfg['label'] for cfg in model_configs],
    output_file='tests/all_models_comparison.png'
)

# Print evidence ranking
evidences = [(cfg['label'], r.get_evidence()['INSlogZ'])
             for cfg, r in zip(model_configs, results_list)]
evidences.sort(key=lambda x: x[1], reverse=True)
print("Model ranking by Bayesian evidence:")
for label, logZ in evidences:
    print(f"  {label}: ln(Z) = {logZ:.2f}")
```

## Notes

- `save_sample_par=True` is required for posterior sample analysis (`plot_posterior_comparison()`, corner plots) but NOT for `get_evidence()` — Bayesian evidence is always written to the HDF5 file
- `standardize_parameter_names()` modifies component IDs in parameter names to enable cross-model comparison
- Bayesian evidence is sensitive to priors — ensure different models use the same prior ranges
- Larger MultiNest `nlive` gives more accurate evidence estimates (recommend `nlive≥400` for production)

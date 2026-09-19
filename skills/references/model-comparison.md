---
# Multi-Configuration Bayesian Comparison — Detailed Guide
---

Compare **model configurations** (SED model + priors + run options) on the same data using Bayesian **Evidence**. Prefer separate `outdir` per configuration.

## Basic Workflow

```python
from bayesed import BayeSEDParams, SEDInference, BayeSEDResults
from bayesed.model import SEDModel
from bayesed.results import standardize_parameter_names, plot_posterior_comparison

input_file = 'observation/test/gal.txt'

def run_config(outdir, sfh_type, dal_law):
    galaxy = SEDModel.create_galaxy(
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type=sfh_type,
        dal_law=dal_law,
    )
    params = BayeSEDParams(
        input_file=input_file,
        outdir=outdir,
        save_sample_par=True,  # required for posterior comparison plots
    )
    params.add_galaxy(galaxy)
    inference = SEDInference()
    inference.priors_init(params)
    return inference.run(params)  # Results

results1 = run_config('tests/output_model1_exp_calzetti', 'exponential', 'calzetti')
results2 = run_config('tests/output_model2_delayed_smc', 'delayed', 'smc')
```

Factory shortcut: `BayeSEDParams.galaxy(..., sfh_type=..., dal_law=...)` then the same `priors_init` + `inference.run`.

## Bayesian Evidence Comparison

```python
# If you only have outdirs (e.g. after batch Execution runs):
# results1 = BayeSEDResults('tests/output_model1_exp_calzetti', catalog_name='gal')
# results2 = BayeSEDResults('tests/output_model2_delayed_smc', catalog_name='gal')

evidence1 = results1.get_evidence()
evidence2 = results2.get_evidence()

# log_evidence prefers INSlogZ, falls back to logZ
delta_logZ = evidence1['log_evidence'] - evidence2['log_evidence']
delta_logZ_err = (evidence1['log_evidence_error']**2 + evidence2['log_evidence_error']**2)**0.5

print(f"Δln(Z) = {delta_logZ:.2f} ± {delta_logZ_err:.2f}")
# Jeffreys scale:
# |Δln(Z)| < 1.0:  inconclusive
# 1.0 < |Δln(Z)| < 2.5:  weak evidence
# 2.5 < |Δln(Z)| < 5.0:  moderate evidence
# |Δln(Z)| > 5.0:  strong evidence
```

## get_evidence() Return Format

```python
evidence = results.get_evidence()
# {'object_id': ..., 'log_evidence': -123.45, 'log_evidence_error': 0.12,
#  'INSlogZ': ..., 'INSlogZerr': ...}  # raw keys only if present in HDF5

evidence_table = results.get_evidence(return_format='table')
evidence_obj = results.get_evidence(object_ids='spec-0285-51930-0184_GALAXY_STARFORMING')
evidence_multi = results.get_evidence(object_ids=['obj1', 'obj2'])
```

## Posterior Distribution Comparison

```python
# Required: parameter names contain component IDs
standardize_parameter_names([results1, results2])

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

objects = results1.list_objects()
samples1 = results1.get_getdist_samples(object_id=objects[0])
samples2 = results2.get_getdist_samples(object_id=objects[0])
samples1.label = 'Exp+Calzetti'
samples2.label = 'Delayed+SMC'

g = plots.get_subplot_plotter()
g.triangle_plot(
    [samples1, samples2],
    ['log(age/yr)', 'log(Mstar)', 'Av'],
    filled=True,
    legend_labels=['Exp+Calzetti', 'Delayed+SMC']
)
plt.savefig('tests/triangle_comparison.png', dpi=150, bbox_inches='tight')
```

## Systematic Multi-Configuration Comparison

```python
model_configs = [
    {'sfh_type': 'exponential', 'dal_law': 'calzetti', 'label': 'Exp+Cal'},
    {'sfh_type': 'delayed',     'dal_law': 'calzetti', 'label': 'Del+Cal'},
    {'sfh_type': 'exponential', 'dal_law': 'smc',      'label': 'Exp+SMC'},
    {'sfh_type': 'delayed',     'dal_law': 'smc',      'label': 'Del+SMC'},
]

results_list = []
for cfg in model_configs:
    outdir = f"tests/output_{cfg['sfh_type']}_{cfg['dal_law']}"
    results_list.append(run_config(outdir, cfg['sfh_type'], cfg['dal_law']))

standardize_parameter_names(results_list)
plot_posterior_comparison(
    results_list,
    labels=[cfg['label'] for cfg in model_configs],
    output_file='tests/all_models_comparison.png'
)

evidences = [(cfg['label'], r.get_evidence()['log_evidence'])
             for cfg, r in zip(model_configs, results_list)]
evidences.sort(key=lambda x: x[1], reverse=True)
print("Model ranking by Bayesian evidence:")
for label, logZ in evidences:
    print(f"  {label}: ln(Z) = {logZ:.2f}")
```

## Notes

- `save_sample_par=True` is required for posterior sample plots, not for `get_evidence()` (evidence is in HDF5 either way)
- `standardize_parameter_names()` rewrites component IDs so posteriors can be overplotted
- Keep prior ranges consistent across configurations when ranking Evidence
- Larger MultiNest `nlive` can tighten evidence; do not jump to 400 on the word “production” — see `advanced-run-modes.md`

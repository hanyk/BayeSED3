---
# Results Loading, Statistics, and Visualization — Detailed Guide
---

## Loading Results

```python
from bayesed import BayeSEDResults

# Basic loading (auto-detect catalog)
results = BayeSEDResults('tests/output_quick_start')

# Specify catalog name (required when multiple catalogs exist)
results = BayeSEDResults('tests/output_quick_start', catalog_name='gal')

# Specify a single object
results = BayeSEDResults(
    'tests/output_quick_start',
    catalog_name='gal',
    object_id='spec-0285-51930-0184_GALAXY_STARFORMING'
)
```

## Basic Queries

```python
# Print summary
results.print_summary()

# List all object IDs
objects = results.list_objects()
# e.g.: ['spec-0285-51930-0184_GALAXY_STARFORMING', ...]

# Get parameter name lists
free_params = results.get_free_parameters()
# e.g.: ['z', 'log(age/yr)[0,1]', 'log(tau/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'Av_2[0,1]']

derived_params = results.get_derived_parameters()
# e.g.: ['log(Mstar)[0,1]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]', ...]
```

## Loading HDF5 Data Tables

```python
# Load all results
hdf5_table = results.load_hdf5_results()

# Filter by SNR (remove low signal-to-noise objects)
high_snr_table = results.load_hdf5_results(filter_snr=True, min_snr=5.0)

# Specify HDF5 file (for multi-model runs)
configs = results.list_model_configurations()
table_model2 = results.load_hdf5_results(hdf5_file=configs['catalog_model2'])
```

## Parameter Statistics

```python
# Get all statistical estimates for a single parameter
age_table = results.get_parameter_values('log(age/yr)[0,1]')
# Returns table with columns: mean, median, std, percentile_16, percentile_84, etc.

# Batch parameter statistics
stats = results.get_parameter_statistics([
    'log(age/yr)[0,1]',
    'log(Z/Zsun)[0,1]',
    'Av_2[0,1]',
    'log(Mstar)[0,1]'
])

# Compute parameter correlation matrix
correlations = results.compute_parameter_correlations([
    'log(age/yr)[0,1]',
    'log(Z/Zsun)[0,1]',
    'Av_2[0,1]'
])
print(correlations)  # correlation coefficient matrix
```

## Bayesian Evidence

```python
# Get evidence (dict format)
evidence = results.get_evidence()
print(f"ln(Z) = {evidence['INSlogZ']:.2f} ± {evidence['INSlogZerr']:.2f}")

# Table format
evidence_table = results.get_evidence(return_format='table')

# Specific object
evidence_obj = results.get_evidence(
    object_ids='spec-0285-51930-0184_GALAXY_STARFORMING'
)
```

## Visualization

### Setting Parameter Labels (LaTeX format)

```python
custom_labels = {
    # Free parameters
    'log(age/yr)[0,1]':  r'\log(age/\mathrm{yr})',
    'log(tau/yr)[0,1]':  r'\log(\tau/\mathrm{yr})',
    'log(Z/Zsun)[0,1]':  r'\log(Z/Z_\odot)',
    'Av_2[0,1]':         r'A_V',
    # Derived parameters
    'log(Mstar)[0,1]':   r'\log(M_\star/M_\odot)',
    'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]': r'\log(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})',
}
results.set_parameter_labels(custom_labels)
```

### Plotting Methods

```python
# Best-fit SED plot
results.plot_bestfit()

# Free parameter posterior distributions (corner plot)
results.plot_posterior_free()

# Derived parameter posterior distributions
results.plot_posterior_derived(max_params=5)

# Specified parameters (can mix free and derived)
results.plot_posterior(params=[
    'log(age/yr)[0,1]',
    'log(Z/Zsun)[0,1]',
    'log(Mstar)[0,1]',
    'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]'
])
```

### Per-Object Analysis

```python
objects = results.list_objects()

for object_id in objects[:3]:  # process first 3 objects
    obj_results = BayeSEDResults(
        'tests/output_quick_start',
        catalog_name='gal',
        object_id=object_id
    )
    obj_results.set_parameter_labels(custom_labels)
    obj_results.plot_bestfit()
    obj_results.plot_posterior_free()
    print(f"Object {object_id}:")
    print(f"  ln(Z) = {obj_results.get_evidence()['INSlogZ']:.2f}")
```

## GetDist Advanced Analysis

```python
from getdist import plots
import matplotlib.pyplot as plt

# Get GetDist sample object (with caching)
samples = results.get_getdist_samples(object_id=objects[0])
samples.label = 'Galaxy Model'

# Triangle plot
g = plots.get_subplot_plotter()
g.triangle_plot(
    [samples],
    ['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'Av_2[0,1]'],
    filled=True
)
plt.savefig('tests/triangle_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# 1D posterior
g2 = plots.get_single_plotter()
g2.plot_1d(samples, 'log(Mstar)[0,1]')
plt.show()
```

## Parameter Name Format

BayeSED3 parameter names follow the format `param_name[igroup,id]`:
- `[0,1]` = group 0, component 1 (typically the galaxy)
- `[1,2]` = group 1, component 2 (typically the AGN accretion disk)
- Derived parameters (e.g., stellar mass) also carry component identifiers

```python
# View actual parameter names
free_params = results.get_free_parameters()
for p in free_params:
    print(p)
# Example output:
# z
# log(age/yr)[0,1]
# log(tau/yr)[0,1]
# log(Z/Zsun)[0,1]
# Av_2[0,1]
```

## Complete Analysis Script Template

```python
from bayesed import BayeSEDResults
from bayesed.results import standardize_parameter_names

# Load results
results = BayeSEDResults('tests/output_quick_start', catalog_name='gal')

# Basic info
results.print_summary()
objects = results.list_objects()
free_params = results.get_free_parameters()
print(f"Objects: {len(objects)}")
print(f"Free parameters: {free_params}")

# Set labels
results.set_parameter_labels({
    'log(age/yr)[0,1]': r'\log(age/\mathrm{yr})',
    'log(Mstar)[0,1]': r'\log(M_\star/M_\odot)',
})

# Statistical analysis
stats = results.get_parameter_statistics(free_params[:4])
print(stats)

# Visualization
results.plot_bestfit()
results.plot_posterior_free()
results.plot_posterior_derived(max_params=5)

# Bayesian evidence
evidence = results.get_evidence()
print(f"ln(Z) = {evidence['INSlogZ']:.2f} ± {evidence['INSlogZerr']:.2f}")
```

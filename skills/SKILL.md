---
name: bayesed3
description: Use this skill for any BayeSED3 Bayesian SED analysis task — galaxy or AGN fitting, stellar mass/SFR/age/metallicity estimation, AGN component decomposition, Bayesian evidence calculation, MultiNest sampling, prior management, posterior visualization, or model comparison. Invoke whenever the user asks how to fit a galaxy or AGN SED, use the BayeSED3 Python interface, set priors, load results, or compare models. Even a vague "how do I fit this galaxy?" should trigger this skill immediately.
---

# BayeSED3 Bayesian SED Analysis Workflow

BayeSED3 performs Bayesian parameter estimation and model comparison for galaxy/AGN spectral energy distributions (SEDs) using MultiNest nested sampling. The Python interface wraps the underlying C/Fortran binary and provides both high-level and low-level usage modes.

## Reference Files

Detailed content in `references/`:
- `galaxy-fitting.md` — Galaxy SED fitting (SSP+SFH+DAL configuration)
- `agn-fitting.md` — AGN/QSO multi-component fitting (accretion disk, BLR, NLR, FeII, torus)
- `prior-management.md` — Prior management API (SEDInference)
- `model-comparison.md` — Multi-model Bayesian comparison (evidence, posterior comparison)
- `results-analysis.md` — Result loading, statistics, visualization
- `binary-cli.md` — Full binary CLI options (advanced output, IGM, physical AGN models, `--import`)

## Two Interface Levels

**High-level** (recommended for most cases):
```python
params = BayeSEDParams.galaxy(input_file=..., outdir=..., ssp_model=..., sfh_type=..., dal_law=...)
params = BayeSEDParams.agn(input_file=..., outdir=..., agn_components=['dsk','blr','nlr','feii'])
```

**Low-level** (for fine-grained control): Construct `SSPParams`, `SFHParams`, `DALParams`, etc. manually and assemble into `BayeSEDParams`. See `references/galaxy-fitting.md`.

## Complete Analysis Workflow

### Step 1: Prepare Input Data

**Option A: Use an existing text file** (recommended for testing)
```python
# Format: one source per line, flux/error columns + filter names
# Example files: observation/test/gal.txt, observation/test/qso.txt
input_file = 'observation/test/gal.txt'
```

**Option B: Create from numpy arrays**
```python
from bayesed.data import SEDObservation
import numpy as np

obs = SEDObservation(
    ids=['galaxy_001', 'galaxy_002'],
    z_min=[0.1, 0.2],
    z_max=[0.15, 0.25],
    phot_filters=['SLOAN/SDSS.g', 'SLOAN/SDSS.r', 'SLOAN/SDSS.i'],
    phot_fluxes=np.array([[12.5, 25.1, 18.3], [15.2, 28.9, 22.1]]),
    phot_errors=np.array([[1.2, 2.5, 1.8], [1.5, 2.9, 2.2]]),
    input_type=0  # 0=flux (μJy), 1=AB magnitude
)
input_file = obs.to_bayesed_input('observation/my_analysis', 'my_catalog')
```

**Download filters** (from SVO database)
```python
bayesed = BayeSEDInterface()
filter_files = bayesed.prepare_filters_from_svo(
    svo_filter_ids=['SLOAN/SDSS.g', 'SLOAN/SDSS.r', 'SLOAN/SDSS.i'],
    output_dir='observation/my_analysis/filters'
)
# Returns filter_files['filters_file'] and filter_files['filters_selected_file']
```

### Step 2: Initialize Interface

```python
from bayesed import BayeSEDInterface

bayesed = BayeSEDInterface(
    mpi_mode='auto',  # 'auto' auto-selects, '1' single-process, 'n' multi-process
    Ntest=2           # Only process first N objects (use during development)
)
```

### Step 3: Build Parameter Configuration

**Galaxy fitting** → see `references/galaxy-fitting.md`
**AGN fitting** → see `references/agn-fitting.md`

Quick reference:

| Parameter | Common values |
|-----------|--------------|
| `ssp_model` | `'bc2003_hr_stelib_chab_neb_2000r'` (high-res), `'bc2003_lr_BaSeL_chab'` (low-res) |
| `sfh_type` | `'exponential'`, `'delayed'`, `'nonparametric'` |
| `dal_law` | `'calzetti'` (starburst), `'smc'` (SMC extinction), `'milky_way'` |
| `agn_components` | `['dsk','blr','nlr','feii']`, any combination |

### Step 4: (Optional) Manage Priors — MUST be done before running

**Critical: `priors_init()` and `set_prior()` MUST be called BEFORE `bayesed.run()`.** Modifying priors after running has no effect on the current analysis.

→ See `references/prior-management.md` for full API

```python
from bayesed import SEDInference
inference = SEDInference()
inference.priors_init(params)          # Step 1: load defaults from params
inference.set_prior('log(age/yr)', min_val=8.5, max_val=9.8)  # Step 2: modify
inference.print_priors()               # Optional: verify settings

# Step 3: run (priors are now active)
result = bayesed.run(params)
# Or equivalently: result = inference.run(params)
```

### Step 5: Run Analysis

```python
result = bayesed.run(params)
# result is a BayeSEDExecution object (execution metadata), NOT scientific results
# Scientific results must be loaded separately via BayeSEDResults
```

### Step 6: Load and Analyze Results

→ See `references/results-analysis.md`

```python
from bayesed import BayeSEDResults

results = BayeSEDResults(
    'tests/output_quick_start',
    catalog_name='gal',    # Required when multiple catalogs exist
    object_id='obj_001'    # Optional: single object
)

results.print_summary()
objects = results.list_objects()
free_params = results.get_free_parameters()
```

### Step 7: Visualize

```python
# Set LaTeX parameter labels (optional)
results.set_parameter_labels({
    'log(age/yr)[0,1]': r'\log(age/\mathrm{yr})',
    'log(Mstar)[0,1]': r'\log(M_\star/M_\odot)',
})

results.plot_bestfit()                        # Best-fit SED
results.plot_posterior_free()                 # Free parameter posteriors
results.plot_posterior_derived(max_params=5)  # Derived parameter posteriors
results.plot_posterior(params=['log(age/yr)[0,1]', 'log(Mstar)[0,1]'])
```

## Model Comparison

→ See `references/model-comparison.md`

```python
from bayesed.results import standardize_parameter_names, plot_posterior_comparison

# Run multiple models on the same data, then compare Bayesian evidence
evidence1 = results1.get_evidence()  # {'INSlogZ': ..., 'INSlogZerr': ...}
evidence2 = results2.get_evidence()
delta_logZ = evidence1['INSlogZ'] - evidence2['INSlogZ']
# delta_logZ > 0 means model 1 is preferred
```

## Key Imports

```python
from bayesed import (
    BayeSEDInterface, BayeSEDParams, BayeSEDResults,
    SEDInference,
    # Low-level parameter classes (for fine-grained control)
    SSPParams, SFHParams, DALParams, MultiNestParams, SysErrParams,
    BigBlueBumpParams, AKNNParams, LineParams, ZParams,
    GreybodyParams, FANNParams, KinParams
)
from bayesed.model import SEDModel
from bayesed.data import SEDObservation
from bayesed.results import standardize_parameter_names, plot_posterior_comparison
```

## Output File Conventions

- Low-level interface output: `observation/<testname>/output/`
- High-level interface output: `tests/output_<name>/` or user-specified `outdir`
- Main result files: `*.hdf5` (posterior samples), `*_bestfit.fits` (best-fit SED)
- Parameter name format: `param_name[igroup,id]`, e.g. `log(age/yr)[0,1]`

## FAQ

**Q: What is the difference between `BayeSEDExecution` and `BayeSEDResults`?**
`bayesed.run()` returns `BayeSEDExecution` (execution metadata: timing, exit code, paths). Scientific results require constructing `BayeSEDResults(outdir, ...)` separately to read HDF5 files.

**Q: How to run a quick test?**
Use `BayeSEDInterface(Ntest=2)` to process only the first 2 objects, or `MultiNestParams(nlive=40)` to reduce sampling points.

**Q: What does `[0,1]` in parameter names mean?**
`[igroup, id]` is the component identifier. Galaxies are typically `[0,1]`; AGN components start from `[1,2]` and increment.

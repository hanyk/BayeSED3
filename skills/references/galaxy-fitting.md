---
# Galaxy SED Fitting — Detailed Guide
---

Default assembly for agents: **`SEDModel`** — happy path and dust emission in `custom-models.md`. This page covers the factory shortcut, low-level `*Params`, and SSP/SFH/DAL tables.

## Factory Shortcut

```python
from bayesed import BayeSEDParams, SEDInference

params = BayeSEDParams.galaxy(
    input_file='observation/test/gal.txt',
    outdir='tests/output_galaxy',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti',
    save_sample_par=True,  # required for posterior plots / model-configuration comparison
)

inference = SEDInference()
inference.priors_init(params)
results = inference.run(params)
```

## Low-Level Interface (Fine-Grained Control)

```python
from bayesed import BayeSEDParams, SEDInference
from bayesed import (SSPParams, SFHParams, DALParams, ZParams,
                     MultiNestParams, SysErrParams, RDFParams)

params = BayeSEDParams(
    input_type=0,                          # 0=flux (μJy), 1=AB magnitude
    input_file='observation/test/gal.txt',
    outdir='observation/test/output/gal',
    save_bestfit=0,                        # 0=FITS, 1=HDF5, 2=both
    save_sample_par=True,  # required for posterior plots / model-configuration comparison
    filters='observation/test/filters.txt',
    filters_selected='observation/test/filters_selected.txt',

    ssp=[SSPParams(
        igroup=0, id=0,
        name='bc2003_hr_stelib_chab_neb_2000r',
        iscalable=1,
        i1=1                               # 1=include nebular emission
    )],
    sfh=[SFHParams(
        id=0,
        itype_sfh=2,                       # 2=exponential decay, 5=delayed exponential
        itype_ceh=0                        # 0=single metallicity
    )],
    dal=[DALParams(
        id=0,
        ilaw=8,                            # 8=Calzetti2000, 7=SMC
        con_eml_tot=2                      # 2=continuum+emission lines
    )],
    z=ZParams(iprior_type=1),
    multinest=MultiNestParams(nlive=40),  # FAQ quick test; omit this block for default nlive=100
    sys_err_obs=SysErrParams(min=0.0, max=0.2)
)

inference = SEDInference()
inference.priors_init(params)
results = inference.run(params)
```

AB mag, phot/spec-only, NNLM/RDF: `advanced-run-modes.md`.

## SSP Model Options

| Model name | Resolution | Library | IMF | Nebular emission |
|-----------|-----------|---------|-----|-----------------|
| `bc2003_hr_stelib_chab_neb_2000r` | High | STELIB | Chabrier | Yes |
| `bc2003_hr_stelib_chab_neb_300r` | High | STELIB | Chabrier | Yes (low-res) |
| `bc2003_lr_BaSeL_chab` | Low | BaSeL | Chabrier | No |

## SFH Types

| `itype_sfh` | `sfh_type` | Description |
|-------------|------------|-------------|
| 0 | `'instantaneous'`, `'instantaneous_burst'`, `'burst'` | Instantaneous burst |
| 1 | `'constant'` | Constant SFR |
| 2 | `'exponential'`, `'exponentially_declining'` | Exponential decay τ |
| 3 | `'exponentially_increasing'`, `'increasing'` | Exponential rise |
| 4 | `'single_burst'`, `'burst_length_tau'` | Single burst of length tau |
| 5 | `'delayed'`, `'delayed_exponential'` | Delayed exponential t·exp(-t/τ) |
| 6 | `'beta'` | Beta function |
| 7 | `'lognormal'`, `'log_normal'` | Log-normal |
| 8 | `'double_powerlaw'`, `'double_power_law'` | Double power-law SFH |
| 9 | `'nonparametric'`, `'non_parametric'` | Non-parametric |

## Dust Attenuation Laws

| `ilaw` | `dal_law` | Description |
|--------|-----------|-------------|
| 0 | `'sed_model'`, `'sed_normalization'` | SED model with L_dust normalization |
| 1 | `'starburst'`, `'starburst_calzetti'`, `'calzetti_fast'` | Starburst (Calzetti+2000, FAST) |
| 2 | `'milky_way'`, `'milky_way_cardelli'`, `'cardelli'` | Milky Way (Cardelli+1989, FAST) |
| 3 | `'star_forming'`, `'star_forming_salim'`, `'salim'` | Star-forming (Salim+2018) |
| 4 | `'mw_allen'`, `'allen'` | MW (Allen+76, hyperz) |
| 5 | `'mw_fitzpatrick'`, `'fitzpatrick_mw'` | MW (Fitzpatrick+86, hyperz) |
| 6 | `'lmc'`, `'lmc_fitzpatrick'`, `'fitzpatrick_lmc'` | LMC (Fitzpatrick+86, hyperz) |
| 7 | `'smc'`, `'smc_fitzpatrick'`, `'fitzpatrick_smc'` | SMC (Fitzpatrick+86, hyperz) |
| 8 | `'calzetti'`, `'calzetti2000'`, `'starburst_calzetti2000'` | SB (Calzetti2000, hyperz) |
| 9 | `'star_forming_reddy'`, `'reddy'` | Star-forming (Reddy+2015) |

String→int mapping for `SEDModel.create_galaxy`: also in `custom-models.md`.

## Spectroscopic Input Catalogs

Build photometry+spectra catalogs with `SEDObservation` — full API in `data-preparation.md`. Minimal shape:

```python
from bayesed.data import SEDObservation
import numpy as np

obs = SEDObservation(
    ids=['galaxy_001'],
    z_min=[0.1], z_max=[0.15],
    phot_filters=['SLOAN/SDSS.g', 'SLOAN/SDSS.r'],
    phot_fluxes=np.array([[12.5, 25.1]]),
    phot_errors=np.array([[1.2, 2.5]]),
    spec_band_names=['SDSS_spec'],
    spec_wavelengths=[np.linspace(3800, 9200, 500)],
    spec_fluxes=[np.ones(500) * 10.0],
    spec_errors=[np.ones(500) * 1.0],
    spec_lsf_sigma=[np.ones(500) * 0.0003],
    input_type=0
)
input_file = obs.to_bayesed_input('observation/my_spec', 'spec_catalog')
```

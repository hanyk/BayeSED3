---
# Galaxy SED Fitting — Detailed Guide
---

## High-Level Interface (Recommended)

```python
from bayesed import BayeSEDInterface, BayeSEDParams, BayeSEDResults

bayesed = BayeSEDInterface(mpi_mode='auto')

params = BayeSEDParams.galaxy(
    input_file='observation/test/gal.txt',
    outdir='tests/output_galaxy',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti',
    save_sample_par=True  # required for posterior plots and model comparison
)

result = bayesed.run(params)
```

## Low-Level Interface (Fine-Grained Control)

```python
from bayesed import BayeSEDParams
from bayesed import (SSPParams, SFHParams, DALParams, ZParams,
                     MultiNestParams, SysErrParams, RDFParams)

params = BayeSEDParams(
    input_type=0,                          # 0=flux (μJy), 1=AB magnitude
    input_file='observation/test/gal.txt',
    outdir='observation/test/output/gal',
    save_bestfit=0,                        # 0=FITS, 1=HDF5, 2=both
    save_sample_par=True,
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
    z=ZParams(iprior_type=1),              # redshift prior
    multinest=MultiNestParams(
        nlive=40,                          # live points (40 for testing, 400+ for production)
        efr=0.05,                          # sampling efficiency
        updInt=100,
        fb=2
    ),
    sys_err_obs=SysErrParams(min=0.0, max=0.2)  # systematic error
)

result = bayesed.run(params)
```

## SSP Model Options

| Model name | Resolution | Library | IMF | Nebular emission |
|-----------|-----------|---------|-----|-----------------|
| `bc2003_hr_stelib_chab_neb_2000r` | High | STELIB | Chabrier | Yes |
| `bc2003_hr_stelib_chab_neb_300r` | High | STELIB | Chabrier | Yes (low-res) |
| `bc2003_lr_BaSeL_chab` | Low | BaSeL | Chabrier | No |
| `bc2003_lr_BaSeL_chab_i0000` | Low | BaSeL | Chabrier | No |

## SFH Types

| `itype_sfh` | `sfh_type` | Description |
|-------------|------------|-------------|
| 0 | `'instantaneous'` | Instantaneous burst |
| 1 | `'constant'` | Constant SFR |
| 2 | `'exponential'` | Exponential decay τ |
| 3 | — | Exponential rise |
| 5 | `'delayed'` | Delayed exponential t·exp(-t/τ) |
| 6 | `'beta'` | Beta function |
| 7 | `'lognormal'` | Log-normal |
| 9 | `'nonparametric'` | Non-parametric |

## Dust Attenuation Laws

| `ilaw` | `dal_law` | Description |
|--------|-----------|-------------|
| 1 | — | Starburst (Calzetti+2000, with scattering) |
| 7 | `'smc'` | SMC (Fitzpatrick+86) |
| 8 | `'calzetti'` | SB (Calzetti2000) |
| 9 | — | Star-forming (Reddy+2015) |

## Adding Dust Emission (SEDModel Interface)

```python
from bayesed.model import SEDModel

galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_lr_BaSeL_chab',
    sfh_type='exponential',
    dal_law='smc'
)
galaxy.add_dust_emission()  # Add greybody dust emission component

params = BayeSEDParams(
    input_type=0,
    input_file='observation/test2/test.txt',
    outdir='tests/output_custom',
    filters='observation/test2/filters.txt',
    filters_selected='observation/test2/filters_selected.txt',
    save_sample_par=True
)
params.add_galaxy(galaxy)
bayesed.run(params)
```

## Spectroscopic Data Fitting

To fit photometry and spectroscopy simultaneously, the input file must include spectral columns (wavelength, flux, error, LSF). Use `SEDObservation` to create input files with spectra:

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
    spec_wavelengths=[np.linspace(0.38, 0.92, 500)],  # units: μm
    spec_fluxes=[np.ones(500) * 10.0],
    spec_errors=[np.ones(500) * 1.0],
    spec_lsf_sigma=[np.ones(500) * 0.0003],  # LSF width in μm
    input_type=0
)
input_file = obs.to_bayesed_input('observation/my_spec', 'spec_catalog')
```

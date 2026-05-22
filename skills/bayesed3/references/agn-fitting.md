---
# AGN/QSO Multi-Component SED Fitting — Detailed Guide
---

## High-Level Interface (Recommended)

```python
from bayesed import BayeSEDInterface, BayeSEDParams

bayesed = BayeSEDInterface(mpi_mode='auto', Ntest=2)

params = BayeSEDParams.agn(
    input_file='observation/test/qso.txt',
    outdir='tests/output_agn',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',  # host galaxy SSP
    sfh_type='exponential',
    dal_law='calzetti',
    agn_components=['dsk', 'blr', 'nlr', 'feii'],  # AGN components
    save_sample_par=True
)

result = bayesed.run(params)
```

## AGN Component Reference

| Component code | Physical meaning | Parameter class |
|---------------|-----------------|----------------|
| `'dsk'` | Accretion disk (Big Blue Bump) | `BigBlueBumpParams` |
| `'blr'` | Broad Line Region (BLR emission lines) | `LineParams` |
| `'nlr'` | Narrow Line Region (NLR emission lines) | `LineParams` |
| `'feii'` | FeII emission | `AKNNParams` + `KinParams` |
| `'tor'` | Dust torus (FANN model) | `FANNParams` |

## Low-Level Interface (Full AGN Configuration)

```python
from bayesed import BayeSEDParams
from bayesed import (SSPParams, SFHParams, DALParams, ZParams,
                     BigBlueBumpParams, LineParams, AKNNParams, KinParams,
                     MultiNestParams, SysErrParams, RenameParams, NNLMParams,
                     SNRmin1Params, RDFParams)

params = BayeSEDParams(
    input_type=0,
    input_file='observation/test/qso.txt',
    outdir='observation/test/output/qso',
    save_bestfit=0,
    save_sample_par=True,

    # Component 0: host galaxy
    ssp=[SSPParams(igroup=0, id=0,
                   name='bc2003_hr_stelib_chab_neb_2000r',
                   iscalable=1, i1=1)],
    sfh=[SFHParams(id=0, itype_sfh=2)],
    dal=[DALParams(id=0, ilaw=8, con_eml_tot=2),
         DALParams(id=1, ilaw=7),   # accretion disk dust attenuation (SMC)
         DALParams(id=2, ilaw=7),   # BLR dust attenuation
         DALParams(id=3, ilaw=7),   # FeII dust attenuation
         DALParams(id=4, ilaw=7)],  # NLR dust attenuation

    # Component 1: accretion disk (Big Blue Bump)
    big_blue_bump=[BigBlueBumpParams(
        igroup=1, id=1, name='bbb',
        iscalable=1, w_min=0.1, w_max=10, Nw=1000
    )],

    # Component 2: Broad Line Region (BLR)
    lines1=[
        LineParams(igroup=2, id=2, name='BLR',
                   iscalable=1, file='lines_BLR.txt',
                   R=300, Nsample=10, Nkin=3),
        # Component 4: Narrow Line Region (NLR)
        LineParams(igroup=4, id=4, name='NLR',
                   iscalable=1, file='lines_NLR.txt',
                   R=2000, Nsample=10, Nkin=2)
    ],

    # Component 3: FeII emission (AKNN model)
    aknn=[AKNNParams(igroup=3, id=3, name='FeII',
                     iscalable=1, k=1, f_run=1)],
    kin=[KinParams(id=3, ikin=2)],  # FeII kinematics

    multinest=MultiNestParams(nlive=40, efr=0.05, updInt=100, fb=2),
    sys_err_obs=SysErrParams(min=0.0, max=0.2),
    snrmin1=SNRmin1Params(snrmin1_phot=0, snrmin1_spec=3)  # spectral SNR threshold
)
```

## Adding a Dust Torus

```python
from bayesed.model import SEDModel

# Option A: high-level interface
agn = SEDModel.create_agn(agn_components=['tor'])
params.add_agn(agn)

# Option B: low-level interface (FANN model)
from bayesed import FANNParams
params.fann = [FANNParams(
    igroup=2, id=2,
    name='clumpy201410tor',  # clumpy dust torus model
    iscalable=1
)]
```

## Full AGN + Galaxy + Torus Example

```python
from bayesed.model import SEDModel

bayesed = BayeSEDInterface(mpi_mode='auto')

# Create host galaxy (with dust emission)
galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_lr_BaSeL_chab',
    sfh_type='exponential',
    dal_law='smc'
)
galaxy.add_dust_emission()

# Create AGN (with torus)
agn = SEDModel.create_agn()
agn.add_torus_fann(name='clumpy201410tor')

# Assemble parameters
params = BayeSEDParams(
    input_type=0,
    input_file='observation/test2/test.txt',
    outdir='tests/output_agn_torus',
    filters='observation/test2/filters.txt',
    filters_selected='observation/test2/filters_selected.txt',
    save_sample_par=True
)
params.add_galaxy(galaxy)
params.add_agn(agn)

bayesed.run(params)
```

## Component ID Assignment Rules

- Host galaxy: `igroup=0, id=0` (shared by SSP/SFH/DAL)
- AGN components start from `igroup=1`, incrementing per component
- Each component requires its own `DALParams` (dust attenuation)
- `id` must be unique across all components

## Common AGN Parameters

| Parameter | Typical value | Description |
|-----------|--------------|-------------|
| BLR line width `R` | 300 | Velocity resolution |
| NLR line width `R` | 2000 | Velocity resolution |
| BLR kinematics `Nkin` | 3 | Kinematic degrees of freedom |
| NLR kinematics `Nkin` | 2 | Kinematic degrees of freedom |
| FeII model | `'FeII'` | AKNN model name |
| Torus model | `'clumpy201410tor'` | FANN model name |

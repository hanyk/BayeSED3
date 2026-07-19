---
# AGN/QSO Multi-Component SED Fitting — Detailed Guide
---

Default assembly for agents: **`SEDModel.create_agn` / `add_agn`** — happy path, disk/torus variants, and ID offsets in `custom-models.md`. This page covers the factory shortcut, low-level layout, and component tables.

## Factory Shortcut

```python
from bayesed import BayeSEDParams, SEDInference

params = BayeSEDParams.agn(
    input_file='observation/test/qso.txt',
    outdir='tests/output_agn',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti',
    agn_components=['dsk', 'blr', 'nlr', 'feii'],
    save_sample_par=True,
)

inference = SEDInference()
inference.priors_init(params)
results = inference.run(params)
```

## AGN Component Reference

| Component code | Physical meaning | Parameter class |
|---------------|-----------------|----------------|
| `'dsk'` | Accretion disk (Big Blue Bump) | `BigBlueBumpParams` |
| `'blr'` | Broad Line Region (BLR emission lines) | `LineParams` |
| `'nlr'` | Narrow Line Region (NLR emission lines) | `LineParams` |
| `'feii'` | FeII emission | `AKNNParams` + `KinParams` |
| `'tor'` | Dust torus (FANN model) | `FANNParams` |

ID offsets and disk/torus method variants: `custom-models.md`.

## Low-Level Interface (Full AGN Configuration)

```python
from bayesed import BayeSEDParams, SEDInference
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

    ssp=[SSPParams(igroup=0, id=0,
                   name='bc2003_hr_stelib_chab_neb_2000r',
                   iscalable=1, i1=1)],
    sfh=[SFHParams(id=0, itype_sfh=2)],
    dal=[DALParams(id=0, ilaw=8, con_eml_tot=2),
         DALParams(id=1, ilaw=7),
         DALParams(id=2, ilaw=7),
         DALParams(id=3, ilaw=7),
         DALParams(id=4, ilaw=7)],

    big_blue_bump=[BigBlueBumpParams(
        igroup=1, id=1, name='bbb',
        iscalable=1, w_min=0.1, w_max=10, Nw=1000
    )],

    lines1=[
        LineParams(igroup=2, id=2, name='BLR',
                   iscalable=1, file='lines_BLR.txt',
                   R=300, Nsample=10, Nkin=3),
        LineParams(igroup=4, id=4, name='NLR',
                   iscalable=1, file='lines_NLR.txt',
                   R=2000, Nsample=10, Nkin=2)
    ],

    aknn=[AKNNParams(igroup=3, id=3, name='FeII',
                     iscalable=1, k=1, f_run=1)],
    kin=[KinParams(id=3, velscale=10, num_gauss_hermites_con=2, num_gauss_hermites_eml=0)],

    multinest=MultiNestParams(nlive=40, efr=0.05, updInt=100, fb=2),
    sys_err_obs=SysErrParams(min=0.0, max=0.2),
    SNRmin1=SNRmin1Params(phot=0, spec=3)
)

inference = SEDInference()
inference.priors_init(params)
results = inference.run(params)
```

Low-level torus (FANN) without SEDModel:

```python
from bayesed import FANNParams
params.fann = [FANNParams(
    igroup=2, id=2,
    name='clumpy201410tor',
    iscalable=1
)]
```

## Component ID Assignment Rules

- Host galaxy: typically `igroup=0` with shared SSP/SFH/DAL id (auto-assigned via `add_galaxy` when using SEDModel)
- AGN component offsets from `base_id` / `base_igroup`: see table in `custom-models.md`
- `id` must be unique across components; each attenuated component needs its own `DALParams` when building low-level

## Common AGN Parameters

| Parameter | Typical value | Description |
|-----------|--------------|-------------|
| BLR line width `R` | 300 | Velocity resolution |
| NLR line width `R` | 2000 | Velocity resolution |
| BLR kinematics `Nkin` | 3 | Kinematic degrees of freedom |
| NLR kinematics `Nkin` | 2 | Kinematic degrees of freedom |
| FeII model | `'FeII'` | AKNN model name |
| Torus model | `'clumpy201410tor'` | FANN model name |

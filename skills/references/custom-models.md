---
# Custom Models — `SEDModel` Builder Reference
---

`SEDModel` (in `bayesed.model`) is the **canonical** builder for the skill happy path. It builds `GalaxyInstance` / `AGNInstance` objects, then `BayeSEDParams.add_galaxy()` / `add_agn()`. Factory methods `.galaxy()` / `.agn()` remain shortcuts (see galaxy/agn refs).

## Basic Pattern (happy path)

Matches the skeleton in `SKILL.md`: one galaxy SED model, `priors_init`, then `inference.run` → Results.

```python
from bayesed import BayeSEDParams, SEDInference
from bayesed.model import SEDModel

galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti',
)
params = BayeSEDParams(
    input_type=0,
    input_file='observation/test/gal.txt',
    outdir='tests/output_skill_happy_path',
    save_sample_par=True,
)
params.add_galaxy(galaxy)

inference = SEDInference()
inference.priors_init(params)
results = inference.run(params)  # BayeSEDResults
```

Batch alternative: `BayeSEDInterface(mpi_mode='auto').run(params)` returns **Execution**, then load `BayeSEDResults` separately.

## Mixed components (dust emission + torus)

Beyond the happy path — greybody dust and/or AGN torus:

```python
galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_lr_BaSeL_chab',
    sfh_type='exponential',
    dal_law='smc',
)
galaxy.add_dust_emission()

agn = SEDModel.create_agn(agn_components=['tor'])

params = BayeSEDParams(
    input_type=0,
    input_file='observation/test2/test.txt',
    outdir='tests/output_custom_model',
    filters='observation/test2/filters.txt',
    filters_selected='observation/test2/filters_selected.txt',
    save_sample_par=True,
)
params.add_galaxy(galaxy)
params.add_agn(agn)

inference = SEDInference()
inference.priors_init(params)
results = inference.run(params)
```

`add_galaxy()` / `add_agn()` default to `auto_assign_ids=True`: if the instance still has default IDs (`igroup=0, id=0`) or would collide with something already added, they're renumbered automatically via `get_next_galaxy_ids()` / `get_next_agn_ids()`. Pass `auto_assign_ids=False` to keep manually assigned IDs (needed when you construct multiple AGN instances yourself and want explicit control — see `add_agn3` example below).

## GalaxyInstance

`SEDModel.create_galaxy(...)` returns a `GalaxyInstance` (`ssp`, `sfh`, `dal`, optional `dem`, `kin`). Its `id` and `igroup` properties both read from `ssp` — SSP/SFH/DAL always share one `id`.

```python
SEDModel.create_galaxy(
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',  # SSP template name
    sfh_type='exponential',                        # str or int, see table below
    dal_law='calzetti',                             # str or int, see table below
    ssp_k=1, ssp_f_run=1, ssp_Nstep=1,
    ssp_i0=0, ssp_i1=0, ssp_i2=0, ssp_i3=0,
    ssp_iscalable=1,       # 0 disables scaling
    sfh_itype_ceh=0,       # chemical evolution history type
    sfh_itruncated=0,
    base_igroup=0, base_id=0
)
```

### `sfh_type` string → int

| string | int | string | int |
|---|---|---|---|
| `instantaneous`/`burst` | 0 | `delayed`/`delayed_exponential` | 5 |
| `constant` | 1 | `beta` | 6 |
| `exponential`/`exponentially_declining` | 2 | `lognormal`/`log_normal` | 7 |
| `exponentially_increasing`/`increasing` | 3 | `double_powerlaw` | 8 |
| `single_burst`/`burst_length_tau` | 4 | `nonparametric` | 9 |

### `dal_law` string → int

| string | int | string | int |
|---|---|---|---|
| `sed_model`/`sed_normalization` | 0 | `mw_fitzpatrick` | 5 |
| `starburst`/`calzetti_fast` | 1 | `lmc`/`lmc_fitzpatrick` | 6 |
| `milky_way`/`cardelli` | 2 | `smc`/`smc_fitzpatrick` | 7 |
| `star_forming`/`salim` | 3 | `calzetti`/`calzetti2000` | 8 |
| `mw_allen`/`allen` | 4 | `star_forming_reddy`/`reddy` | 9 |

Any unrecognized string is passed through `int()` as a last resort; unmappable values raise `ValueError`.

### `add_dust_emission()`

```python
galaxy.add_dust_emission(
    model_type='greybody',   # or 'blackbody'
    iscalable=-2,             # -2 = scaled by dust mass
    w_min=1.0, w_max=1000.0, Nw=200,
    ithick=0                  # greybody only
)
```

DEM always shares the SSP's `igroup` and takes `id = ssp.id + 1` — this is a fixed convention, not configurable. `add_galaxy()` preserves an existing `dem`/`kin` across auto-reassignment by rebuilding the DEM at the new SSP id.

## AGNInstance

`SEDModel.create_agn(agn_components=[...])` builds from a components list; each component maps to a fixed id offset from `base_id`/`base_igroup`:

| offset | component | attribute | default builder |
|---|---|---|---|
| `+0` | disk | `dsk` | `add_disk_bbb()` (BigBlueBump) |
| `+1` | BLR | `blr` | `add_broad_line_region()` |
| `+2` | FeII | `feii` | `add_feii_template()` |
| `+3` | NLR | `nlr` | `add_narrow_line_region()` |
| `+4` | torus | `tor` | `add_torus_fann()` |

`agn_components` accepts `'dsk'`/`'disk'`/`'bbb'`, `'blr'`, `'nlr'`, `'feii'`, `'tor'`/`'torus'` (aliases normalized internally). Default is `['dsk', 'blr', 'nlr', 'feii']` (no torus) when `agn_components=None`.

To swap in a different disk or torus type, call the instance method after `create_agn()` — it overwrites `dsk`/`tor` in place:

```python
agn = SEDModel.create_agn(base_igroup=7, base_id=8, agn_components=['dsk', 'nlr', 'feii'])
agn.add_disk_agn(name='agnsed', imodel=0, icloudy=0)  # replace default BBB disk
params.add_agn(agn, auto_assign_ids=False)  # keep the manual base_igroup/base_id
```

Disk variants: `add_disk_bbb()` (`BigBlueBumpParams`), `add_disk_agn()` (`AGNParams`), `add_disk_fann()` (`FANNParams`), `add_disk_aknn()` (`AKNNParams`). Torus variants: `add_torus_fann()`, `add_torus_aknn()`. Only BBB/AGN disks get a DAL entry (`get_dal_params()`, `ilaw=7`, AGN-specific law) — FANN/AKNN disks are pre-attenuated emulators and skip DAL entirely. FeII gets kinematics (`get_kin_params()`) automatically when present.

## SEDModel Instance Settings

Beyond galaxy/AGN components, a plain `SEDModel()` instance carries global settings, each returning `self` for chaining:

```python
model = SEDModel()
model.set_igm(igm_model=1)                                  # 0=None,1=Madau95,2=Meiksin06,3=hyperz,4=FSPS,5=Inoue+2014
model.set_cosmology(H0=70.0, omigaA=0.7, omigam=0.3)
model.set_redshift_prior(iprior_type=1, min=0.0, max=4.0, nbin=40)
model.set_sys_err_mod(iprior_type=1, min=0.0, max=0.1, nbin=40)
model.set_kinematics(id=0, velscale=10, num_gauss_hermites_con=0, num_gauss_hermites_eml=0)
model.set_luminosity(id=-1, w_min=..., w_max=...)            # w_min/w_max required
model.set_sfr_over(past_Myr1=10.0, past_Myr2=100.0)
model.set_lw_max(lw_max=...)
model.set_line_list(file=..., type=0)                        # 0=intrinsic,1=emergent,2/3=cumulative variants
```

These aren't consumed by `add_galaxy()`/`add_agn()`. Apply them with `params.add_model(model)`, which copies `igm`, `cosmology`, `z` (redshift prior), `sys_err_mod`, `kin`, `luminosity`, `sfr_over`, `lw_max`, and `line_list` onto the `BayeSEDParams` instance:

```python
model = SEDModel()
model.set_igm(igm_model=1).set_cosmology(H0=70.0, omigaA=0.7, omigam=0.3)
params.add_model(model)
```

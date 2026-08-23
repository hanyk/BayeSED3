# Data Preparation

Build an **Observation** in memory, then write an **Input catalog** (text file) for the binary. Filter setup lives here too.

AB mag / phot-only / spec-only run flags after the catalog exists: `advanced-run-modes.md`.

## SEDObservation

`SEDObservation` (in `bayesed.data`) is the primary class for building an input catalog from numpy arrays. It can hold photometry, spectroscopy, or both together. `PhotometryObservation` and `SpectrumObservation` are thin wrappers that raise if you accidentally pass the other kind of data.

```python
from bayesed.data import SEDObservation
import numpy as np

obs = SEDObservation(
    ids=['galaxy_001', 'galaxy_002'],
    z_min=[0.1, 0.2],       # required
    z_max=[0.15, 0.25],     # required
    distance_mpc=None,      # optional, defaults to zeros
    ebv=None,               # optional, defaults to zeros

    phot_filters=['SLOAN/SDSS.g', 'SLOAN/SDSS.r', 'SLOAN/SDSS.i'],
    phot_fluxes=np.array([[12.5, 25.1, 18.3], [15.2, 28.9, 22.1]]),  # shape (N, Nphot)
    phot_errors=np.array([[1.2, 2.5, 1.8], [1.5, 2.9, 2.2]]),        # shape (N, Nphot)
    input_type=0  # 0 = flux in μJy, 1 = AB magnitude
)

input_file = obs.to_bayesed_input('observation/my_analysis', 'my_catalog')
# Writes observation/my_analysis/my_catalog.txt, returns that path
```

### Photometry nondetections

Only meaningful when `input_type=0` (flux/μJy):

```python
obs = SEDObservation(
    ...,
    phot_flux_limits=np.array([0.5, 0.8, 0.6]),  # per-band limit in μJy, shape (Nphot,) or (N, Nphot)
    phot_nsigma=2.0,  # N-sigma threshold
)
```
Bands where S/N < 1 are rewritten as `F = Flim`, `sigma = -Flim/N` (the standard BayeSED nondetection convention). For `input_type=1` (AB mag) use `phot_mag_limits` instead, with the same `phot_nsigma`; nondetection is triggered when `sigma_m > 1.08574`.

### Spectroscopy

```python
obs = SEDObservation(
    ids=['galaxy_001'],
    z_min=[0.1], z_max=[0.15],
    spec_band_names=['B', 'R'],                          # one entry per band
    spec_wavelengths=[np.linspace(3800, 5500, 500),       # Angstrom, (Nw,) for N=1 or (N, Nw)
                       np.linspace(5500, 9200, 500)],
    spec_fluxes=[np.ones(500) * 10.0, np.ones(500) * 8.0],
    spec_errors=[np.ones(500) * 1.0, np.ones(500) * 0.8],
    spec_lsf_sigma=[np.ones(500) * 0.0003, np.ones(500) * 0.0003],  # LSF width in microns
    spec_flux_type='fnu',  # 'fnu' (μJy, default) or 'flambda' (erg s^-1 cm^-2 Å^-1)
)
```

Note the unit mismatch to watch for: `spec_wavelengths` and the input `phot`/`spec` flux arrays are specified in Angstrom/μJy at the `SEDObservation` level, but `spec_lsf_sigma` is in **microns**, and internally `create_input_catalog` stores wavelength and LSF columns in microns in the output file (the conversion happens for you — just don't pre-convert wavelengths yourself).

### other_columns

Arbitrary extra per-object columns (e.g. `ra`/`dec`) written to the catalog:

```python
obs = SEDObservation(..., other_columns={'ra': [150.1, 150.2], 'dec': [2.1, 2.2]})
```

### SNR / systematic error controls

`SEDObservation` also accepts `SNRmin1`, `SNRmin2` (both `SNRmin1Params`/`SNRmin2Params` from `bayesed.params`, with `phot`/`spec` float fields) and `sys_err_obs` (`SysErrParams`) — these are stored on the observation object for convenience but are actually applied via the `BayeSEDParams` config, not by `to_bayesed_input()`. Set them on `params` directly (see `references/galaxy-fitting.md`).

## Filters

### Download from SVO

```python
bayesed = BayeSEDInterface()
filter_files = bayesed.prepare_filters_from_svo(
    svo_filter_ids=['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r'],
    output_dir='observation/my_analysis/filters'
)
params.filters = filter_files['filters_file']
params.filters_selected = filter_files['filters_selected_file']
```

`svo_filter_ids` must match SVO Filter Profile Service IDs (`PROVIDER/FILTER_NAME`, e.g. `SLOAN/SDSS.g`) — this is also the format expected in `phot_filters`/input catalog band names.

### Build a selection file from an existing filters file

Use `bayesed.prepare_filters_selected(filters_file=..., selected_indices=[...])` (or `filter_names=[...]`) when filters are already downloaded and you just need to choose a subset, instead of re-downloading from SVO.

### Filter calibration constants

`bayesed/utils.py` exposes filter-file constants: `FILTER_TYPE_ENERGY` (itype=0) and `FILTER_TYPE_PHOTON` (itype=1, the default); `FILTER_CALIB_STANDARD` (icalib=0, B_ν = constant — the default). Other `FILTER_CALIB_*` names (IRAC, SUBMM, BLACKBODY, MIPS, SCUBA) exist for hand-built filter files. These matter only if constructing filter definition files by hand rather than via SVO download.

## Low-level: create_input_catalog

`SEDObservation.to_bayesed_input()` is a thin wrapper around `bayesed.utils.create_input_catalog()`. Call the function directly if you need catalog creation without the dataclass (e.g. building rows incrementally).

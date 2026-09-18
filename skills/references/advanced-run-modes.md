---
# Advanced Run Modes — Python API
---

Should-level recipes for production-style runs. Use after the Must happy path in `SKILL.md`. For Launcher Inspect (`results` / `plot` / `priors`), see `launcher-inspect.md`. For `run` / binary flags and `--import`, see `binary-cli.md`.

These patterns mirror `tests/run_test.py` (test1 / test3). Prefer assembling with `SEDModel` + `SEDInference` when possible; low-level `*Params` shown where tests use them.

## AB Magnitude Input

`input_type=1` when the input catalog stores AB magnitudes (not μJy flux).

```python
from bayesed import BayeSEDParams, SEDInference, ZParams
from bayesed.model import SEDModel

galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_lr_BaSeL_chab',
    sfh_type='exponential',
    dal_law='calzetti',
)
params = BayeSEDParams(
    input_type=1,  # AB magnitude catalog
    input_file='observation/test1/test_inoise1.txt',
    outdir='observation/test1/output',
    filters='observation/test1/filters_COSMOS_CSST_Euclid_LSST_WFIRST.txt',
    filters_selected='observation/test1/filters_COSMOS_seleted.txt',
    save_sample_par=True,
    save_bestfit=2,
    z=ZParams(min=0, max=4),
    suffix='_COSMOS',
    no_spectra_fit=True,  # photometry-only catalog
)
params.add_galaxy(galaxy)
params.configure_multinest(nlive=50, efr=0.1, updInt=1000)

inference = SEDInference()
inference.priors_init(params)
results = inference.run(params)
```

## Photometry-Only vs Spectroscopy-Only

| Flag | Effect |
|------|--------|
| `no_spectra_fit=True` | Ignore spectral columns (phot-only fit) |
| `no_photometry_fit=True` | Ignore photometry (spec-only fit) |

Do not set both. Typical CSST-style switch (`run_test.py` test3):

```python
params.no_spectra_fit = True       # itype == 'phot'
# or
params.no_photometry_fit = True    # itype == 'spec'
```

`BayeSEDParams.add_observation(obs)` copies these flags from an `SEDObservation` when present.

## Output Suffix

`suffix` disambiguates parallel runs that share an outdir / catalog stem:

```python
params = BayeSEDParams(
    ...,
    suffix='_phot',  # appears in output filenames
)
```

## NNLM, RDF, and SNR Cuts

Useful for spectroscopic residual modeling and quality cuts (test3):

```python
from bayesed import NNLMParams, SNRmin1Params, RDFParams

params.NNLM = NNLMParams(1, 1000, 0.0, 10, 0.01, 0.025, 0.975)
params.SNRmin1 = SNRmin1Params(0, 3)   # phot SNR min, spec SNR min
params.rdf = RDFParams(-1, 0)
```

Pair with HR SSP and chemical-evolution SFH when matching test3:

```python
galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_hr_stelib_chab_neb_300r',
    sfh_type='exponential',
    dal_law='calzetti',
    ssp_iscalable=0,
    sfh_itype_ceh=1,
)
```

## MultiNest: Quick vs Production

Via params (recommended on happy path):

```python
params.configure_multinest(
    nlive=40,    # quick test; production often 400–1000
    efr=0.3,     # ~0.3 for evidence; ~0.8 for parameter estimation
    updInt=1000,
    fb=2,
    INS=1,
)
```

Via `SEDInference` sampler helpers (applied at `inference.run`):

```python
inference = SEDInference()
inference.multinest(nlive=100, efr=0.1, INS=1, fb=2)
inference.priors_init(params)
results = inference.run(params)
```

Also available on `SEDInference`: `gsl()`, `nnlm()`, `ndumper()`, `set_logzero()`, `set_unweighted_samples()`, `set_nsample()`, `set_niteration()` — see method docstrings in `bayesed/inference.py`. Happy-path quick test: `inference.run(params, Ntest=2)`.

## Batch Path: Execution then Results

When you need exit codes / paths before loading science products:

```python
from bayesed import BayeSEDInterface, BayeSEDResults

execution = BayeSEDInterface(mpi_mode='auto', Ntest=2).run(params)
# execution is BayeSEDExecution — not posteriors
results = BayeSEDResults(params.outdir, catalog_name='...')
```

## Related

- Input catalogs and filters: `data-preparation.md`
- SEDModel assembly: `custom-models.md`
- Launcher Inspect: `launcher-inspect.md`
- Binary CLI / `run` / `--import`: `binary-cli.md`

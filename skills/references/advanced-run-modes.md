---
# Advanced Run Modes — Python API
---

Should-level recipes for production-style runs. Use after the Must happy path in `SKILL.md`. For Launcher Inspect (`results` / `plot` / `priors`), see `launcher-inspect.md`. Open this file for AB mag / phot-spec-only / NNLM-RDF-SNR or *when to change* `nlive` / `efr` beyond the SKILL FAQ — not for ordinary Skeleton fits, and not for slot *meanings* (`multinest-params.md`). For assembling Run argv, see `binary-cli.md` (only when a Prefer `run` gate in `SKILL.md` holds).

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

## MultiNest: when to change `nlive` / `efr`

`nlive` is nested-sampling resolution; wall-clock cost scales with it. Unattended fits leave library defaults (`nlive=100`, `efr=0.1`, `INS=1`, `ceff=0`). FAQ quick test: `Ntest=1` and `MultiNestParams(nlive=40)` — not `configure_multinest(...)`, which overwrites other fields. “Production” / “evidence” does not mean `nlive=400` or `efr=0.3` (that relaxes the default `efr=0.1`). Raise `nlive` only if the user wants stabler evidence (then check `logZerr`); noisy photometry can get worse with extra live points (Han et al. 2023). Raise `efr` to `0.8` only for faster parameter estimation.

```python
from bayesed import MultiNestParams

params.multinest = MultiNestParams(efr=0.8)  # faster PE; nlive stays 100
# or, if multinest already exists: params.multinest.efr = 0.8
```

Slot meanings: `multinest-params.md`. Assemble-run: patch the `--help` tuple via the `binary-cli.md` index; omit `--multinest` when the user did not state a sampling intent.

`SEDInference.multinest(...)` still exists; like `configure_multinest`, it is not a single-field patch.

Also available on `SEDInference`: `gsl()`, `nnlm()`, `ndumper()`, `set_logzero()`, `set_unweighted_samples()`, `set_nsample()`, `set_niteration()` — see method docstrings. Happy-path quick test stays the FAQ (`Ntest=1`, `nlive=40`).

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
- Assembling Run argv: `binary-cli.md` (only when a Prefer `run` gate in `SKILL.md` holds)
- MultiNest slot meanings: `multinest-params.md`

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BayeSED3 is a Bayesian Spectral Energy Distribution (SED) synthesis and analysis tool for galaxies and AGNs. It performs Bayesian parameter estimation using posterior probability distributions and Bayesian model comparison using Bayesian evidence. The project consists of compiled binaries (MultiNest-based), a Python interface package (`bayesed/`), and data files (models, filters, neural network emulators).

## Architecture

### Two Interface Levels

There are two distinct ways to use BayeSED3, and it's important not to confuse them:

- **Low-level** (`tests/run_test.py` style): Direct construction of parameter objects (`SSPParams`, `SFHParams`, `DALParams`, etc.) and manual assembly into `BayeSEDParams`. Full control, more verbose.
- **High-level Python API** (all other `tests/test_*.py`): Factory methods like `BayeSEDParams.galaxy()` and `BayeSEDParams.agn()` with sensible defaults, plus `SEDModel` builder.

### Core Package: `bayesed/`
- **`core.py`**: `BayeSEDInterface` (runs the binary), `BayeSEDParams` (config/factory methods), `BayeSEDExecution` (execution metadata returned by `run()`). Handles MPI parallelization, config file generation, and binary invocation.
- **`params.py`**: Dataclasses for every component (SSP, SFH, DAL, AGN, MultiNest, cosmology, etc.). These generate `.iprior` and config files consumed by the C/Fortran backend.
- **`model.py`**: `SEDModel` — high-level builder with `create_galaxy()` / `create_agn()` and `add_dust_emission()`.
- **`data.py`**: `SEDObservation`, `PhotometryObservation`, `SpectrumObservation` — convert numpy arrays to BayeSED input catalog format via `.to_bayesed_input()`.
- **`results/`**: Subpackage (not a single file). `BayeSEDResults` in `bayesed_results.py` handles HDF5 loading, posterior plotting, GetDist integration, parameter analytics (correlations, statistics), and SNR filtering. `standardize_parameter_names()` and `plot_posterior_comparison()` live here too.
- **`inference.py`**: `SEDInference` — prior management. Call `priors_init(params)` then `set_prior(name, ...)` with regex support.
- **`prior.py` / `prior_manager.py`**: Prior type definitions (Uniform, Gaussian, Gamma, Beta, Student's t, Weibull) and management utilities.
- **`plotting.py`**: `plot_bestfit()` and related visualization helpers.
- **`utils.py`**: Filter management (SVO download), `create_input_catalog()`, filter calibration constants (`FILTER_TYPE_ENERGY`, `FILTER_CALIB_STANDARD`, etc.).

### Key Object Lifecycle

`BayeSEDInterface.run(params)` → returns `BayeSEDExecution` (execution metadata: timing, exit code, paths). To access scientific results, construct `BayeSEDResults(outdir, catalog_name='...', object_id='...')` separately — it reads the HDF5 files written by the binary.

### Data Directories
- **`models/`**: Template SED models and pre-trained emulator data (FANN, AKNN, PCA).
- **`nets/`**: AKNN and FANN ML emulation models for fast SED prediction.
- **`filters/`**: Filter transmission curves (cigale, eazy-photoz).
- **`data/`**: Extinction curves, emission line templates.
- **`bin/linux/` and `bin/mac/`**: Compiled executables (`bayesed_mn_1` single run, `bayesed_mn_n` parallel MultiNest).

## Common Commands

### Installation
```bash
pip install -e .       # Editable install for development
pip install .          # Regular install for production
conda build conda/ && conda install --use-local bayesed3  # Conda install
```
OpenMPI 4.1.6 is required. Install via `conda install -c conda-forge openmpi=4.1.6` or system package manager.

### Running Tests
```bash
# Low-level interface tests (compiled binary + MPI)
python tests/run_test.py gal plot       # Galaxy spectroscopic fit
python tests/run_test.py qso plot       # QSO/AGN spectroscopic fit
python tests/run_test.py test1 plot     # Mock photometric fit
python tests/run_test.py test2 plot     # Real galaxy photometric fit
python tests/run_test.py test3 phot plot  # CSST mock photometric
python tests/run_test.py test3 spec plot  # CSST mock spectroscopic

# High-level Python interface examples
python tests/quick_start.py
python tests/test_agn_fitting.py
python tests/test_data_arrays.py
python tests/test_custom_model.py
python tests/test_multi_model_comparison.py
python tests/test_advanced_analytics.py
python tests/test_prior_management.py
python tests/test_bayesed_bagpipes_comparison.py
python tests/run_all_examples.py
```

Use `BayeSEDInterface(Ntest=2)` to limit processing to the first N objects — useful for quick iteration during development.

## Key Patterns

- **Parameter construction**: `BayeSEDParams.galaxy()` or `BayeSEDParams.agn()` for quick setup; construct individual `SSPParams`, `SFHParams`, `DALParams`, etc. manually for full control.
- **Result loading**: `BayeSEDResults(output_dir, catalog_name='...', object_id='...')` reads HDF5 files. Results are cached via `_lazy_load_hdf5()`.
- **Observation creation**: `SEDObservation` converts numpy arrays to BayeSED input catalogs via `.to_bayesed_input()`.
- **Prior management**: `SEDInference.priors_init(params)` then `set_prior(name, ...)` with regex pattern support — avoids manual `.iprior` file editing.
- **Model comparison**: Run multiple models on same data, use `from bayesed.results import standardize_parameter_names, plot_posterior_comparison`, then compare Bayesian evidence via `results.get_evidence()`.

## File Conventions

- Test output goes to `observation/<testname>/output/` where `<testname>` is the first argument to `run_test.py`.
- Input catalogs are text files (one source per line) with flux/error columns referencing filter names.
- Filter references follow `PROVIDER/FILTER_NAME` format (e.g., `SLOAN/SDSS.g`).
- Posterior samples stored as HDF5; best-fit SEDs as FITS files.
- Version follows date format: `YYYY.MM.DD` (e.g., `2026.01.31`).

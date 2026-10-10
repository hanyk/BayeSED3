# BayeSED3

Bayesian spectral energy distribution (SED) synthesis and analysis for galaxies and AGNs: parameter estimation via posterior distributions and model comparison via Bayesian evidence.

## Language

**Observation**:
A set of photometry and/or spectroscopy for one or more sources to be fitted (`SEDObservation` and related types).
_Avoid_: Calling the on-disk catalog file an observation

**Input catalog**:
The text file consumed by the binary (one source per line; flux/error columns and filter names).
_Avoid_: Input (alone); Results catalog; catalog when meaning HDF5 Results products

**Results catalog**:
The name that distinguishes one set of Results products under an outdir.
_Avoid_: Input catalog; catalog when meaning the on-disk text file

**SED model**:
The physical component assembly used for a fit (`SEDModel` / `*Params` components such as SSP, SFH, DAL, AGN).
_Avoid_: Model alone when comparing statistical model configurations

**Model configuration**:
One complete fitting setup (SED model + priors + run options) used in Bayesian model comparison.
_Avoid_: SED model when the subject is evidence ranking across setups

**Execution**:
Metadata from one binary run (`BayeSEDExecution`: exit code, paths, timing).
_Avoid_: Results, analysis results, fit output (when meaning scientific posteriors)

**Results**:
Scientific posterior products loaded from HDF5 (`BayeSEDResults` facade over products + presentation).
_Avoid_: Execution, run output (when meaning only process metadata); treating plot helpers as the definition of Results

**Evidence**:
Bayesian evidence (ln Z) for one Model configuration on one source (one Input catalog row). The Python API exposes `get_evidence()` with `log_evidence` / `log_evidence_error` (mapped from table columns) for a single source. Launcher `results … evidence --json` uses the HDF5 table column names (`ID`, and present `INSlogZ` / `logZ` / `INSlogZerr` / `logZerr` — two estimators and their errors) as a JSON array of row objects.
_Avoid_: Treating a multi-source Results catalog as one evidence number; treating raw `INSlogZ` as the only public API key name; treating Inspect JSON column names as the Python dict API

**Priors**:
The prior lifecycle for a model configuration — load/generate/validate/bootstrap of `.iprior` specifications (`Priors`). Interactive edit/print is an internal `PriorEditor` (not a public type). Working copies are the editable files beside the Input catalog (`set_prior` writes these). Factory defaults are the unmodified set for that Model configuration. On Launcher `run`, the compiled binary creates any missing `.iprior` files.
_Avoid_: Treating `PriorManager` or `PriorEditor` as the public entry name; treating sampler settings (`SEDInference.multinest`, etc.) as priors; treating `.iprior` as a free-form agent-edit surface (custom bounds use Python `set_prior`); treating Python `priors_init` as the Launcher `run` prior procedure; treating factory defaults as the files `set_prior` edits; treating a shared Input catalog directory as per-configuration isolation for working copies

**Prior type**:
One of the discrete prior distribution families (Uniform, Gaussian, …) with fixed hyperparameter slots. `bayesed3 priors list` with no directory prints the Prior type table; with a directory it loads `.iprior` files there and prints their Priors contents — one `list [dir]` form, two meanings by presence of the directory.
_Avoid_: Calling `.iprior` file contents a prior type; calling either mode “list prior files”; splitting into separate user-facing verbs when the single optional-directory form is intended

**Launcher**:
The `bayesed3` / `python -m bayesed` console entry of the installed Python package. It is the cross-platform unified interface to the compiled binary (OpenMPI, OS-arch, process count, data-root cwd). First token selects the path: `version` / `--version` report Package version and Binary banner; `results` / `plot` / `priors` present existing Results or Priors; `run` takes Launcher-owned `--mpi` / `--np` and forwards **Run argv** to the compiled binary. Empty argv and top-level `--help` are Launcher overview: one product tagline plus Version/Inspect/Run syntax sketches (verb names allowed on Inspect lines; Inspect’s first sketch may list `--log-level` enum values; no per-entry prose, no other Inspect flag lists, no multi-line subcommand usage), not binary help. Coexists with the Python analysis API (`SEDInference` / `BayeSEDParams`) and with `BayeSEDInterface` (Params → Execution); they share internal `runtime`.
_Avoid_: Treating `run` as a no-Python or Python-free path; treating every invocation as a binary run; treating `results` / `plot` / `priors` as fitting or as a second SED-model API; treating bare binary argv (without `run`) as the fit path; routing the binary path through `BayeSEDInterface.run(list)`; treating the Launcher as only a `--help` helper; conflating package name `bayesed3` with the Python happy path; calling the `run` first token "binary" in user-facing grammar; treating Package version and Binary banner as one “version”; putting full Inspect usage or `--cat`/`--par` details on the overview; treating `--log-level` as controlling the compiled binary; treating Inspect `plot --save` as Run argv `--save_*`

**Run argv**:
The argument list after `bayesed3 run` that the Launcher forwards to the compiled binary, except Launcher-owned `--mpi` / `--np`. One encoding of a Model configuration. Live flag shapes come from `bayesed3 run --help`. `--z` / `--sys_err_*` may appear here.
_Avoid_: Inspect argv; top-level `bayesed3 --help`; treating Run argv as a no-Python path; stuffing binary USAGE into the agent skill as the source of truth

**Inspect loose match**:
The Launcher inspect name ladder for `--cat` / `--cfg` / `--obj` / `--par`: exact, then contains, then regex; first non-empty hit list wins. Usage shows these value slots as `PATTERN`.
_Avoid_: Treating PATTERN as regex-only; conflating with the Python Results / Model configuration soft-match; putting the ladder on the Launcher overview

**Plot save**:
Inspect `plot` boolean `--save`: when present, write one image per plotted object beside that object's Results products. When omitted, display only. The user does not choose a path. Presentation computes the sidecar; Inspect only passes `save=`. Bestfit uses the FITS stem. Posterior uses the loaded `*_sample_par` config stem (strip `_sample_par`) plus `_posterior_free` / `_posterior_derived` / `_posterior`. Python `plot_bestfit` and `plot_posterior*` default to writing that sidecar (`save=True`); Inspect without `--save` passes `save=False`.
_Avoid_: Results outdir as the save directory; path-valued `--out`; treating Inspect `--save` as binary `--save_bestfit` / `--save_sample_par`; treating Python default sidecar PNGs as Inspect behaviour when `--save` is omitted; Inspect assembling sidecar paths; a second lookup of sample_par for the PNG; stashing the sample_par path on GetDist samples; `{stem}_free.png` / `sample_par_free`; one PNG for every posterior kind; treating Python `plot_posterior*` as save-off by default

**Bestfit appearance**:
Inspect `plot … bestfit` flags that change how the best-fit SED is drawn. When a flag is present, Inspect passes that value through to `plot_bestfit`. When it is omitted, Inspect does not pass the argument, so the Python default stands. Distinct from Plot save, where omitting `--save` forces display only.
_Avoid_: These flags on posterior plots; `--hide` of one component; `--linear` with an axis; a user-chosen image path; font size, figure size, or dpi on Inspect

**Component name**:
The legend string for one `model:` extension on a best-fit SED, excluding `model:total`. The list follows those extensions in file order. `list_model_components()` returns the extension names that order and that `show_components` keys use.
_Avoid_: Posterior `--lab`; SED model; Model configuration; the FITS extension name itself

**Fit quality**:
Per-object goodness-of-fit facts computed from bestfit FITS residuals (`BayeSEDResults.get_fit_quality()`; Launcher `results <outdir> fit_quality --json`). Residuals are `(obs−mod)/err` on native FITS columns (sigma units, invariant under flux-unit conversion); a dataset record aggregates n / χ² / χ²_red / median / RMS / `frac_outside` at `sigma_threshold` (default 3, `--sigma`), plus a ranked `outliers` top-N (`--top`, default 20) with obs+rest wavelengths. χ² uses the header `XMIN²/ND` when present (source "header"), else computed. Datasets: spectroscopy splits by `iband` by default, or by `--band` rest-frame μm boundary list into contiguous half-open segments `[min,b1),…,` last closed (a partition — an edge point counts once, right-hand side; replacing iband grouping); photometry stays per `band` name. `line_residual_outliers` flags spectroscopy datasets only (|n_σ| > threshold within 0.005 μm of an `emission_lines.py` line; photometric bandpasses make "near a line" meaningless). Distinct from Evidence: Evidence ranks Model configurations, Fit quality judges one object's fit.
_Avoid_: `plot bestfit --json` (images stay images); paired-interval band syntax; applying `--band` to photometry; prose improvement advice in the payload; full per-point arrays; conflating `frac_outside` thresholding with the always-present ranked `outliers` list; per-object χ² versus MultiNest header χ² confusion

**Inspect wide table**:
Inspect `results … par_values --wide`: human Astropy table with no row/column elision (`pprint(max_width=-1, max_lines=-1)`), so middle columns and middle rows are not replaced by `...`. Default human `par_values` keeps Astropy terminal truncation. Ignored with `--json`.
_Avoid_: `--full`; Bestfit `--full-range`; treating `--wide` as changing JSON shape or HDF5 contents

**Inspect transpose table**:
Inspect `results … par_values --transpose`: human table from `get_parameter_values(..., transpose=True)` — rows are parameter/statistic columns, columns are object IDs. Independent of Inspect wide table (does not imply `--wide`). Ignored with `--json` (JSON stays row=object records).
_Avoid_: Transposing `--json`; auto-enabling `--wide`; per-parameter block printing as the default Inspect behaviour

**Package version**:
The installed `bayesed3` distribution identity from package metadata (calendar versioning via setuptools-scm). Shown as the `python` line of `bayesed3 version`.
_Avoid_: Python version (the interpreter); binary version; “BayeSED3 version” when only one of the two identities is meant

**Binary banner**:
The compiled program’s identity line from its help output (commit, branch, build time, OS, Copyright / citations). Not a CalVer package number and not a dedicated binary `--version` flag. Shown as the `binary` line of `bayesed3 version`.
_Avoid_: Binary version (when implying CalVer); Package version; treating help USAGE as the version identity

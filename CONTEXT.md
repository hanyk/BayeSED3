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
Bayesian evidence for a model configuration (`get_evidence()` → `log_evidence`, with error).
_Avoid_: Treating raw `INSlogZ` as the only public key name

**Priors**:
The prior lifecycle for a model configuration — load/generate/validate/bootstrap of `.iprior` specifications (`Priors`). Interactive edit/print is an internal `PriorEditor` (not a public type).
_Avoid_: Treating `PriorManager` or `PriorEditor` as the public entry name; treating sampler settings (`SEDInference.multinest`, etc.) as priors

**Prior type**:
One of the discrete prior distribution families (Uniform, Gaussian, …) with fixed hyperparameter slots. `bayesed3 priors list` with no directory prints the Prior type table; with a directory it loads `.iprior` files there and prints their Priors contents — one `list [dir]` form, two meanings by presence of the directory.
_Avoid_: Calling `.iprior` file contents a prior type; calling either mode “list prior files”; splitting into separate user-facing verbs when the single optional-directory form is intended

**Launcher**:
The `bayesed3` / `python -m bayesed` console entry. First token selects the path: `version` / `--version` report Package version and Binary banner; `results` / `plot` / `priors` present existing Results or Priors; `run` forwards the rest to the compiled binary (after resolving OpenMPI / OS-arch / process count). Empty argv and top-level `--help` are Launcher overview: one product tagline plus Version/Inspect/Run syntax sketches (verb names allowed on Inspect lines; Inspect’s first sketch may list `--log-level` enum values; no per-entry prose, no other Inspect flag lists, no multi-line subcommand usage), not binary help. Coexists with `BayeSEDInterface` (Python Params → Execution); they share internal `runtime`.
_Avoid_: Treating every invocation as a binary run; treating `results` / `plot` / `priors` as fitting or as a second SED-model API; treating bare binary argv (without `run`) as the fit path; routing the binary path through `BayeSEDInterface.run(list)`; treating the Launcher as only a `--help` helper; conflating package name `bayesed3` with the Python happy path; calling the `run` first token "binary" in user-facing grammar; treating Package version and Binary banner as one “version”; putting full Inspect usage or `--cat`/`--par` details on the overview; treating `--log-level` as controlling the compiled binary

**Inspect loose match**:
The Launcher inspect name ladder for `--cat` / `--cfg` / `--obj` / `--par`: exact, then contains, then regex; first non-empty hit list wins. Usage shows these value slots as `PATTERN`.
_Avoid_: Treating PATTERN as regex-only; conflating with the Python Results / Model configuration soft-match; putting the ladder on the Launcher overview

**Package version**:
The installed `bayesed3` distribution identity from package metadata (calendar versioning via setuptools-scm). Shown as the `python` line of `bayesed3 version`.
_Avoid_: Python version (the interpreter); binary version; “BayeSED3 version” when only one of the two identities is meant

**Binary banner**:
The compiled program’s identity line from its help output (commit, branch, build time, OS, Copyright / citations). Not a CalVer package number and not a dedicated binary `--version` flag. Shown as the `binary` line of `bayesed3 version`.
_Avoid_: Binary version (when implying CalVer); Package version; treating help USAGE as the version identity

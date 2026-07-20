# BayeSED3

Bayesian spectral energy distribution (SED) synthesis and analysis for galaxies and AGNs: parameter estimation via posterior distributions and model comparison via Bayesian evidence.

## Language

**Observation**:
A set of photometry and/or spectroscopy for one or more sources to be fitted (`SEDObservation` and related types).
_Avoid_: Calling the on-disk catalog file an observation

**Input catalog**:
The text file consumed by the binary (one source per line; flux/error columns and filter names).
_Avoid_: Input (alone), catalog when meaning HDF5 results

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

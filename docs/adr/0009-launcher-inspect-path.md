# Launcher first-token inspect path

ADR-0007 made `bayesed3` a thin binary forwarder and forbade treating it as a Results / Priors workflow. We now split on the first token: `results`, `plot`, and `priors` present existing products; every other argv (including `-h` / `--help`) stays the binary path.

Inspect grammar uses flags for catalog / model configuration / object wherever Results are loaded: `--cat` / `--cfg` / `--obj`. No trailing positional catalog or object names.

- **results query** (`summary` / `evidence` / `par_names` / `objects`): verbs first (stackable), then optional `--cat` / `--cfg` / `--obj` (`--obj` is exact `object_id` on the constructor).
- **par_values**: isolated; `--par` required; optional `--cat` / `--cfg` / `--obj` (`--obj` filters via `get_parameter_values`, loose match).
- **catalogs** / **model_configs**: discovery only (`list_catalog_names` / `list_model_configs`), no HDF5 load; `model_configs` takes optional `--cat`; neither combines with other commands.
- **plot**: `bestfit` or `posterior` (not combined); same `--cat` / `--cfg` / `--obj`; `posterior` is `free` / `derived` / repeating `--par` with optional adjacent `--lab`; `derived` may take `--max`. `--cfg` is passed through to `BayeSEDResults` (no CLI pre-check). Multiple model configurations without `--cfg` raise from the library.

They call existing `BayeSEDResults` / `list_prior_types` / `PriorManager.load_prior_file` + `PriorEditor.print_priors` — not a new `.iprior` parser, not fit, not `priors_init`.

We rejected a second console script (easy to miss), a named-fit / YAML surface (a second option language), and `priors init` on this path (needs a Model configuration the binary already accepts via `--priors_only`).

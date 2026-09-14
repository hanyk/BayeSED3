# Launcher first-token paths (inspect + run)

ADR-0007 made `bayesed3` a thin binary forwarder. We split on the **first token**: `results`, `plot`, and `priors` present existing products; **`run`** forwards the rest to the compiled binary (Launcher-owned `--mpi` / `--np` only after `run`). Bare binary argv without `run` is rejected with a short error that rewrites the line as `bayesed3 run <original argv>`. This supersedes ADR-0007’s rule that every non-inspect argv (including `-h` / `--help`) goes to the binary.

**Help / discovery (human + agent):** Empty argv and top-level `--help` / `-h` print the same short Launcher overview (exit 0)—inspect entries, `run`, and pointers to sub-usages—not the full flag lists and not binary USAGE. `bayesed3 run` and `bayesed3 run --help` are equivalent and show binary help. Flag details (`--cat` / `--cfg` / `--obj` / `--par` / …) appear only on the relevant subcommand usage or when an error must prompt the next step. Missing catalog / model configuration errors keep the library message and add one copy-pasteable hint to `catalogs` / `model_configs` (no auto-pick). `results <outdir>` with no subcommand lists catalogs (optional `--json`, exit 0); bare `plot <outdir>` still prints plot usage.

Inspect grammar uses flags for catalog / model configuration / object wherever Results are loaded: `--cat` / `--cfg` / `--obj`. No trailing positional catalog or object names.

- **results query** (`summary` / `evidence` / `par_names` / `objects`): verbs first (stackable), then optional `--cat` / `--cfg` / `--obj` (`--obj` is exact `object_id` on the constructor); optional `--json` on all `results` commands.
- **par_values**: isolated; `--par` required; optional `--cat` / `--cfg` / `--obj` (`--obj` filters via `get_parameter_values`, loose match).
- **catalogs** / **model_configs**: discovery only (`list_catalog_names` / `list_model_configs`), no HDF5 load; `model_configs` takes optional `--cat`; neither combines with other commands.
- **plot**: `bestfit` or `posterior` (not combined); same `--cat` / `--cfg` / `--obj`; `posterior` is `free` / `derived` / repeating `--par` with optional adjacent `--lab`; `derived` may take `--max`. `--cfg` is passed through to `BayeSEDResults` (no CLI pre-check). Multiple model configurations without `--cfg` raise from the library (plus discovery hint above).

They call existing `BayeSEDResults` / `list_prior_types` / `PriorManager.load_prior_file` + `PriorEditor.print_priors` — not a new `.iprior` parser, not fit, not `priors_init`.

We rejected a second console script (easy to miss), a named-fit / YAML surface (a second option language), `priors init` on this path (needs a Model configuration the binary already accepts via `--priors_only`), optional `run` prefix (two equivalent fit spellings), and keeping top-level `--help` as binary help (inspect stayed undiscoverable).

Inspect argv stays hand-written (no argparse / click / typer). Shared `--flag value` walking and usage lines live in `inspect_cli`; `--par` / `--lab` interleaving stays custom.

# Platform-independent Launcher beside BayeSEDInterface

Superseded in part by ADR-0009: first-token `results` / `plot` / `priors` present existing products; the binary-forwarder rules below still apply to every other argv.

`bayesed3` (and `python -m bayesed`) is the **Launcher**: a thin, platform-independent shell that is fully capable relative to the compiled binary — not a second Python analysis API and not Observation / SED model / fitting. It owns only `--mpi` and `--np` (plus a short usage message when invoked with no args); all other argv, including `-h`/`--help`, is forwarded to the binary. Empty argv → short usage that points at `bayesed3 --help`; help flags → binary USAGE. I/O is streamed; the process exit code is the binary/mpirun return code — no Interface Step/progress/`BayeSEDExecution` summary.

**Coexistence:** Launcher (CLI argv → binary) and `BayeSEDInterface` (Python `BayeSEDParams` → Execution) both stay. They share internal `bayesed.runtime`, not each other. `BayeSEDInterface.run` accepts only `BayeSEDParams`; the broken `run(list)` passthrough is removed rather than repaired. ADR-0004’s rejection of a “public CLI entry point” means no public Python export of argv serialization (`to_cli_args`); it does not forbid this console-script Launcher.

**`runtime` ownership (extends ADR-0005):** Beyond OpenMPI / OS-arch binary / process-count resolution, `runtime` also owns data-root cwd for the binary process, relative→absolute path preparation for that cwd change, and `build_mpi_command` (mpirun + scheduler flags + `-np` + executable + binary argv). Callers (`Launcher`, `BayeSEDInterface`) each own their own `Popen` and presentation. MPI *mode selection* heuristics (`_select_mpi_mode`) remain on the Interface; the Launcher takes `--mpi` / `--np` from the user and passes them into the same builders.

We rejected: routing the Launcher through `BayeSEDInterface.run(list)`; duplicating cwd/path/`mpirun` assembly outside `runtime`; folding Interface progress UX into `runtime`; treating the Launcher as help-only or as a replacement for the Python happy path.

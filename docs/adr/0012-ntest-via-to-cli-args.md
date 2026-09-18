# Ntest reaches the binary via to_cli_args

`BayeSEDParams.Ntest` is a compiled-binary flag (`--Ntest`). `to_cli_args` emits it when set. `BayeSEDInterface(Ntest=…)` / `SEDInference.run(..., Ntest=…)` override `params.Ntest` before serialization so `--Ntest` appears once in the argv list. We rejected leaving `params.Ntest` as a silent no-op while only `build_mpi_command(..., ntest=)` injected the flag (dual homes; Consistency A/B needed a special case). The MPI helper may still append `ntest=` for older callers, but the Interface run path no longer uses that argument.

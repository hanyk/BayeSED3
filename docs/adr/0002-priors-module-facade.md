# Priors module with SEDInference facade

Prior lifecycle (init, validate, binary `.iprior` bootstrap) lives in a deep `Priors` module. `SEDInference.priors_init` / `set_prior` / `print_priors` / `validate_priors` / `list_prior_types` remain as thin facades that delegate to a `Priors` instance held on the inference object, so existing callers and the skill happy path stay unbroken. `PriorManager` stays an internal adapter for `.iprior` I/O; binary bootstrap stays a private implementation detail of `Priors` (not an injectable seam until a second adapter exists). We rejected breaking the public API onto `Priors` alone, thickening `PriorManager` into the sole entry point, and injecting a fake generator in the first cut.

Interactive edit/print (`set_prior` / `print_priors` implementation) later moved to `PriorEditor`; `Priors` keeps thin delegates. See ADR-0006.

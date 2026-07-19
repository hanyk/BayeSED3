# CLI argv adapter outside BayeSEDInterface

Serializing a `BayeSEDParams` model configuration to the binary argv list lives in an internal `bayesed.cli_args` module (`to_cli_args` plus private `_format_*` helpers). `BayeSEDInterface` invokes that function from the run path and no longer owns `_params_to_args` / formatters. The adapter is not exported from the public `bayesed` package. We rejected a public CLI entry point, attaching format methods to each params dataclass, and leaving a thin `_params_to_args` middle-man on the interface.

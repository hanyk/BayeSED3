# BayeSEDParams: declared fields only

`BayeSEDParams` is the Model configuration object. It accepts writes only to names in its dataclass fields; unknown attribute names raise `AttributeError` at the write (with a close-name hint when one exists). `add_model` must assign onto those declared fields (e.g. `IGM`, `SFR_over`), never invent parallel names — otherwise settings never reach `cli_args` / the binary, and Evidence can silently describe the wrong Model configuration.

We rejected catching this only in `validate()`: after a typo write, there is no signal left to distinguish an undeclared attribute from a real field. We rejected warn-only: wrong Evidence must not pass quietly.

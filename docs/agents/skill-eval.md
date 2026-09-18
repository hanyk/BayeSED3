# Skill agent eval playbook

How to check that `skills/SKILL.md` guides an agent through a full Bayesian SED workflow. Decision record: ADR-0011.

## Protocol

- Fresh agent session per task (bayesed3 skill + this repo).
- **Skill-only knowledge (simulates a real user):** the agent may use `skills/SKILL.md`, `skills/references/*`, the installed `bayesed` package (import / run / Inspect), and paths given in the prompt / fixtures. It must **not** search or read `bayesed/*.py`, `tests/`, or `docs/adr/` to discover the workflow. Doing so is a correctness failure (or severe protocol violation). Scorers may use `to_cli_args` / Evidence diffs **outside** the agent session.
- Run each task at least twice; record variance.
- Score **correctness** first. Compare **efficiency** only when both sides completed correctly.
- Primary efficiency: token count + tool rounds. Secondary: session wall clock. Control: binary fit wall clock.
- Also record which `skills/references/*` files the agent opened.

## Fixtures

| Label | Input catalog | Role |
|-------|---------------|------|
| gal | `observation/test/gal.txt` | Simple galaxy happy path |
| qso | `observation/test/qso.txt` | Complex AGN / multi-component (required) |

Use small runs for iteration: `Ntest=1` or `2`, `nlive=40`, and a fixed MultiNest `seed` (≥0, same on both paths for consistency checks).

## Task set

Run **gal and qso** for tasks 1, 2, 5, and the consistency A/B. Task 3 may reuse an outdir from task 1. Task 4 probe is TBD.

| # | Name | Prompt (intent) | Pass if |
|---|------|-----------------|---------|
| 1 | Happy path | Fit with the skill; print summary; get Evidence. Prefer `SEDInference`. | `priors_init` then `run` → Results; summary + Evidence |
| 2 | Python branch | Same catalog; phot-only **or** add dust / mixed component; stay on Python API. | Opens the right reference; still Results, not bare `run` |
| 3 | Batch + Inspect | Do not refit. From an `outdir`, use `bayesed3 results … --json` for summary / evidence. | Inspect path; JSON Evidence uses HDF5 column names |
| 4 | `run` positive (TBD probe) | Require a binary-only flag / `--import` that Params does not cleanly expose. | Uses `bayesed3 run` + `binary-cli.md`; states why Python was insufficient |
| 5 | `run` negative | Ordinary fit prompt; do **not** mention `run`. | Must **not** default to `bayesed3 run` |

### Consistency A/B (hard gate)

Same Model configuration, same `nlive` / `Ntest` / `seed`, separate outdirs:

1. Path A: Python API → capture golden argv via `to_cli_args(params)`.
2. Path B: `bayesed3 run` with equivalent argv (or `--import` expanded to equivalent).
3. **Argv:** normalize (stable flag order; drop Launcher `--mpi` / `--np`); must match.
4. **Science (hard):**
   - Evidence: `|Δlog_evidence| ≤ max(0.05, 2·√(σ₁²+σ₂²))` using each side’s error.
   - Key free params: median difference ≤ the larger 68% posterior width (name params per configuration; qso ≠ gal).
   - If normalized argv are identical: apply a stricter near-identity bar on Evidence / summaries.

Efficiency A/B uses this same pair: differences should be agent overhead, not MultiNest, when argv match.

## Correctness scorecard

| Check | Fail if |
|-------|---------|
| Entry Prefer | Ordinary new fit uses `bayesed3 run`; in-process Results uses Inspect; shell-only outdir uses Python load when Inspect was enough |
| Skill-only | Reads `bayesed/*.py`, `tests/`, or ADRs to learn the fit workflow |
| Priors | Skips `priors_init` before fit / `set_prior` |
| Results vs Execution | Analyzes posteriors from Execution metadata |
| Products | No summary / no Evidence when asked |
| Task 5 | Defaults to `bayesed3 run` |

## Result log (copy per run)

```
task: <1-5 or A/B>
fixture: gal | qso
attempt: 1 | 2
correctness: pass | fail — <reason>
entry_used: SEDInference | BayeSEDInterface | Inspect | bayesed3 run
skill_only: yes | no — <if no, what was read>
refs_opened: …
tokens: …
tool_rounds: …
session_s: …
fit_wall_s: …
evidence: logZ=… ± … (path A) / … ± … (path B) | Δ=…
argv_match: yes | no
notes:
```

## Phase 2 (later)

Automate prompt files, `to_cli_args` capture, dual runs, and Evidence checks only after this manual playbook is stable. Do not build the harness first.

# Skill agent eval playbook

How to check that `skills/SKILL.md` guides an agent through a full Bayesian SED workflow. Decision record: ADR-0011. Related fix from Phase 1: ADR-0012 (`Ntest` via `to_cli_args`).

## How to run (operator checklist)

1. **Sync skill** — repo `skills/` is canonical; overwrite `~/.agents/skills/bayesed3/` and `~/.claude/skills/bayesed3/` after skill edits.
2. **Fresh agent per task** — parent session is scorched (contaminated). Spawn a **new** subagent / chat as the agent under test; parent only scores.
3. **Prompt template** — paste hard protocol + one task from the table below. Always pin: `outdir`, `Ntest`, `nlive`, MultiNest `seed` (≥0, same for A/B).
4. **Skill-only** — agent may use `skills/SKILL.md`, `skills/references/*`, installed `bayesed` / `bayesed3`, and fixture paths in the prompt. Reading `bayesed/*.py`, `tests/*.py`, `docs/adr/**`, or `CLAUDE.md` to learn the workflow = correctness fail (or severe protocol violation). Scorers may use `to_cli_args` / Evidence diffs **outside** the agent session.
5. **Score** — fill the result-log block; correctness first. Efficiency (tokens / tool rounds) only if both sides of a comparison completed correctly.
6. **Variance** — run each critical task at least twice (attempt 1 / 2) before calling the fixture stable.
7. **Outdirs** — keep under `tests/output_skill_eval_*` (local / gitignored-style litter; do not commit fit products).

Suggested order: task 1 gal → attempt 2 → Consistency A/B gal → task 1 qso → attempt 2 → Consistency A/B qso → tasks 2, 3, 5, 4.

## Protocol

- Fresh agent session per task (bayesed3 skill + this repo).
- **Skill-only knowledge (simulates a real user):** see checklist step 4.
- Run each task at least twice; record variance.
- Score **correctness** first. Compare **efficiency** only when both sides completed correctly.
- Primary efficiency: token count + tool rounds. Secondary: session wall clock. Control: binary fit wall clock.
- Also record which `skills/references/*` files the agent opened.

## Fixtures

| Label | Input catalog | Role |
|-------|---------------|------|
| gal | `observation/test/gal.txt` | Simple galaxy happy path / Inspect / A/B |
| qso | `observation/test/qso.txt` | Complex AGN / multi-component (required) |
| phot | `observation/test1/test_inoise1.txt` | Photometry + AB mag branch (task 2) |

Use small runs for iteration: `Ntest=1` or `2`, `nlive=40`, and a fixed MultiNest `seed` (≥0, same on both paths for consistency checks).

## Task set

Run **gal and qso** for tasks 1, 5, and Consistency A/B. Task 2 uses **phot**. Task 3 may reuse an outdir from task 1. Task 4 uses the ready-made `--import` probe below.

| # | Name | Prompt (intent) | Pass if |
|---|------|-----------------|---------|
| 1 | Happy path | Fit with the skill; print summary; get Evidence. Prefer `SEDInference`. | `priors_init` then `run` → Results; summary + Evidence |
| 2 | Python branch | Photometry catalog `observation/test1/test_inoise1.txt` with AB mag + phot-only (`advanced-run-modes.md`), **or** gal + dust/mixed. Stay on Python API. Do **not** use `gal.txt` + `no_spectra_fit` alone (spectrum-dominated catalog → degenerate Evidence). | Opens the right reference; still Results, not bare `run` |
| 3 | Batch + Inspect | Do not refit. From an `outdir`, use `bayesed3 results … --json` for summary / evidence. | Inspect path; JSON Evidence uses HDF5 column names |
| 4 | `run` positive | Require `bayesed3 run --import <opts>` with a ready-made options file (escape hatch). Probe: write exact contents to `tests/output_skill_eval_gal_t4_opts.txt` (see Phase 1 log / prior run), then `bayesed3 run --import …`. Do **not** rebuild via `SEDInference`. | Uses `bayesed3 run` + `binary-cli.md`; states why `--import` / run path was required |
| 5 | `run` negative | Ordinary fit prompt; do **not** mention `run`. | Must **not** default to `bayesed3 run` |

### Consistency A/B (hard gate)

Same Model configuration, same `nlive` / `Ntest` / `seed`, separate outdirs:

1. Path A: Python API → capture golden argv via `to_cli_args(params)`.
2. Path B: `bayesed3 run` with equivalent argv (or `--import` expanded to equivalent).
3. **Argv:** normalize (stable flag order; drop Launcher `--mpi` / `--np`); must match.
   - `Ntest` is part of that argv when set on `BayeSEDParams` (`to_cli_args` emits `--Ntest`). `inference.run(..., Ntest=N)` / `BayeSEDInterface(Ntest=N)` override `params.Ntest` before serialization. Do not also inject a second `--Ntest` in the MPI wrapper.
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
fixture: gal | qso | phot
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

## Phase 1 results (2026-09-18)

Manual skill-only board. Tokens were **not** instrumented this round (tool rounds / fit wall only).

| Task | Fixture | Attempts | Correctness | Notes |
|------|---------|----------|-------------|-------|
| 1 | gal | 2 | pass | Evidence −16287.4 ± 0.092 both; `SEDInference` |
| Consistency A/B | gal | 1 | pass | Δ Evidence = 0 after ADR-0012 |
| 1 | qso | 2 | pass | Evidence −86304.5 ± 0.056; Factory `agn` branch OK |
| Consistency A/B | qso | 1 | pass | Δ Evidence = 0; argv includes `--Ntest` |
| 2 | gal + `no_spectra_fit` | 1 | routing pass / science fail | Degenerate Evidence ≈ 0 — bad fixture |
| 2 | phot (`test_inoise1`) | 1 | pass | Retry; AB mag + phot-only |
| 3 | gal outdir | 1 | pass | `bayesed3 results … --json` |
| 5 | gal ordinary | 1 | pass | Did not default to `bayesed3 run` |
| 4 | gal `--import` | 1 | pass | `bayesed3 run --import`; Evidence matches gal A/B |

## Findings (what Phase 1 surfaced)

### Product (fixed)

1. **`params.Ntest` was a dead field** — setting `BayeSEDParams.Ntest` did not emit `--Ntest` via `to_cli_args`; only `inference.run(..., Ntest=)` / `BayeSEDInterface(Ntest=)` / MPI helper injected it. Consistency A/B needed a manual `--Ntest` special case. **Fix:** ADR-0012 — serialize in `to_cli_args`; Interface/`run` override once; skill FAQ updated.

### Eval design (playbook fixed)

2. **`gal.txt` + `no_spectra_fit` is not a photometry-only test** — spectrum-dominated catalog → MultiNest collapses (Evidence ≈ 0, ~nlive evals). Task 2 must use a photometric catalog (`observation/test1/test_inoise1.txt`) or a mixed/dust branch on gal, not phot-only on `gal.txt`.

3. **Task 4 probe must force `--import`** — otherwise agents rebuild with `SEDInference` and never exercise the escape hatch. Pin a ready-made opts file + mandatory `bayesed3 run --import`.

### Skill / agent behaviour (observations, not blockers)

4. **Prefer routing held** — ordinary fits used `SEDInference`; Inspect used for existing outdir; task 5 did not default to `bayesed3 run`; task 4 used `run` when required.
5. **Skill-only is enforceable and informative** — agents completed fits without reading `bayesed/*.py` / ADRs when the skill tree was enough.
6. **Reference fan-out varies** — attempt 1 sometimes opens an extra ref (e.g. `custom-models.md`) that attempt 2 skips; correctness stable; efficiency comparison needs token metering next round.
7. **Same seed ⇒ identical Evidence** across agent re-runs and A/B paths when argv match — science hard gate is practical at `nlive=40`.

### Still open for Phase 2

- Instrument **token count** (and tighter session wall clock) per attempt.
- Optional: scripted golden-argv capture + Evidence tolerance (do not build until this manual board stays useful across skill edits).
- Re-run Consistency A/B after any change to `to_cli_args` / Launcher argv assembly.

## Phase 2 (later)

Automate prompt files, `to_cli_args` capture, dual runs, and Evidence checks only after this manual playbook is stable. Do not build the harness first.

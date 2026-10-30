# Constraints

Last reviewed: 2026-10-13 by Doniele Antonio

## Floor (always enforced, no setup required)

Every file this plan produces; checked at task end and in CI:

- No new suppression comments: `# noqa`, `# type: ignore`, `# ruff: noqa`
- No unimplemented stubs: `raise NotImplementedError`, `except: pass`, empty catch-style swallows
- No skipped or deleted tests without a reason in the commit message
- No secrets in source (`OPENAI_API_KEY` never in a committed file)
- **This file does not get weakened to make a change pass.** Loosening it is a reviewed decision, never a fix.

## Enforced with numbers

Every row names the command producing the verdict. A number with no command is an aspiration, not a constraint.

| Dimension | Rule | Checked by | Runs at |
|---|---|---|---|
| Lint | Zero errors | `ruff check src tests scripts app.py` | task end, CI |
| Format | Zero diffs | `black --check src tests scripts app.py` | task end, CI |
| Tests | All pass; none need network | `pytest -q` | task end, CI |
| Coverage | Changed lines ≥ 80% | `pytest --cov=src --cov-report=term-missing` intersected with `git diff` | task end, CI |
| Types | Zero errors in scoped modules | `mypy src/utils src/chains` | task end, CI |
| Secrets | Zero hits | `python scripts/check_floor.py --repo-wide` | CI |
| Floor guard | Zero hits | `python scripts/check_floor.py --diff` | task end |
| Golden QA | Zero golden regressions; wrong-year contamination = 0 | `python scripts/eval_rag.py` | before every checkpoint gate |
| Demo | 6-step demo script green | manual run, journal §6 records the date | before submission |
| Docker | Image builds | `docker build -t isipfin .` | CI, before submission |

## Measured, not yet enforced

Recorded during build (T2, T6, T17 outputs). Ratchet rule: must not fall without a journal note.

| Metric | Today | Direction |
|---|---|---|
| Index build time (MSFT corpus) | not yet measurable | record at Checkpoint 1 |
| Answer latency p50 (simple) | not yet measurable | record at Checkpoint 2 |
| Answer latency (decomposed) | not yet measurable | record at Checkpoint 2 |
| Tokens in/out per answer (median) | not yet measurable | record at Checkpoint 2 |
| Golden-QA retrieval hit-rate@5 | not yet measurable | ≥ 0.9 target (see TESTING-AND-EVAL.md) |
| Citation presence rate | not yet measurable | 1.0 target |

## Floor guard mechanism

`scripts/check_floor.py` (created at T1, wired into CI at T23 as first step):

- **diff-scoped at task end**, repo-wide in CI.
- Patterns (any hit exits nonzero): `# noqa`, `# type: ignore`, `raise NotImplementedError`, `TODO`, `FIXME`, `except:` immediately followed by `pass`, `Not implemented`.
- Permitted exception: `TODO` inside prompt files under `prompts/` is a deliberate "record during build" marker per `docs/prompts/README.md` — markdown, not source. Guard scans `src/`, `tests/`, `scripts/`, `app.py` only.

## Adaptations to the exam (stated, not hidden)

Two deviations from a production bar, each with its reason:

1. **mypy scoped to `src/utils` and `src/chains` signatures**, not the whole tree. Reason: 46-hour clock; full strictness on Streamlit UI code burns hours on `Any`-heavy framework types for no grading credit. Contract-bearing code (gateway, graph state) is fully typed.
2. **Accessibility dimension dropped.** `axe` and Lighthouse need a running URL; CI cannot serve one. Per the constraint skill's rule, the dimension is dropped and named, not given an invented check. UI follows WCAG basics as good practice (labels, contrast); no automated gate claimed.

## Build-time checklist

Gates the build must pass, not claims it has passed:

- [ ] `ruff` and `black` clean on the union of changed files
- [ ] `pytest -q` green with **zero network-dependent tests**
- [ ] Coverage of changed lines ≥ 80%
- [ ] `mypy` clean on scoped modules
- [ ] `check_floor.py` clean (diff at task end, repo-wide in CI)
- [ ] `scripts/eval_rag.py` green including the $211,915M year-confusion canary
- [ ] `docker build` succeeds

Read before writing code; referenced by `docs/DEFINITION-OF-DONE.md` and task T1 in `docs/PLAN.md` (which creates the lint/format configs these checks depend on).

# Plan Review — redo audit after the skill update

> Audit of the 31-doc plan set against the **current** skill bindings and internal consistency.
> Trigger: the user updated `using-agent-skills`; it now binds the Definition of Done as a
> per-task gate, `CONSTRAINTS.md` as the written quality bar, the product-lifecycle worksheets,
> and explicit assumption-surfacing. Verdicts below are evidence-based, citing the exact
> sections inspected. Findings marked **FAIL** are fixed by this redo; the rest hold.

## Audit table

| File | Holds | Finding | Required fix |
|---|---|---|---|
| `docs/CAPABILITY-MAP.md` | Yes | Module ids, dependency direction, frozen interfaces; consistent with every other doc | none |
| `docs/SPEC.md` | Partial | Never names the graph nodes — a reader cannot connect the spec to the architecture | FAIL-2: add the 7 node names + ADR-001 pointer in §1 |
| `docs/ARCHITECTURE.md` | Yes | Node contract table + mermaid; `decompose_question` present here and in ADR-001 | none (it is the source of truth for node names) |
| `docs/PLAN.md` | **No** | **`decompose_question` has no task.** T10 covers `analyze_question`, T11 jumps to `hybrid_retrieve` — the exam's headline capability ("decomposing complex questions") has no task, no file, no test | FAIL-1: add T10b/retask so every ARCHITECTURE node maps to a task |
| `docs/TODO.md` | No | Inherits the missing decomposition task from PLAN | FAIL-1 (mirrored) |
| `docs/TIMELINE.md` | No | 10:30–14:30 block assumes decomposition inside T10/T12 | FAIL-1 (mirrored) |
| `docs/REPO-STRUCTURE.md` | Partial | `nodes/` directory listed but `decompose.py` absent | FAIL-3: add the file |
| `docs/adr/ADR-001..008` | Yes | Consistent with ARCHITECTURE; ADR-003 carries the vectorless trade-off | none |
| `docs/prompts/*` + README | Yes | P1–P5 versioned; `decompose_question` referenced from P1 | none |
| `docs/JOURNAL.md` | Partial | Correctly seeded with `[FILL AFTER BUILD]` slots, but no section records the redo itself or the lifecycle/constraints decisions | FAIL-4: add redo + play sections |
| `docs/TESTING-AND-EVAL.md` | Partial | `test_graph.py` row says "routing table edges per ADR-001" but no row tests decomposition output shape | FAIL-1 (mirrored: add assertion) |
| `docs/OBSERVABILITY.md` | Yes | Event shape, node trace contract consistent with ADR-007 | none |
| `docs/SECURITY-PRIVACY.md` | Yes | Assets/controls consistent; env vars match SPEC | none |
| `docs/RUNBOOK.md` | Yes | Env var table matches ADR-005 + SPEC | none |
| `docs/DELIVERABLES-MATRIX.md` | Partial | Row "Query decomposition" cites `decompose_question (T10)` — the wrong task id after the fix | FAIL-1 (mirrored) |
| `docs/GRADING-ALIGNMENT.md` | Partial | Same T10 mis-citation for requirement 4 | FAIL-1 (mirrored) |
| `docs/STRESS-TEST-RAG.md` | Yes | Mitigations reference node/state names that all exist | none |
| `docs/STRESS-TEST-DELIVERY.md` | Yes | Demo script maps to SPEC §10 criteria | none |
| `docs/README.md` | Partial | Index lacks the five new redo docs | FAIL-5 |
| **Missing:** `docs/DEFINITION-OF-DONE.md` | **No** | Updated skill binds the DoD as a gate on every task; absent | FAIL-6: create (now exists) |
| **Missing:** `docs/CONSTRAINTS.md` | **No** | `constraint-driven-development` requires a written, numbered bar | FAIL-6: create (now exists) |
| **Missing:** `docs/LIFECYCLE-PLAYS.md` | **No** | `product-lifecycle-plays` requires the three worksheets | FAIL-6: create (now exists) |
| **Missing:** `docs/DEPENDENCY-PINS.md` | **No** | Reproducibility is a graded value; no pinning strategy exists | FAIL-6: create (now exists) |

## Conflicts

1. **Task mapping conflict (real):** `decompose_question` exists in `docs/ARCHITECTURE.md` line 83 and `docs/adr/ADR-001` line 44, but `docs/PLAN.md` tasks, `docs/REPO-STRUCTURE.md` tree, `docs/TESTING-AND-EVAL.md` rows, and the two grading docs' task citations all omit it. Resolution: one task added, citations repointed.
2. **Version claims:** no hardcoded library versions exist anywhere in the 31 docs (the only version-looking strings are the prompt semver protocol and Docker's port 8501). Nothing to un-stale — but also nothing *pinned*, which the reproducibility criterion rewards. Resolution: `docs/DEPENDENCY-PINS.md` with PyPI-verified numbers.
3. **Env var names:** consistent across SPEC/ADR-005/RUNBOOK (verified by grep: `MOCK_LLM` 25 uses, `OPENAI_BASE_URL` 7, plus `OPENAI_MODEL`, `EMBEDDING_MODEL`, `DATA_DIR`, `LOG_JSON`, `LANGSMITH_TRACING`, `RETRIEVAL_K`). No conflict.
4. **SQLite schema:** `node_trace_json` and the 6 tables appear identically in ARCHITECTURE, ADR-006, ADR-007, OBSERVABILITY, PLAN. No conflict.
5. **Task ids:** T1–T26 all present, contiguous, referenced consistently in PLAN/TODO/TIMELINE/DELIVERABLES. No conflict.

## Gaps vs the updated skill bindings

| Binding | Status before redo | Fix |
|---|---|---|
| Definition of Done as per-task gate | missing | `docs/DEFINITION-OF-DONE.md` (created) |
| Numbered quality bar + floor guard | missing | `docs/CONSTRAINTS.md` (created) |
| Lifecycle worksheets | missing | `docs/LIFECYCLE-PLAYS.md` (created) |
| Reproducibility / pinned deps | missing | `docs/DEPENDENCY-PINS.md` (created) |
| Assumption surfacing (behavior 1) | implicit only | ledger below + inline ASSUMPTIONS in new docs |
| Post-node decomposition coverage | missing | FAIL-1 fix |
| Docs index completeness | 25 of 31 files listed | FAIL-5 fix |

## Assumptions ledger (previously implicit; now written)

Each row: the assumption, where it now lives.

| # | Assumption | Now written in |
|---|---|---|
| A1 | The grader can reach either the OpenAI API or accepts the keyless `MOCK_LLM=1` demo | `docs/SPEC.md` §1 (pre-existing), restated in `docs/RUNBOOK.md` |
| A2 | The reference corpus is text-layer PDF (not scanned); image-only uploads are a degraded, warned case | `docs/SPEC.md` §11, `docs/adr/ADR-004` |
| A3 | Reviewers run Docker (or the native pip path) — no Kubernetes/cloud deployment is graded | `docs/adr/ADR-008` |
| A4 | One user per container; multi-user auth is out of scope | `docs/SPEC.md` §8 Won't, `docs/SECURITY-PRIVACY.md` |
| A5 | Chunk sizes/tau are starting values tuned against the golden set — not final claims | `docs/SPEC.md` §11, `docs/PLAN.md` T17 |
| A6 | Journal numbers are filled from measured build output, never estimated | `docs/JOURNAL.md` header note, `docs/DEFINITION-OF-DONE.md` |
| A7 | Coverage bar applies to changed lines, not repo-wide (46h solo constraint) | `docs/CONSTRAINTS.md` |
| A8 | mypy strictness is scoped to `src/utils` + `src/chains` signatures, not the whole tree | `docs/CONSTRAINTS.md` |
| A9 | This redo was authored by the orchestrator, not subagents (pool rate-limited) — recorded for honesty | `docs/JOURNAL.md` §2 |

## Fixes applied by this redo

`FAIL-1` PLAN/TODO/TIMELINE/TESTING/DELIVERABLES/GRADING: `decompose_question` task added · `FAIL-2`
SPEC node names · `FAIL-3` REPO-STRUCTURE file · `FAIL-4` JOURNAL redo section · `FAIL-5` README
index · `FAIL-6` four new docs.

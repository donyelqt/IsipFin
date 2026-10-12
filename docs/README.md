# Docs Index

> Planning documentation for the Thinking Machines GenAI Engineer take-home: **FinDoc Assistant**
> — a Streamlit chatbot answering financial-PDF questions (reference corpus: Microsoft FY2023
> Annual Report) with cited answers, on a LangGraph hybrid-RAG pipeline.

## Reading order

**Reviewers (10 minutes):** `docs/JOURNAL.md` → `docs/GRADING-ALIGNMENT.md` →
`docs/STRESS-TEST-RAG.md`. That is the decision record, the requirement mapping, and the
adversarial review of the design.

**Engineers implementing it:** `docs/SPEC.md` → `docs/ARCHITECTURE.md` → `docs/PLAN.md` →
`docs/TODO.md`, consulting the ADRs per module as you reach it.

## Contents

### Design contract (frozen — change these first, everything else follows)

| Doc | What it is |
|---|---|
| `docs/CAPABILITY-MAP.md` | 7 module ids, dependency direction, build order, frozen interfaces |
| `docs/SPEC.md` | Six-area spec: objective, stack, commands, structure, tests, boundaries, MoSCoW scope, interface dataclasses, success criteria |
| `docs/ARCHITECTURE.md` | System context, ingestion + query pipelines (mermaid), node contract table, `GraphState`, data model, file ownership |

### Decisions (ADRs — `docs/adr/`)

| ADR | Decision |
|---|---|
| [ADR-001](adr/ADR-001-langgraph-orchestration.md) | LangGraph `StateGraph` orchestration; 7 named nodes; routing table |
| [ADR-002](adr/ADR-002-hybrid-vector-bm25-retrieval.md) | Hybrid FAISS + BM25, RRF fusion, parent-child chunks |
| [ADR-003](adr/ADR-003-vectorless-rag-evaluated.md) | Vectorless/tree RAG evaluated and rejected; section-path idea adopted |
| [ADR-004](adr/ADR-004-pdf-parsing-and-tables.md) | pypdf + pdfplumber markdown tables, cleaning rules, provenance invariants |
| [ADR-005](adr/ADR-005-llm-gateway-and-params.md) | One OpenAI-compatible gateway, typed outputs, `MOCK_LLM` mode |
| [ADR-006](adr/ADR-006-conversation-memory-sessions.md) | SQLite chat store; history windowed and contamination-free |
| [ADR-007](adr/ADR-007-observability-errors.md) | JSON events, node tracing, debug panel, typed error taxonomy |
| [ADR-008](adr/ADR-008-packaging-docker-ci.md) | One Docker image, compose data volume, CI pipeline |
| [ADR-009](adr/ADR-009-streamlit-native-ui.md) | Streamlit as the native UI; no React, no FastAPI layer — and why "the framework improves quality" is the wrong reason for it |
| [ADR-010](adr/ADR-010-neuro-symbolic-pipeline.md) | The graph is a deterministic neuro-symbolic pipeline and an agentic workflow, not a multi-agent fleet; symbolic owns control, neural owns semantics, plus a deterministic numeric-check pre-pass before LLM verification, plus the settled vocabulary decision (accepted labels: "agentic workflow", "deterministic neuro-symbolic pipeline") |

### Stress tests (adversarial reviews)

| Doc | What it is |
|---|---|
| `docs/STRESS-TEST-RAG.md` | 8 ranked failure modes of financial QA, each with a real trigger question and a mitigation in design vocabulary |
| `docs/STRESS-TEST-DELIVERY.md` | Delivery risk register with Plan Bs, the 6-step demo script, kill-switch order |
| `docs/GRADING-ALIGNMENT.md` | Every brief bullet → design element → demo evidence; over/under-built checks; journal hooks |
| `docs/PLAN-REVIEW.md` | Redo audit of the full doc set: verdicts per file, the one real defect found (missing decomposition task), conflicts, gaps, and the assumptions ledger |

### Quality bar (read before writing code)

| Doc | What it is |
|---|---|
| `docs/DEFINITION-OF-DONE.md` | The standing gate every task must clear *in addition to* its acceptance criteria; exam-specific gates; "not done despite looking done" red flags |
| `docs/CONSTRAINTS.md` | The numbered quality bar: floor rules, enforced-with-commands table, measured ratchets, floor-guard mechanism, the two exam adaptations stated openly |
| `docs/DEPENDENCY-PINS.md` | PyPI-verified pins (2026-10-13), base-image decision, floor-guard wiring, version re-verification protocol |
| `docs/LIFECYCLE-PLAYS.md` | Stage diagnosis, module portfolio view, and the prompt-iteration replacement risk register — play verdict: Extend |

### Execution

| Doc | What it is |
|---|---|
| `docs/PLAN.md` | 26 tasks (T1–T26) in 6 phases, acceptance + verification per task, checkpoints, risks |
| `docs/TODO.md` | The same tasks as a flat checklist (the task list target) |
| `docs/TIMELINE.md` | 46-hour schedule with checkpoint gates and buffer policy |

### Prompts and behavior

| Doc | What it is |
|---|---|
| `docs/prompts/README.md` | Prompt inventory P1–P5, versioning protocol, change log |
| `docs/prompts/query-analysis.md` | P1 — classify + plan (kind, sub_queries, fiscal_period, missing_aspect) |
| `docs/prompts/follow-up-contextualization.md` | P2 — standalone rewrite with citation disambiguation |
| `docs/prompts/answer-synthesis-with-citations.md` | P3 — grounded, cited synthesis + year-aware extraction contract |
| `docs/prompts/table-and-number-formatting.md` | P4 — number/table formatting contract |
| `docs/prompts/summarization-and-verification.md` | P5 — claim-level grounding verification + summary format |

### Operations and submission

| Doc | What it is |
|---|---|
| `docs/OBSERVABILITY.md` | Event shape, node trace, debug panel contract, metrics, failure→UI mapping |
| `docs/TESTING-AND-EVAL.md` | Hermetic test layers, golden QA set (10 cases incl. year-confusion canaries), pass bars |
| `docs/SECURITY-PRIVACY.md` | Assets, controls, prompt-injection posture, out-of-scope honesty |
| `docs/RUNBOOK.md` | Target-state commands, env vars, common failures and fixes |
| `docs/REPO-STRUCTURE.md` | Target repo tree with module ownership and enforced structural rules |
| `docs/JOURNAL.md` | The journal: decisions, hard problems, iteration templates, AI-use disclosure, measurements (§6 filled at build) |
| `docs/DELIVERABLES-MATRIX.md` | Every exam requirement → artifact → verification, plus submission checklist |

## Status

Planning complete; implementation not started (by design — this is the plan phase).
`docs/JOURNAL.md` sections marked `[FILL AFTER BUILD]` are intentionally empty slots to be
filled with measured data during the build, not placeholder claims.

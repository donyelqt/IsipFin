# Docs Index

> Planning documentation for the Thinking Machines GenAI Engineer take-home: **IsipFin**
> — a Streamlit chatbot answering financial-PDF questions (reference corpus: Microsoft FY2023
> Annual Report) with cited answers, on a LangGraph hybrid-RAG pipeline.

## Reading order

**Reviewers (10 minutes):** `JOURNAL.md` → `GRADING-ALIGNMENT.md` → `STRESS-TEST-RAG.md`.
The decision record, the requirement mapping, and the adversarial review.

**Engineers:** `SPEC.md` → `ARCHITECTURE.md` → `PLAN.md` → `TODO.md`, consulting the ADRs per
module as you reach it.

## Contents

### Design contract (frozen — change these first, everything else follows)

| Doc | What it is |
|---|---|
| `CAPABILITY-MAP.md` | 7 module ids, dependency direction, build order, frozen interfaces |
| `SPEC.md` | Six-area spec: objective, stack, commands, structure, tests, boundaries, MoSCoW scope, interface dataclasses, success criteria |
| `ARCHITECTURE.md` | System context, ingestion + query pipelines (mermaid), node contract table, `GraphState`, data model, file ownership |

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
| [ADR-010](adr/ADR-010-neuro-symbolic-pipeline.md) | The graph is a deterministic neuro-symbolic pipeline and an agentic workflow (not a multi-agent fleet); symbolic owns control, neural owns semantics, with a deterministic numeric-check pre-pass before LLM verification. Accepted labels: "agentic workflow", "deterministic neuro-symbolic pipeline" |

### Stress tests (adversarial reviews)

| Doc | What it is |
|---|---|
| `STRESS-TEST-RAG.md` | 8 ranked failure modes of financial QA, each with a real trigger question and a mitigation in design vocabulary |
| `STRESS-TEST-DELIVERY.md` | Delivery risk register with Plan Bs, the 6-step demo script, kill-switch order |
| `GRADING-ALIGNMENT.md` | Every brief bullet → design element → demo evidence; over/under-built checks; journal hooks |

### Quality bar (read before writing code)

| Doc | What it is |
|---|---|
| `DEFINITION-OF-DONE.md` | The standing gate every task must clear *in addition to* its acceptance criteria; exam-specific gates; "not done despite looking done" red flags |
| `CONSTRAINTS.md` | The numbered quality bar: floor rules, enforced-with-commands table, measured ratchets, floor-guard mechanism, the two exam adaptations stated openly |
| `DEPENDENCY-PINS.md` | PyPI-verified pins (2026-10-13), base-image decision, floor-guard wiring, version re-verification protocol |
| `LIFECYCLE-PLAYS.md` | Stage diagnosis, module portfolio view, and the prompt-iteration replacement risk register — play verdict: Extend |

### Execution

| Doc | What it is |
|---|---|
| `PLAN.md` | 26 tasks (T1–T26) in 6 phases, acceptance + verification per task, checkpoints, risks |
| `TODO.md` | The same tasks as a flat checklist (the task list target) |
| `TIMELINE.md` | 46-hour schedule with checkpoint gates and buffer policy |

### Prompts and behavior

| Doc | What it is |
|---|---|
| `docs/prompts/README.md` | Prompt inventory P1–P5, versioning protocol, change log |
| `docs/prompts/query-analysis.md` | P1 — classify + plan (kind, sub_queries, fiscal_period, missing_aspect) |
| `docs/prompts/follow-up-contextualization.md` | P2 — standalone rewrite with citation disambiguation |
| `docs/prompts/answer-synthesis-with-citations.md` | P3 — grounded, cited synthesis + year-aware extraction contract |
| `docs/prompts/table-and-number-formatting.md` | P4 — number/table formatting contract |
| `docs/prompts/summarization-and-verification.md` | P5 — claim-level grounding verification + summary format |

### Provider decisions (measured, post-build)

| Doc | What it is |
|---|---|
| `LOCAL-INFERENCE.md` | Local inference record (`qwen2.5:3b` on Ollama: per-node verdicts) and §7, the provider decision: the token constraint, the four options (Groq free / Groq Developer / local / hybrid) quantified with cost, latency and which pass bars each, and the recommendation with its exact `LLM_PROVIDER` / `LLM_MODEL` config |

### Operations and submission

| Doc | What it is |
|---|---|
| `OBSERVABILITY.md` | Event shape, node trace, debug panel contract, metrics, failure→UI mapping |
| `TESTING-AND-EVAL.md` | Hermetic test layers, golden QA set (10 cases incl. year-confusion canaries), pass bars |
| `SECURITY-PRIVACY.md` | Assets, controls, prompt-injection posture, out-of-scope honesty |
| `RUNBOOK.md` | Target-state commands, env vars, common failures and fixes |
| `REPO-STRUCTURE.md` | Target repo tree with module ownership and enforced structural rules |
| `JOURNAL.md` | The journal: decisions, hard problems, iteration templates, AI-use disclosure, measurements (§6 filled at build) |
| `DELIVERABLES-MATRIX.md` | Every exam requirement → artifact → verification, plus submission checklist |

## Status

Planning complete; implementation not started (by design — plan phase).
`docs/JOURNAL.md` sections marked `[FILL AFTER BUILD]` are intentionally empty slots to be
filled with measured data during the build, not placeholder claims.

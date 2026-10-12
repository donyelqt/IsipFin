# Implementation Plan: IsipFin

> Per-module tasks in dependency order. Task ids (`T1`…`T26`) are referenced by
> `docs/TODO.md` and `docs/TIMELINE.md`. Interfaces frozen in `docs/SPEC.md` §9; build order
> from `docs/CAPABILITY-MAP.md`. Nothing here is code — these are the tasks the build executes.

## Overview

Six modules built bottom-up: ingest → retrieval (+ gateway/store in parallel) → orchestration
→ UI → packaging, with prompts landing inside their owning task and observability attached per
node from the first node onward. Every task ends with a runnable verification; no task is
"done" on inspection alone.

## Phase 1 — Foundation: `doc-ingest` + `chunking-index` (serial chain)

| Id | Task | Files | Acceptance | Verify |
|---|---|---|---|---|
| T1 | Repo scaffold: `src/` tree, `requirements.txt` (pinned), `.env.example`, `.gitignore` (`data/`, `*.pdf`, `.env`), `pytest` config, `ruff`/`black` config | all | `pip install -r requirements.txt` + `pytest -q` (0 tests, green) + `ruff check` pass on empty tree | run install + pytest + ruff |
| T2 | `doc-ingest`: pypdf per-page text, pdfplumber tables → markdown, cleaning rules, section detection, `Block`/`ParsedDocument` dataclasses, sha256 `doc_id`, `text_layer: false` warnings | `src/core/pdf_parser.py`, `src/core/tables.py` | Ingesting the corpus yields per-page blocks; ≥30 table blocks detected; header/footer boilerplate absent from block texts; scanned-page path flags `text_layer: false` | unit test: ingest corpus fixture, assert table blocks exist, pages 1-based, no repeated header text |
| T3 | `chunking-index`: heading-aligned parents (~1000 tok), child chunks (~250/50), tables atomic in parents, page ranges | `src/core/chunking.py` | Every child maps to a parent; no parent cut inside a table; `page_end >= page_start`; overlap measured | unit test on fixture PDF: table-integrity + overlap + linkage invariants |
| T4 | Index build + hybrid retrieval: FAISS embed children, BM25 pickle, `chunks.jsonl` source of truth, RRF fusion (k=60), parent expansion, `retrieve(query, k, filters)` | `src/core/index.py`, `src/core/retrieval.py` | "revenue FY2023" returns the income-statement parent at rank 1 (sparse+dense both hit); exact token "Note 13" retrieves via BM25; index reload from disk works | script: build index, run 10 probe queries, print fused top-5 with scores |
| T5 | `scripts/download_corpus.py` + `scripts/build_index.py` + `docs.jsonl` catalog | `scripts/` | Clone → one command → corpus downloaded and indexed; re-running is a no-op (sha256 idempotent) | run both commands twice; second run reports "no changes" |

**Checkpoint 1 (after T5):** index builds from a cold clone; retrieval probe script shows sane
top-k for 10 financial queries; `pytest -q` green; `ruff` clean.

## Phase 2 — Contracts in parallel: `llm-gateway` + `chat-store` (+ prompts)

| Id | Task | Files | Acceptance | Verify |
|---|---|---|---|---|
| T6 | `llm-gateway`: `build_llm(params)`, OpenAI-compatible client, JSON-schema outputs, retry/backoff, typed errors, `MOCK_LLM=1` scripted mode with real FY2023 figures, token/latency accounting | `src/utils/llm.py`, `src/utils/errors.py`, `src/utils/mock_llm.py`, `src/utils/settings.py` | Real call returns usage+latency; mock mode answers the revenue question deterministically; 429 path retries then raises typed error; no node constructs clients | unit tests with mock transport; `MOCK_LLM=1 python -c "…"` revenue call |
| T7 | `chat-store`: SQLite schema (6 tables), accessors, session lifecycle, WAL mode | `src/store/db.py`, `src/store/sessions.py` | Create/switch/rename/delete cascades; citations rows persist and reload; `events` writes from the logger | unit test: full session lifecycle + reload assertions |
| T8 | Prompt assets (the five prompts live in `prompts/` with version headers; wired by nodes as they land in Phase 3) | `prompts/*.md` | Each prompt has id/version/model-target/changelog; mock-mode fingerprints resolve | manual: load each prompt, assert non-empty schema fields referenced by T9–T12 exist |

## Phase 3 — Core: `orchestration` (serial, highest risk)

| Id | Task | Files | Acceptance | Verify |
|---|---|---|---|---|
| T9 | `GraphState`, node skeletons with `@trace_node`, error mapping into `node_trace` | `src/chains/state.py`, `src/utils/tracing.py`, `src/chains/graph.py` | Empty graph runs; every node emits `NodeEvent`; exceptions become typed errors, never empty answers | unit test each node with stub gateway |
| T10 | `analyze_question` (+ T8's `query-analysis.md`): classification `simple/multi_part/out_of_scope`, `required_sections`, `missing_aspect`, JSON schema | `src/chains/nodes/analyze.py` | Revenue Q → simple; "compare X and Y" → multi_part; 2019 question → out_of_scope with `missing_aspect=time_period` | unit test: 12-question classification fixture, exact-kind assertions |
| T10b | `decompose_question`: rule-based split of `query_plan.sub_queries`, fan-out cap ≤4, one retrieval pass per sub-query | `src/chains/nodes/decompose.py` | multi-part plans yield ≤4 atomic sub-queries; a comparison never splits into bare operands (both sides kept together); sub-queries carry `required_sections` | unit test: fan-out cap enforced; operand-splitting rejected; 4-question fixture of real multi-part financial questions |
| T11 | `hybrid_retrieve` node + routing, score gate `tau`, one rewrite retry, context budget packing | `src/chains/nodes/retrieve.py` | Multi-part question fans out ≤4 and merges parents under budget; below-tau retrieval triggers exactly one retry; wide query never overflows context | unit test: fan-out cap, retry count = 1, budget cap |
| T12 | `synthesize_answer` (+ `answer-synthesis-with-citations.md`, `table-and-number-formatting.md`): citation-constrained synthesis, year-aware extraction contract | `src/chains/nodes/synthesize.py` | "revenue in 2023" → **$211,915 million** with citation; comparison question returns both years correctly; numbers without metric label in span get rejected | golden QA subset (see `docs/TESTING-AND-EVAL.md`), run in mock+real |
| T13 | `groundedness_check` (+ `summarization-and-verification.md`): deterministic numeric pre-pass in `src/chains/checks.py`, then per-claim span verification, trim-and-flag on failure | `src/chains/nodes/verify.py`, `src/chains/checks.py` | Claim supported only when span contains metric+figure; unsupported claims trimmed with visible marker; pessimistic default on schema violation; **deterministic test: an FY2024 figure ($245,122M) asserted as FY2023 is rejected by the pre-pass with zero LLM calls** | unit test: 8 claims, mixed support, assert exactly the right ones trimmed; pre-pass test asserts rejection without any gateway call |
| T14 | `contextualize_question` (+ `follow-up-contextualization.md`): 6-turn history + `cited_pages`, degradation to raw question on failure | `src/chains/nodes/contextualize.py` | "and the margin trend?" → "What was Microsoft's operating margin trend in FY2023?"; LLM failure leaves the turn alive with raw question | golden follow-up pairs; failure-injection test |
| T15 | `refuse` node: honest refusals, coverage-naming copy for time vs topic | `src/chains/nodes/refuse.py` | 2019 question → "documents cover FY2021–FY2023"; competitor question → topic-scoped refusal; no fabricated content | unit test both refusal classes |
| T16 | Graph wiring: edges + conditional routing per ADR-001 table, `recursion_limit`, params flow from UI into state | `src/chains/graph.py` | `invoke()` returns complete `AnswerEnvelope` incl. `node_trace`; all 6 demo-script questions pass end-to-end in mock mode | run demo script steps 3–6 against mock LLM |
| T17 | Golden QA harness `scripts/eval_rag.py` + dataset | `scripts/eval_rag.py`, `tests/data/golden_qa.jsonl` | Runs N questions, reports retrieval hit-rate, citation presence, figure match; exits nonzero on golden regressions | run against real LLM; record baseline numbers in journal |

**Checkpoint 2 (after T17):** all `docs/SPEC.md` §10 success criteria pass end-to-end (mock at
minimum); golden baseline recorded; `pytest -q` green.

## Phase 4 — `app-ui` (serial, thin)

| Id | Task | Files | Acceptance | Verify |
|---|---|---|---|---|
| T18 | `app.py` shell: chat pane, upload widget (multi-PDF), empty state. UI boundary per ADR-009: `app.py`/components contain zero business logic, no `import streamlit` outside the UI layer | `app.py`, `src/components/chat.py`, `src/components/upload.py` | Upload → index build progress → empty state → first Q&A with citation cards; `grep -rl "import streamlit" src/` returns only `src/components/` | manual run: full demo script + boundary grep |
| T19 | Sidebar: model params (model, temp, top_p, max_tokens, k) → `GraphState.params`; session manager (new/switch/rename/delete) | `src/components/sidebar.py` | Changing model/temperature affects the next answer; sessions persist across reload | manual: change param, reload, switch session |
| T20 | Debug panel: node_trace, query_plan, retrieval scores per chunk | `src/components/debug_panel.py` | Works for current and historical answers (reads `node_trace_json`) | manual: inspect two different answers' traces |
| T21 | Error surfacing: banners per error class with Retry; per-file upload errors; ungrounded markers | `src/components/errors.py` | Missing key → visible error, no blank answer; bad PDF → per-file card, other uploads proceed | manual: unset key, upload corrupt PDF |

**Checkpoint 3 (after T21):** demo script (`docs/STRESS-TEST-DELIVERY.md`) runs green locally
in both real and mock mode.

## Phase 5 — `packaging` (serial last)

| Id | Task | Files | Acceptance | Verify |
|---|---|---|---|---|
| T22 | `Dockerfile` (multi-stage, non-root, healthcheck) + `docker-compose.yml` (named volume for `data/`, env passthrough) | `Dockerfile`, `docker-compose.yml` | Cold clone → `cp .env.example .env` → `docker compose up --build` → app at :8501; volume survives rebuild with chat history intact | actual build + run + rebuild-with-history check |
| T23 | CI: `ruff` + `black --check` + `pytest -q` + docker build smoke | `.github/workflows/ci.yml` | CI green on PR; no network-dependent tests | push branch, observe CI |
| T24 | `README.md`: 4-command quick start, architecture summary, mock-mode demo, repo layout, journal pointer | `README.md` | A reviewer with Docker and no API key can run the app via `MOCK_LLM=1` | cold-clone dry run following README verbatim |

**Checkpoint 4 (after T24):** CI green; cold-clone reproduction green; docker rebuild keeps data.

## Phase 6 — Journal + submission

| Id | Task | Files | Acceptance | Verify |
|---|---|---|---|---|
| T25 | `JOURNAL.md`: decisions (ADR-linked), measured build data (index build time, latency, token counts), ≥1 documented prompt iteration that changed output, vector-vs-vectorless section, cut-list with reasons | `JOURNAL.md` | Every ADR referenced; contains real measured numbers (no invented benchmarks); AI-use disclosure section written | read-through against `docs/GRADING-ALIGNMENT.md` journal hooks |
| T26 | Submission: private repo, collaborators `crcheng`/`levymedina`/`jgcosme`, final gate run of demo script + CI | repo settings | Private repo accessible to the three reviewers; demo script green; CI green | re-run demo script + CI after last commit |

## Risks and Mitigations

| Risk | Impact | Mitigation | Plan B |
|---|---|---|---|
| Year-confusion failure (STRESS-TEST-RAG #1) | Wrong headline figure at demo | Year-aware extraction contract + golden tests pin all three years | Quick prompt fix; golden test catches regression before demo |
| Context overflow on wide queries | Crash mid-demo | Context budget packing (T11) | Auto-lower k on overflow, logged |
| Subagent-free solo build slips | Incomplete features | Kill-switch order in `docs/STRESS-TEST-DELIVERY.md` | Cut Could → Should in listed order; journal records cuts |
| API quota at grading time | Demo fails with key | `MOCK_LLM=1` path | Mock mode as default demo |
| Journal left to last minute | Weighted deliverable weak | T25 scheduled before submission with Checkpoint-2 data already recorded | Journal skeleton pre-seeded in `docs/JOURNAL.md`, fill during build |

## Open Questions

- Final chunk sizes/`tau` recorded in the journal after golden-QA tuning (T17), not assumed here.
- Whether to enable LangSmith env hooks — env-gated only, no build time.

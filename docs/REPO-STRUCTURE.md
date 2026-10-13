# Repo Structure (target state)

> The tree the build produces. Task ids from `docs/PLAN.md` create each item; every directory
> has one owning module from `docs/CAPABILITY-MAP.md`.
>
> **Shipped vs pending.** Every unmarked file below exists in the working tree today. Lines
> marked `⏳ pending` are planned by `docs/PLAN.md` (mostly Phase 4 app-ui and Phase 5
> packaging) but have not been built yet — the planned task id is named so the gap is
> unambiguous. Nothing is described as shipped that is not on disk.

```
├── app.py                          # Streamlit entrypoint — thin: UI only, no business logic.
│                                   #   Placeholder body printing a not-yet-built notice until T18;
│                                   #   `import streamlit` finally lands here (ADR-009 boundary).
├── src/
│   ├── components/                 # app-ui (T18–T21) — ⏳ pending, only __init__.py shipped
│   │   ├── chat.py                 # chat pane: message list, citation cards, ungrounded badges (T18)
│   │   ├── upload.py               # multi-PDF uploader, per-file status/error cards, progress (T18)
│   │   ├── sidebar.py              # model params (model/temp/top_p/max_tokens/k), session manager (T19)
│   │   ├── debug_panel.py          # node_trace, query_plan, retrieval scores (OBSERVABILITY.md) (T20)
│   │   └── errors.py               # error banners per typed error class + Retry action (T21)
│   ├── core/                       # doc-ingest + chunking-index (T2–T4)
│   │   ├── pdf_parser.py           # pypdf per-page text, pdfplumber grid → markdown,
│   │   │                            #   text_layer detection (ADR-004). Table serialization
│   │   │                            #   (table_to_markdown/extract_tables) shipped here.
│   │   ├── tables.py               # pdfplumber table → markdown; cross-page merge — ⏳ pending (T2);
│   │   │                            #   table work landed inside pdf_parser.py instead
│   │   ├── chunking.py             # parents/children, atomic tables, page ranges (ADR-002)
│   │   ├── index.py                # FAISS + BM25 build/load; chunks.jsonl source of truth
│   │   └── retrieval.py            # hybrid retrieve(), RRF fusion, parent expansion
│   ├── chains/                     # orchestration (T9–T16)
│   │   ├── state.py                # GraphState TypedDict (frozen in docs/ARCHITECTURE.md)
│   │   ├── schemas.py              # QueryPlan, GroundingResult JSON schemas — ⏳ pending; the nodes
│   │   │                            #   that use them declare QUERY_PLAN_SCHEMA (analyze.py) and
│   │   │                            #   GROUNDING_SCHEMA (verify.py) until this module lands
│   │   ├── graph.py                # StateGraph wiring, edges, conditional routing (ADR-001)
│   │   ├── checks.py               # symbolic predicates: gates, caps, budget, citation match, numeric check (ADR-010)
│   │   └── nodes/
│   │       ├── contextualize.py    # follow-up rewrite (P2)
│   │       ├── analyze.py          # classify + plan (P1)
│   │       ├── decompose.py        # sub-query fan-out, cap 4 (T10b)
│   │       ├── retrieve.py         # hybrid retrieval node, score gate, retry bound
│   │       ├── synthesize.py       # citation-constrained answer (P3 + P4)
│   │       ├── verify.py           # claim-level grounding (P5)
│   │       └── refuse.py           # honest refusals
│   ├── utils/                      # llm-gateway + observability (T6, T9)
│   │   ├── llm.py                  # build_llm(params), OpenAI-compatible client (ADR-005).
│   │   │                            #   Holds the typed error taxonomy (ADR-005): LLMError base
│   │   │                            #   + RateLimited/ContextLength/Auth/Timeout subclasses, plus
│   │   │                            #   SchemaViolationError — deliberately not in a separate
│   │   │                            #   errors module.
│   │   ├── mock_llm.py             # MOCK_LLM=1 deterministic scripted gateway
│   │   ├── tracing.py              # @trace_node decorator → node_trace
│   │   └── prompts.py              # load_prompt(id) from prompts/
│   └── store/                      # chat-store (T7)
│       ├── db.py                   # SQLite schema + accessors (6 tables, WAL)
│       └── sessions.py             # session lifecycle: create/switch/rename/delete
├── prompts/                        # versioned prompt assets (P1–P5, docs/prompts/*.md is the spec)
│   ├── query-analysis.md
│   ├── follow-up-contextualization.md
│   ├── answer-synthesis-with-citations.md
│   ├── table-and-number-formatting.md
│   └── summarization-and-verification.md
├── scripts/
│   ├── download_corpus.py          # install a PDF into data/pdfs/; no default URL — Microsoft
│   │                               #   serves DOCX, so the drop-in path is manual (T5)
│   ├── build_index.py              # ingest + index build (T5). Also owns env loading for the
│   │                               #   index path (MOCK_LLM, OPENAI_API_KEY, base-url/model)
│   ├── check_floor.py              # floor guard — enforces docs/CONSTRAINTS.md on added lines
│   ├── prompts_meta.json          # per-node prompt version/temperature registry
│   ├── eval_rag.py                 # golden QA runner (T17) — ships; runs invoke() over
│   │                              #   tests/data/golden_qa.jsonl, exits nonzero on regression
│   └── demo_script.py              # the 6-step demo run (STRESS-TEST-DELIVERY.md)
├── tests/                          # hermetic pytest suite (TESTING-AND-EVAL.md)
│   ├── test_parser.py      # Block/ParsedDocument parse + table markdown
│   ├── test_chunking.py   # parent/child invariants, table atomicity, page ranges
│   ├── test_retrieval.py  # RRF fusion, score gate, determinism, embedder resolution
│   ├── test_gateway.py    # build_llm, retry/backoff, typed errors, MOCK_LLM scripts
│   ├── test_store.py      # SQLite sessions, messages, citations, events
│   ├── test_prompts.py    # header parse, version assertions, node/prompt registry
│   ├── fixtures.py       # shared fixtures (no test_ prefix → not collected)
│   └── data/golden_qa.jsonl  # golden QA dataset (T17) — ships; 10 cases incl. the
│                             #   year-confusion and adjacent-year canaries
├── data/                           # gitignored runtime state (volume-mounted)
│   ├── pdfs/                       # manual PDF drop (2023_Annual_Report.pdf); download_corpus.py
│   │                               #   records digests in pdfs/manifest.json
│   ├── chat.sqlite3               # chat store (T7); path is $DATA_DIR/chat.sqlite3, created at
│   │                               #   store init — absent until the app or a test opens it
│   └── index/                      # faiss/{index.faiss,meta.json}, bm25.pkl, chunks.jsonl,
│                                   # docs.jsonl
├── docs/                           # this plan set: SPEC, ARCHITECTURE, ADRs, PLAN, JOURNAL…
├── Dockerfile                      # multi-stage, non-root, healthcheck (ADR-008) — ⏳ pending (T22)
├── docker-compose.yml              # app service + data volume + env passthrough — ⏳ pending (T22)
├── .env.example                    # every env var documented (T1)
├── .github/workflows/ci.yml        # ruff + black + pytest + docker build smoke (T23) — ⏳ pending
├── requirements.txt                # pinned (T1)
├── pyproject.toml                  # setuptools + ruff/black/pytest/mypy config (T1)
└── README.md                       # 4-command quick start, mock mode, architecture (T24) — ⏳ pending
```

## Ownership map

| Path | Module | Task |
|---|---|---|
| `src/core/pdf_parser.py`, `chunking.py`, `index.py`, `retrieval.py` | doc-ingest, chunking-index | T2–T5 |
| `src/utils/llm.py`, `mock_llm.py`, `prompts.py` | llm-gateway | T6 |
| `src/store/db.py`, `sessions.py` | chat-store | T7 |
| `prompts/*.md` | (prompt assets — spec in `docs/prompts/`) | T8 |
| `src/chains/*`, `src/utils/tracing.py` | orchestration | T9–T16 |
| `scripts/eval_rag.py`, `tests/data/golden_qa.jsonl` | (evaluation) | T17 |
| `app.py`, `src/components/*.py` | app-ui | T18–T21 |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `README.md` | packaging | T22–T24 |
| `docs/JOURNAL.md` | (submission) | T25 |

Notes on this map: it lists what each module's files *are* today. `src/utils/errors.py` and
`src/utils/settings.py` are named in `docs/PLAN.md`'s T6 row but were never created — the
error taxonomy shipped inside `src/utils/llm.py` and env loading inside
`scripts/build_index.py` / `src/utils/llm.py`.

## Rules the tree enforces

- `app.py` and `src/components/` never import retrieval, prompts, or the LLM gateway directly —
  UI calls `orchestration.invoke()` and renders `AnswerEnvelope` fields. This is what keeps the
  demo script's 6 steps testable without a browser.
- No inline system prompts in `src/chains/nodes/*.py` — `load_prompt("P3")` only
  (`docs/prompts/README.md` protocol).
- Nothing outside `data/` is written at runtime; the container's only writable path is the
  mounted volume (SECURITY-PRIVACY.md).

# Repo Structure (target state)

> The tree the build produces. Task ids from `docs/PLAN.md` create each item; every directory
> has one owning module from `docs/CAPABILITY-MAP.md`.

```
├── app.py                          # Streamlit entrypoint — thin: UI only, no business logic
├── src/
│   ├── components/                 # app-ui (T18–T21)
│   │   ├── chat.py                 # chat pane: message list, citation cards, ungrounded badges
│   │   ├── upload.py               # multi-PDF uploader, per-file status/error cards, progress
│   │   ├── sidebar.py              # model params (model/temp/top_p/max_tokens/k), session manager
│   │   ├── debug_panel.py          # node_trace, query_plan, retrieval scores (OBSERVABILITY.md)
│   │   └── errors.py               # error banners per typed error class + Retry action
│   ├── core/                       # doc-ingest + chunking-index (T2–T4)
│   │   ├── pdf_parser.py           # pypdf per-page text, text_layer detection (ADR-004)
│   │   ├── tables.py               # pdfplumber table → markdown; cross-page merge
│   │   ├── chunking.py             # parents/children, atomic tables, page ranges (ADR-002)
│   │   ├── index.py                # FAISS + BM25 build/load; chunks.jsonl source of truth
│   │   └── retrieval.py            # hybrid retrieve(), RRF fusion, parent expansion
│   ├── chains/                     # orchestration (T9–T16)
│   │   ├── state.py                # GraphState TypedDict (frozen in docs/ARCHITECTURE.md)
│   │   ├── schemas.py              # QueryPlan, GroundingResult JSON schemas
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
│   │   ├── llm.py                  # build_llm(params), OpenAI-compatible client (ADR-005)
│   │   ├── mock_llm.py             # MOCK_LLM=1 deterministic scripted gateway
│   │   ├── errors.py               # LLMError/RetrievalError/ParseError/GraphNodeError (ADR-007)
│   │   ├── tracing.py              # @trace_node decorator → node_trace
│   │   ├── logging.py              # JSON formatter, setup_logging()
│   │   ├── prompts.py              # load_prompt(id) from prompts/
│   │   └── settings.py             # env loading, validation, defaults
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
│   ├── download_corpus.py          # fetch MSFT FY2023 annual report (T5)
│   ├── build_index.py              # ingest + index build (T5)
│   └── eval_rag.py                 # golden QA runner (T17)
├── tests/                          # hermetic pytest suite (TESTING-AND-EVAL.md)
│   ├── test_parser.py  test_chunking.py  test_retrieval.py
│   ├── test_gateway.py test_store.py    test_nodes.py  test_graph.py
│   └── data/golden_qa.jsonl
├── data/                           # gitignored runtime state (volume-mounted)
│   ├── pdfs/  index/{faiss/,bm25.pkl,chunks.jsonl,docs.jsonl}  chat.sqlite3
├── docs/                           # this plan set: SPEC, ARCHITECTURE, ADRs, PLAN, JOURNAL…
├── Dockerfile                      # multi-stage, non-root, healthcheck (ADR-008)
├── docker-compose.yml              # app service + data volume + env passthrough
├── .env.example                    # every env var documented (T1)
├── .github/workflows/ci.yml        # ruff + black + pytest + docker build smoke (T23)
├── requirements.txt                # pinned (T1)
└── README.md                       # 4-command quick start, mock mode, architecture (T24)
```

## Ownership map

| Path | Module | Task |
|---|---|---|
| `src/core/*` | doc-ingest, chunking-index | T2–T5 |
| `src/utils/llm.py`, `mock_llm.py`, `errors.py`, `settings.py` | llm-gateway | T6 |
| `src/store/*` | chat-store | T7 |
| `prompts/*` | (prompt assets — spec in `docs/prompts/`) | T8 |
| `src/chains/*`, `src/utils/tracing.py` | orchestration | T9–T16 |
| `scripts/eval_rag.py`, `tests/data/` | (evaluation) | T17 |
| `app.py`, `src/components/*` | app-ui | T18–T21 |
| `Dockerfile`, `docker-compose.yml`, CI, `README.md` | packaging | T22–T24 |
| `JOURNAL.md` | (submission) | T25 |

## Rules the tree enforces

- `app.py` and `src/components/` never import retrieval, prompts, or the LLM gateway directly —
  UI calls `orchestration.invoke()` and renders `AnswerEnvelope` fields. This is what keeps the
  demo script's 6 steps testable without a browser.
- No inline system prompts in `src/chains/nodes/*.py` — `load_prompt("P3")` only
  (`docs/prompts/README.md` protocol).
- Nothing outside `data/` is written at runtime; the container's only writable path is the
  mounted volume (SECURITY-PRIVACY.md).

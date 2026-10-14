# Repo Structure (target state)

> The tree the build produces. Task ids from `docs/PLAN.md` create each item; every directory
> has one owning module from `docs/CAPABILITY-MAP.md`.
>
> **Shipped vs pending.** Every unmarked file below exists in the working tree today. Lines
> marked `⏳ pending` are planned by `docs/PLAN.md` but have not been built yet — the planned
> task id is named so the gap is unambiguous. Phase 4 (`app.py` + `src/components/`) is
> shipped; Phase 5's packaging files (T22–T24) exist on disk **and are committed** (an earlier
> version of this note called them uncommitted; `git ls-files` lists `Dockerfile`,
> `docker-compose.yml`, `README.md`, `requirements.txt`, `requirements-dev.txt`,
> `pyproject.toml`, `AGENTS.md` and `.github/workflows/ci.yml`).
> Nothing is described as shipped that is not on disk.

```
├── app.py                          # Streamlit entrypoint — thin shell, shipped (ADR-009):
│                                   #   page config (`initial_sidebar_state=260`), theme
│                                   #   injection, the four component calls, session state,
│                                   #   persistence; reaches the pipeline only through
│                                   #   `src.chains.graph.invoke()` (AnswerEnvelope).
├── src/
│   ├── components/                 # app-ui (T18–T21) — shipped (Streamlit-native, no build step)
│   │   ├── __init__.py             # package marker (empty; components are imported by name)
│   │   ├── theme.py                # design system: the token table, the one injected
│   │   │                            #   stylesheet, panel/label/figure/badge helpers
│   │   ├── chat.py                 # canvas (T18): stored transcript, live answer, citation
│   │   │                            #   cards, composer, empty-state suggestions, pending
│   │   │                            #   bubble, ungrounded/refusal markers, error banner
│   │   │                            #   with Retry (T21)
│   │   ├── upload.py               # ingest surface (T18): multi-PDF uploader, index-build
│   │   │                            #   progress, document catalog, per-file status and
│   │   │                            #   error cards (T21)
│   │   ├── debug_panel.py           # Diagnostics (T20): plan, path, retrieval, grounding for
│   │   │                            #   the selected answer (live envelope or stored message)
│   │   ├── errors.py                # failure classification + error banner (T21) — see the
│   │   │                            #   ownership notes below
│   │   └── sidebar.py              # left rail (T19): brand row, primary action, 4-view nav
│   │                               #   (Documents, Retrieval, Parameters, Diagnostics), session
│   │                               #   manager, index-sourced workspace status card; renders the
│   │                               #   Parameters view via render_model_params() → LLMParams
│   ├── core/                       # doc-ingest + chunking-index (T2–T4)
│   │   ├── pdf_parser.py           # pypdf per-page text, pdfplumber grid → markdown,
│   │   │                            #   text_layer detection (ADR-004). Table serialization
│   │   │                            #   (table_to_markdown/extract_tables) shipped here.
│   │   ├── tables.py               # pdfplumber table → markdown; cross-page merge — ⏳ pending (T2);
│   │   │                            #   table work landed inside pdf_parser.py instead
│   │   ├── chunking.py             # parents/children, atomic tables, page ranges (ADR-002)
│   │   ├── index.py                # FAISS + BM25 build/load; chunks.jsonl source of truth
│   │   ├── retrieval.py            # hybrid retrieve(), RRF fusion, parent expansion
│   │   └── local_embedder.py       # cached fastembed embedder (get_local_embedder),
│   │                               #   LocalEmbeddingError — named by no PLAN.md task row
│   ├── chains/                     # orchestration (T9–T16)
│   │   ├── state.py                # GraphState TypedDict (frozen in docs/ARCHITECTURE.md)
│   │   ├── graph.py                # StateGraph wiring, edges, conditional routing (ADR-001)
│   │   ├── checks.py               # symbolic predicates: gates, caps, budget, citation match, numeric check (ADR-010)
│   │   └── nodes/
│   │       ├── contextualize.py    # follow-up rewrite (P2)
│   │       ├── analyze.py          # classify + plan (P1); declares QUERY_PLAN_SCHEMA inline (ADR-005)
│   │       ├── decompose.py        # sub-query fan-out, cap 4 (T10b)
│   │       ├── retrieve.py         # hybrid retrieval node, score gate, retry bound
│   │       ├── synthesize.py       # citation-constrained answer (P3 + P4)
│   │       ├── verify.py           # claim-level grounding (P5); declares GROUNDING_SCHEMA inline (ADR-005)
│   │       └── refuse.py           # honest refusals
│   ├── utils/                      # llm-gateway + observability (T6, T9)
│   │   ├── llm.py                  # build_llm(params), OpenAI-compatible client (ADR-005).
│   │   │                            #   Holds the typed error taxonomy (ADR-005): LLMError base
│   │   │                            #   + RateLimited/ContextLength/Auth/Timeout subclasses, plus
│   │   │                            #   SchemaViolationError — deliberately not in a separate
│   │   │                            #   errors module.
│   │   ├── mock_llm.py             # MOCK_LLM=1 deterministic scripted gateway
│   │   ├── tracing.py              # @trace_node decorator → node_trace
│   │   ├── prompts.py              # load_prompt(id) from prompts/
│   │   └── settings.py             # load_env(): reads the repo-root .env once (python-dotenv),
│   │                               #   from REPO_ROOT; called first by app.py and both scripts
│   └── store/                      # chat-store (T7)
│       ├── db.py                   # SQLite schema + accessors (6 tables, WAL)
│       └── sessions.py             # session lifecycle: create/switch/rename/delete
├── scripts/
│   ├── download_corpus.py          # install a PDF into data/pdfs/; no default URL — Microsoft
│   │                               #   serves DOCX, so the drop-in path is manual (T5)
│   ├── build_index.py              # ingest + index build (T5). Calls load_env() first, so the
│   │                               #   repo-root .env reaches MOCK_LLM, OPENAI_API_KEY,
│   │                               #   base-url/model (src/utils/settings.py)
│   ├── check_floor.py              # floor guard — enforces docs/CONSTRAINTS.md on added lines
│   ├── prompts_meta.json           # per-node prompt version/temperature registry
│   ├── eval_rag.py                 # golden QA runner (T17) — ships; runs invoke() over
│   │                               #   tests/data/golden_qa.jsonl, exits nonzero on regression;
│   │                               #   two modes: MOCK_LLM=1 hermetic, or a keyed real LLM
│   └── demo_script.py              # the 6-step demo run (STRESS-TEST-DELIVERY.md)
├── tests/                          # hermetic pytest suite (TESTING-AND-EVAL.md)
│   ├── test_parser.py      # Block/ParsedDocument parse + table markdown
│   ├── test_chunking.py   # parent/child invariants, table atomicity, page ranges
│   ├── test_retrieval.py  # RRF fusion, score gate, determinism, embedder resolution
│   ├── test_gateway.py    # build_llm, retry/backoff, typed errors, MOCK_LLM scripts
│   ├── test_store.py      # SQLite sessions, messages, citations, events
│   ├── test_prompts.py    # header parse, version assertions, node/prompt registry
│   ├── test_settings.py   # env precedence, REPO_ROOT resolution
│   ├── test_errors.py     # failure classification / render_error
│   ├── test_nodes_analyze.py    # planner classification, year-coverage gate, degraded marker
│   ├── test_nodes_verify.py     # claim-level grounding verdicts (ADR-010)
│   ├── test_degraded_planner_marker.py  # planner_degraded end-to-end through the envelope
│   ├── test_answer_survives_store_failure.py  # answer rendered when the store cannot be written
│   ├── test_doc_summary_sections.py  # catalog DocSummary section reading
│   ├── test_eval_rag.py  # eval harness verdict/exit logic (mock mode, hermetic)
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
├── Dockerfile                      # multi-stage builder/runtime, non-root appuser, Streamlit
│                                   #   healthcheck (ADR-008) (T22)
├── docker-compose.yml              # app service + named data volume + env passthrough (T22)
├── .env.example                    # every env var documented (T1)
├── .github/workflows/ci.yml        # one job: ruff + black + mypy + floor guard + pytest with
│                                   #   coverage + docker build smoke (T23)
├── requirements.txt                # pinned (T1)
├── requirements-dev.txt            # dev toolchain (ruff/black/mypy/pytest); installed in the
│                                   #   Dockerfile builder stage and in CI, never the runtime image
├── pyproject.toml                  # setuptools + ruff/black/pytest/mypy config (T1)
├── AGENTS.md                       # agent-facing repo rules; named by no PLAN.md task row
└── README.md                       # reviewer quick start, corpus, repo layout, commands,
                                    #   honest limitations (T24)
```

## Ownership map

| Path | Module | Task |
|---|---|---|
| `src/core/pdf_parser.py`, `chunking.py`, `index.py`, `retrieval.py`, `local_embedder.py` | doc-ingest, chunking-index | T2–T5 |
| `src/utils/llm.py`, `mock_llm.py`, `prompts.py`, `settings.py` | llm-gateway | T6 |
| `src/store/db.py`, `sessions.py` | chat-store | T7 |
| `prompts/*.md` | (prompt assets — spec in `docs/prompts/`) | T8 |
| `src/chains/*`, `src/utils/tracing.py` | orchestration | T9–T16 |
| `scripts/eval_rag.py`, `tests/data/golden_qa.jsonl` | (evaluation) | T17 |
| `app.py`, `src/components/*.py` | app-ui | T18–T21 |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `README.md` | packaging | T22–T24 |
| `docs/JOURNAL.md` | (submission) | T25 |

Notes on this map: it lists what each module's files *are* today. `src/utils/settings.py`
(named in `docs/PLAN.md`'s T6 row) now exists: `load_env()` is the one place the repo-root
`.env` is read, called first in `app.py::main()` and in `scripts/build_index.py`'s and
`scripts/eval_rag.py`'s `main()`. `src/utils/errors.py` is named in the same T6 row but was
never created — the error taxonomy shipped inside `src/utils/llm.py`.

**Two corrections to an earlier version of this note, which said two Phase-4 files were never
created.** Both now exist, and both are wired; the bullets note the one limit that remains in each:

- `src/components/debug_panel.py` (T20) exists and **is rendered for the selected answer**:
  the app's **Diagnostics** view calls `render_debug_panel(envelope=…)` for the live answer
  (the only shape carrying the plan and retrieval scores) and `render_debug_panel(message=…)`
  for a stored one, chosen with a picker that leads with `Current answer (live)` and then
  lists the session's stored answers labelled by their questions (a lone candidate renders
  with no control). A stored answer renders its persisted node trace and grounding record;
  the plan and retrieval sections say they were not recorded, because the store keeps
  neither. The same work closed a stale-live bug: `_LAST_SESSION_KEY` gates the live
  envelope to its own session, so a conversation switch can no longer show one
  conversation's live panel under another's name.
- `src/components/errors.py` (T21) also exists (419 lines) and **is wired**: `app.py` imports
  it and routes both catch-alls that render a banner through it — a failed transcript read
  calls `errors.render_error(exc)`, and a failed turn calls `errors.classify_error(exc)` then
  `errors.render_error(exc, on_retry=view.retryable)` — so the banner and its Retry action
  come from the failure class (`Auth` names `OPENAI_API_KEY` / `MOCK_LLM=1`;
  `chat.render_answer_error` remains a public component API that `app.py` no longer calls).
  The T21 surface that reviewers actually see is split: failed turns render here, while
  `chat.py`'s ungrounded and refusal markers and `upload.py`'s per-file status and error
  cards keep their own rendering. The parser's branch is the one still untaken —
  `classify_error` has seen no `ParseError` from a call site, because upload failures render
  through `upload._failed` — recorded here rather than implied complete.

`src/components/theme.py`, shipped with the T18–T21 UI, is not named in `docs/PLAN.md` at all.

## Rules the tree enforces

- `app.py` and `src/components/` render what the pipeline already decided: the pipeline is
  reached through `src.chains.graph.invoke()`, and the object that crosses back into the UI is
  the frozen `AnswerEnvelope`. No UI module calls a retrieval, prompt, or gateway *function*.
  The two documented exceptions: `upload.py` drives the ingest modules for T18's "upload →
  index build progress" step (it imports `scripts/build_index.py` lazily for the catalog format
  and embedder policy rather than reimplementing them), and `app.py` sets
  `src.chains.nodes.retrieve.RETRIEVE_K` for the duration of one call instead of changing the
  frozen graph signature — that is how the `k` slider reaches the retriever. This is what keeps
  the demo script's 6 steps testable without a browser.
- No inline system prompts in `src/chains/nodes/*.py` — `load_prompt("P3")` only
  (`docs/prompts/README.md` protocol).
- Nothing outside `data/` is written at runtime; the container's only writable path is the
  mounted volume (SECURITY-PRIVACY.md).

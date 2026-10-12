# Task List

> Flat checklist. Task ids match `docs/PLAN.md`. Check off only after the task's verification
> ran — never from inspection.

## Phase 1 — Foundation: `doc-ingest` + `chunking-index`

- [x] T1. Repo scaffold: `src/` tree, pinned `requirements.txt`, `.env.example`, `.gitignore` (`data/`, `*.pdf`, `.env`), pytest/ruff/black config — **merged via PR #1** (`6f66663`). Verified: `pip check` clean, all 14 pins installed at pinned versions, all 14 imports work, faiss 1.15.1 runs a real `IndexFlatIP` search, `pytest -q` exit 0, ruff + black clean, floor guard exit 0 and mutation-tested. Python 3.11 pinned at three levels (`.python-version`, `requires-python ==3.11.*`, `python:3.11-slim` in the Dockerfile spec).
- [x] T2. `doc-ingest`: pypdf text, pdfplumber markdown tables, cleaning rules, section detection, `Block`/`ParsedDocument`, sha256 idempotency, `text_layer: false` warnings — **merged via PR #3** (`fa9f9cf`). Verified: 25 pytest passed; ruff + black + mypy clean; floor guard exit 0. APIs checked against official docs (pypdf `PdfReader`/`extract_text`, pdfplumber `extract_tables`/`.page_number` 1-based). Suite mutation-tested — the run exposed a real hole (page-number stripping unasserted, regex missing `re.MULTILINE`), now fixed and guarded.
- [x] T3. `chunking-index`: heading-aligned parents (~1000 tok), children (~250/50), atomic tables, page ranges — **merged via PR #6** (`96362c6`). Verified on merged main: 47 pytest passed; ruff/black/mypy clean; floor guard exit 0. Mutation-tested (breaking row-boundary atomicity and inverting `page_end` each turn the suite red). Inspecting real parsed output exposed two T2 defects — fixture header drawn outside pdfplumber's ruled grid (first data row promoted to header) and a phantom all-empty row serialized as em-dashes — both fixed. Token counting is offline (`len//4`, not tiktoken, whose vocab downloads at runtime); `langchain-text-splitters==1.1.3` pinned as a **direct** dep since langgraph doesn't declare it.
- [x] T4. Index build + hybrid retrieval: FAISS children, BM25 pickle, `chunks.jsonl` truth, RRF k=60, parent expansion, `retrieve(query, k, filters)` — **merged via PR #8** (`9479cb1`). Verified on merged main: 96 pytest passed; ruff/black/mypy clean; floor guard exit 0; `pip check` clean. Inspected real output: the revenue question returns the income-statement parent at fused rank 0 with `$ 211,915` and page 2. `IndexFlatIP` cosine-equivalence and reload reproducibility verified against brute force; `BM25Okapi([])` `ZeroDivisionError` and `add_with_ids` `RuntimeError` reproduced and handled. Mutation-tested: caught `RRF_K` breakage immediately and closed a real gap — expansion-before-fusion survived 48 tests because the fixture had one child per parent, so a discriminating test now asserts siblings get distinct dense scores.
- [x] T5. `scripts/download_corpus.py`, `scripts/build_index.py`, `docs.jsonl` catalog — **merged via PR #10** (`b1d7d7b`). Verified on merged main: 96 pytest passed; ruff/black/mypy clean; floor guard exit 0 (19 files); `pip check` clean. Ran on the user-supplied 85-page corpus: 81 parents / 383 children, second run "no changes (sha256 match)" with byte-identical artifacts, `docs.jsonl` carries exactly the five frozen fields. Scripts are URL-independent because **no official FY2023 PDF is reachable** — Microsoft's `ar23/download-center` serves DOCX for both the Annual Report and the 10-K (`PK\x03\x04`, `pypdf` rejects it), the ar23 page has no `.pdf` link, and SEC EDGAR's 10-K (`0000950170-23-035122`) is HTML-only. `download_corpus.py` therefore has no default URL and a `%PDF` guard that was verified live against the real Microsoft DOCX (rejected, exit 1, nothing installed). Heading gate tightened after measurement: word count rejected (breaks the fixture's one-word `CONTINGENCIES`), recurrence rejected (only 2 all-caps lines repeat), length floor shipped — residual noise documented in `docs/JOURNAL.md` §9. Repo installed editable so scripts import `src` with zero `# noqa` suppressions.

**Checkpoint 1:** cold-clone index build; 10-query retrieval probe sane; `pytest -q` green; ruff clean.

## Phase 2 — `llm-gateway` + `chat-store` (parallel)

- [ ] T6. `llm-gateway`: `build_llm(params)`, OpenAI-compatible client, JSON-schema outputs, retry/backoff, typed errors, `MOCK_LLM=1` scripted mode, token/latency accounting
- [ ] T7. `chat-store`: 6-table SQLite schema, accessors, session lifecycle, WAL
- [ ] T8. Prompt assets in `prompts/` with version headers (5 prompts)

## Phase 3 — `orchestration` (serial, highest risk)

- [ ] T9. `GraphState`, node skeletons with `@trace_node`, error mapping into `node_trace`
- [ ] T10. `analyze_question`: classification, `required_sections`, `missing_aspect`, JSON schema
- [ ] T10b. `decompose_question`: sub-query fan-out (cap 4), retrieval pass per sub-query
- [ ] T11. `hybrid_retrieve` node: routing, score gate `tau`, one rewrite retry, context budget packing
- [ ] T12. `synthesize_answer`: citation-constrained synthesis, year-aware extraction contract
- [ ] T13. `groundedness_check`: per-claim span verification, trim-and-flag
- [ ] T14. `contextualize_question`: 6-turn history + `cited_pages`, graceful degradation
- [ ] T15. `refuse`: honest refusals, coverage-naming copy
- [ ] T16. Graph wiring: edges + conditional routing, `recursion_limit`, params flow
- [ ] T17. Golden QA harness `scripts/eval_rag.py` + dataset

**Checkpoint 2:** `docs/SPEC.md` §10 criteria pass end-to-end (mock minimum); golden baseline recorded; `pytest -q` green.

## Phase 4 — `app-ui`

- [ ] T18. `app.py` shell: chat pane, multi-PDF upload, empty state
- [ ] T19. Sidebar: model params, session manager
- [ ] T20. Debug panel: node_trace, query_plan, retrieval scores (current + historical)
- [ ] T21. Error surfacing: banners per error class, per-file upload errors, ungrounded markers

**Checkpoint 3:** demo script green locally, real + mock modes.

## Phase 5 — `packaging`

- [ ] T22. `Dockerfile` (multi-stage, non-root, healthcheck) + `docker-compose.yml` (data volume, env passthrough)
- [ ] T23. CI: ruff + black + pytest + docker build smoke
- [ ] T24. `README.md`: 4-command quick start, mock mode, architecture, journal pointer

**Checkpoint 4:** CI green; cold-clone reproduction green; rebuild keeps data.

## Phase 6 — Journal + submission

- [ ] T25. `JOURNAL.md`: ADR-linked decisions, measured build data, prompt iteration, vector-vs-vectorless section, cut-list, AI-use disclosure
- [ ] T26. Submission: private repo, three collaborators, final demo-script + CI gate

**Checkpoint 5 (final):** every Must item in `docs/SPEC.md` §8 demonstrably green; demo script passes; CI green; journal complete with measured data.

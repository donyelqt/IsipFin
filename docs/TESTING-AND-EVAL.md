# Testing & Evaluation

> Test strategy for the build. Two layers: **correctness tests** (fast, hermetic, no network)
> and the **golden QA eval** (the honest quality gate that also produces the journal's numbers).

## Principles

- Every test runs offline: nodes are tested against `MockLLM` (`MOCK_LLM=1` mode,
  `src/utils/mock_llm.py`) or stub transports; embeddings in unit tests come from a
  deterministic fake. CI proves this on every push.
- Tests target **behavior and invariants**, never wiring or text copies.
- The golden QA set is the only test allowed to be slow and network-dependent; it runs as a
  script (`scripts/eval_rag.py`), not in the pytest suite.

## Layer 1 — Unit tests (hermetic, `pytest -q`)

| File | Target | Invariants asserted |
|---|---|---|
| `tests/test_parser.py` | `doc-ingest` | page counts; table blocks detected (≥1 per known table page); header/footer boilerplate absent; `text_layer: false` flagged on image pages; `page` 1-based |
| `tests/test_chunking.py` | `chunking-index` | every child has a parent; **no parent cut inside a table block**; overlap ≈50 tokens; `page_end >= page_start`; total content within slack |
| `tests/test_retrieval.py` | `chunking-index` | exact-token query ("Note 13") retrieves via BM25; "revenue FY2023" income-statement parent at rank 1 in fused list; index reload from disk reproduces results; `filters={"section": …}` restricts correctly |
| `tests/test_gateway.py` | `llm-gateway` | mock transport returns usage + latency; 429 → retry with backoff then typed `LLMError`; schema-violation path retries once then raises; `MOCK_LLM` revenue answer is exact and deterministic |
| `tests/test_store.py` | `chat-store` | session create/switch/rename/delete cascades to turns/messages/citations; citations survive reload byte-identical; `events` rows written |
| `tests/test_checks.py` | `src/chains/checks.py` (symbolic layer, ADR-010) | numeric pre-pass rejects a figure absent from its cited span and a period-label mismatch (FY2024 `$245,122M` claimed as FY2023); citation marker↔block correspondence rejects orphans; follow-up heuristic fires only on real pronouns/ellipsis; tau gate and single retry bound; context budget packs greedily to cap and drops the rest with a note |
| `tests/test_nodes_verify.py` | `groundedness_check` (ADR-010's verifier) | **verdicts resolved by the `claim_id` P5 sends, never by list position** — an off-by-one or reordered id must not re-point a verdict onto another claim, a duplicated id keeps the pessimistic verdict, an id naming no handed-over claim is ignored, and `all_supported: true` never rescues a claim whose own verdict is `unsupported` (STRESS-TEST-RAG #2); **the P5 payload carries only the blocks the claims it judges cite** — uncited blocks omitted, the cited union sent unrenumbered, and every block sent when no claim cites anything |
| `tests/test_nodes.py` | each graph node | classification fixture 12/12 (`analyze_question`); fan-out ≤4, never splits a comparison into operands (`decompose_question`); single retry (`hybrid_retrieve`); **pre-pass trims wrong-figure claims before the LLM verifier runs** (`groundedness_check`); trim-and-flag on mixed claims; standalone rewrite on follow-up pairs (`contextualize_question`); both refusal classes (`refuse`); node failure → typed error, never empty answer (all nodes) |
| `tests/test_graph.py` | `orchestration` | routing table edges per `docs/adr/ADR-001`; `recursion_limit` set; `AnswerEnvelope` complete incl. `node_trace` on all 6 demo-script questions (mock mode) |

## Layer 2 — Golden QA eval (`scripts/eval_rag.py`)

Runs N golden questions against the MSFT FY2023 corpus with the **real** LLM, reports:

| Metric | Definition | Pass bar |
|---|---|---|
| Retrieval hit-rate@5 | the chunk containing the printed answer is in fused top-5 | ≥ 0.9 |
| Citation presence | every factual answer carries ≥1 valid marker | 1.0 |
| Figure match | the expected figure appears verbatim | ≥ 0.95 |
| Wrong-year contamination | an adjacent-year figure appears where the asked year's figure was expected | 0 (hard fail) |
| Refusal correctness | out-of-scope questions refused; in-scope questions answered | 1.0 / ≥ 0.95 |

### Seed golden set (`tests/data/golden_qa.jsonl`)

| # | Question | Expected | Notes |
|---|---|---|---|
| 1 | "What was Microsoft's revenue in FY2023?" | $211,915 million | income statement page; **year-confusion canary** |
| 2 | "What was operating income in fiscal year 2023?" | $88,523 million | vs FY2022 $83,383 million |
| 3 | "Compare revenue and operating income between FY2023 and FY2022." | both pairs, cited | decomposition path |
| 4 | "What was net income and diluted EPS in FY2023?" | $72,361 million / $9.68 | two facts, one row |
| 5 | "Which segment had the highest revenue in FY2023?" | segment named + figure, MD&A citation | requires scope matching, not narrative Leakage |
| 6 | "Summarize FY2023 key results." | structured summary, all cited | summarization mode |
| 7 | "And the margin trend?" (after Q1) | operating-margin trend, resolves "that" | follow-up path |
| 8 | "Which segment drove that?" (after Q7) | correct segment from prior context | pronoun-drift canary |
| 9 | "What was revenue in FY2019?" | honest refusal, `missing_aspect=time_period` | refuse path |
| 10 | "What was Amazon's revenue in 2023?" | honest refusal, `missing_aspect=entity` | refuse path |

Golden values are quoted from Microsoft's published FY2023 annual report (revenue
$211,915M, operating income $88,523M, net income $72,361M, diluted EPS $9.68; FY2022 operating
income $83,383M). If the corpus disagrees with a golden value, **the corpus wins** — that
discrepancy is itself a finding for the journal (wrong chunk captured the wrong table).

## UI smoke (manual, documented in runbook)

1. `streamlit run app.py` (or compose) → empty state renders.
2. Upload corpus → progress + doc card; index build latency visible in debug panel.
3. Q1 → $211,915M + citation card + node trace.
4. Q7 after Q1 → resolves follow-up.
5. Q9 → refusal with coverage naming.
6. Reload browser → session + citations intact.
7. Unset key → visible error banner, no blank answer.

## Commands

```bash
pytest -q                          # full hermetic suite (must never need network)
pytest -q tests/test_retrieval.py  # focused
python scripts/eval_rag.py         # golden QA against real LLM (needs OPENAI_API_KEY)
ruff check src tests scripts app.py && black --check src tests scripts app.py
```

## Anti-patterns this suite refuses

- No test asserts a prompt's literal wording or a node's internal call order.
- No test passes because "it didn't throw"; each failure-mode assertion names the observable
  wrong behavior it prevents.
- No snapshot tests of chunk texts (they change with tuning and teach nothing).

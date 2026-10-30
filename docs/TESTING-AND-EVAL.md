# Testing & Evaluation

> Test strategy. Two layers: **unit tests** (fast, hermetic, no network) and the
> **golden QA eval** (the honest quality gate that also produces the journal's numbers).

## Principles

- Every test runs offline: nodes are tested against `MockLLM` (`MOCK_LLM=1`,
  `src/utils/mock_llm.py`) or stub transports; embeddings in unit tests come from a
  deterministic fake. CI proves this on every push.
- Tests target **behavior and invariants**, never wiring or text copies.
- The golden QA set is the only test allowed to be slow and network-dependent. It has **two
  modes**, one harness (`scripts/eval_rag.py`): `MOCK_LLM=1` runs hermetic and keyless
  (deterministic mock + sha256 embedder), and a keyed run uses the real LLM. It is a script rather
  than part of pytest, but the verdict/exit logic is pinned in `tests/test_eval_rag.py`, so each
  metric's pass bar — and that the harness exits nonzero on regression — is a hermetic assertion.

## Layer 1 — Unit tests (hermetic, `pytest -q`)

| File | Target | Invariants asserted |
|---|---|---|
| `tests/test_parser.py` | `doc-ingest` | page counts; table blocks detected (≥1 per known table page); header/footer boilerplate absent; `text_layer: false` on image pages; `page` 1-based |
| `tests/test_chunking.py` | `chunking-index` | every child has a parent; **no parent cut inside a table block**; overlap ≈50 tokens; `page_end >= page_start`; total content within slack |
| `tests/test_retrieval.py` | `chunking-index` | exact-token query ("Note 13") retrieves via BM25; "revenue FY2023" income-statement parent at rank 1 in fused list; reload from disk reproduces; `filters={"section": …}` restricts correctly |
| `tests/test_gateway.py` | `llm-gateway` | mock transport returns usage + latency; 429 → retry with backoff then typed `LLMError`; schema-violation path retries once then raises; `MOCK_LLM` revenue answer exact + deterministic |
| `tests/test_store.py` | `chat-store` | session create/switch/rename/delete cascades to turns/messages/citations; citations survive reload byte-identical; `events` rows written |
| `tests/test_checks.py` | `src/chains/checks.py` (symbolic, ADR-010) | numeric pre-pass rejects a figure absent from its span and a period mismatch (FY2024 `$245,122M` as FY2023); citation marker↔block correspondence rejects orphans; follow-up heuristic fires only on real pronouns; tau gate + single retry bound; budget packs greedily to cap then drops the rest with a note |
| `tests/test_nodes_verify.py` | `groundedness_check` (ADR-010) | **verdicts resolved by `claim_id`, never list position** — off-by-one/reorder must not re-point a verdict; a duplicated id keeps the pessimistic verdict; an id naming no handed claim is ignored; `all_supported` never rescues a `unsupported` claim (STRESS-TEST-RAG #2); **the P5 payload carries only blocks cited by its claims** — uncited blocks omitted, the cited union sent unrenumbered, all blocks sent when none cited |
| `tests/test_nodes.py` | each graph node | 12/12 classification fixture (`analyze_question`); fan-out ≤4, never splits a comparison into operands (`decompose_question`); single retry (`hybrid_retrieve`); **pre-pass trims wrong-figure claims before the LLM verifier**; trim-and-flag on mixed claims; standalone rewrite on follow-ups; both refusal classes; node failure → typed error, never empty answer |
| `tests/test_graph.py` | `orchestration` | routing table edges per `docs/adr/ADR-001`; `recursion_limit` set; complete `AnswerEnvelope` incl. `node_trace` on all 6 demo-script questions (mock mode) |

## Layer 2 — Golden QA eval (`scripts/eval_rag.py`)

Runs the golden questions against the MSFT FY2023 corpus in **either** mode — `MOCK_LLM=1`
(keyless, deterministic, CI path) or a real keyed LLM — and reports the same table in both.
Each mode limits what a metric can mean; the harness prints `NA` with its reason rather than
a number it cannot decide. No bar is lowered in either mode; an `NA` cell is "no verdict", never
a pass.

| Metric | Definition | Pass bar |
|---|---|---|
| Retrieval hit-rate@5 | chunk containing the printed answer is in fused top-5 | ≥ 0.9 |
| Citation presence | every factual answer carries ≥1 valid marker | 1.0 |
| Figure match | expected figure appears verbatim | ≥ 0.95 |
| Wrong-year contamination | adjacent-year figure where the asked year's was expected | 0 (hard fail) |
| Refusal correctness | out-of-scope refused; in-scope answered | 1.0 / ≥ 0.95 |

### What each mode can measure

| Metric | `MOCK_LLM=1` (keyless) | Real LLM (keyed) |
|---|---|---|
| Retrieval hit-rate@5 | **N/A** — mock embedder hashes each text, so dense vector is noise; fused top-5 is BM25 ordering with noise on top | fully measurable |
| Citation presence | **N/A** — mock cites only blocks containing its figure | fully measurable |
| Figure match | **N/A** — mock answers from scripted phrases + retrieved contexts | fully measurable |
| Wrong-year contamination | **fully measurable** — reads the answer's figures + layout (sentence, table cell vs column header), so lexical and mode-independent; can FAIL under `MOCK_LLM=1` | fully measurable |
| Refusal correctness | **partially measurable** — mock planner refuses only on a fiscal period the corpus lacks, so Q9 (`time_period`) is scoreable; Q10 (`entity`) is not | fully measurable |

Every `NA` cell prints its observed value labelled as an observation, never as a verdict, so
"this mode cannot answer it" is never read as "it failed".

### Seed golden set (`tests/data/golden_qa.jsonl`)

| # | Question | Expected | Notes |
|---|---|---|---|
| 1 | "What was Microsoft's revenue in FY2023?" | $211,915 million | income statement page; **year-confusion canary** |
| 2 | "What was operating income in fiscal year 2023?" | $88,523 million | vs FY2022 $83,383 million |
| 3 | "Compare revenue and operating income between FY2023 and FY2022." | both pairs, cited | decomposition path |
| 4 | "What was net income and diluted EPS in FY2023?" | $72,361 million / $9.68 | two facts, one row |
| 5 | "Which segment had the highest revenue in FY2023?" | segment named + figure, MD&A citation | requires scope matching, not narrative leakage |
| 6 | "Summarize FY2023 key results." | structured summary, all cited | summarization mode |
| 7 | "And the margin trend?" (after Q1) | operating-margin trend, resolves "that" | follow-up path |
| 8 | "Which segment drove that?" (after Q7) | correct segment from prior context | pronoun-drift canary |
| 9 | "What was revenue in FY2019?" | honest refusal, `missing_aspect=time_period` | refuse path |
| 10 | "What was Amazon's revenue in 2023?" | honest refusal, `missing_aspect=entity` | refuse path |

Golden values are quoted from Microsoft's published FY2023 report (revenue $211,915M, operating
income $88,523M, net income $72,361M, diluted EPS $9.68; FY2022 operating income $83,383M). If the
corpus disagrees with a golden value, **the corpus wins** — that discrepancy is itself a
finding for the journal.

## UI smoke (manual, documented in runbook)

1. `streamlit run app.py` (or compose) → empty state renders.
2. Upload corpus → progress + doc card; index-build latency visible in debug panel.
3. Q1 → $211,915M + citation card + node trace.
4. Q7 after Q1 → resolves follow-up.
5. Q9 → refusal with coverage naming.
6. Reload browser → session + citations intact.
7. Unset key → visible error banner, no blank answer.

## Commands

```bash
pytest -q                          # full hermetic suite (never needs network)
pytest -q tests/test_retrieval.py  # focused
MOCK_LLM=1 python scripts/eval_rag.py  # keyless golden run (hermetic, deterministic)
python scripts/eval_rag.py         # golden QA against real LLM (needs OPENAI_API_KEY)
ruff check src tests scripts app.py && black --check src tests scripts app.py
```

## Anti-patterns this suite refuses

- No test asserts a prompt's literal wording or a node's internal call order.
- No test passes because "it didn't throw"; each failure-mode assertion names the observable
  wrong behavior it prevents.
- No snapshot tests of chunk texts (they change with tuning and teach nothing).

# Stress Test: RAG failure modes for financial QA

> Adversarial review of the design in `docs/ARCHITECTURE.md`. Every row is a way the pipeline
> produces a **wrong, uncited, or misleading** answer on the Microsoft FY2023 corpus. Ranked by
> probability × severity. Mitigations use design vocabulary and land in `docs/PLAN.md` tasks.

## Ranked failure modes

| # | Failure mode | Trigger (real question) | Why the design produces it | Severity | Mitigation (concrete change) |
|---|---|---|---|---|---|
| 1 | **Multi-year figure confusion** | "What was revenue in 2023?" | FY2023 report's MD&A tables contain FY2023 ($211,915M) alongside FY2022 ($198,270M) and comparative columns; FY2024 report also contains a FY2023 comparative ($219,790M pro forma). A parent chunk spanning an income-statement table returns all three years; the model picks the wrong column | **High** | Three layers, in order: synthesis prompt's "line-item extraction contract" (name the column header before quoting); **deterministic numeric pre-pass** in `src/chains/checks.py` rejects any claim whose figure is absent from its cited span or whose period label contradicts the claim — zero LLM calls (ADR-010); golden set pins all three years. Expected-value tests assert $211,915M an… |
| 2 | **Groundedness check passes a plausible wrong number** | "What was operating income in FY2023?" → "$88,523M" vs "$85,993M" (both appear in the report; one is FY2023, one is a segment/quarter) | `groundedness_check` verifies a claim *appears in* retrieved text — not that it answers the question asked. Substring presence ≠ correctness | **High** | Strengthen the schema: `GroundingResult` per claim gets `supports_claim: bool`, `claim_id`, and `matched_span`; claims pass only when the span contains **both** the metric name and the figure. Failing claims trimmed with a "(unsupported claim removed)" marker; the prompt instructs "a number is supported only if the same span contains the metric label from the question… |
| 3 | **Follow-up pronoun drift** | "What drove that increase?" | `contextualize_question` sees only last-6-turns **content**, not the prior answer's cited figures/pages; "that increase" resolves to the wrong entity when the prior turn compared two segments | **High** | Extend `Turn` with `cited_pages`/`entities`; pass the previous `AnswerEnvelope.citations` (doc, page, section) into the contextualization prompt as disambiguation context. Add golden follow-up pairs ("revenue → operating income?" … "which segment drove that?") |
| 4 | **Table fragmentation** | "Show revenue by segment for FY2023" | If a child chunk lands mid-table, the child retrieves well but the parent (~1000 tokens) may cut the table in half if parent boundaries are heading-based and the table is longer than 1000 tokens | **High** | Parent rule: parents are *never* cut inside a table block — a table always lives in one parent (tables are atomic from `doc-ingest`). If a table exceeds ~1200 tokens, split at a row boundary and carry the header row into the continuation chunk. Invariant tested in the chunker unit tests |
| 5 | **Citation points at the wrong page** | "according to page 45" but answer came from page 44's section spanning two pages | `Block.page` is the page where the parent chunk *starts*; multi-page tables/sections lose their range | **Med** | Store `page_start`/`page_end` on parents; citations render "pp. 44–45" and the UI snippet shows the exact sub-span. `chunking.py` unit test asserts `page_end >= page_start` and merged cross-page tables keep the full range |
| 6 | **Decomposition fan-out cost blowup** | "Compare revenue, gross margin, operating income, and net income across FY23 and FY22, and explain drivers" — quietly 6–8 facts | Fan-out capped at 4 sub-queries but each triggers retrieval + the merge prompt ships all parents; on a long report this overflows context or rate limit | **Med** | Cap at 4 sub-queries (already), add a **global context budget** in `GraphState`: parents packed greedily by fused score until a token budget (default ~6k) is hit, remainder dropped with a note in `retrieval_debug`. `synthesize_answer` also gets a "context may be truncated" instruction |
| 7 | **Out-of-scope classification whiplash** | "What was revenue in 2019?" → graph says `out_of_scope` (correct) but for the wrong reason (thinks it's off-topic, not year-missing) | `analyze_question` plans against `doc_catalog` section names, not temporal coverage; a year it can't see looks like a topic it can't see | **Med** | `QueryPlan` gains `missing_aspect: "time_period" | "topic" | null`; `refuse` renders different copy: "The documents cover fiscal years 2021–2023" (time) vs "These documents cover Microsoft's financials, not competitors" (topic). Both honest refusals; only the first is a period issue |
| 8 | **BM25 keyword hijack on rare tokens** | "What does the report say about Copilot revenue?" — Copilot wasn't a revenue line in FY2023 | BM25 boosts chunks containing "Copilot" (shareholder letter emphasis); vectors pull business-overview text; synthesis assembles a confident-sounding revenue claim from narrative | **Med** | Citation contract: narrative-only answers for revenue questions must cite the MD&A/financial statements section, else downgraded to a qualitative summary with an explicit "no revenue figure is disclosed" line. Enforced in the synthesis prompt and tested with a golden "cannot answer" case |

## What would make a reviewer distrust this system

1. **"What was revenue in 2023?" answered with $245,122M** (FY2024's figure, in the FY2024 comparative) — instant loss of trust. Defense: failure mode 1's extraction contract + the golden test asserting $211,915M.
2. **An answer with zero citations.** Any uncited claim undermines the citation story. Defense:
   citation schema requires ≥1 supporting chunk per factual sentence; UI renders an "ungrounded" badge when `grounded == False`.
3. **A hallucinated figure that reads plausible** (e.g., "operating margin was 41.8%"). Defense:
   groundedness check requires the span to contain metric + figure; unverified numbers are removed, not softened.
4. **Refusing a question the report clearly answers** (e.g., "What was net income?"). Defense:
   `missing_aspect` classification + a golden "should answer" set that fails CI when refusals regress.
5. **A 15-second answer with no explanation of the wait.** Defense: debug panel shows the node
   trace live (decomposition + retrieval + synthesis timings), so the wait is visibly structured
   work; `streaming` stays off for verifiability (ADR-005).

## Residual risks accepted

| Risk | Why accepted | Detection signal |
|---|---|---|
| Scanned pages (future image-only upload) yield poor text | Reporting corpus has a clean text layer; generic uploads warn | `text_layer: false` count in `data/index/docs.jsonl` + upload warning card |
| RRF is untuned against a cross-encoder reranker | Reranker deferred (ADR-002); golden QA is the tuning signal | Golden-set hit-rate regression in `scripts/eval_rag.py` |
| Embedding provider changes shift retrieval distributions | Gateway makes swaps env-only; index rebuild is one command | Retrieval score distributions in debug panel shifting cluster-wide |
| 6-turn history window loses long-range context | Beyond 6 turns, the topic is effectively new; full history risks drift | Refusal rate on follow-ups > threshold in dev sessions |
| Parent expansion inflates tokens on wide queries | Context budget (failure mode 6) caps it | tokens_in per node in `node_trace` |

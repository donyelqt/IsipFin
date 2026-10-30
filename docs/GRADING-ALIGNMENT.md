# Grading Alignment

> Every brief bullet mapped to a design element, demo-time evidence, and gap risk.
> Source of truth for design vocabulary: `docs/CAPABILITY-MAP.md`, `docs/ARCHITECTURE.md`,
> `docs/SPEC.md`.

## Requirements coverage matrix

| # | Requirement (from brief) | Design element | Evidence at demo | Gap risk |
|---|---|---|---|---|
| 1 | Streamlit chat app — upload a PDF | `app-ui`: `app.py` + `src/components/upload.py`, multi-file `st.file_uploader`, per-file status card | Grader drops the annual report; card shows pages parsed, sections found, index build time | Low |
| 2 | Ask questions in natural language | `app-ui` chat pane (`st.chat_message`), `orchestration.invoke` | Free-text questions in the chat pane, into the graph | Low |
| 3 | Accurate and concise LLM answers | `synthesize_answer` node, citation-constrained prompt `docs/prompts/answer-synthesis-with-citations.md`, temperature 0.2 | "$211,915 million" with page citation; answers 2–6 sentences | Med — STRESS-TEST-RAG #1/#2; golden QA gates it |
| 4 | Decompose complex questions | `analyze_question` + `decompose_question` nodes; `QueryPlan.kind == multi_part`; fan-out ≤ 4 (T10, T10b) | "Compare FY23 vs FY22 operating income and drivers" → debug panel shows sub-queries; both figures cited | Low |
| 5 | Plan retrieval and synthesis dynamically | `QueryPlan` (kind, sub_queries, required_sections) drives routing + retrieval filters | Debug panel shows the plan before retrieval; different shapes take different graph paths | Low |
| 6 | Manage conversation context and follow-ups | `contextualize_question` + `chat-store` 6-turn window + `cited_pages` disambiguation | "and the margin trend?" answers correctly after a revenue question | Med — STRESS-TEST-RAG #3; golden follow-up pairs gate it |
| 7 | Accurate answers with proper citations | `AnswerEnvelope.citations` → `citations` table; citation cards render doc/page/section/snippet | `[1]` cards under every answer; click shows page + snippet | Low |
| 8 | Integrate LangChain/LangGraph for agentic workflows | LangGraph `StateGraph`; 7 nodes with conditional routing (`docs/adr/ADR-001`) | Debug panel shows node trace; graph routing visible in code + mermaid | Low |
| 9 | Document retrieval using a vector database | FAISS dense `data/index/faiss/` + BM25 sparse + RRF fusion (`docs/adr/ADR-002`) | Retrieval scores in debug panel; vectors persist across restarts | Low |
| 10 | Document processing pipelines (extraction, splitting, cleaning) | `doc-ingest`: pypdf + pdfplumber markdown tables + cleaning rules + section paths (`docs/adr/ADR-004`) | Upload card: pages, tables detected, cleaning applied; tables render as markdown | Low |
| 11 | Prompt engineering: financial QA | `prompts/query-analysis.md` + `prompts/answer-synthesis-with-citations.md` — year-aware extraction contract | Correct-year figures on multi-year questions; refusals on missing coverage | Med |
| 12 | Prompt engineering: summarizing key facts/insights | `prompts/summarization-and-verification.md` — structured summary schema | "Summarize FY2023 results" → structured bullets with citations | Low |
| 13 | Prompt engineering: text/table output formatting | `prompts/table-and-number-formatting.md`; tables stored as markdown blocks | "Show revenue by segment" → markdown table in the answer | Low |
| 14 | Prompt engineering: follow-ups/context | `prompts/follow-up-contextualization.md` + history window | Demo step 5 | Med |
| 15 | Upload multiple documents | `doc-ingest` idempotent by sha256; multi-file uploader; per-doc retrieval filters | Upload 2 PDFs, ask a cross-doc question, see both cited | Low |
| 16 | Change LLM model parameters | Sidebar controls (model, temperature, top_p, max_tokens, k) → `GraphState.params` (`docs/adr/ADR-005`) | Change temperature/model, next answer reflects it; params recorded per message | Low |
| 17 | Manage chat history | `chat-store` sessions/turns; UI session manager | Switch sessions, see full transcript + citations intact | Low |
| 18 | Persist chat sessions | SQLite `data/chat.sqlite3` on compose volume (`docs/adr/ADR-006`) | Reload browser/app → sessions and citations intact | Low |
| 19 | Containerize for deployment | `Dockerfile`, `docker-compose.yml`, named volume, healthcheck (`docs/adr/ADR-008`) | `docker compose up --build` works cold | Low |
| 20 | Comprehensive error handling | Typed error taxonomy; no empty answers; visible banners (`docs/adr/ADR-007`) | Kill the API key → visible error, retry, no blank answer | Low |
| 21 | Observability | JSON logs, `node_trace`, debug panel, events table (`docs/adr/ADR-007`) | Debug panel shows latencies/tokens/scores for any answer incl. history | Low |
| 22 | JOURNAL.md — decisions, challenges, solutions, prompt strategies | `docs/JOURNAL.md` (seeded; finalized during build) | Journal exists, structured, cross-references ADRs with measured data | Med — must be filled with build-time data |
| 23 | Private GitHub repo shared with crcheng, levymedina, jgcosme | README, .gitignore (data/ excluded), no secrets | Repo private + collaborators added; clone → 4-command run | Low |
| 24 | AI Use Principles | Journal "AI use disclosure" section documents what AI did and what stayed human | Journal section present, specific | Low |

## Over-engineering check

| Element | Cut or keep | Reason |
|---|---|---|
| RRF score gate + single rewrite retry | **Keep** | Cheap (one extra retrieval), prevents "no relevant chunks → hallucination" |
| `out_of_scope` classification node | **Keep** | Refusals are a visible grading moment; low build cost |
| `cited_pages` enrichment of history | **Keep** | Directly fixes the worst failure mode (#3) at ~10 lines |
| Global context budget | **Keep, minimal form** | Prevents context-length crash on wide queries; ~15 lines in graph state packing |
| Cross-encoder reranker | **Cut** (deferred, documented) | 46h budget; golden QA is the honest proxy |
| Streaming synthesis | **Cut** (documented as Could) | Complicates groundedness gating; no grading credit |
| LangSmith | **Cut to env-gated stub** | Optional; needs external account |
| Postgres | **Cut** | Compose simplicity wins |

## Under-built check

| Item | Minimum that must exist |
|---|---|
| Citations | ≥1 numbered citation per factual answer; source snippet visible — not just "page 45" |
| Decomposition | Real fan-out (≥2 sub-queries) demonstrated on one question in the demo script, visible in debug panel |
| Observability | Debug panel renders node_trace for **current and past** answers; JSON log line per graph run |
| Docker | `docker build` succeeds in CI — the one check the grader's path works |
| Journal | Real build data: measured index build time, measured answer latency, ≥1 documented prompt iteration changing output |

## Journal hooks (ranked)

1. **Vector vs vectorless trade-off** (`docs/adr/ADR-003`) — research depth; strongest artifact.
2. **The deterministic neuro-symbolic split** (`docs/adr/ADR-010`) — why the graph is not
   multi-agent, what stays symbolic, and the deterministic numeric check that caught a wrong-year
   figure with zero LLM calls. Terminology grounded in literature (NeSy-RAG 2026; TrustTable,
   ACL 2026), not coined.
3. **Why parent-child chunks**, with the table-fragmentation failure and fix — pipeline judgment.
4. **The year-confusion failure** + the extraction contract + deterministic pre-pass that killed it.
5. **Retrieval fusion choice** (RRF over weighted sum) — numerical literacy.
6. **Observability design** — what was measured and what the numbers said.
7. **What was cut and why** — scoping judgment; graders trust engineers who cut.

# Spec: FinDoc Assistant — Financial PDF RAG Chatbot (TM GenAI Take-Home)

> Implementation-ready specification. Module ids and interface contracts come from
> `docs/CAPABILITY-MAP.md`; this document freezes them. ADRs, the plan, the task list, and the
> prompt library all reference *this* spec — when a decision changes, change the spec first.

## 1. Objective

A Streamlit chat application that lets a user upload one or more financial PDFs (reference
corpus: **Microsoft FY2023 Annual Report, fiscal year ended June 30, 2023**) and ask natural-
language questions, receiving **concise, citation-backed answers** grounded in the document.

The LangGraph graph (nodes named in `docs/ARCHITECTURE.md`, designed in
`docs/adr/ADR-001-langgraph-orchestration.md`): `contextualize_question` → `analyze_question` →
(`decompose_question` for multi-part) → `hybrid_retrieve` → `synthesize_answer` →
`groundedness_check`, with `refuse` for out-of-scope questions. Every node maps to exactly one
task in `docs/PLAN.md` and one file in `docs/REPO-STRUCTURE.md`.

Control is **symbolic**, semantics is **neural** (`docs/adr/ADR-010`): routing, caps, gates,
fusion, citation mapping, and a deterministic numeric verification pre-pass contain no LLM call;
the model handles rewriting, classification, generation, and the semantic half of grounding. This
is a deterministic neuro-symbolic pipeline and an **agentic workflow** in the LangGraph sense — the
graph is a *single fixed DAG with tool-calling nodes*, not a fleet of independently coordinating
agents; the difference is a scoping economy (one document, fan-out ≤ 4, one API key), not a
semantic one, and ADR-010 argues it in full.

The system demonstrates agentic behavior, not a naive "top-k → prompt → answer" wrapper:

- **Query contextualization** — follow-ups like *"and how did that change YoY?"* resolve against
  conversation history and the prior answer's cited pages.
- **Query analysis and planning** — the graph classifies each question (simple fact, multi-part
  comparison, summary/insight, out-of-scope) and routes it: single retrieval pass, decomposed
  multi-pass retrieval, or refusal.
- **Hybrid retrieval** — dense vector search + BM25 sparse keyword search, fused (RRF), over
  parent-child chunks; parent chunks (full table/section context) are what reach the LLM.
- **Grounded synthesis with citations** — answers cite `[1]`, `[2]`… mapped to
  (document, page range, section), rendered in the UI with the source snippet.
- **Conversation memory** — multi-turn context, session persistence across reloads.
- **Observability and error handling** — every graph run emits a structured event stream with
  per-node latency, token usage, retrieval scores; failures degrade gracefully and visibly.

**User story.** *As an analyst, I upload Microsoft's annual report, ask "What was revenue in 2023
and how did operating margin change?", and get a concise answer with per-claim citations linking
to the exact pages — and I can follow up with "which segment drove that?" without repeating
myself.*

## 2. Tech Stack

| Layer | Choice | Rationale (see ADR) |
|---|---|---|
| Language / runtime | Python 3.11+ (`.venv`, pinned `requirements.txt`) | ecosystem, reviewer familiarity |
| Orchestration | **LangGraph** (`langgraph` standalone package, `StateGraph` + `MessagesState`), narrow use of LangChain primitives for splitters/retrievers only | ADR-001 |
| LLM access | OpenAI-compatible `ChatOpenAI`-style client via `base_url` (OpenAI / Azure OpenAI / vLLM / Ollama); embeddings from same provider | ADR-005 |
| Extraction | `pypdf` (+ `pdfplumber` fallback for table pages) | ADR-004 |
| Splitting | `RecursiveCharacterTextSplitter` for child chunks; structure-aware parents from detected headings | ADR-002 |
| Retrieval | **FAISS** (dense) + **rank_bm25** (sparse), Reciprocal Rank Fusion; parent-document expansion | ADR-002 |
| Storage | SQLite (`chat.sqlite3`) for sessions/turns/events; index artifacts + parsed JSON under `data/` | ADR-006 |
| UI | **Streamlit** (`st.chat_message`, `st.file_uploader`, `st.session_state`, sidebar) | exam requirement |
| Observability | Python `logging` + structured JSON events + LangSmith hooks (optional) + in-app debug panel | ADR-007 |
| Packaging | Docker + docker-compose (one app service, named volumes for `data/`), GitHub Actions CI | ADR-008 |
| Tests | `pytest`; graph tests with a scripted/mock LLM | TESTING doc |

Non-negotiable constraint from the exam brief: *"Implement document retrieval using a vector
database to store and search document content"* — satisfied by FAISS. Vectorless RAG
(tree/PageIndex-style) is **evaluated and explicitly rejected** in ADR-003 with measurements,
not hand-waved.

## 3. Commands (target state — after Phase 1 of `docs/PLAN.md`)

```bash
# Setup
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # set OPENAI_API_KEY / OPENAI_BASE_URL

# Data (reference corpus; script downloads, PDF is gitignored)
python scripts/download_corpus.py

# Run
streamlit run app.py                                   # http://localhost:8501

# Test / lint
pytest -q                                              # full suite
pytest -q tests/test_graph.py -k citation              # focused
ruff check src tests app.py && black --check src tests app.py

# Docker
docker compose up --build                              # http://localhost:8501
```

## 4. Project Structure

```
├── app.py                     # Streamlit entrypoint; thin, no business logic
├── src/
│   ├── components/            # UI-only components (chat, sidebar, debug panel)
│   ├── core/                  # doc-ingest + chunking-index modules
│   ├── chains/                # LangGraph nodes, graph builder, prompt assembly
│   ├── utils/                 # llm-gateway client, logging/observability, settings
│   └── store/                 # chat-store: SQLite schema + accessors
├── prompts/                   # Prompt assets (markdown, versioned) — see docs/prompts/
├── scripts/                   # download_corpus.py, build_index.py, eval_rag.py
├── tests/                     # pytest: unit (parse/chunk/retrieve), graph (mock LLM), eval
├── data/                      # gitignored runtime: PDFs, FAISS/BM25 artifacts, chat.sqlite3
├── docs/                      # this spec, ADRs, plan, journal seed
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## 5. Code Style

- Type hints on every public function; fail fast with explicit exceptions.
- Modules expose pure functions where practical (parse → blocks, blocks → chunks,
  question → plan) so they are unit-testable without network.
- No business logic in `app.py` or `src/components/`: UI calls graph/gateway interfaces only.
- Structured logging via `structlog`-style JSON records (stdlib `logging` + JSON formatter is
  acceptable); every record carries `trace_id`, `session_id`, `node`.
- Comments explain **why** (trade-offs, invariants, gotchas), never restate the code.
- Errors: typed exception hierarchy in `src/utils/errors.py`; every module boundary catches,
  logs with context, and re-raises domain errors; the UI renders user-safe messages.

```python
def retrieve(query: str, k: int = 8, filters: dict | None = None) -> list[ScoredChunk]:
    """Hybrid dense+sparse retrieval with parent expansion.

    Dense and sparse ranks are fused with RRF *before* parent expansion so expansion
    cannot reorder by parent granularity alone. Returns child chunks annotated with
    their parent payload; caller decides what to send to the LLM.
    """
```

## 6. Testing Strategy

- **Unit (no network):** PDF parsing invariants (page counts, table blocks detected), chunker
  invariants (overlap, parent-child linkage, no content loss beyond configured slack), RRF fusion
  ordering, citation formatter, settings loading.
- **Graph tests (scripted LLM):** routing correctness (simple/multi-part/refusal), follow-up
  contextualization, decomposition fan-out, citation completeness — run against a deterministic
  fake gateway.
- **Golden QA eval:** `scripts/eval_rag.py` runs a fixed question set with expected
  figure/citation answers against the MSFT corpus; reports retrieval hit-rate, citation
  presence, and answer-vs-golden diff for manual review. See `docs/TESTING-AND-EVAL.md`.
- **UI smoke:** launch Streamlit, upload corpus, ask the revenue question, assert cited answer
  appears (scripted via Playwright or documented manual step).
- Coverage bar: every module in the capability map has at least one unit test file; graph nodes
  have one test each. No coverage-percentage target — behavior coverage, not line coverage.

## 7. Boundaries

- **Always:** run `pytest` + lint before any commit; keep prompts versioned in `prompts/`;
  ground every generated claim to a retrieved chunk or mark it ungrounded; log retrieval scores;
  never commit `data/` or `.env`.
- **Ask first (in a real team):** adding a paid dependency, changing the citation format,
  changing the chunking contract, switching embedding provider.
- **Never:** silently swallow LLM errors into an empty answer, return answers without citations
  when grounding exists, commit the PDF corpus or secrets, weaken the groundedness check to make
  a test pass.

## 8. Scope — MoSCoW Against Exam Requirements

| Exam requirement | Priority | Where |
|---|---|---|
| Upload PDF, ask questions, concise cited answers | **Must** | core slice (Phase 2–4 of plan) |
| LLM answers grounded in PDF content | **Must** | ADR-002/005, synthesis node |
| Query decomposition for complex questions | **Must** | orchestration graph |
| Follow-up handling + conversation context | **Must** | contextualization node + ADR-006 |
| Citations to document sections/pages | **Must** | citation contract §3 |
| Vector database retrieval | **Must** | FAISS (ADR-002) |
| LangChain/LangGraph orchestration | **Must** | ADR-001 |
| Document processing pipelines | **Must** | doc-ingest (ADR-004) |
| Prompt engineering (financial QA, summaries, tables) | **Must** | `prompts/`, ADR-003 notes |
| Error handling + observability | **Must** | ADR-007, OBSERVABILITY.md |
| Docker / compose | **Must** | ADR-008 |
| **Should** | Multi-document upload; model parameter controls (temperature/top-p/model); chat history management; persisted sessions; custom logging decorators for tokens/latency | plan phases 3–4 |
| **Could** | Streaming answers; re-ingestion of updated PDFs; LangSmith integration; Perplexity-style source snippet cards | plan phase 4, only after Must+Should |
| **Won't (exam)** | Fine-tuning, cloud deployment beyond Docker, auth/multi-user server, non-PDF formats | out of scope |

## 9. Interface Contracts (frozen)

Produced by modules named in `docs/CAPABILITY-MAP.md`; all downstream code and tests target
these signatures.

```python
# doc-ingest → chunking-index
@dataclass
class Block:
    kind: Literal["text", "table"]
    text: str                       # table → markdown/pipe rendering
    page: int                       # 1-based
    section_path: list[str]         # e.g. ["Business", "What We Offer"]

@dataclass
class ParsedDocument:
    doc_id: str                     # content hash
    filename: str
    page_count: int
    blocks: list[Block]

# chunking-index
@dataclass
class ScoredChunk:
    chunk_id: str
    child_text: str
    parent_text: str
    page: int
    section_path: list[str]
    doc_id: str
    dense_score: float
    sparse_score: float
    fused_rank: int

def retrieve(query: str, k: int = 8, filters: dict | None = None) -> list[ScoredChunk]: ...
def build_index(doc: ParsedDocument) -> HybridIndexHandle: ...   # idempotent by doc_id
def load_index() -> HybridIndexHandle | None: ...

# llm-gateway
@dataclass
class ChatResult:
    text: str
    parsed: dict | None            # when response schema requested & validated
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str

async def chat(messages: list[dict], params: LLMParams, schema: dict | None = None) -> ChatResult: ...

# orchestration
@dataclass
class AnswerEnvelope:
    answer_md: str                 # markdown; tables rendered with markdown tables
    citations: list[Citation]      # [1] → doc_id, page, section_path, snippet
    grounded: bool                 # self-check result
    refusal_reason: str | None
    plan: QueryPlan                # classification + sub-queries (debug panel)
    retrieval_debug: list[ScoredChunk]
    node_trace: list[NodeEvent]    # per-node latency, tokens, errors

def invoke(question: str, history: list[Turn], params: LLMParams) -> AnswerEnvelope: ...
```

## 10. Success Criteria

1. `docker compose up --build` serves the app at :8501 with no manual steps beyond `.env`.
2. *"What was Microsoft's revenue in FY2023?"* → **$211,915 million** with citation to the
   income-statement page.
3. *"Compare FY2023 and FY2022 operating income and explain the drivers"* → decomposed into
   sub-questions; both figures cited ($88,523M vs $83,383M); drivers grounded in MD&A text.
4. *"And the margin trend?"* (follow-up) → resolves "margin" from prior turn without repeating
   the product name.
5. A question with no supporting passage (e.g. *"What was the 2020 revenue?"*) → explicit,
   honest refusal naming what the documents cover — **never a fabricated number**.
6. Sidebar changes model/temperature and the next answer reflects them (observable in debug panel).
7. Every answer carries citations or a visible "ungrounded" marker; debug panel shows node trace.
8. `pytest -q` green; `ruff`/`black` clean; README reproduces all of the above.

## 11. Open Questions (implementation-phase, do not block planning)

- Embedding dimensionality/model default (`text-embedding-3-small` recommended; overridable via env).
- Whether to ship a mock-LLM mode (`MOCK_LLM=1`) for offline demo — **yes**, planned in `prompts/README.md`;
  it doubles as the graph-test double.
- Exact chunk sizes — start at child ~250 tokens / parent ~1000 tokens / overlap 50; tune against
  golden QA; record final values in the journal.

## 12. Traceability

Every exam bullet maps to at least one module id, one plan task, and one acceptance criterion —
see `docs/DELIVERABLES-MATRIX.md`.

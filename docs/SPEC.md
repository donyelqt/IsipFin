# Spec: IsipFin — Financial PDF RAG Chatbot (TM GenAI Take-Home)

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

# Data (reference corpus — see note below: Microsoft serves DOCX, not PDF)
# (no fetch command: drop the PDF in by hand, then index it)
python scripts/build_index.py                        # index data/pdfs/*.pdf → data/index

# Run
streamlit run app.py                                   # http://localhost:8501

# Test / lint
pytest -q                                              # full suite (304 tests, 6 files)
pytest -q tests/test_retrieval.py -k rrf               # focused: RRF fusion arithmetic
pytest -q tests/test_store.py -k citation              # focused: citation persistence
ruff check src tests scripts app.py && black --check src tests scripts app.py

# Docker
docker compose up --build                              # http://localhost:8501
```

The corpus is acquired **by hand**: drop the Microsoft FY2023 Annual Report PDF into
`data/pdfs/` and run `scripts/build_index.py`, which indexes whatever PDFs it finds there.
There is no download command because no official FY2023 PDF URL exists — Microsoft's own
ar23 download center serves a **DOCX** (a ZIP whose first bytes are `PK\x03\x04`, not `%PDF`)
for both the Annual Report and the 10-K, and SEC EDGAR is HTML-only. A hand-placed file is
**adopted**, not re-downloaded: `scripts/download_corpus.py` accepts `--url` for an opt-in
PDF fetch and validates the response's magic bytes before writing a `.pdf`, so a DOCX can
never be installed as corpus. The PDF itself is gitignored (`.gitignore`: `*.pdf`).

## 4. Project Structure

```
├── app.py                     # Streamlit entrypoint; thin, no business logic
├── src/
│   ├── components/            # UI-only components (Phase 4 — T18-T21, not yet built)
│   ├── core/                  # doc-ingest + chunking-index modules
│   │   ├── pdf_parser.py      # parse_pdf() → Block/ParsedDocument, ParseError
│   │   ├── chunking.py        # chunk_document() → ParentChunk/ChildChunk
│   │   ├── index.py           # ChunkIndex: build, save, load, replace_document
│   │   └── retrieval.py       # HybridRetriever: RRF fusion, parent expansion, filters
│   ├── chains/                # LangGraph nodes, graph builder, prompt assembly
│   │   ├── state.py           # GraphState, NodeEvent, AnswerEnvelope, Citation
│   │   ├── checks.py          # deterministic numeric verification pre-pass
│   │   ├── graph.py           # build/compile + routing tables, GraphError taxonomy
│   │   └── nodes/             # contextualize, analyze, decompose, retrieve, synthesize, verify, refuse
│   ├── utils/                 # llm-gateway client, mock gateway, prompt loader, node tracing
│   │   ├── llm.py             # build_llm(), ChatLLM, LLMError taxonomy, backoff
│   │   ├── mock_llm.py        # MockLLM: offline deterministic scripted gateway
│   │   ├── prompts.py         # load_prompt() over docs/prompts/, PromptError taxonomy
│   │   ├── settings.py        # load_env() → repo-root .env once; REPO_ROOT (python-dotenv)
│   │   └── tracing.py         # @trace_node → node_trace (ADR-007 §2)
│   └── store/                 # chat-store: SQLite schema + accessors
├── scripts/                   # download_corpus.py, build_index.py, demo_script.py, check_floor.py
├── tests/                     # pytest: parser, chunking, retrieval, gateway, store, prompts
├── data/                      # gitignored runtime: PDFs, FAISS/BM25 artifacts, chat.sqlite3
├── docs/                      # this spec, ADRs, plan, journal seed
├── Dockerfile                 # Phase 5 (T22) — not yet built
├── docker-compose.yml         # Phase 5 (T22) — not yet built
├── .env.example
└── requirements.txt
```

Prompt assets live in `docs/prompts/` (P1–P5 markdown, version-headered), loaded by
`src/utils/prompts.py`; `scripts/prompts_meta.json` records the minimum version and the
consuming node per id. There is deliberately **no** top-level `prompts/` directory.
Environment variables are read by the modules that use them (`build_llm()` reads
`MOCK_LLM`/`OPENAI_API_KEY`/`OPENAI_BASE_URL`/`OPENAI_MODEL`, `scripts/build_index.py` reads
the same names plus the `EMBEDDING_*` pair), but the `.env` file itself is loaded from the
repo root by `src/utils/settings.py::load_env()`: it resolves the root from its own
`__file__` (so `streamlit run app.py`, `python scripts/build_index.py` and pytest all find
the same file), is idempotent, and never overrides an already-set variable. It is called
first in `app.py::main()` and in `scripts/build_index.py::main()` /
`scripts/eval_rag.py::main()` — that call is what makes the documented
`cp .env.example .env` setup path supply the key and `MOCK_LLM` at all.

## 5. Code Style

- Type hints on every public function; fail fast with explicit exceptions.
- Modules expose pure functions where practical (parse → blocks, blocks → chunks,
  question → plan) so they are unit-testable without network.
- No business logic in `app.py` or `src/components/`: UI calls graph/gateway interfaces only.
- Structured logging via **stdlib `logging`** — no third-party logging dependency: each graph
  node's module owns `logger = logging.getLogger(__name__)` and logs degradations through it,
  tagging records with `extra={"node": NODE_NAME, ...}` so the emitted context (target: `ts`,
  `level`, `trace_id`, `session_id`, `node`) identifies the run. The JSON formatter /
  `setup_logging` described in ADR-007 §1 is **not yet built** (Phase 4, with the UI) — until
  then the stdlib records are the observability output, alongside the per-node trace in
  `NodeEvent` (see §9).
- Comments explain **why** (trade-offs, invariants, gotchas), never restate the code.
- Errors: a **typed exception hierarchy, defined in the module that raises it** — the gateway
  taxonomy (`LLMError` plus `RateLimited` / `ContextLength` / `Auth` / `Timeout` /
  `SchemaViolationError`) in `src/utils/llm.py`, `ParseError` in `src/core/pdf_parser.py`,
  `RetrievalError` in `src/core/retrieval.py`, `GraphError` / `GraphNodeError` /
  `GraphStateError` in `src/chains/graph.py`, `StoreError` / `SessionNotFoundError` in
  `src/store/`, and `PromptError` / `UnknownPromptError` / `PromptParseError` /
  `PromptVersionError` in `src/utils/prompts.py`. Each base lets a caller catch the domain
  without catching `Exception`; a failure is always typed, never a bare `RuntimeError` and
  never a swallowed one. Every module boundary catches, logs with context, and re-raises the
  domain error; no error is ever turned into an empty answer.

```python
def retrieve(query: str, k: int = 8, filters: dict | None = None) -> list[ScoredChunk]:
    """Hybrid dense+sparse retrieval with parent expansion.

    Dense and sparse ranks are fused with RRF *before* parent expansion so expansion
    cannot reorder by parent granularity alone. Returns child chunks annotated with
    their parent payload; caller decides what to send to the LLM.
    """
```

## 6. Testing Strategy

Every test is hermetic — no network, no API key. The offline scripted gateway
(`MOCK_LLM=1` → `src/utils/mock_llm.py`) and an injected fake embedder are the seams that make
that possible. Six files, all under `tests/`, all collected by `pytest -q` (304 tests):

| File | Covers |
|---|---|
| `tests/test_parser.py` | **doc-ingest**: cleaning (dehyphenation, boilerplate, page numbers), heading detection, table→markdown serialization, sha256 `doc_id` idempotency, 1-based pages, section-path inheritance, scanned-page flagging, `ParseError` carrying the page number |
| `tests/test_chunking.py` | **chunking**: parent/child linkage (every child resolves to a parent, every parent owns a child), section-aligned parents, cross-page merge, table atomicity in parents (row-boundary split with the header repeated), child overlap, no page loss, determinism |
| `tests/test_retrieval.py` | **chunking-index retrieval**: RRF arithmetic verified by hand against `1/(k+rank)` with `k=60`, fused-rank ordering and tie-breaks, section/`doc_id` filters applied *before* ranking, parent expansion after fusion, save→load→retrieve round-trip, `k` bounds, empty-document and zero-BM25-rank degenerate cases, no `openai` import |
| `tests/test_gateway.py` | **llm-gateway**: the frozen `ChatResult` fields and mock determinism (across calls *and* across processes), the keyless `MOCK_LLM=1` path importing no `openai`, retry policy (429/5xx retried; auth, context-length and timeout never), exponential backoff arithmetic, schema-mode single strict retry, token/latency accounting, the exact FY2023 revenue figure through the mock, and the real keyed path built from the environment |
| `tests/test_store.py` | **chat-store**: the six documented tables with columns *in order*, WAL + foreign-key pragmas, schema idempotency, session lifecycle reloaded through a fresh connection, delete cascade, the history window, per-session 1-based gap-free `seq`, unknown ids raising rather than no-oping, two connections on one file, the events stream, and the document registry |
| `tests/test_prompts.py` | **prompt assets**: all five prompts (P1–P5) load with a complete semver header, the minimum-version floor, unknown-id rejection, node/meta agreement, no prompt text duplicated into `src/`, numeric version comparison, and header-parsing edge cases |

- **Graph tests (scripted LLM):** routing correctness (simple/multi-part/refusal), follow-up
  contextualization, decomposition fan-out, and citation completeness are driven end to end by
  `scripts/demo_script.py` (T16), which runs the graph's nodes in the graph's own order under
  `MOCK_LLM=1` against the real persisted index. Dedicated per-node unit tests land with the
  graph nodes in Phase 3 of `docs/PLAN.md`; `pytest` itself holds the six hermetic unit files
  above, and `testpaths = ["tests"]` keeps network-touching golden QA out of the suite.
- **Golden QA eval:** `scripts/eval_rag.py` (T17) runs the fixed question set in
  `tests/data/golden_qa.jsonl` end to end through real `invoke()` calls over the real
  persisted index in `data/index/` — nothing is mocked. It reports retrieval hit-rate@5
  (bar ≥ 0.9), citation presence (1.0), figure match (≥ 0.95), wrong-year contamination
  (0, hard fail) and refusal correctness, and exits nonzero on a golden regression.
  Under `MOCK_LLM=1` it runs hermetically with no key; mock-mode retrieval metrics are
  reported `N/A (mock mode)` with the reason rather than silently passed. It is deliberately
  outside `pytest`'s `testpaths`. See `docs/TESTING-AND-EVAL.md`.
- **UI smoke:** launch Streamlit, upload corpus, ask the revenue question, assert cited answer
  appears (scripted via Playwright or documented manual step) — lands with Phase 4.
- Coverage bar: every module in the capability map has at least one unit test file; graph nodes
  have one test each. No coverage-percentage target — behavior coverage, not line coverage.

## 7. Boundaries

- **Always:** run `pytest` + lint before any commit; keep prompts versioned in
  `docs/prompts/`; ground every generated claim to a retrieved chunk or mark it ungrounded;
  log retrieval scores; never commit `data/` or `.env`.
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

- Embedding dimensionality/model default — `text-embedding-3-small` (shipped, overridable via
  `EMBEDDING_MODEL` / `EMBEDDING_BASE_URL`); `MOCK_LLM=1` instead uses the 64-dim offline
  embedder in `scripts/build_index.py`.
- ~~Whether to ship a mock-LLM mode~~ — **resolved and shipped**: `MOCK_LLM=1` returns
  `src/utils.mock_llm.MockLLM`, a deterministic, keyless, network-free scripted gateway
  (ADR-005). It doubles as the graph-test double and is documented in `docs/prompts/README.md`.
- Exact chunk sizes — child ~250 tokens / parent ~1000 tokens / overlap 50 are the shipped
  values in `src/core/chunking.py` (ADR-002 parameter table); any retune goes through the
  golden-QA score in `scripts/eval_rag.py` and gets recorded in the journal.

## 12. Traceability

Every exam bullet maps to at least one module id, one plan task, and one acceptance criterion —
see `docs/DELIVERABLES-MATRIX.md`.

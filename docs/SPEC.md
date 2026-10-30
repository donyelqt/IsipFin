# Spec: IsipFin — Financial PDF RAG Chatbot (TM GenAI Take-Home)

> Implementation-ready specification. Module ids and interface contracts come from
> `docs/CAPABILITY-MAP.md`; this document freezes them. ADRs, the plan, the task list, and the
> prompt library all reference *this* spec — when a decision changes, change the spec first.

## 1. Objective

A Streamlit chat app that lets a user upload one or more financial PDFs (reference corpus:
**Microsoft FY2023 Annual Report, fiscal year ended June 30, 2023**) and ask natural-language
questions, receiving **concise, citation-backed answers** grounded in the document.

The LangGraph graph (nodes named in `docs/ARCHITECTURE.md`, designed in
`docs/adr/ADR-001`): `contextualize_question` → `analyze_question` →
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
  parent-child chunks; parent chunks (full table/section context) reach the LLM.
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
| Orchestration | **LangGraph** (`langgraph` standalone, `StateGraph` over a local `GraphState` — no `MessagesState`), narrow LangChain primitives for splitters/retrievers only | ADR-001 |
| LLM access | OpenAI-compatible client via `base_url` (OpenAI / Azure OpenAI / vLLM / Ollama); chat and embeddings separate — `EMBEDDING_PROVIDER=local` (`fastembed`, default, in-process, no key) or `api` for the OpenAI-compatible path | ADR-005 |
| Extraction | `pypdf` (+ `pdfplumber` for table pages) | ADR-004 |
| Splitting | `RecursiveCharacterTextSplitter` for child chunks; structure-aware parents from detected headings | ADR-002 |
| Retrieval | **FAISS** (dense) + **rank_bm25** (sparse), Reciprocal Rank Fusion; parent-document expansion | ADR-002 |
| Storage | SQLite (`chat.sqlite3`) for sessions/turns/events; index artifacts + parsed JSON under `data/` | ADR-006 |
| UI | **Streamlit** (`st.chat_message`, `st.file_uploader`, `st.session_state`, sidebar) | exam requirement |
| Observability | stdlib `logging` + structured JSON events + LangSmith hooks (optional) + in-app debug panel | ADR-007 |
| Packaging | Docker + docker-compose (one app service, named volumes for `data/`), GitHub Actions CI | ADR-008 |
| Tests | `pytest`; graph tests with a scripted/mock LLM | TESTING doc |

Non-negotiable from the brief: *"Implement document retrieval using a vector database"* — satisfied
by FAISS. Vectorless RAG (tree/PageIndex-style) is **evaluated and rejected** in ADR-003 with
measurements, not hand-waved.

## 3. Commands (current state — Phase 1 of `docs/PLAN.md` is done)

```bash
# Setup
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # set LLM_PROVIDER / LLM_API_KEY (LLM_MODEL optional)

# (no fetch command: drop the PDF in by hand, then index it)
# Data (reference corpus — see note: Microsoft serves DOCX, not PDF)
python scripts/build_index.py                        # index data/pdfs/*.pdf → data/index

# Run
streamlit run app.py                                   # http://localhost:8501

# Test / lint
pytest -q                                              # full suite (634 tests, 15 files)
pytest -q tests/test_retrieval.py -k rrf               # focused: RRF fusion arithmetic
pytest -q tests/test_store.py -k citation              # focused: citation persistence
ruff check src tests scripts app.py && black --check src tests scripts app.py

# Docker
docker compose up --build                              # http://localhost:8501
```

The corpus is acquired **by hand**: drop the Microsoft FY2023 Annual Report PDF into
`data/pdfs/` and run `scripts/build_index.py`, which indexes whatever PDFs it finds there.
There is no default download because no official FY2023 PDF URL exists — Microsoft's ar23
download center serves a **DOCX** (a ZIP whose first bytes are `PK\x03\x04`, not `%PDF`)
for both the Annual Report and the 10-K, and SEC EDGAR is HTML-only. A hand-placed file is
**adopted**, not re-downloaded: `scripts/download_corpus.py` accepts `--url` for an opt-in
PDF fetch and validates the response's magic bytes before writing a `.pdf`, so a DOCX can
never be installed as corpus. The PDF itself is gitignored (`.gitignore`: `*.pdf`).

## 4. Project Structure

```
├── app.py                     # Streamlit entrypoint; thin, no business logic
├── src/
│   ├── components/            # seven modules: chat, upload, sidebar, theme, errors, debug_panel, __init__
│   ├── core/                  # doc-ingest + chunking-index modules
│   │   ├── pdf_parser.py      # parse_pdf() → Block/ParsedDocument, ParseError
│   │   ├── chunking.py        # chunk_document() → ParentChunk/ChildChunk
│   │   ├── index.py           # ChunkIndex: build, save, load, replace_document
│   │   ├── retrieval.py       # HybridRetriever: RRF fusion, parent expansion, filters
│   │   └── local_embedder.py  # EMBEDDING_PROVIDER=local: fastembed BAAI/bge-small-en-v1.5, LocalEmbeddingError
│   ├── chains/                # LangGraph nodes, graph builder, prompt assembly
│   │   ├── state.py           # GraphState, NodeEvent, AnswerEnvelope, Citation
│   │   ├── checks.py          # deterministic numeric verification pre-pass
│   │   ├── graph.py           # build/compile + routing tables, GraphError taxonomy
│   │   └── nodes/             # contextualize, analyze, decompose, retrieve, synthesize, verify, refuse
│   ├── utils/                 # llm-gateway client, mock gateway, prompt loader, node tracing
│   │   ├── llm.py             # build_llm(), chat client, LLMError taxonomy, backoff (ADR-005)
│   │   ├── mock_llm.py        # MockLLM: offline deterministic scripted gateway
│   │   ├── prompts.py         # load_prompt() over docs/prompts/, PromptError taxonomy
│   │   ├── settings.py        # load_env() → repo-root .env once; REPO_ROOT (python-dotenv)
│   │   └── tracing.py         # @trace_node → node_trace (ADR-007 §2)
│   └── store/                 # chat-store: SQLite schema + accessors
├── scripts/                   # download_corpus.py, build_index.py, demo_script.py, check_floor.py
├── tests/                     # pytest: 15 files — parser, chunking, retrieval, gateway, store, prompts, nodes
├── data/                      # gitignored runtime: PDFs, FAISS/BM25 artifacts, chat.sqlite3
├── docs/                      # this spec, ADRs, plan, journal seed, LOCAL-INFERENCE, NV-INFERENCE
├── Dockerfile                 # two-stage build; runtime uid=10001, healthcheck
├── docker-compose.yml         # one app service, named data:/app/data volume, full env passthrough
├── .env.example
└── requirements.txt
```

Prompts live in `docs/prompts/` (P1–P5 markdown, version-headered), loaded by
`src/utils/prompts.py`; `scripts/prompts_meta.json` records the minimum version and consuming
node per id. There is deliberately **no** top-level `prompts/` directory.

Environment variables are read by the modules that use them (`build_llm()` reads `MOCK_LLM`
first, then the chat surface `LLM_PROVIDER` / `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` — with
`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` and `GROQ_API_KEY` as lower-precedence
aliases — `scripts/build_index.py` reads the same names plus the `EMBEDDING_*` pair). The `.env`
file itself is loaded from the repo root by `src/utils/settings.py::load_env()`: it resolves the
root from its own `__file__` (so `streamlit run app.py`, `python scripts/build_index.py` and
pytest all find the same file), is idempotent, and never overrides an already-set variable —
except for the provider-aiming names, where a non-blank value **in the file** is authoritative so an
ambient `OPENAI_*` export cannot silently replace a credential. It is called first in
`app.py::main()` and in `scripts/build_index.py::main()` / `scripts/eval_rag.py::main()` — that
call is what makes the documented `cp .env.example .env` setup supply the key and `MOCK_LLM`.

`MOCK_LLM` is checked before the chat names: with `MOCK_LLM=1` no `LLM_PROVIDER`,
`LLM_API_KEY`, `LLM_BASE_URL` or `LLM_MODEL` is needed — the keyless demo/test path. `LLM_PROVIDER`
(`openai` · `groq` · `nvidia` · `ollama` · `vllm` · `azure`) selects the provider's own endpoint;
`LLM_BASE_URL` overrides it; `LLM_API_KEY` is the single chat credential; `LLM_MODEL` is the model
id, still overridable per call through `params.model`. Because Groq exposes no `/embeddings`,
chat and embeddings are separate — a Groq chat config pairs with `EMBEDDING_PROVIDER=local`, and
the Groq model must accept `response_format: json_schema` (`analyze_question`, `groundedness_check`).
`LLM_PROVIDER=ollama` is **not supported**: measured against `qwen2.5:3b`, only P2 is servable
locally — P1 misroutes superlatives, P3 emits no citation markers, P5 over-trims — so a keyless
local run today needs `MOCK_LLM=1`; full per-node verdict + measurement traps in
`docs/LOCAL-INFERENCE.md`. `.env.example` is the worked reference for all of it.

## 5. Code Style

- Type hints on every public function; fail fast with explicit exceptions.
- Modules expose pure functions where practical (parse → blocks, blocks → chunks, question → plan)
  so they are unit-testable without network.
- No business logic in `app.py` or `src/components/`: UI calls graph/gateway interfaces only.
- Structured logging via **stdlib `logging`** — no third-party logging dependency: each node's
  module owns `logger = logging.getLogger(__name__)` and logs degradations, tagging records with
  `extra={"node": NODE_NAME, ...}` so the emitted context (target: `ts`, `level`, `trace_id`,
  `session_id`, `node`) identifies the run. The JSON formatter / `setup_logging` from ADR-007 §1 is
  **not built** — the stdlib records are the observability output, alongside the per-node trace in
  `NodeEvent` (§9). `LOG_JSON` therefore names nothing: no module reads it
  (`grep -rn LOG_JSON --include="*.py" .` returns no matches), so it was removed from
  `Dockerfile` and `docker-compose.yml` and is documented as unbuilt here.
- Comments explain **why** (trade-offs, invariants, gotchas), never restate the code.
- Errors: a **typed exception hierarchy, defined in the module that raises it** — the gateway
  taxonomy (`LLMError` + `RateLimited` / `ContextLength` / `Auth` / `Timeout` /
  `SchemaViolationError`) in `src/utils/llm.py`, `ParseError` in `src/core/pdf_parser.py`,
  `RetrievalError` in `src/core/retrieval.py`, `GraphError` / `GraphNodeError` /
  `GraphStateError` in `src/chains/graph.py`, `StoreError` / `SessionNotFoundError` in
  `src/store/`, and `PromptError` / `UnknownPromptError` / `PromptParseError` /
  `PromptVersionError` in `src/utils/prompts.py`. Each base lets a caller catch the domain
  without catching `Exception`; a failure is always typed, never a bare `RuntimeError` and never
  swallowed. Every module boundary catches, logs with context, and re-raises the domain error; no
  error is ever turned into an empty answer.

```python
def retrieve(query: str, k: int = 8, filters: dict | None = None) -> list[ScoredChunk]:
    """Hybrid dense+sparse retrieval with parent expansion.

    Dense and sparse ranks are fused with RRF *before* parent expansion so expansion cannot
    reorder by parent granularity alone. Returns child chunks annotated with their parent
    payload; caller decides what to send to the LLM.
    """
```

## 6. Testing Strategy

Every test is hermetic — no network, no API key. The offline scripted gateway
(`MOCK_LLM=1` → `src/utils/mock_llm.py`) and an injected fake embedder are the seams that make
that possible. Fifteen files, all under `tests/`, all collected by `pytest -q` (634 tests):

| File | Covers |
|---|---|
| `tests/test_parser.py` | **doc-ingest**: cleaning, heading detection, table→markdown, sha256 `doc_id` idempotency, 1-based pages, section-path inheritance, scanned-page flagging, `ParseError` |
| `tests/test_chunking.py` | **chunking**: parent/child linkage, section-aligned parents, cross-page merge, table atomicity, child overlap, page ranges, determinism |
| `tests/test_retrieval.py` | **retrieval**: RRF arithmetic vs `1/(k+rank)` with `k=60`, fused-rank ordering + tie-breaks, section/`doc_id` filters before ranking, parent expansion after fusion, save→load→retrieve round-trip, `k` bounds, empty/zero-BM25 degenerate cases |
| `tests/test_gateway.py` | **llm-gateway**: frozen `ChatResult` fields, mock determinism (across calls + processes), keyless `MOCK_LLM=1` importing no `openai`, retry policy (429/5xx retried; auth/context/timeout never), backoff arithmetic, schema-mode single strict retry, token/latency accounting, exact FY2023 revenue through mock, real keyed path from environment |
| `tests/test_store.py` | **chat-store**: 6 tables with columns *in order*, WAL + foreign-key pragmas, schema idempotency, session lifecycle reloaded through a fresh connection, delete cascade, history window, per-session 1-based gap-free `seq`, unknown ids raising, two connections on one file, events stream, document registry |
| `tests/test_prompts.py` | **prompts**: all five (P1–P5) load with a complete semver header, min-version floor, unknown-id rejection, node/meta agreement, no prompt text duplicated into `src/`, numeric version compare, header edge cases |
| `tests/test_nodes_analyze.py` | **analyze node**: classification + routing, incl. the fiscal-year coverage gate before any planner POST |
| `tests/test_nodes_verify.py` | **verify node**: groundedness checks + deterministic numeric pre-pass |
| `tests/test_eval_rag.py` | **golden-QA harness**: both modes, degradation, pacing, exit codes (suite also holds `test_settings.py`, `test_errors.py`, `test_degraded_planner_marker.py`, `test_doc_summary_sections.py`, `test_answer_survives_store_failure.py`) |

- **Graph tests (scripted LLM):** routing correctness (simple/multi-part/refusal), follow-up
  contextualization, decomposition fan-out, and citation completeness are driven end to end by
  `scripts/demo_script.py` (T16), which runs the nodes in their own order under `MOCK_LLM=1`
  against the real persisted index. Nodes also carry per-node unit files
  (`tests/test_nodes_analyze.py`, `tests/test_nodes_verify.py`); `pytest` holds the fifteen
  hermetic unit files above, and `testpaths = ["tests"]` keeps network-touching golden QA out.
- **Golden QA eval:** `scripts/eval_rag.py` (T17) runs the fixed question set
  (`tests/data/golden_qa.jsonl`) through real `invoke()` over the persisted index in `data/index/`.
  It reports retrieval hit-rate@5 (≥ 0.9), citation presence (1.0), figure match (≥ 0.95),
  wrong-year contamination (0, hard fail) and refusal correctness, and exits nonzero on regression.
  Two modes: `MOCK_LLM=1` runs hermetic with no key, reporting `NA` with its reason for every
  metric it cannot measure (the wrong-year canary is lexical, so it can still fail); keyed mode
  scores the full set. A case whose planner could not run (429/auth/context/timeout) is
  **not measured** and excluded from its metrics, never scored as a semantic verdict, but still
  drives exit 1 (`EXIT_REGRESSION`) — a degraded case can never be laundered into a pass. A real-LLM
  run is *paced*: before each case the harness projects that case's token cost and waits for room
  in the rolling window vs `EVAL_TPM_LIMIT` (default 8000) and `EVAL_TPD_LIMIT` (default 200000),
  with `EVAL_TPD_BUDGET` (tokens for today) / `EVAL_TPD_FRACTION` (default 1.0) sizing the day's
  usable share and `EVAL_TPD_OVERRIDE=1` disabling the daily gate; it prints the arithmetic so a
  wait is not confused with a hang. The pacer never re-scores a throttled case. Under `MOCK_LLM=1`
  the pacer is inert. Outside `pytest`'s `testpaths`. See `docs/TESTING-AND-EVAL.md`.
- **UI smoke:** launch Streamlit, upload corpus (or `scripts/build_index.py`), ask the revenue
  question, confirm the cited answer — the manual path is `docs/RUNBOOK.md`; the docker path is
  a `docker build` smoke in CI.
- Coverage bar: every module in the capability map has at least one unit test file; graph nodes
  have one test each. No coverage-percentage target — behavior coverage, not line coverage.

## 7. Boundaries

- **Always:** run `pytest` + lint before any commit; keep prompts versioned in `docs/prompts/`;
  ground every generated claim to a retrieved chunk or mark it ungrounded; log retrieval scores;
  never commit `data/` or `.env`.
- **Ask first:** adding a paid dependency, changing the citation format, changing the chunking
  contract, switching embedding provider.
- **Never:** swallow LLM errors into an empty answer, return answers without citations when
  grounding exists, commit the PDF corpus or secrets, weaken the groundedness check to make a test
  pass.

## 8. Scope — MoSCoW Against Exam Requirements

| Exam requirement | Priority | Where |
|---|---|---|
| Upload PDF, ask questions, concise cited answers | **Must** | core slice (Phase 2–4) |
| LLM answers grounded in PDF content | **Must** | ADR-002/005, synthesis node |
| Query decomposition for complex questions | **Must** | orchestration graph |
| Follow-up handling + conversation context | **Must** | contextualization node + ADR-006 |
| Citations to document sections/pages | **Must** | citation contract §3 |
| Vector database retrieval | **Must** | FAISS (ADR-002) |
| LangChain/LangGraph orchestration | **Must** | ADR-001 |
| Document processing pipelines | **Must** | doc-ingest (ADR-004) |
| Prompt engineering (financial QA, summaries, tables) | **Must** | `docs/prompts/`, ADR-003 notes |
| Error handling + observability | **Must** | ADR-007, OBSERVABILITY.md |
| Docker / compose | **Must** | ADR-008 |
| **Should** | Multi-document upload; model parameter controls (temperature/top-p/model); chat history; persisted sessions; per-node token/latency decorators | plan phases 3–4 |
| **Could** | Streaming answers; re-ingestion of updated PDFs; LangSmith integration; source-snippet cards | plan phase 4, only after Must+Should |
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
5. A question with no supporting passage (e.g. *"What was 2020 revenue?"*) → honest refusal
   naming what the documents cover — **never a fabricated number**.
6. Sidebar changes model/temperature and the next answer reflects them (observable in debug panel).
7. Every answer carries citations or a visible "ungrounded" marker; debug panel shows node trace.
8. `pytest -q` green; `ruff`/`black` clean; README reproduces all of the above.

## 11. Open Questions (implementation-phase, do not block planning)

- ~~Embedding model~~ — **resolved**: embeddings are a separate setting from chat. `EMBEDDING_PROVIDER=local`
  (default) runs `fastembed` 0.9.0 with `BAAI/bge-small-en-v1.5` (384-dim) in-process via
  `src/core/local_embedder.py`, no key; `EMBEDDING_PROVIDER=api` is OpenAI-compatible
  (`EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`, default `text-embedding-3-small`). `MOCK_LLM=1`
  uses the offline deterministic embedder in `scripts/build_index.py`. Guidance:
  `docs/LOCAL-INFERENCE.md`, `docs/NV-INFERENCE.md`.
- ~~Mock-LLM mode~~ — **resolved**: `MOCK_LLM=1` returns `src/utils/mock_llm.py::MockLLM`, a
  deterministic, keyless, network-free scripted gateway (ADR-005). It is also the graph-test
  double and is documented in `docs/prompts/README.md`.
- Chunk sizes — child ~250 tokens / parent ~1000 tokens / overlap 50, shipped in
  `src/core/chunking.py` (ADR-002). Any retune goes through golden-QA (`scripts/eval_rag.py`) and
  gets recorded in the journal.

## 12. Traceability

Every exam bullet maps to at least one module id, one plan task, and one acceptance criterion —
see `docs/DELIVERABLES-MATRIX.md`.

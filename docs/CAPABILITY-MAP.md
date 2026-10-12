# Capability Map: IsipFin (TM GenAI Take-Home)

> Phase 0 artifact of the spec-driven workflow. Every later document (spec, architecture, ADRs,
> plan, task list) selects work by the `module id` below. Ids are stable — rename only by
> superseding this document.

## Objective

A Streamlit chat application that ingests financial PDFs (Microsoft FY2023 Annual Report as the
reference corpus) and answers natural-language questions with concise, citation-backed answers,
built on a LangGraph agentic RAG pipeline with production-grade error handling and observability.

## Module Map

| module id | Responsibility | Produces | Depends on |
|---|---|---|---|
| `doc-ingest` | PDF open, text/page extraction, cleaning, table extraction, heading/section detection, provenance metadata | `ParsedDocument`, `Block[]` (text/table, page, section path) | — |
| `chunking-index` | Parent-child chunking, hybrid index build (dense + BM25), persistence + reload, re-ingest/dedup by content hash | `Chunk[]`, `HybridIndex` (retrieval API + artifacts under `data/`) | `doc-ingest` |
| `llm-gateway` | Provider-agnostic chat client (OpenAI-compatible `base_url`, Azure, local vLLM/Ollama), JSON-schema outputs, streaming, retries/backoff, token+latency accounting | `LLMClient`, `ChatResult`, typed errors | — |
| `orchestration` | LangGraph graph: query contextualization, analysis/planning, decomposition, hybrid retrieval + parent expansion, citation synthesis, groundedness self-check, formatting | `GraphState`, compiled graph, node functions | `chunking-index`, `llm-gateway` |
| `chat-store` | Sessions/turns/messages/events tables, document registry, LangGraph checkpointer binding | SQLite schema + accessors (`data/chat.sqlite3`) | — |
| `app-ui` | Streamlit UI: upload (multi-PDF), chat, follow-ups, model-parameter sidebar, session manager, retrieval debug panel, error surfacing | `app.py` + components | `orchestration`, `chat-store`, `chunking-index` |
| `packaging` | Docker image, compose (app + volumes + healthcheck), env template, CI workflow | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` | all |

## Dependency Direction (no cycles)

```
doc-ingest ──► chunking-index ──┐
llm-gateway ────────────────────►├──► orchestration ──► app-ui ──► packaging
chat-store ─────────────────────┘
```

Build order: `doc-ingest` → `chunking-index` (+ `llm-gateway` in parallel, it is a leaf) →
`orchestration` → `chat-store` → `app-ui` → `packaging`.

## Slice Assignments (who builds what, in what order)

1. **Foundation (serial):** `doc-ingest`, `chunking-index` — retrieval quality gates depend on them.
2. **Parallel with 1:** `llm-gateway`, `chat-store` — independent leaves; no shared code.
3. **Core (serial):** `orchestration` — consumes the retrieval contract from (1) and the client
   contract from (2). This is the highest-risk module; build after contracts are frozen.
4. **Serial:** `app-ui` — thin adapter over the graph; must not contain retrieval or prompt logic.
5. **Serial last:** `packaging` — pins what already builds.
6. **Cross-cutting:** observability and prompt assets land with their owning module (see
   `docs/OBSERVABILITY.md`, `docs/prompts/`), never as a separate phase.

## Interfaces Frozen Here (full contracts in `docs/SPEC.md` §3)

- `chunking-index.retrieve(query: str, k: int, filters: dict | None) -> list[ScoredChunk]` where
  `ScoredChunk` carries child text, parent text, page range, section path, dense+sparse scores.
- `llm-gateway.chat(messages, params, schema | None) -> ChatResult` where `ChatResult` carries
  text, parsed JSON when a schema is given, token usage, latency, model id.
- `orchestration.invoke(question: str, history: list[Turn], params) -> AnswerEnvelope` where
  `AnswerEnvelope` carries answer markdown, citations `[n] → (doc, page, section)`, retrieval
  diagnostics, node trace, and a refusal flag.

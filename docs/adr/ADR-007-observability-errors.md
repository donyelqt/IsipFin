# ADR-007: Observability and error handling — JSON event stream, node tracing, typed errors

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none

## Context

The brief requires *"comprehensive error handling and observability standards"* and the grading
criteria call out *"proper error handling and observability."* For a RAG app the interesting
failures are not exceptions — they are **plausible wrong answers**: a wrong-year number, an
uncited claim, a refusal that should have answered. The system must make all of this visible
in-product, with zero external infrastructure.

## Decision

Three mechanisms, no external dependencies:

### 1. Structured logging

`src/utils/tracing.py` is the observability entry point: it owns the node-trace JSON event stream that `app.py` initializes at startup (no separate logging module exists). Every record carries:
`ts`, `level`, `trace_id` (per graph run, uuid4), `session_id`, `turn_seq`, `node`, `event`,
`duration_ms`, plus node-specific fields, and is emitted by the `@trace_node` decorator (see §2). One `setup_trace()` call at startup from
`app.py`.

### 2. Node tracing

`@trace_node` decorator in `src/utils/tracing.py` wraps every graph node:

```python
@trace_node("synthesize_answer")
def synthesize_answer(state: GraphState) -> dict:
    ...  # returns partial state update
```

It times the node, extracts token usage from the gateway result, catches exceptions into a
`NodeEvent(node, started_at, duration_ms, tokens_in, tokens_out, error | None)` appended to
`GraphState.node_trace`, and re-raises typed errors. `node_trace` is persisted to
`messages.node_trace_json` and mirrored into the `events` table — the debug panel can show
history, not just the latest turn.

### 3. In-app debug panel

Streamlit sidebar section (collapsed by default) rendering the last answer's:
`query_plan` (kind + sub-queries), per-node latency/tokens, and retrieved chunks with
**dense score, sparse score, and fused RRF score** — so a grader can see *why* an answer was
produced, and *which* chunks grounded it. This is the demo artifact for observability.

### Metrics (all derivable from `node_trace` — no Prometheus)

| Metric | Source | Use |
|---|---|---|
| Per-node latency p50/p95 | `NodeEvent.duration_ms` | spot the slow node (usually embeddings or synthesis) |
| Tokens in/out per turn | gateway accounting in `NodeEvent` | cost awareness, context-window guard |
| Retrieval score distribution | `retrieval_debug` | detect weak retrieval (all scores below `tau`) |
| Groundedness-failure rate | `grounded` flag counts | prompt-quality signal over time |
| Refusal rate | `refusal_reason` histogram | distinguishes true out-of-scope from classification errors |
| Path counts | `query_plan.kind` mix | proves decomposition is exercised |

### Error taxonomy (in `src/utils/llm.py` — `src/utils/errors.py` was never created)

| Error | Raised by | Log | User-facing behavior |
|---|---|---|---|
| `LLMError.RateLimited` | gateway | warning + retry log | inline "model busy, retrying…"; after max attempts, error banner with Retry button |
| `LLMError.ContextLength` | gateway/synthesis | error | user banner: "context too large for model — lower k or ask a narrower question" |
| `LLMError.Auth` | gateway | error | banner naming the env var; never leaks the key |
| `LLMError.Timeout` | gateway | warning | retry once, then banner |
| `LLMError.SchemaViolation` | JSON nodes | warning | one stricter retry, then classification `out_of_scope` + flag |
| `RetrievalError` | index load/search | error | banner: "index not built — upload a PDF"; app stays usable for other sessions |
| `ParseError` | doc-ingest | error w/ page number | per-file upload error card; other uploads still process |
| `GraphNodeError` | any node | error w/ node name | banner with node name + Retry |

**Hard rule, enforced by review:** no failure path yields an empty answer. The response is one
of: a grounded answer, a partial answer with unsupported claims trimmed and marked `ungrounded`,
an explicit refusal with reason, or a visible error banner. `groundedness_check` failure routes
to trim-and-flag, never to silent output.

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| LangSmith tracing | Genuinely good and cheap to add — kept as an **optional** env-gated integration (`LANGSMITH_TRACING=true` adds env vars only); rejected as the *sole* mechanism because it requires an external account/key the grader may not have |
| Prometheus + Grafana | Infrastructure weight out of proportion to a single-container exam app |
| Print/`st.write` debugging | Not durable, not queryable, leaks into the UI |
| Sentry | Cloud dependency; overkill |

## Consequences

- Graders can verify observability from the UI alone — no log spelunking required.
- The `events` table + `node_trace` give the journal real material (measured latencies and
  failure counts during development).
- Metric set is demo-scale by design; the ADR states the upgrade path (OTel export) instead of
  building it.

# Observability

> The runtime contract for the three mechanisms decided in
> `docs/adr/ADR-007-observability-errors.md`. This document is what the code implements and
> what the debug panel renders.

## Event shape

Every observability record — JSON log line, `events` table row, and trace entry — carries the
same fields:

```json
{
  "ts": "2026-10-13T02:31:04.812Z",
  "level": "info",
  "event": "node.synthesize_answer",
  "trace_id": "b3f1...",
  "session_id": "0f6c...",
  "turn_seq": 12,
  "node": "synthesize_answer",
  "duration_ms": 1843,
  "tokens_in": 5117,
  "tokens_out": 214,
  "error": null,
  "extra": { "citations": 3, "grounded": true }
}
```

`event` vocabulary: `graph.start`, `node.<name>`, `node.<name>.error`, `retrieval.query`,
`synthesis.complete`, `numeric_prepass.trimmed`, `grounding.trimmed`, `refusal.issued`,
`upload.parsed`, `session.switched`.
Anything not in the vocabulary goes in `extra`, never as a new top-level `event` type.

**Implemented today:** only `upload.parsed` is ever written, from `src/components/upload.py`'s
`_record_upload_events` (one `events` row per indexed document). The `node.<name>` /
`node.<name>.error` shapes are **not emitted as `events` rows either** — a node run is recorded
as a `NodeEvent` and persisted per assistant message in `messages.node_trace_json`, which is not
the same column as the `events` stream. Every other name above is aspirational vocabulary for a
sink that does not yet exist: nothing in `src/` emits `graph.start`, `retrieval.query`,
`synthesis.complete`, `numeric_prepass.trimmed`, `grounding.trimmed`, `refusal.issued` or
`session.switched`, so a reader greps the `events` table for them and finds nothing.
`numeric_prepass.trimmed` is the deterministic checker firing (`extra`: `{ "claim": "...",
"reason": "figure_not_in_span | period_mismatch" }`) — evidence in the UI that a wrong figure was
caught without a model call (ADR-010). It stays the documented name for what the vocabulary
*specifies*, not a claim about what a run writes.

## Node trace (`GraphState.node_trace`)

Appended by the `@trace_node` decorator (`src/utils/tracing.py`) wrapping every node:

```python
NodeEvent(node: str, started_at: float, duration_ms: float,
          tokens_in: int | None, tokens_out: int | None, error: str | None)
```

- persisted per assistant message in `messages.node_trace_json` (the `events` table is a
  separate sink and holds no node-trace rows today),
- rendered by the debug panel for **any** message the user selects, not just the latest.

## Debug panel contract

Sidebar section, collapsed by default. The panel renders the selected answer: its picker
leads with the live answer when there is one (`Current answer (live)`) and then lists the
session's stored answers, each labelled by its question, so a historical answer is inspected
by selecting it; a lone candidate renders with no control. A stored answer carries its
persisted node trace, grounded flag, refusal reason and citations — not the plan or the
retrieval scores, which the store does not keep — and those sections say "not recorded for
this answer" rather than drawing an empty block. For the selected answer:

1. **Plan** — `query_plan`: kind, sub_queries, required_sections, fiscal_period,
   missing_aspect.
2. **Path** — ordered node list with per-node ms and tokens; the route taken (simple /
   decomposed / refused) highlighted.
3. **Retrieval** — for each selected chunk: dense score, sparse score, fused RRF score, parent
   page range, section path, and a 200-char snippet. Sorted by fused score.
4. **Grounding** — the `grounded` flag (badge: grounded / ungrounded), the `refusal_reason`, the
   citation count, and the `planner_degraded` badge where the live answer carries it. It does
   **not** render verdicts: the deterministic pre-pass rejections and the per-claim LLM verdicts
   are not carried on the answer record — they surface only through the node events and the
   answer text — so the section says so with an explicit note rather than inventing a verdict
   count (ADR-007).

This panel is the demo artifact for the observability criterion: a reviewer can verify *why* an
answer was produced without reading logs.

## Metrics derived (no Prometheus)

| Metric | Computed from | Surfaced in |
|---|---|---|
| Per-node latency | `NodeEvent.duration_ms`, as the per-node ms the path list renders for one selected answer | debug panel (per selected answer; no p50/p95 aggregate, and no cross-session scope) |
| Tokens in/out per turn | gateway accounting lifted into `NodeEvent` | debug panel (per node row, for the live answer) |
| Retrieval score distribution | `retrieval_debug` fused scores | debug panel (per retrieved card: `dense · sparse · fused` text, plus the 0-based rank; **no sparkline**, and the fused score itself is absent from a stored answer) |
| Groundedness-failure rate | `grounded` flags on messages | not surfaced — the panel shows one answer's flag, the aggregate is not computed anywhere in the app |
| Refusal rate | `refusal_reason` values on messages | not surfaced as an aggregate — the panel names one answer's refusal reason |
| Path mix | `query_plan.kind` | debug panel (the route badge inferred from which nodes ran: simple / decomposed / refused — not a `kind` count, and only for the live answer whose plan is carried) |

## Logging rules

- `setup_logging(json=True)` at startup; one JSON object per line to stdout; Docker captures
  it (`docker compose logs app`).
- No secrets, no PDF text, no full context blocks in logs — counts and ids only. (The
  `events.payload_json` may hold chunk ids and scores, never document text.)
- Errors log at `error` with the typed error class name; expected degradations
  (`contextualization_failed`, `retrieval.rewrite_retry`) log at `warning`.

## Failure → user mapping

| Failure | User sees |
|---|---|
| Any node error | inline error banner: node name + "Retry" action (ADR-007 taxonomy) |
| Grounding trim | answer body with `(unsupported claim removed)` markers + `ungrounded` badge |
| Refusal | honest refusal naming coverage (time_period / topic / entity) |
| Missing index | "Upload a PDF to begin" empty state — never a blank chat |

**The invariant:** no code path renders an empty answer. Refusal, partial-with-markers, or
visible error — one of the three, always.

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
`numeric_prepass.trimmed` is the deterministic checker firing (`extra`: `{ "claim": "...",
"reason": "figure_not_in_span | period_mismatch" }`) — evidence in the UI that a wrong figure was
caught without a model call (ADR-010).

## Node trace (`GraphState.node_trace`)

Appended by the `@trace_node` decorator (`src/utils/tracing.py`) wrapping every node:

```python
NodeEvent(node: str, started_at: float, duration_ms: float,
          tokens_in: int | None, tokens_out: int | None, error: str | None)
```

- persisted per assistant message in `messages.node_trace_json`,
- mirrored into the `events` table,
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
4. **Grounding** — deterministic pre-pass verdicts first (`figure_not_in_span` /
   `period_mismatch` rejections, trimmed before any model call), then per-claim LLM verdicts
   (supported / partially / unsupported), with trimmed claims marked.

This panel is the demo artifact for the observability criterion: a reviewer can verify *why* an
answer was produced without reading logs.

## Metrics derived (no Prometheus)

| Metric | Computed from | Surfaced in |
|---|---|---|
| Per-node latency p50/p95 | `NodeEvent.duration_ms` | debug panel (session-scoped) |
| Tokens in/out per turn | gateway accounting in `NodeEvent` | debug panel |
| Retrieval score distribution | `retrieval_debug` fused scores | debug panel sparkline |
| Groundedness-failure rate | `grounded` flags on messages | journal §6 |
| Refusal rate | `refusal_reason` histogram | journal §6 |
| Path mix | `query_plan.kind` counts | journal §6 |

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

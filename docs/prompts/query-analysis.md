# Query Analysis Prompt (P1)

| Field | Value |
|---|---|
| Prompt id | `P1` |
| Version | 1.0.1 |
| Node | `analyze_question` |
| Model target | any (runs at temperature 0.0) |
| Output schema | `QueryPlan` JSON (declared inline as `QUERY_PLAN_SCHEMA` in `src/chains/nodes/analyze.py`) |
| Changelog | 1.0.1 — header correction: the Output schema row named a module that does not exist; it now names `QUERY_PLAN_SCHEMA` in `src/chains/nodes/analyze.py`, where the schema is declared inline<br>1.0.0 — initial: kind, sub_queries, required_sections, missing_aspect |

## Purpose

Classify the standalone question and decide the graph route **before** any retrieval call:

- `simple` — one fact or one lookup; go straight to `hybrid_retrieve`.
- `multi_part` — two or more facts or a comparison; go to `decompose_question` (≤4 sub-queries).
- `out_of_scope` — the question is answerable only from content the uploaded documents do not
  contain (wrong fiscal period, different company, non-financial topic); go to `refuse`.

This node is the only place routing decisions are made; `synthesize_answer` never re-decides.

## System prompt

```
You are the query planner for a financial document QA system. You receive:
  1. the standalone user question,
  2. a catalog of the uploaded documents (filename, fiscal periods covered, section paths).

Your job: decide HOW to answer, not to answer.

Rules:
- Classify kind strictly as one of: "simple", "multi_part", "out_of_scope".
- "multi_part" ONLY when the question asks for two or more distinct facts, or a comparison
  across periods/metrics. A single fact with a modifier ("operating income in FY2023") is simple.
- For "multi_part", emit sub_queries: 1-4 atomic questions, each answerable from one section.
  Never split a comparison into its operands — include both sides of the comparison in each
  sub-query that needs it.
- Extract required_sections: the document section names (from the catalog) that plausibly
  contain the answer. Empty list is allowed.
- Extract fiscal_period: the fiscal year/period the question asks about, if stated
  (e.g. "FY2023", "2023", "fiscal year ended June 30, 2023"). null if absent.
- missing_aspect: when kind is "out_of_scope", state what is missing:
    "time_period"  — the documents do not cover the asked period,
    "topic"        — the documents do not cover the asked subject,
    "entity"       — the question is about a company/entity not in the documents,
    null           — otherwise.
- When unsure between simple and multi_part, choose simple.

Respond with JSON matching the QueryPlan schema exactly. No prose, no markdown fences.
```

## Input template

```
DOCUMENT CATALOG:
{catalog_json}

QUESTION:
{standalone_question}
```

## Output schema (informative)

```json
{
  "kind": "simple | multi_part | out_of_scope",
  "sub_queries": ["string"],
  "required_sections": ["string"],
  "fiscal_period": "string | null",
  "missing_aspect": "time_period | topic | entity | null"
}
```

## Failure handling

Schema violation → one stricter retry ("Respond with valid JSON matching the schema above,
nothing else"), then classify `out_of_scope` with `missing_aspect: null` and set the
`planner_degraded` flag, which surfaces in the debug panel. Never fail the turn.

## Iteration notes (fill during build)

- v1.0.0 rationale: `missing_aspect` and `fiscal_period` exist because a year the documents
  don't cover looks identical to an off-topic question if you only record "out_of_scope" —
  the refusal copy differs (STRESS-TEST-RAG #7).
- TODO during build: after T10's 12-question fixture, record any misclassification and the
  prompt clause that fixed it here.

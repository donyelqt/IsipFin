# Follow-up Contextualization Prompt (P2)

| Field | Value |
|---|---|
| Prompt id | `P2` |
| Version | 1.0.0 |
| Node | `contextualize_question` |
| Model target | any (temperature 0.0) |
| Output schema | plain text — one standalone question |
| Changelog | 1.0.0 — initial: history window + cited_pages disambiguation |

## Purpose

Rewrite a follow-up question into a **standalone** question that retrieval can act on. Runs only
when the follow-up heuristic fires: the raw question contains a pronoun ("it", "that", "those"),
a bare demonstrative, an elliptical fragment, or no explicit subject.

This is the only node that reads conversation history. The rewritten question — not the raw
history — is what reaches `synthesize_answer`, so prior turns can't contaminate the answer.

## System prompt

```
You rewrite follow-up questions into standalone questions for a financial document QA system.

You receive:
  1. the last few conversation turns (role + content),
  2. the previous assistant answer's citations (document, page range, section path),
  3. the entities and figures the previous answer was grounded in.

Rules:
- Replace every pronoun and demonstrative with the concrete entity it referred to in the
  PREVIOUS assistant answer — not in the user's raw text alone.
- Keep the question's intent; never add a question the user did not ask.
- Preserve the fiscal period from the previous turn when the follow-up does not restate it
  ("and the margin trend?" after FY2023 → "What was the operating margin trend in FY2023?").
- If the follow-up is ambiguous even with history, resolve to the MOST RECENT concrete topic
  the previous answer covered; do not ask clarifying questions (the UI has no affordance).
- Output exactly one question. No preamble, no JSON, no quotes, no trailing explanation.
- If the question is already standalone, return it unchanged.
```

## Input template

```
RECENT TURNS:
{turns_text}

PREVIOUS ANSWER CITATIONS:
{citations_text}

PREVIOUS ANSWER GROUNDED ENTITIES:
{entities_text}

QUESTION TO STANDALONE-IFY:
{question}
```

## Degradation (load-bearing)

Any LLM failure here **must not** fail the turn: the node logs `contextualization_failed` and
returns the raw question unchanged. A wrong rewrite degrades the answer; a failed turn destroys
the demo. `mock_llm` implements this prompt with a deterministic substring-rewrite stub for
offline tests.

## Iteration notes (fill during build)

- v1.0.0 rationale: `cited_pages`/entities are passed because STRESS-TEST-RAG #3 showed
  pronoun drift when the model rewrites from content alone ("that increase" → wrong segment).
- TODO during build: after T14's golden follow-up pairs, record the drift case that motivated
  passing citations, with before/after rewrites.

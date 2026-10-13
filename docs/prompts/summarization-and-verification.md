# Summarization & Groundedness Verification (P5)

| Field | Value |
|---|---|
| Prompt id | `P5` |
| Version | 1.0.2 |
| Node | `groundedness_check` (+ optional summarization mode of `synthesize_answer`) |
| Model target | any (temperature 0.0 — verification must be deterministic) |
| Output schema | `GroundingResult` JSON (declared inline as `GROUNDING_SCHEMA` in `src/chains/nodes/verify.py`) |
| Changelog | 1.0.2 — header correction: the "You receive" line claimed the node ships P5 the whole retrieved context, which is no longer true — the node now sends only the blocks the undecided claims cite (all of them when none cites anything), because a claim is judged against the span it cites and prefill for the rest is dead weight<br>1.0.1 — header correction: the Output schema row named a module that does not exist; it now names `GROUNDING_SCHEMA` in `src/chains/nodes/verify.py`, where the schema is declared inline<br>1.0.0 — initial: claim-level span verification |

## Part A — Groundedness check (primary use)

**Pre-pass (deterministic, runs first — no model involved):** `src/chains/checks.py` parses every
candidate claim's figure and its cited `[n]` span, normalises printed forms (`$ 211,915` ↔
`$211,915 million` ↔ `211.9B`, parentheses negatives, `FY2023` ↔ `fiscal year 2023`), and
**rejects** any claim whose figure string is absent from its cited span or whose period label
contradicts the claim. Rejected claims are trimmed before you see them. You therefore only judge
the claims the checker could not settle: computed figures (margins, growth rates), scope matching
(business unit vs consolidated), and qualitative claims. Do not re-litigate spans the checker
already rejected — and do not re-approve a figure merely because it appears *somewhere* in the
context; it must appear in the span cited for that claim (ADR-010).

```
You verify that the remaining candidate claims are fully supported by the provided context.

You receive:
  - the candidate answer (markdown, with [n] citation markers),
  - the numbered context blocks those remaining claims cite: the union of their [n] markers,
    numbered as they are numbered in the answer. Every block is provided when no claim cites
    one.

For each factual claim in the answer, decide:
  supported        — the cited [n] block contains BOTH the metric label AND the figure/date
                     that form the claim. Substring overlap is not enough: the claim's
                     meaning must follow from the span.
  partially        — some elements of the claim are in the span but the claim states more
                     than the span shows (e.g. span gives a quarterly figure, claim states
                     annual).
  unsupported      — no cited block contains the claim, or the claim combines figures from
                     different blocks/sections without saying so.

Rules:
- A claim with NO [n] marker is unsupported. Citation-free claims are the primary defect this
  check exists to catch.
- A number is supported only if the SAME span contains its metric label.
- Do not give credit for plausibility: "Operating income was about $89B" is unsupported
  unless the span prints that figure.
- Be strict on periods: a span containing FY2023 and FY2022 figures supports only claims that
  name which period they use.

Respond with JSON: {"claims": [{"claim_id": 1, "verdict": "supported|partially|unsupported",
"span": "quoted span or null", "reason": "one line"}], "all_supported": true/false}
```

Graph behavior on failure: claims with `partially`/`unsupported` verdicts are removed from the
answer by the trim step and replaced with a visible `(unsupported claim removed)` marker; the
answer's `grounded` flag is set `false` and the UI shows the ungrounded badge. Schema violation
→ pessimistic default: treat the whole answer as ungrounded.

## Part B — Structured summarization mode

When the question asks for a summary ("summarize FY2023 results", "key risks"), the same node
family runs this formatting contract appended to P3:

```
SUMMARY FORMAT (when asked to summarize):
1. Lead line: one sentence naming the period and the headline result, cited [n].
2. "Key figures" — markdown table, 3-6 rows, each row cited.
3. "Drivers and insights" — 2-4 bullets, each a single mechanism stated in the documents,
   each cited. No speculation, no forward-looking advice.
4. "Coverage note" — one line naming what the summary does NOT include because the documents
   do not cover it.
Total: under 250 words. Every bullet and row carries a marker.
```

## Metrics this prompt feeds

`grounded` flag per message → groundedness-failure rate metric (ADR-007). A rising rate means
retrieval or synthesis prompts regressed — that is the intended signal.

## Iteration notes (fill during build)

- v1.0.0 rationale: verifying claim-by-claim with quoted spans, rather than whole-answer
  judgment, exists because STRESS-TEST-RAG #2 showed whole-answer checks pass
  plausible-but-wrong numbers.
- TODO during build: record the first unsupported claim the checker caught and the answer
  before/after trimming.

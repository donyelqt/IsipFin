# Prompt Library

> Prompts are versioned assets, not string literals buried in code. Each lives in its own file
> with a version header; `src/chains/nodes/*.py` loads them via
> `src/utils/prompts.py` (`load_prompt("P3")`) so a prompt change is a diffable file change.

## Inventory

| Id | File | Node | Role | Failure exposure |
|---|---|---|---|---|
| P1 | `query-analysis.md` | `analyze_question` | Classify + plan the route | Wrong classification → wasted retrieval or wrong refusal |
| P2 | `follow-up-contextualization.md` | `contextualize_question` | Rewrite follow-ups standalone | Pronoun drift → wrong-entity answer |
| P3 | `answer-synthesis-with-citations.md` | `synthesize_answer` | Grounded, cited answer | Wrong-year figure, uncited claims — the headline risk |
| P4 | `table-and-number-formatting.md` | appended to P3/P5 | Number/table formatting contract | Unverifiable restatements, unreadable tables |
| P5 | `summarization-and-verification.md` | `groundedness_check` (+ summary mode) | Claim-level verification | Hallucinated plausible numbers passing review |

## Versioning protocol

1. **One change per version.** A version bump addresses one observed failure; the changelog
   line names the failure it fixes (e.g., "1.1.0 — added period-column rule after FY2024
   comparative figure appeared in an FY2023 answer").
2. **Semver in the header.** `major` = output contract change (schema/format), `minor` = rule
   added, `patch` = wording clarification. Bump on every edit, no exceptions.
3. **Prompt files are the source of truth.** Node code contains no inline system prompts; if a
   prompt needs to change for a demo, that's a version bump, not a code edit.
4. **Evidence-gated bumps.** Every minor bump must reference a golden-QA case or a debug-panel
   observation that motivated it. No speculative prompt edits — that is how prompts rot.
5. **Load, don't copy.** `load_prompt(id)` reads the file at startup; tests assert the loaded
   prompt's `version` matches the node's expected minimum, so a stale deployment fails loudly.

## Change log (cumulative)

| Id | Version | Date | Change | Motivated by |
|---|---|---|---|---|
| P1 | 1.0.0 | 2026-10-13 | Initial: kind/sub_queries/required_sections/fiscal_period/missing_aspect | design (STRESS-TEST-RAG #7) |
| P2 | 1.0.0 | 2026-10-13 | Initial: citations + grounded entities as disambiguation context | design (STRESS-TEST-RAG #3) |
| P3 | 1.0.0 | 2026-10-13 | Initial: citation constraint + year-aware extraction contract | design (STRESS-TEST-RAG #1) |
| P4 | 1.0.0 | 2026-10-13 | Initial: number/table contract | exam "table formatting" bullet |
| P5 | 1.0.0 | 2026-10-13 | Initial: claim-level span verification | design (STRESS-TEST-RAG #2) |

Rows below v1.0.0 lines are filled during the build — they are the evidence the journal's
prompt-engineering section cites. If a prompt is never iterated, that is also worth a line:
"unchanged after golden QA" is a result.

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
| P1 | 1.0.1 | 2026-10-13 | Header correction: Output schema row named a module that does not exist; now names the inline `QUERY_PLAN_SCHEMA` declared in `src/chains/nodes/analyze.py` | docs review of the header (metadata only, no body change) |
| P5 | 1.0.1 | 2026-10-13 | Header correction: Output schema row named a module that does not exist; now names the inline `GROUNDING_SCHEMA` declared in `src/chains/nodes/verify.py` | docs review of the header (metadata only, no body change) |
| P3 | 1.0.1 | 2026-10-13 | Added grounding rule 3a: a figure quoted from a multi-period context block must name its period in the sentence itself, because a fully grounded claim naming no period is trimmed by the verifier | live provider run: FY2023 revenue answer returned as "(unsupported claim removed)", grounded=False, citations=0 (groundedness_check trimmed "$ 211,915 million [3]") |
| P5 | 1.0.2 | 2026-10-14 | Header correction: the "You receive" line described the payload as the same numbered context blocks synthesis used; it now names the union of the blocks the undecided claims cite, with every block provided when no claim cites anything | live provider run: the verifier shipped all 7 retrieved blocks (21.0 KB, ~6,900 prompt tokens, 20.7s prefill-dominated call) for an answer citing only one; the payload was trimmed in `src/chains/nodes/verify.py` |
| P2 | 1.0.2 | 2026-10-14 | Added metric-fidelity rules: when the previous answer carried figures, the rewrite must keep the same metric family (dollar margin vs percentage margin are different questions) and must not switch units or scope — the follow-up's figures are what the answer depends on | golden case q07 `And the margin trend?`: repeated keyed runs at temperature 0 rewrote the follow-up into a gross-margin percentage question, so synthesis returned percentages with neither expected figure ($88,523, $211,915) present, and another run trimmed the answer to "(unsupported claim removed)"; the systematic loss is the metric-family switch, the provider's residual run-to-run variation is not claimed fixed |
| P4 | 1.0.1 | 2026-10-14 | Corrected rule 1's citation placement: rule 1 told synthesis "a source note under the table lists the [n] markers for the rows it covers", a placement `checks.split_units` cannot read — the table block ends before the note, so its rows inherit no marker and each is trimmed as `figure_not_in_span`. It also contradicted rule 4, which already prescribed the placement that works. Rule 1 now points at the table's own rows, per rule 4 | golden case q03 `Compare revenue and operating income between FY2023 and FY2022`: the real run placed every marker in the source note below the table, so both body rows were trimmed (`figure_ok: False`) and the answer rendered `\|\|(unsupported claim removed)\|\|` twice. The contract was on the prompt side, so the fix is a prompt bump, not a loosened checker |

Rows below v1.0.0 lines are filled during the build — they are the evidence the journal's
prompt-engineering section cites. If a prompt is never iterated, that is also worth a line:
"unchanged after golden QA" is a result.

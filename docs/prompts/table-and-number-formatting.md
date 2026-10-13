# Table & Number Formatting Prompt (P4)

| Field | Value |
|---|---|
| Prompt id | `P4` |
| Version | 1.0.1 |
| Node | shared formatting contract used by `synthesize_answer` and the summarizer |
| Model target | any |
| Output schema | markdown (tables + prose) |
| Changelog | 1.0.1 — corrected rule 1's citation placement: "a source note under the table lists the [n] markers" prescribed a placement the verifier's `split_units` cannot read (the table block ends before the note, so the rows inherit no marker and every row is trimmed as `figure_not_in_span`), and it contradicted rule 4, which already prescribed the placement that works — a marker in each row's label cell, or per-cell. Motivated by golden case q03 `Compare revenue and operating income between FY2023 and FY2022`: the real run's two table rows were both trimmed, leaving `figure_ok: False` and the answer `&#124;&#124;(unsupported claim removed)&#124;&#124;`<br>1.0.0 — initial |

## Purpose

A shared formatting contract for how numbers and tables appear in answers. Loaded as an
appendix to P3 (synthesis) and P5 (summary) so one rule set governs both — this is the
exam's *"improving text and table output formatting"* bullet.

## Rules (verbatim appendix to other prompts)

```
FORMATTING CONTRACT (tables and numbers):

1. TABLES: when comparing 2+ metrics or periods, answer with a markdown table:
   | Metric | FY2023 | FY2022 |
   |---|---|---|
   | Revenue | $ 211,915 | $ 198,270 |
   - Row label first, periods as columns; align numbers right; keep header row.
   - Put the [n] marker in the table itself — in each row's label cell when one block
     supports the whole row, per-cell when the cells come from different blocks (rule 4).
     A source note listed *below* the table does not cite the rows: the verifier reads a
     table's markers from inside the table, so a note underneath leaves every row
     uncited and each row is trimmed as an unsupported claim.
   - Never split a table the context provides into prose, and never rebuild a table the
     context does not fully support — partial tables must name the missing cells "(not
     disclosed)".

2. NUMBERS: reproduce exactly as printed — "$ 211,915" not "$211.915B" or "about $212B";
   "9.68" not "$9.7". Parentheses are negatives. Units: use the document's unit ("in millions")
   stated once, not per number. Percentages: one decimal place, copied from the document.

3. ORDER: when listing periods, newest first (FY2023 before FY2022).

4. CITATIONS IN TABLES: place the [n] marker in the row label cell when one block supports the
   whole row; place it per-cell when cells come from different blocks.

5. PROSE: no bullet lists for single facts; no bold for every phrase (bold at most the headline
   figure); no "Sure!"/"Certainly!" openers; answer in the analyst's voice.

6. UNITS & SCALE: if the question asks for billions but the document prints millions, convert
   ONLY arithmetically exact values and show both ("$ 211,915 million (~$212 billion)").
   Never convert approximate values.
```

## Anti-patterns (each one burned a build iteration)

| Anti-pattern | Why it fails | Rule that prevents it |
|---|---|---|
| "$211.9B" restatement | Rounding loses verifiability against the printed table | Rule 2 |
| Prose narrative of a 4-row table | Unreadable, hard to cite | Rule 1 |
| Bold on every number | Noise; the grader can't find the headline figure | Rule 5 |
| FY2024 comparative column quoted for FY2023 | Silent wrong-year answer | Rule 1 + P3 rules 2–4 |
| "About $212 billion" alone | Conversion without the printed figure | Rule 6 |
| The table's `[n]` markers in a source note below it | The verifier reads a table's markers from inside the table, so a note below it leaves every row uncited and each row is trimmed as `figure_not_in_span` | Rule 1 + rule 4 |

## Iteration notes (fill during build)

- TODO during build: record the first answer that violated each rule and the prompt edit that
  fixed it — this is direct evidence for the journal's prompt-engineering section.

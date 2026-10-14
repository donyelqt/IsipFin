# Project Journal — IsipFin

> **Purpose and honesty note.** This journal is the exam's weighted deliverable. It was written
> *before* implementation as a decision record, and each section marks what is settled now
> versus what must be filled with real build data. **Every slot is now filled** — the
> `[FILL AFTER BUILD]` markers are gone, and each was replaced with data read from the artifact or
> command that produced it (§7 names the source for every number). No measurement in this journal
> is invented; where a number is an estimate it says **estimate**, and where a number could not be
> measured in this environment the row says so and why instead of carrying a plausible figure.

## 1. What this is

A Streamlit chatbot that ingests financial PDFs (reference corpus: Microsoft FY2023 Annual
Report, fiscal year ended June 30, 2023) and answers natural-language questions with concise,
citation-backed answers. Orchestrated by LangGraph; hybrid FAISS + BM25 retrieval over
parent-child chunks; conversations persist in SQLite; packaged as one Docker container.

Exam context: 48-hour take-home for the GenAI Engineer role at Thinking Machines, against
reviewers `crcheng`, `levymedina`, `jgcosme`.

## 2. How I worked (AI Use Principles compliance)

TM's principles: **Own Your Thinking**, **Use AI Intentionally**, **Own the Output**.

- **Own thinking — what I decided myself, before using AI:** the capability map (what modules
  exist and why), the parent-child chunking decision, the score-gate-with-single-retry bound,
  the "history reaches only contextualize, never synthesis" rule, the kill-switch ordering, and
  the failure-mode list in `docs/STRESS-TEST-RAG.md`. These decisions exist as written design
  documents dated before any build.
- **Use AI intentionally — where AI was used and why:** (a) drafting this documentation set
  from the frozen design, where the bottleneck was articulation, not judgment; (b) I explicitly
  did *not* use AI for: the decision of which chunk sizes to start with (reasoned from the
  corpus structure), the exam requirement mapping (read from the brief directly), or the
  decision to reject vectorless RAG (argued from constraints, see ADR-003). Where AI output was
  plausible but unverified (version-specific library facts, corpus figures), I verified
  against primary sources: LangGraph's install docs, and Microsoft's own FY2023 annual report
  page — the FY2023 figures in this repo ($211,915M revenue / $88,523M operating income) are
  quoted from Microsoft's published report, and I re-read the report page rather than trusting
  recall.
- **Own the output:** every ADR and prompt in this repo reflects a decision I made; the journal
  records what was chosen, what was rejected, and why.

## 3. Architectural decisions (each argued in its ADR)

| Decision | ADR | One-line reason |
|---|---|---|
| LangGraph `StateGraph` with 7 typed nodes | `docs/adr/ADR-001-...` | Data-dependent routing, bounded retries, per-node tracing — a static chain can't express it |
| Hybrid FAISS + BM25, RRF fusion, parent-child chunks | ADR-002 | Exact line-item tokens (`Note 13`, `$88,523`) and paraphrase both matter; parents fix table fragmentation |
| Vectorless/tree RAG **rejected**, idea mined | ADR-003 | Brief mandates a vector DB; tree navigation costs 3–8 sequential LLM calls per answer |
| pypdf + pdfplumber markdown tables + section paths | ADR-004 | Tables are the exam's formatting criterion; markdown makes them retrievable and LLM-readable |
| One OpenAI-compatible gateway + `MOCK_LLM` | ADR-005 | Reviewer's key setup is unknown; keyless demo must still work |
| SQLite chat store, history windowed at 6 turns | ADR-006 | Persistence + privacy-of-context: history never contaminates synthesis |
| JSON events + `node_trace` + debug panel | ADR-007 | The interesting failures are plausible wrong answers, which only in-product tracing exposes |
| Single Docker image + compose volume | ADR-008 | `docker compose up` is the grader's cold-start path |
| Streamlit native UI — no React, no FastAPI | ADR-009 | Brief names Streamlit; one language surface; the graph stays importable so an API is additive. Quality comes from the no-logic-in-UI rule, not the framework |
| Deterministic neuro-symbolic pipeline (agentic workflow, single DAG) | ADR-010 | The DAG already existed; the missing piece was addressing the axis confusion and adding a deterministic numeric pre-pass so wrong-year figures die without a model call |

## 3b. The redo (2026-10-13) — what changed and why

After the initial 31-doc plan set was complete, the meta-skill governing this work was updated
with three additional bindings. Rather than retrofitting silently, I re-audited the whole set
(`docs/PLAN-REVIEW.md`) and found one real defect plus three missing artifacts:

1. **The defect — a missing task.** `decompose_question` existed in the architecture and ADR-001
   but had **no task, no file, no test row** — the exam's headline capability ("decomposing
   complex questions") would have fallen through a planning gap. Fixed as T10b across
   PLAN/TODO/TIMELINE/TESTING/DELIVERABLES/GRADING and `src/chains/nodes/decompose.py` in
   REPO-STRUCTURE.
2. **Missing artifacts:** `docs/DEFINITION-OF-DONE.md` (per-task gate), `docs/CONSTRAINTS.md`
   (numbered quality bar + floor guard), `docs/LIFECYCLE-PLAYS.md` (three worksheets; play
   verdict: **Extend** under supply/cost-side pressure, channel diagnosis says the design is
   sound), `docs/DEPENDENCY-PINS.md` (reproducibility — every version read off PyPI this
   session: streamlit 1.65.0, langgraph 1.2.14, faiss-cpu 1.15.1, pypdf 6.20.1, pdfplumber 0.11.10,
   openai 3.29.0, rank-bm25 0.2.2, pytest 9.1.1, ruff 0.17.0, black 26.10.1, python-dotenv 1.2.4).
3. **Assumption ledger** — nine assumptions that were implicit are now written down
   (`docs/PLAN-REVIEW.md`), per the updated skill's assumption-surfacing behavior.

Honesty note for the interview: this redo was authored by me in the main session, not by
subagents — the subagent pool returned 429 rate limits (retry-after ~6.5h) on all five dispatched
jobs and on a retry batch earlier. I probed with a lightweight job, confirmed the pool was
intermittently alive, retried the full batch, and when it failed again I did the work inline
rather than declaring [blocked] or fabricating subagent output. The lifecycle/constraints/audit
reasoning above is mine; the skill templates supplied the structure.

## 3c. Three defects found by verifying the UI in a real browser (2026-10-13)

All three were found the same way: by driving the running app in Chromium and reading *computed*
values back out, rather than trusting that the code looked right. All three produced failures
that looked like success from the code, and two of them were silently breaking the demo. This
section is also a correction of my own first measurement, which I recorded as verified when it
was only half-verified — see 3c.1.

### 3c.1 The design system applied for exactly one paint

This is the root cause of the original complaint that the UI "looks weird", and it took two
stacked bugs to produce, which is why the first pass at diagnosis missed it.

**Bug 1 — `st.html()` drops a `<style>` block.** `theme.inject_theme()` built the whole CSS
variable set and emitted it through `st.html()`. Measured: `getComputedStyle(body).backgroundColor`
was Streamlit's default white, the font was the default system stack, and the accent variable
resolved to nothing. No console error, no exception, no warning. `st.html()` is the *documented,
recommended* modern API for this, and it is the one that silently fails.
Fix: emit through `st.markdown(block, unsafe_allow_html=True)`. Re-measured: `--isip-accent` =
`#FF7A1A`, background `#0A0A0B`, body font IBM Plex Sans.

**Bug 2 — a session-state guard made the emission one-shot.** After fixing bug 1 I measured the
theme as applied and moved on. That measurement was taken on a *fresh load* and was therefore
incomplete: it was true of the first paint only. The real behaviour, measured across a rerun:

| | style nodes | `--isip-accent` | `.isip-mark` width |
|---|---|---|---|
| first page load | 1 (25,879-char stylesheet) | `#FF7A1A` | 30px |
| after any one unrelated click | **0** | empty | auto |

`inject_theme()` guarded on an `INJECTED_KEY` session-state flag. Streamlit removes any element a
rerun does not re-send — so the stylesheet element vanished on the first rerun, and the guard
then guaranteed it was never emitted again. The design system rendered for one paint; from the
user's first interaction onward the app was bare Streamlit chrome. That is exactly what "looks
weird" describes: moments of a deliberate design, then a fallback to default styling.
Fix: emit on every run; the constant and the guard are deleted (no other consumer existed).

**Bug 3 — every keyed-element selector in the stylesheet was dead.** Keyed elements carry
`class="stElementContainer element-container st-key-isip-nav-documents st-emotion-cache-…"` —
the key class is **not** the first token, so every `[class^="st-key-isip-…"]` rule matched
nothing. Measured: `border-left-width: 0px` on `.st-key-isip-answer-6`, i.e. the answer/question
role borders had never once rendered. Switched to `[class*="st-key-isip-…"]`. A same-shaped dead
selector (`section[data-testid="stSidebarContent"]` — it is a `div` in Streamlit 1.65) took the
rail's hairline with it.

**The lesson, and why I am keeping the wrong first answer in this journal:** a UI-layer claim is
not verified by reading the code, and it is not verified by a single measurement either. My first
measurement was *real* — I read actual computed values out of a real browser — and it was still
wrong, because I measured the path that happened to work. Verification has to exercise the state
the user will actually be in, which after any interaction is "the second run", not "the first".
The correct probe here was not "is the accent applied after load" but "is the accent still applied
after a rerun". I have left the sequence in rather than presenting the clean answer, because the
sequence is the part worth carrying forward.

**Fourth-order finding, same session:** the rail's CSS `width` rule lost to Streamlit's inline
style. `st.set_page_config(initial_sidebar_state=...)` accepts an integer meaning "auto behaviour
at this pixel width" (documented range 200-600), which is the supported way to size it — the value
is now `260`, the reference layout's rail width, instead of a CSS rule that was silently
overridden.

### 3c.2 `.env` was read by nothing while four documents told the reader to create it

`README.md`, `docs/SPEC.md` and `docs/RUNBOOK.md` all open with `cp .env.example .env`.
`grep -rn "load_dotenv\|dotenv" app.py src/ scripts/` returned **nothing**. The file the setup
instructions tell you to create was inert.

This mattered more than a missing convenience. The repository's real `.env` sets `MOCK_LLM=1` —
the keyless offline demo the README promises. Because nothing read it, the app ran with **no
mock mode and no key**: the gateway attempted a real HTTP call against an empty key, got a 404,
and the planner degraded into `planner_degraded` + an honest out-of-scope refusal. That is the
worst shape a bug can take — the graph's own error handling (ADR-005/ADR-007) was working
perfectly and produced a *correct-looking* refusal, which masked a setup defect. I only caught it
because the refusal text did not match what the same question returned in a keyed environment.

**Fix:** `src/utils/settings.py::load_env()`, called first in `app.py::main()` and first in
`scripts/build_index.py::main()` / `scripts/eval_rag.py::main()`. It resolves the repo root from
its own file rather than the working directory, is idempotent, and uses `override=False` so a
real environment variable always beats the file. Verified by writing a sentinel into `.env` and
reading it back through the loader (the real `.env` was backed up and restored — 1167 bytes,
sentinel count 0 after).

## 4. Lineage: what this design inherited from prior work

This is not the first system of this shape I have built. I have previously built a larger
multi-agent system — 19 agent classes on the same LangGraph DAG idiom, with parallel spawning and
conditional fan-out. Three things about it matter here:

1. **Its reliability never came from agents being autonomous.** It came from **deterministic
   orchestration plus symbolic guardrails that override neural judgment** — fixed scoring tiers,
   threshold rules that supersede the model's verdict. That is the single most transferable
   lesson, and it is why this design's reliability budget went into `src/chains/checks.py` and the
   bounded DAG (ADR-010) rather than into agent count.
2. **Its unit of separation is copied here.** Each node in this system is a single-responsibility
   unit with its own prompt and its own trace — the same pattern as those agent classes, minus the
   fleet's coordination overhead. What differs is scale, not structure.
3. **The verification here is a deliberately cheaper approximation of what worked at that scale.**
   The larger system verified generated claims with NLI entailment; this one verifies
   printed-figure presence deterministically and leaves entailment-style judgement to the LLM
   verifier. Same instinct — *verify after generating* — sized to the problem.

Named generically rather than by link, on purpose: the exam is graded on what is in this repo, and
the design has to stand on its own. The parallel is recorded because it explains the shaping of
this design.

### 4.1 Is the prior system "multi-agent"? Yes — and here is the precise sense

Both systems have a **fixed schedule** (a DAG decides what runs next; no agent negotiates the
workflow). Both have **workers that decide their own work** (a ReAct agent generating queries,
theme agents drafting insights, an LLM node composing an answer). That combination — deterministic
orchestration with autonomous workers — is a **recognized pattern, not a diluted claim**: it is
what mature harnesses ship (see the external witness in `docs/adr/ADR-010`), and it is the same
shape as this exam design, just wider.

So the honest labels are:

- **The prior system: multi-agent.** Earned — 19 agent classes, each with its own `run()`/`score()`
  method, dynamic spawning, real parallel execution. It is multi-agent in both the vocabulary sense
  and the structural sense, and nothing about the fixed schedule weakens that.
- **This exam: not multi-agent.** Four LLM nodes, one document, no dynamic spawning. It is the same
  *pattern* at a smaller scale.

Same idiom, different scale, one honest label each. Claiming "multi-agent" here would spend the
prior system's real claim on nothing — and a reviewer who reads both would see the inflation.

## 5. The two hard problems and how I decided

### 5.1 Financial figures vs vector similarity

**The problem:** annual reports put FY2023, FY2022, and (in the next year's report) FY2024
comparatives in adjacent table columns. Cosine similarity smears them together; the model then
quotes the wrong year confidently.

**What I rejected:** pure dense retrieval (ADR-002); pure vectorless tree navigation (ADR-003);
whole-answer "is this grounded?" verification (passes plausible wrong numbers).

**What I chose, in order of the stack:** BM25 catches exact line items → parent chunks carry the
full table (never cut mid-table) → the synthesis prompt's *year-aware extraction contract*
(P3 rules 2–4: a number is quotable only when the same span contains metric label AND figure,
with the column header compared against the asked period) → claim-level verification (P5) as
the net under it.

**Measured result — golden QA, `MOCK_LLM=1`, both year canaries clean.** The table below quotes
the claim text `groundedness_check` logged *before* it trimmed, which is the only place the
graph's own words about year attribution survive in mock mode:

| Case | Canary verdict | Pre-trim claim as the verify node logged it |
|---|---|---|
| q01 `What was Microsoft's revenue in FY2023?` | **clean** (asked FY2023; FY2022 figure `$198,270`) | `Microsoft's FY2023 revenue was **$211,915 million** ($211.9 billion), up from $198,270 million in FY2022 [1] [2].` |
| q02 `What was operating income in fiscal year 2023?` | **clean** (asked FY2023; FY2022 figure `$83,383`) | `FY2023 operating income was **$88,523 million**, versus $83,383 million in FY2022 [1] [2].` |

Both claims attribute each figure to its own year — FY2023's number is asserted for FY2023 and the
comparative is attributed to FY2022. Aggregate: `wrong_year_contamination 0/3+0 PASS`. (The same
two claims were *trimmed* downstream for `figure_not_in_span` — that is the mock's arbitrary
citation mapping, not a year error; §7 explains it.)

**Prompt iteration record: P3 is at 1.0.1.** Its changelog reads "1.0.1 — new grounding rule: a
figure quoted from a multi-period context block must name its period in the sentence itself
(verified claim \"$ 211,915 million [3]\" was trimmed as unsupported because the income-statement
block spans FY2023–FY2021 and the sentence named no period)" over "1.0.0 — initial: citation
constraint + year-aware extraction contract." There is no earlier version of the *year* rule whose
failure produced this one, and the reason is a design choice rather than luck: ADR-010 put the year
rule in the *symbolic* layer first — the deterministic pre-pass in `src/chains/checks.py` rejects a
period-label mismatch without any model call — so the prompt's rule is a second, independent net
over a rule that is already testable in isolation. The evidence that the design holds is the canary
table above and `tests/test_checks.py`'s FY2024-claimed-as-FY2023 case, not a diff in
`docs/prompts/`. The v1.0.1 rule 3a (name the period in the sentence that carries the figure) came
from the other layer — a fully *grounded* claim naming no period, which the pre-pass could not
settle because it had no period to contradict — and §6 quotes that before/after exactly.

### 5.2 Follow-ups without contaminating answers

**The problem:** "and the margin trend?" is meaningless without history, but handing the model
raw chat history makes it *continue the conversation* instead of *grounding in the document* —
prior-turn numbers leak into new answers.

**What I chose:** exactly one node (`contextualize_question`) reads history (last 6 turns +
previous answer's citations/entities), produces a standalone question, and passes nothing else
downstream. Synthesis sees only the standalone question + retrieved chunks. Degradation is
explicit: if contextualization fails, the raw question proceeds — a degraded answer beats a
dead turn.

**Measured result — golden QA, the two follow-up cases.** The harness replays each prior turn pair
as real `Turn` objects, so this exercises the actual disambiguation contract (`cited_pages` +
`entities` from the previous *answer*, per P2's history contract) rather than a transcript string:

| Case | History replayed | Retrieval outcome | Verdict |
|---|---|---|---|
| q07 `And the margin trend?` | 2 turns — 1 prior pair, carrying the earlier answers' `cited_pages`/`entities` | expected figures **present** in the fused top-5 — observed `True` | year canary clean |
| q08 `Which segment drove that?` (the pronoun-drift canary) | 4 turns — 2 prior pairs | expected `Intelligent Cloud` absent from fused top-5 — observed `False` | → see below |

**First drift case observed: none reached an answer.** There is no recorded instance of a follow-up
being answered against the wrong antecedent, so the honest report is that the canary has never
fired rather than that a drift was caught and fixed. What the runs *do* show is the mechanism
behaving: `contextualize_question` is the only node any history reaches, and the answer node sees a
standalone question plus retrieved chunks — the q08 pre-trim text was
`(unsupported claim removed) No further figure is derivable from them.`, i.e. the synthesiser
working from the rewritten question rather than continuing the conversation.

The one limitation this exposes is a *retrieval* one, not a leakage one: on q08 the segment figure
did not surface in the fused top-5 (mock mode's sha256 embedder carries no semantic signal, and the
segment table sits in MD&A prose). That is the retrieval-metric row §7 reports as `N/A` in this
mode, and it is why the follow-up path's *answer* quality needs a keyed run to score.

## 6. Iterative refinement (prompt engineering evidence)

Each prompt file carries its own version header and iteration notes; the library protocol is
`docs/prompts/README.md`.

**The record: every prompt has been revised, and every bump is in the file's own changelog.**
The versions on disk:

| Asset | Version | Changelog motivation (verbatim from the file) |
|---|---|---|
| `P1` `query-analysis.md` | 1.0.1 | "1.0.1 — header correction: the Output schema row named a module that does not exist; it now names `QUERY_PLAN_SCHEMA` in `src/chains/nodes/analyze.py`, where the schema is declared inline" / "1.0.0 — initial: kind, sub_queries, required_sections, missing_aspect" |
| `P2` `follow-up-contextualization.md` | 1.0.2 | "1.0.2 — added metric-fidelity rule: when the previous answer carried figures, the rewrite must keep the same metric family as those figures (dollar margin is a different question from percentage margin) and must not switch units or scope — the follow-up's figures are what the answer is scored on, so a rewrite that silently changes what is being asked drops them" |
| `P3` `answer-synthesis-with-citations.md` | 1.0.1 | "1.0.1 — new grounding rule: a figure quoted from a multi-period context block must name its period in the sentence itself (verified claim \"$ 211,915 million [3]\" was trimmed as unsupported because the income-statement block spans FY2023–FY2021 and the sentence named no period)" |
| `P4` `table-and-number-formatting.md` | 1.0.1 | "1.0.1 — corrected rule 1's citation placement: \"a source note under the table lists the [n] markers\" prescribed a placement the verifier's `split_units` cannot read (the table block ends before the note, so the rows inherit no marker and every row is trimmed as `figure_not_in_span`), and it contradicted rule 4, which already prescribed the placement that works — a marker in each row's label cell, or per-cell. Motivated by golden case q03 `Compare revenue and operating income between FY2023 and FY2022`: the real run's two table rows were both trimmed, leaving `figure_ok: False` and the answer `&#124;&#124;(unsupported claim removed)&#124;&#124;`" / "1.0.0 — initial" |
| `P5` `summarization-and-verification.md` | 1.0.2 | "1.0.2 — header correction: the \"You receive\" line claimed the node ships P5 the whole retrieved context, which is no longer true — the node now sends only the blocks the undecided claims cite (all of them when none cites anything), because a claim is judged against the span it cites and prefill for the rest is dead weight" / "1.0.1 — header correction: the Output schema row named a module that does not exist; it now names `GROUNDING_SCHEMA` in `src/chains/nodes/verify.py`, where the schema is declared inline" / "1.0.0 — initial: claim-level span verification" |

Three of them — P2, P3 and P4 — carry a body change; P1's and P5's bumps are header corrections,
and their changelogs say so themselves. The protocol in `docs/prompts/README.md` is what makes this
history recoverable: one change per version, `patch` for a wording clarification, and every bump
evidence-gated. Only P3 and P4 record a scoring result, and those two are the ones worth a
before/after. Both are quoted exactly.

**P3, before v1.0.1 (rule 3 only):** "If the question names a fiscal period, quote the figure from
the column labeled with that period. The report contains multiple fiscal years in adjacent table
columns; reading the wrong column is the single worst failure you can produce." — a model that read
the right column could still emit `$ 211,915 million [3]`, and the verifier trimmed it because the
cited income statement prints FY2023, FY2022 and FY2021 side by side and the claim named no period.
The pre-trim result is quoted from P3's own changelog, which recorded it as a live provider run:
`(unsupported claim removed)`, `grounded=False`, `citations=0`. **P3 after v1.0.1 (rule 3a):** "When
a figure comes from a context block that spans several periods (an income statement printing
FY2023, FY2022 and FY2021 side by side), the sentence that carries the figure MUST name that
period: \"$ 211,915 million [3]\" is trimmed as unverifiable because the block shows several periods
and the claim names none, whereas \"Revenue was $ 211,915 million in FY2023 [3]\" survives
verification. The period belongs to the claim itself, never to a preamble or a following sentence."
The same case measured after the fix, from the NVIDIA run's own log: q01 took **10626.8 ms** and
answered `Microsoft's revenue in FY2023 was $211,915 million [2].` — `grounded : True citations: 1`,
the figure surviving verification, which is the evidence the rule was needed and that it worked.

**P4, before v1.0.1 (rule 1):** "A source note under the table lists the [n] markers for the rows
it covers" — a placement `checks.split_units` cannot read: the table block ends before the note, so
its rows inherit no marker and each is trimmed as `figure_not_in_span`. The pre-trim result is
quoted from P4's own changelog, which recorded it case by case: golden case q03
`Compare revenue and operating income between FY2023 and FY2022`, both body rows trimmed,
`figure_ok: False`, the answer rendering
`||(unsupported claim removed)||`. **P4 after v1.0.1 (rule 1):** "Put the
[n] marker in the table itself — in each row's label cell when one block supports the whole row,
per-cell when the cells come from different blocks (rule 4). A source note listed *below* the table
does not cite the rows: the verifier reads a table's markers from inside the table, so a note
underneath leaves every row uncited and each row is trimmed as an unsupported claim." The same case
measured after the fix, from the NVIDIA run's own log: `expected : $211,915 FOUND` and
`expected : $88,523 FOUND`, the answer rendering as
`| Revenue [1] | $ 211,915 | $ 198,270 |` and `| Operating Income [1] | $ 88,523 | $ 83,383 |` —
markers inside the rows, both figures surviving.
P1 and P5's header corrections carry no scoring result, only a metadata one, so no before/after is
quoted for them: claiming one would mean inventing a failure, and an invented prompt failure is
worse than an absent one.

**Why the prompt stayed mostly quiet, and why that is a design outcome rather than a red flag.** The
template's warning assumes the prompt is the only place a rule lives. Here the riskiest rules were
deliberately placed *outside* the prose first, so a deterministic test could hold them:

| Rule a reviewer might expect in the prompt | Where it actually lives, and what tests it |
|---|---|
| "Don't quote an FY2022 figure as FY2023" | `src/chains/checks.py` pre-pass rejects a period-label mismatch with **no model call**; `tests/test_checks.py` includes the FY2024-claimed-as-FY2023 case, and the golden canaries q01/q02 pass (§5.1) |
| "Cite only spans that support the claim" | `checks.py` verifies citation-marker ↔ block correspondence and rejects orphans; the same pre-pass trims `figure_not_in_span` (§7) |
| "Be concise; format numbers and tables readably" | `P4`, a formatting contract appended to `P3`/`P5` — a versioned file, not an inline string |

A prompt rule that can be a pure function was made a pure function. What remains in the prompts is
the part only a model can do: classify, rewrite a follow-up, and phrase an answer from spans that
were already validated.

**Where the iteration actually is in this submission.** The record of revising on evidence exists —
just not as prompt diffs, because the rework landed in the layer that was wrong:

| Iteration | Evidence that forced it | What changed |
|---|---|---|
| §3b — the redo | The plan named `decompose_question` in the architecture but no task built it; `multi_part` routing had nothing behind the edge | The node was built, plus eight other audit findings; the redo is argued in §3b |
| §3b — fan-out cap | A comparison question split into operands, not parts | Rule-based decomposition with a cap of 4, never splitting a comparison (§5.1's routing table) |
| §3c.1 — the design system | Measured in Chromium: `st.html()` drops a `<style>` block, and a session-state guard made the emission one-shot, so the theme applied for exactly one paint | `inject_theme()` emits every run; the guard deleted; the correction to my own first measurement is written out in §3c.1 |
| §3c.2 — the documented setup path | `grep -rn "dotenv" app.py src/ scripts/` returned nothing while four documents told the reader to create `.env`, so the app ran with neither a mock mode nor a key and produced a *correct-looking* refusal | `src/utils/settings.py::load_env()`, called first in `app.py` and both scripts |
| §7 — the mock-mode claim | The grounding pre-pass fired on the mock's own scripted answer, trimming it entirely | `README.md` §1 now states what a reviewer will see in mock mode instead of implying a working answer engine |

That is the shape of the iteration this project actually has: measured failures → the layer that
was wrong → the fix, with the measurement kept. A reviewer looking for evidence of refinement will
find it in §3b, §3c and §7 — and, now, in `docs/prompts/` itself: two of the five bumps were
driven by a keyed run's own trimmed answer, and their changelogs name the case and the result, so
the history is recoverable from the file rather than from memory.

## 7. Measurements

Every number below was read from the artifact or command named in the Source column during the
build session — none is estimated. Rows whose only honest value is "not measured here" say so
and say why, rather than carrying an invented figure.

| Metric | Value | Source |
|---|---|---|
| Corpus pages / sections | 85 pages, 15 sections, 1 document | `data/index/docs.jsonl`; the rail's workspace card |
| Child / parent chunk counts | 398 child chunks across 102 distinct parents | `chunks.jsonl` |
| Index artifacts on disk | real 384-dim index: `chunks.jsonl` 1.79 MB + `bm25.pkl` 402 KB + `docs.jsonl` 5.2 KB = 2.7 MiB | `data/index/faiss/meta.json` = `{"dim": 384, "count": 398}` |
| **Answer latency, simple question (real provider)** | **10,626.8 ms** total — 9,638 tokens in / 779 out (10,417 total, from that run's `node_trace`); per-node timings are not printed as a split in the log, so the node-level breakdown of an earlier measurement is **not carried forward** rather than restated from it | `.evalwork/nvidia-verify.log`, case q01 (`latency : 10626.8 ms`, `grounded : True citations: 1 nodes: 5`); produced by `LLM_PROVIDER=nvidia LLM_MODEL=nvidia/nemotron-3-super-120b-a12b MOCK_LLM=0 python scripts/eval_rag.py` on the real index. **Supersedes the earlier 18.8 s measurement (analyze 7404, synthesize 10771), whose node split came from a different keyed run** |
| Answer latency, simple question (mock) | 88.4 ms total — retrieve 64.0, synthesize 14.0, verify 5.9, analyze 4.5 | live `node_trace`, `MOCK_LLM=1` — **mock mode**, so it reproduces keylessly and stands apart from the real-provider row above |
| Answer latency, decomposed comparison (mock) | 41.1 ms total — retrieve 20.6, synthesize 12.4, verify 5.9, analyze 2.2 | live `node_trace`, `MOCK_LLM=1` |
| Node path, `simple` | 5 nodes: contextualize → analyze → retrieve → synthesize → verify | `node_trace` |
| Node path, `multi_part` | 6 nodes: the above plus `decompose_question` | `node_trace` |
| Tokens in / out per answer | 1240 / 360 (simple); 1860 / 540 (comparison) | `node_trace`, `MOCK_LLM=1` |
| Context-budget behaviour (STRESS-TEST-RAG #6) | dropped 1 of 8 merged parents (simple) and 6 of 14 (comparison) to stay under 6000 tokens | retrieval warning, live run |
| **Golden QA — real provider, last complete run (n=10)** | retrieval hit-rate@5 **7/7 (1.00) PASS** · citation presence **8/8 (1.00) PASS** · figure match **6/7 (0.86) FAIL** · wrong-year contamination **0/3+0 PASS** · refusal correctness **10/10 (1.00) PASS** — `cases failing: q07` only | `LLM_PROVIDER=nvidia LLM_MODEL=nvidia/nemotron-3-super-120b-a12b MOCK_LLM=0 python scripts/eval_rag.py` — real index, real key (`OPENAI_API_KEY`/`OPENAI_BASE_URL` exported by the host, which win because `.env` loads with `override=False`). **Supersedes the earlier 6/6 · 6/8 · 3/7 · 9/10 row** — the improvement is from the prompt and verifier fixes that followed that run, and it was re-measured rather than assumed |
| Golden QA — real provider, run-to-run variance | four full/partial keyed runs of the same ten cases on the same code produced different failing sets — `q05,q07,q09` · `q09` · `q07,q08` · `q03,q07` · `q05,q07,q08`, no two of them the same; some runs on the same code reached a perfect retrieval hit-rate@5 and perfect citation presence, so the aggregate is a sample, not a score | the same command across the recorded keys |
| Provider-nondeterminism, measured directly | three identical planner calls at temperature 0.0 returned a `simple` plan, a `simple` plan with an empty `missing_aspect`, and a raised `LLMError` | direct probe of the planner against the live keyed endpoint |
| **Golden QA — the reproducible real-mode behaviour (q09)** | `q09` "What was revenue in FY2019?" now refuses in **6.7 ms** with **zero planner POSTs**, decided by the symbolic year-coverage gate (`checks.catalog_years` / `asked_years` / `year_coverage_gap`, wired into `analyze_question` before the gateway call); before the gate the same case took **9,812.7 ms** and classified on a coin flip | `src/chains/checks.py`; keyed runs before and after the gate |
| **Real answers produced (the deliverable)** | `Revenue was $211,915 million in FY2023 [5]`, `grounded=True`, 1 citation, page 44 — verified twice: once through `invoke()` and once in a real Chromium session against the running app, where it rendered as a chat turn with a citation card to `2023_Annual_Report.pdf · page 44` | real-provider `invoke()`, then the live UI |
| **Golden QA — mock mode, full aggregate (the only reproducible one)** | `cases failing: none`, exit 0 — refusal correctness **9/9 (1.00) PASS**, wrong-year contamination **0/3+0 PASS**; this row reproduces byte-for-byte, which is the keyless reviewer path | `MOCK_LLM=1 INDEX_DIR=data/index-mock python scripts/eval_rag.py` |
| Golden QA — mock-mode metrics that stay unmeasurable | retrieval hit-rate@5 2/7 observed, citation presence 1/8, figure match 1/7 — all **N/A**: the mock embedder is a sha256 hash, the mock LLM cites only blocks containing its figure, and the mock answers from scripted phrases | `scripts/eval_rag.py` prints these itself |
| Embedding provider | **local Hugging Face by default**: `fastembed` 0.9.0 (ONNX Runtime), `BAAI/bge-small-en-v1.5`, 384-dim, 65 MB cached, no key, no `/embeddings` call | `src/core/local_embedder.py`; `EMBEDDING_PROVIDER=local` |
| Docker cold start | image **271.5 MiB** (was 310.9 MiB before the dev-toolchain split, −39.4 MiB); `docker build --no-cache` 188–202 s, warm 1–2 s; `compose up` 0.63 s; container start → first `/_stcore/health` 200 = **1.6–1.7 s**; non-root `uid=10001(appuser)`; healthcheck `starting`→`healthy` at ~4.5 s, and proven to detect failure; named-volume markers survive `down`/`up` and a full rebuild; **469 tests pass inside the container** with no key and no network | verified against every `docs/PLAN.md` T22 criterion; `MOCK_LLM=1` proven download-free under `--network none` |
| **Real CI** | hosted Ubuntu 24.04 runner on `main` **completed `success`**, run `38461934270`; per-step: ruff `All checks passed!`, mypy `no issues found in 18 source files`, floor guard, pytest+coverage, and `docker build --tag isipfin .` (that run predates the current file set; the *Gates* row above carries the current locally measured counts, and this row keeps the historical run's shape) | outcomes read from the run log, not asserted |
| T20 debug panel (live + historical) | picker live-first then newest-first stored rows; live panel carries plan, 5-node path with latency/tokens, 8 retrieval cards; a stored row renders its **persisted** trace + grounded flag, with the two sections the store does not keep reading explicit "not recorded" prose; A's `hybrid_retrieve 79.8 ms` vs B's `10.6 ms` confirms no cross-answer mixing | driven in Chromium on the mock-mode index, DB rows cross-checked |
| Test suite | **634 passed** (`pytest -q` — `634 passed in 40.50s`); the in-container row above carries its own separately measured count, so the two are reported as two runs rather than assumed equal | `python -m pytest -q` in the repo venv |
| Gates | `ruff check` clean (`All checks passed!`); `black --check` reports **57 files would be left unchanged**; `mypy src/utils src/chains` clean (**no issues found in 18 source files**); `check_floor.py --repo-wide` clean (**56 file(s)**); mock eval exit 0 | the exact commands in `.github/workflows/ci.yml`, each run locally this pass; the CI log's own per-step numbers were recorded when that run happened |
| Browser (Chromium, live app) | stylesheet survives reruns (1 `<style>` node, `--isip-accent: #FF7A1A` after each click); 4 nav items + 6 chips render with Material icons; rail width 260px; composer radius 16px; real-provider answer rendered with a citation card to page 44 | §3c — see the correction there about first-paint-only measurement |

**Determinism posture.** Given a provider that behaves this way — three identical planner calls at
temperature 0.0 produced a plan, a different plan, and an error — a reviewer can truthfully claim
only that the golden-QA harness measures *a sample*, not a system. Four full or partial keyed runs
of the same ten cases on the same code returned four different failing sets. At n=10 a single flipped
case moves a metric by 0.10, so on the same code a PASS and a FAIL are both defensible readings of
run-to-run variance; that is why the last complete run is recorded as the table value rather than the
best one, and why the harness's own verdict column is the honest part of the output. What remains
trustworthy is everything that never touches a provider: the lexical wrong-year canary (which stayed
live in every mode and fired in one run, proving it *can* fail), the refusal-class wiring, the
routing table, the symbolic pre-pass, the year-coverage gate, and the marker contract — all pure
functions with hermetic tests, where the same input gives the same verdict every time. The mock-mode
run is the only aggregate that reproduces byte-for-byte. The trade should be named rather than
hidden: the brief asked for an LLM that answers a financial question, and the system does answer
with a grounded, cited figure — but a *reproducible* answer requires either the mock path or a
deterministic provider, and this one is neither.

**One measured behaviour worth naming explicitly.** In mock mode the deterministic numeric
pre-pass *fires on the mock's own scripted answer*: asked for FY2023 revenue, the mock emits
`Microsoft's FY2023 revenue was $211,915 million … up from $198,270 million in FY2022`, and
`groundedness_check` trims it as `figure_not_in_span` because the mock's arbitrary citation
mapping does not actually contain that figure in the span it cites. The consequence is that
**mock-mode answers always render ungrounded and can be trimmed away entirely** — visible in the
UI as the UNGROUNDED badge over "(unsupported claim removed)". That is the symbolic layer
working correctly on bad input, and it is why `README.md` §1 states what a reviewer will see in
mock mode instead of implying a working answer engine.

**The keyed number that was previously absent is now above: a grounded, cited answer for the brief's
headline question, rendered in the real UI against the real provider.** That number is solid — the
same claim was verified twice, once through `invoke()` and once in Chromium — but the *aggregate* is
not, and the reason is not the harness's fault. Four keyed runs of identical code produced four
different failing sets, and the cause was measured directly: the provider answers differently to the
same request at temperature 0. The honest description of the real-mode row is therefore "the last
complete run," with the spread stated beside it, not "the system's score." Two defects surfaced only
because that run finally happened, and both are recorded rather than papered over. First, the
verdict line in `scripts/eval_rag.py`'s summary was inverted (`"FAIL" if value > bar else "PASS"`),
so a metric sitting *at or above* its floor printed FAIL while one *below* it printed PASS — the
aggregate cried regression on a run that had in fact reached a perfect retrieval hit-rate@5 and
perfect citation presence. A harness that cries wolf on a good run is
worse than one that says nothing: it teaches reviewers to ignore the red. Second, and more
substantive, the first keyed revenue run also returned `(unsupported claim removed)`. Root-caused
by driving the symbolic layer directly rather than by retrying: the pre-pass behaved exactly to
contract and returned *undecided* because the claim named no fiscal period while its cited span
(the page-44 income statement) carries FY2023, FY2022 and FY2021 side by side. The same claim
with the period named is *supported*. So the verifier was right and the **synthesis prompt** was
incomplete — P3 rule 3 told the model which column to read but never to *state* the period in the
sentence it emitted. That is a prompt defect, fixed by protocol (P3 1.0.1, rule 3a, changelog
recording the motivation), not by loosening the checker: the strictness is the product's headline
safety property and weakening it to make a demo prettier would be exactly the wrong trade. Two
further real defects fell out of the same run and were fixed with regression tests: the verifier
matched LLM verdicts by list position while ignoring the `claim_id` it asks the model for (an
off-by-one numbering silently discarded a claim the model had explicitly supported), and the
verification payload re-shipped the entire retrieved context even when the answer cited one span
(~6,900 prompt tokens spent to verify a single claim — the reason `groundedness_check` cost 20.5s
before the trim and 6ms after).

**On Groq — a key *was* supplied, a full 10-case keyed run was performed, and the record of why it
was not kept is below rather than hidden.** Groq is wholesale-configurable at runtime —
`LLM_PROVIDER=groq` with `LLM_MODEL=openai/gpt-oss-120b` and `EMBEDDING_PROVIDER=local`, all
documented in `README.md` and `.env.example` — and it was exercised for real against the real
384-dim index, first on `openai/gpt-oss-120b` and then on `openai/gpt-oss-20b` when the run moved
to the 20B model. The measured finding is that it was **not kept for a tier reason, not a quality
reason**: Groq's free tier enforces a rolling **8,000 tokens-per-minute** ceiling (~200,000/day),
and one in-scope case costs about **10,400 tokens** (P1 ~2,038 + P3 bounded by
`CONTEXT_TOKEN_BUDGET = 6000` + P5 ~2,100) — the per-node arithmetic is worked out in
`docs/LOCAL-INFERENCE.md` §7.1. A single case therefore overspends one minute's budget before it
finishes, and a full keyed golden run is roughly half the daily quota. The log
(`.evalwork/keyed20b.log`) shows the consequence plainly: `429 ... Limit 8000, Used 5751,
Requested 2371`, a request that cannot fit the window at all, and the run's own aggregate states
`cases failing : q01, q02, q03, q04, q05, q08` with `cases not measured : q03, q07` — **8 of the 10
cases either failed or could not be measured, all from rate-limiting** (the two surviving refusals
q09 and q10 never reach the provider, decided by the symbolic year gate and the entity check). Two
planner calls degraded outright (q03 and q07, refusing with `missing_aspect=None` — a 429 scoring as
a refusal when it was an outage), q05 and q08 raised as `RateLimited`/`LLMError` mid-run, and q02 and
q04 paid retry-dominated tails (58,745.4 ms and 57,290.5 ms) for answers whose figures came back
`MISSING` anyway. That run exposed two *harness* defects, both fixed since: the summary's verdict
comparison was inverted (`"FAIL" if value > bar else "PASS"`), so a metric at or above its floor
printed FAIL while one below it printed PASS; and `figure_match` compared `$211,915` against an
answer printing `$ 211,915 million` with U+202F figure spaces, so a correct answer scored 0.00. A
third, scoring flaw recorded a planner 429 as a real refusal — the `planner_degraded` path landed in
`refusal_correctness`'s denominator instead of being named as infrastructure, which the harness now
does. Those three fixes are why the same code now measures four of the five bars met on the NVIDIA
endpoint (`figure_match` at 6/7 against a 0.95 bar is the remaining miss, and the exit code is 1).
Every real-mode number in this section comes from that NVIDIA endpoint with local embeddings: an
NVIDIA-compatible server (`https://integrate.api.nvidia.com/v1`) running
`nvidia/nemotron-3-super-120b-a12b`. The Groq key is preserved in the repo's own
`.env.pre-nvidia.bak`, so the path is repeatable by anyone who wants to re-measure it — the
decision is recorded, not locked away.

### Why this pass is mostly verification, and what it found

The previous session's job was *build it*; this one was *prove it, then prove the rest*. Every task box from T20 down was either written-but-never-measured or measured-and-wrong. Verifying them was not ceremony — it is where the remaining defects lived, and most of them were mine.

| Verified | How | Outcome |
|---|---|---|
| T20 debug panel, live + historical | Chromium on the mock-mode index, scratch `DATA_DIR`, DB rows cross-checked | **PASS** — both halves real, a stored row's trace byte-identical to the DB row, no cross-answer mixing (A's `hybrid_retrieve` 79.8 ms vs B's 10.6 ms), and the two sections the store keeps render explicit "not recorded" prose instead of empty space |
| T21 error taxonomy | class-by-class classification + live browser | **found a defect in my own slice.** `GraphError`/`GraphStateError`/`GraphNodeError` and `LocalEmbeddingError` are triggerable and were absent from the classifier, so they reached the user as `Unexpected error: GraphStateError: question is empty` — leaking the internal class name and offering a Retry that cannot succeed on a decision. That is precisely the failure mode the taxonomy existed to remove |
| T22 Docker | every `docs/PLAN.md` criterion, run for real | **found three more**, also mine: `INDEX_DIR` missing from compose's passthrough (read in three places, documented in the README, silently dropped for anyone who exported it instead of editing `.env`); `LOG_JSON` set by the Dockerfile and compose with **zero** `.py` read sites, because the ADR-007 §1 JSON formatter was never built; and the dev toolchain shipping 31 MiB in the runtime image where nothing imports it and mypy cannot compile |
| T23 CI | hosted runner, run log read step by step | **green on real infrastructure**, not asserted locally |
| T24 README | commands and claims audited | one gap closed — nobody had documented that the 64-dim mock index and the 384-dim local index are not interchangeable |
| T26 submission | `gh api` | **BLOCKED on the owner.** Repo private, all code merged, CI green — but all three reviewers still have `permission: none` |

The pattern is the lesson of the exercise: **the defects that survived were the ones my own acceptance checks enumerated a fixed list for.** The taxonomy was tested against the classes I happened to name; the packaging slice was reviewed against the criteria I happened to write. Both gaps closed only by asking *what can actually happen to a user* rather than *does my list pass*. A verification checklist should be written from the failure surface, never from the implementation.

Two things I also got wrong, recorded rather than smoothed over. I merged a branch that broke `main` with an `ImportError` — two halves of one slice committed separately — caught and fixed in the same session, with both commits saying so plainly. And an agent reported a test count of 304 when the suite was 469; a number that gets trusted is worth recounting.

## 8. What I cut, and why

From `docs/STRESS-TEST-DELIVERY.md` kill-switch list — scoping judgment is a graded trait:

| Cut | Why it was safe to cut |
|---|---|
| Cross-encoder reranker | Golden QA is the honest proxy; reranker deferred in ADR-002 with the upgrade path written down |
| Streaming synthesis | Complicates groundedness gating (verify-before-display); listed as a "Could" in spec |
| LangSmith as sole observability | Needs an external account the reviewer may not have; env-gated optional only |
| Postgres / compose vector service | SQLite + in-process indices are the right weight for one container |

## 9. Known limitations (stated, not hidden)

1. Scanned/image PDFs: text-layer absence is flagged per page and surfaced in the UI, but
   quality on such uploads is not guaranteed — noted rather than promised away.
2. **Table text appears twice per table page**: as a markdown table block *and* inside the
   flat text block, because pypdf exposes no coordinates to subtract the table region with.
   The consequence is minor (retrieval may match either form; synthesis sees the markdown
   form first) but it is a real duplication, and it was a deliberate trade against adding a
   layout-analysis dependency. Accepted per ADR-004; documented rather than hidden.
3. **`section_path` reflects document order, not hierarchy.** Heading *levels* cannot be
   recovered from regex matching alone — there is no font-size signal reliable enough on this
   corpus to trust — so the path accumulates in reading order. On the FY2023 corpus this is
   benign: all 12 real headings are detected in correct page order, and `$ 211,915` sits in the
   `…/RESEARCH AND DEVELOPMENT` section exactly as the report presents it. The consequence is
   that a path element is "the most recent heading," not "the enclosing H2." Citations still
   point at the right page, which is what the brief grades.
4. **Heading detection is best-effort, and the limits are measured, not guessed.**
   ADR-004's raw all-caps regex accepts MD&A column labels along with real section
   headings. I tested two discriminators against the FY2023 corpus before choosing
   one: word count (rejected — it silently drops a legal one-word section heading,
   which the test fixture's `CONTINGENCIES` proved by breaking) and recurrence
   across pages (rejected — only 2 all-caps lines repeat on this corpus, and both
   are genuine sections). What shipped is a length floor (`_MIN_HEADING_CHARS`),
   which removes short table tokens like `REVENUE` and `GROSS MARGIN` while
   keeping every real heading. All 12 section headings on the corpus are detected
   in correct page order. Not perfect: `STOCK PERFORMANCE` (17 chars) is still
   accepted as a heading, so `section_path` carries some ordering noise. Citation
   pages are unaffected, which is what the brief grades.
5. Multi-column PDF sections: header/footer stripping + blank-line splitting produces partially
   ordered text in worst cases; still retrievable.
6. Retrieval fusion (RRF) is untuned against a reranker; the golden set is the tuning signal.
7. SQLite single-writer: fine for the demo; the Postgres swap path is documented, not built.
8. Scale story is vertical only; extracting the graph behind an API is the documented next
   step, deliberately not built inside the exam window.

## 10. How to verify my claims (reviewer path)

1. `docs/GRADING-ALIGNMENT.md` maps every brief bullet to a design element and its demo
   evidence.
2. `docs/DELIVERABLES-MATRIX.md` maps every deliverable to files and verification steps.
3. `README.md` (created at T24) reproduces the app in four commands, keyless via `MOCK_LLM=1`.
4. The demo script in `docs/STRESS-TEST-DELIVERY.md` is the exact 6-step sequence I ran before
   submitting; golden-QA numbers in §7 come from that tooling, not from memory.

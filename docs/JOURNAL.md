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

**Prompt iteration record: none — P3 is at v1.0.0, changelog "initial: citation constraint +
year-aware extraction contract."** There is no earlier version whose failure produced this one, and
I am not going to invent one. The reason is a design choice rather than luck: ADR-010 put the year
rule in the *symbolic* layer first — the deterministic pre-pass in `src/chains/checks.py` rejects a
period-label mismatch without any model call — so the prompt's rule is a second, independent net
over a rule that is already testable in isolation. The evidence that the design holds is the canary
table above and `tests/test_checks.py`'s FY2024-claimed-as-FY2023 case, not a diff in
`docs/prompts/`. §6 addresses the "zero iteration looks like a red flag" objection directly.

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
`entities` from the previous *answer*, per P2 v1.0.0) rather than a transcript string:

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

**The record: all five prompts are at v1.0.0 and no prompt was revised.** Their changelogs read
`1.0.0 — initial` (`P1` kind/sub_queries/required_sections/missing_aspect; `P2` history window +
`cited_pages` disambiguation; `P3` citation constraint + year-aware extraction contract; `P5`
claim-level span verification). I have no earlier version's failing output to quote, so this
section carries no before/after prompt diff — producing one would mean inventing a failure, and an
invented prompt failure is worse than an absent one.

**Why there is no prompt diff, and why that is a design outcome rather than a red flag.** The
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
find it in §3b, §3c and §7 rather than in `docs/prompts/`.

## 7. Measurements

Every number below was read from the artifact or command named in the Source column during the
build session — none is estimated. Rows whose only honest value is "not measured here" say so
and say why, rather than carrying an invented figure.

| Metric | Value | Source |
|---|---|---|
| Corpus pages / sections | 85 pages, 15 sections, 1 document | `data/index/docs.jsonl`; the rail's workspace card |
| Child / parent chunk counts | 398 child chunks across 102 distinct parents | `chunks.jsonl` |
| Index artifacts on disk | real 384-dim index: `chunks.jsonl` 1.79 MB + `bm25.pkl` 402 KB + `docs.jsonl` 5.2 KB = 2.7 MiB | `data/index/faiss/meta.json` = `{"dim": 384, "count": 398}` |
| **Answer latency, simple question (real provider)** | **18.8 s** total — analyze 7404, synthesize 10771, retrieve 584, verify 6 | live `node_trace`, keyed run |
| Answer latency, simple question (mock) | 88.4 ms total — retrieve 64.0, synthesize 14.0, verify 5.9, analyze 4.5 | live `node_trace`, `MOCK_LLM=1` |
| Answer latency, decomposed comparison (mock) | 41.1 ms total — retrieve 20.6, synthesize 12.4, verify 5.9, analyze 2.2 | live `node_trace`, `MOCK_LLM=1` |
| Node path, `simple` | 5 nodes: contextualize → analyze → retrieve → synthesize → verify | `node_trace` |
| Node path, `multi_part` | 6 nodes: the above plus `decompose_question` | `node_trace` |
| Tokens in / out per answer | 1240 / 360 (simple); 1860 / 540 (comparison) | `node_trace`, `MOCK_LLM=1` |
| Context-budget behaviour (STRESS-TEST-RAG #6) | dropped 1 of 8 merged parents (simple) and 6 of 14 (comparison) to stay under 6000 tokens | retrieval warning, live run |
| **Golden QA — real provider, last complete run (n=10)** | retrieval hit-rate@5 6/6 (1.00) **PASS** · citation presence 6/8 (0.75) **FAIL** · figure match 3/7 (0.43) **FAIL** · wrong-year contamination 0/3+0 **PASS** · refusal correctness 9/10 (0.90) **FAIL** | `MOCK_LLM=0 OPENAI_MODEL=nvidia/nemotron-3-super-120b-a12b python scripts/eval_rag.py` — real index, real key |
| Golden QA — real provider, run-to-run variance | four full/partial keyed runs of the same ten cases on the same code produced different failing sets — `q05,q07,q09` · `q09` · `q07,q08` · `q03,q07` · `q05,q07,q08`, no two of them the same; some runs on the same code reached a perfect retrieval hit-rate@5 and perfect citation presence, so the aggregate is a sample, not a score | the same command across the recorded keys |
| Provider-nondeterminism, measured directly | three identical planner calls at temperature 0.0 returned a `simple` plan, a `simple` plan with an empty `missing_aspect`, and a raised `LLMError` | direct probe of the planner against the live keyed endpoint |
| **Golden QA — the reproducible real-mode behaviour (q09)** | `q09` "What was revenue in FY2019?" now refuses in **6.7 ms** with **zero planner POSTs**, decided by the symbolic year-coverage gate (`checks.catalog_years` / `asked_years` / `year_coverage_gap`, wired into `analyze_question` before the gateway call); before the gate the same case took **9,812.7 ms** and classified on a coin flip | `src/chains/checks.py`; keyed runs before and after the gate |
| **Real answers produced (the deliverable)** | `Revenue was $211,915 million in FY2023 [5]`, `grounded=True`, 1 citation, page 44 — verified twice: once through `invoke()` and once in a real Chromium session against the running app, where it rendered as a chat turn with a citation card to `2023_Annual_Report.pdf · page 44` | real-provider `invoke()`, then the live UI |
| **Golden QA — mock mode, full aggregate (the only reproducible one)** | `cases failing: none`, exit 0 — refusal correctness **9/9 (1.00) PASS**, wrong-year contamination **0/3+0 PASS**; this row reproduces byte-for-byte, which is the keyless reviewer path | `MOCK_LLM=1 INDEX_DIR=data/index-mock python scripts/eval_rag.py` |
| Golden QA — mock-mode metrics that stay unmeasurable | retrieval hit-rate@5 2/7 observed, citation presence 1/8, figure match 1/7 — all **N/A**: the mock embedder is a sha256 hash, the mock LLM cites only blocks containing its figure, and the mock answers from scripted phrases | `scripts/eval_rag.py` prints these itself |
| Embedding provider | **local Hugging Face by default**: `fastembed` 0.9.0 (ONNX Runtime), `BAAI/bge-small-en-v1.5`, 384-dim, 65 MB cached, no key, no `/embeddings` call | `src/core/local_embedder.py`; `EMBEDDING_PROVIDER=local` |
| Docker cold start | image 1.41 GB uncompressed / 311 MiB compressed; build 321 s; `compose up` 0.63 s; **3.58 s** to `/_stcore/health` 200; `fastembed` model cached on the `data` volume, second load **1.4 s** offline vs 85.6 s first | measured end-to-end; `MOCK_LLM=1` proven download-free under `--network none` |
| Test suite | **306 passed** (`pytest -q`) | CI command |
| Gates | `ruff check` clean; `black --check` 52 files unchanged; `mypy src/utils src/chains` clean (18 files); `check_floor.py --repo-wide` clean (51 files); mock eval exit 0 | the exact commands in `.github/workflows/ci.yml` |
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

**On Groq.** Groq was requested by the submission's owner and is fully wired in — base URL, model
`openai/gpt-oss-120b`, and the `LLM_API_KEY` / `LLM_BASE_URL` / `GROQ_API_KEY` precedence chain are
all in place — but no Groq key was ever supplied, so nothing in this submission was verified
against Groq and no claim here rests on it. Every real-mode number in this section came from the one
keyed endpoint that was actually available: an NVIDIA-compatible server
(`https://integrate.api.nvidia.com/v1`) running `nvidia/nemotron-3-super-120b-a12b`, with
embeddings local throughout.

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

# Lifecycle Plays

> The product is the 48-hour exam submission and its module family. Run per
> `product-lifecycle-plays`: diagnosis before play, extension tested explicitly, honest
> pressure source. "Not yet measurable" marks what can only be recorded at a checkpoint —
> that is a fact about timing, not an omission.

## Worksheet 1 — Stage diagnosis: the exam deliverable

**Trigger:** the submission ships once; the question this worksheet answers is *what kind of
finishing play the remaining hours call for* — polish what exists, rebuild part of it, or cut.

**Revenue trend** (analogy for demand/signal on this artifact): six months of interview outcomes,
zero deployments; each build session is a "quarter." The last three sessions produced planning
artifacts (31 docs), i.e., investment rising while shippable output was flat — a classic
mature-and-softening pattern on the *artifact*.

### Transition questions

| # | Lever | Ask | Yes/No | Evidence |
|---|---|---|---|---|
| 1 | Marketing objective | Is defending the full intended scope still profitable? | **Yes** | The brief's Must set is fixed; extending beyond it costs hours that the journal (weighted deliverable) needs. Defending *only* the Must set is profitable |
| 2 | Competition | Are loyalty efforts no longer retaining? | No | No competitors — this is a graded exam, not a market. The reviewer's attention is retained by demonstrated quality |
| 3 | Product | Are legacy support costs becoming unsustainable? | Yes | The plan is 31 docs / 26 tasks against 46 remaining hours; supporting *all* planned surface (streaming, reranker, Postgres path) is unsustainable, supporting the Must set is not |
| 4 | Promotion | Should we run migration campaigns instead of loyalty? | No | Nothing to migrate; one shot, one submission |
| 5 | Place | Are we losing money on certain channels? | Yes | Two redo rounds spent on doc polish without code = the "channel" of pure planning is now costing more than it returns |
| 6 | Price | Are value-add bundles losing effectiveness? | **Yes** | Nice-to-have features (multi-doc UI polish, session rename/delete) have stopped adding grading value relative to the core: cited answers, decomposition, Docker, journal |
| 7 | Data strategy | Is data shifting from predictive to transitional? | Yes | Golden-QA metrics don't exist yet — the data we need for the journal is build-time measurement, i.e., forward-looking, not historical |

**Yes count:** 5/7 — **crossing into decline** on the artifact-as-planned. The pattern matters
more than the count: every yes is about **scope support cost**, none is about the core problem
becoming irrelevant. The core (cited financial QA) still solves a real, graded problem.

### Pressure source

- [x] **Supply/cost-side** — the 46-hour clock and the doc-to-code ratio. `docs/STRESS-TEST-DELIVERY.md`'s kill-switch list exists precisely because of this pressure
- [x] Demand-side — ruled out: needs have not shifted; the brief is the same brief
- [x] Capability-side — ruled out: the architecture is not at its limit; it has never been built

**Primary pressure: supply/cost-side.** Say it out loud, per the skill's trap warning: this is
*"our costs are bad,"* not *"the customer's need changed."* Success looks like a complete Must
set, not a maximal feature set.

### Extension test (all four, explicitly)

1. Is there a segment a variant could serve? **No —** one reviewer set, one brief; a "lite exam
   variant" and a "full variant" would split the same grading attention with no different buyer
2. Is there a need an added capability would meet without re-architecture? **Yes, narrowly** —
   the golden-QA harness and the debug panel are extensions on the existing graph that add
   grading credit without re-architecture. That is the *only* extension worth hours
3. Would a repackage or rebrand reach a different buyer? **No** — same reviewers, same rubric
4. Is the decline in the product, or the channel around it? **The channel.** Pure-documentation
   effort was the failing channel; the plan itself is sound. Fixing the lever (code now, docs
   alongside) is cheaper than replacing any part of the design

### Recommended play

**Play: Extend** — keep the existing design, add the two extensions that matter (golden-QA
harness, debug panel), and cut the rest via the kill-switch order.

**Because:** the pressure is supply-side and the channel diagnosis (Q4) says the design is
sound; extension is the cheapest play and the pattern of yeses points at scope, not concept.

**What this play costs:** roughly the hours budgeted in `docs/TIMELINE.md` 16:00–00:30, with
features cut in the published kill order if a checkpoint goes red.

**What we're accepting:** no streaming, no reranker, no multi-user, no Postgres — all already
documented as cut in `docs/PLAN.md` risks and `docs/adr/ADR-002/006`.

## Worksheet 2 — Portfolio view: the module family

| Product (module) | Stage | Yes-count | Pressure | Extend because | Replace because | Retire because | Play |
|---|---|---|---|---|---|---|---|
| `doc-ingest` | Design | 0/7 | none | tables → markdown is already the shape | — | — | **Extend** |
| `chunking-index` | Design | 0/7 | none | parent-child chosen before build; no reason to disrupt | — | — | **Build (as-is)** |
| `llm-gateway` | Design | 0/7 | none | `MOCK_LLM` already covers the keyless case | — | — | **Build (as-is)** |
| `orchestration` | Design | 1/7 | mild (scope of nodes) | node set is complete after FAIL-1 fix | — | — | **Build (as-is)** |
| `chat-store` | Mature design | 1/7 | none | SQLite schema already sized for the demo | — | — | **Build (as-is)** |
| `app-ui` | Mature soft. | 2/7 | supply (hours vs credit) | extras (rename/delete) survive | only if hours explode — unwrap to chat-only | never: it's the deliverable | **Extend, cut tail** |
| `packaging` | Design | 1/7 | none | compose already the grader's path | — | — | **Build (as-is)** |
| Documentation set (31 docs) | **Mature softening** | 5/7 (worksheet 1) | **supply/cost-side** | 5 new redo docs closed the skill gaps | — | no doc gets retired; all trace to a requirement | **Extend (done), now harvest** |

### Interaction checks

- [x] No two products are the replacement for the same customers — nothing is being replaced
- [x] No extension undercuts another's replacement — no replacement play exists
- [x] No two retirements land in the same window — the kill-switch list sequences cuts
- [x] Sequencing across the line is deliberate — `docs/PLAN.md` phases + checkpoints

## Worksheet 3 — Replacement risk register

The genuine replacement in this plan is **the prompt iteration cycle**: untested v1.0.0 prompts
(`docs/prompts/*`, now versioned) are replaced by golden-QA-validated v1.x during the build
(T12/T17). It is a replacement because the old artifact stops being used, not because it changes
in place.

| # | Hazard | Applies? | Prob | Impact | Mitigation | Contingency (Plan B) |
|---|---|---|---|---|---|---|
| 1 | GTM process failure — new prompts written but never wired | Yes | M | H | `load_prompt("P3")` pattern; version asserted in node tests (`docs/prompts/README.md` rule 5) | Node test fails loudly → revert to last prompt version; prompts are files, revert is one commit |
| 2 | Delayed market entry — prompts left at v1.0.0, golden never run | Yes | M | H | T17 scheduled before journal writing; journal §5 template requires observed before/after | Ship v1.0.0 with an honest "no iteration was needed" line rather than an invented iteration |
| 3 | Internal misalignment — prompt and node disagree on schema | Yes | L | M | `QueryPlan`/`GroundingResult` schemas live in `src/chains/schemas.py`, referenced by both | Schema-validation unit test at T10 catches it |
| 4 | External forces — model provider changes JSON behavior | Yes | M | M | MOCK_LLM mode decouples tests from provider quirks; gateway retries once on schema violation | Bridge shim: coerce provider-specific JSON into our schema, recorded in journal |
| 5 | Regulatory risk | No | — | — | n/a (no compliance surface) | — |
| 6 | Unexpected cannibalization — P4 formatting rules contradict P3 synthesis rules | Yes | M | L | P4 is explicitly an *appendix* to P3/P5, one rule set, versioned together | Merge P4 into P3 as one file — duplication removed, not patched |
| 7 | Poor EOL management — old prompt versions linger in code as inline strings | Yes | L | H | Floor-guard pattern scan + rule "no inline system prompts anywhere in src" | `grep -rn` at review; inline string found = task not done (`docs/DEFINITION-OF-DONE.md`) |

### Cannibalization plan

- Which prompt wins which question: P3 owns every factual answer; P5 owns verification; P4 owns
  formatting rules inside both. No question is claimed by two prompts.
- Margin during overlap: during T12 both v1.0.0 and v1.x exist in files; cost is two prompt loads
  in tests, zero runtime cost.
- Overlap duration: one task. v1.0.0 is superseded in place (changelog line), never forked.

### Retirement funding

- Who owns the EOL of v1.0.0 prompts: the same task that supersedes them (T12/T17).
- Is it funded: **yes** — supersede-in-place costs the changelog line it always costs; there is no
  separate migration workstream to under-fund. Recorded so hazard 7 has an owner, not a hope.

## Quality checks

- [x] All seven transition questions answered with evidence from the plan, not vibes
- [x] The play was chosen after the diagnosis; the yes-pattern (all scope-cost, none concept-death) drove Extend
- [x] All four extension questions answered explicitly; Q2's narrow "yes" is where the two remaining extensions came from
- [x] Pressure source named out loud: our costs, not the reviewer's changed needs
- [x] What Extend costs and what it trades away are both stated
- [x] Every applicable hazard has a contingency; hazard 5 marked n/a with a reason rather than forced low/low
- [x] Cannibalization planned in both directions; the retirement has a named owner and is funded

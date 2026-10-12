# ADR-010: The orchestration is a deterministic neuro-symbolic pipeline — not a multi-agent system

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none
- **Amends:** ADR-001 (names the layer it already implies)

## Context

The question was asked directly: *should orchestration be a deterministic multi-agent DAG so it
isn't pure LLM agents?* Two corrections and one enrichment follow from examining what ADR-001
already specifies.

**Correction 1 — the DAG already exists.** ADR-001 fixes a 7-node `StateGraph` with a static
topology: `contextualize_question` → `analyze_question` → (`decompose_question`) →
`hybrid_retrieve` → `synthesize_answer` → `groundedness_check`, with `refuse` as a terminal
branch. Control is data-dependent but not learned: edges fire on `query_plan.kind`, fused
retrieval scores (`tau` gate), and counted state (`retries`, fan-out cap, context budget).
Nothing in the system decides its own next tool call at runtime. It is a deterministic DAG by
construction.

**Correction 2 — "multi-agent" is the wrong label.** A multi-agent system is several
autonomous agents coordinating through messages; here there is one process, one graph, one state
object, fixed node identities, and deterministic routing. Calling it "multi-agent" would invite
exactly the reviewer question we cannot defend ("which agent negotiates with which, and how is
that bounded?"). The accurate and *stronger* framing is **neuro-symbolic**: deterministic
structure owns control and verification; neural models do the semantic work where they earn
their place. The pattern has standing in current literature for precisely this task — NeSy-RAG
(arXiv 2608.06292) and TrustTable (ACL 2026, neuro-symbolic auditing for faithful table QA) —
so the label is defensible, not a coinage.

## The split, stated precisely

| Layer | Owns | Components |
|---|---|---|
| **Symbolic (deterministic, no LLM)** | control, cost bounds, ranking arithmetic, citation plumbing, **numeric verification** | graph topology and edges; `recursion_limit`; `retries` counter; fan-out cap ≤ 4; score gate `tau` + single rewrite retry; RRF fusion and parent expansion; citation index mapping; context-budget packing; **`src/chains/checks.py`** |
| **Neural (LLM)** | the four places semantics genuinely require it | `contextualize_question` (rewrite), `analyze_question` (classification → structured plan), `synthesize_answer` (generation), and the **semantic half** of `groundedness_check` (claims that are not pure figures) |

Everything a grader could ask "why did it do that?" about is in the symbolic column and is
testable with `MOCK_LLM=1` and no network. Only answers and judgements require a model.

## Decision

Keep the DAG exactly as specified in ADR-001. Add three things:

1. **Name it correctly, everywhere — and keep the two labels independent.** The architecture is a
   *deterministic neuro-symbolic pipeline*: the symbolic/neural split is orthogonal to how many
   actors exist. **A multi-agent system can be neuro-symbolic** — a fleet whose agents are
   constrained by fixed symbolic rules (scoring tiers, threshold overrides that supersede neural
   judgement) is neuro-symbolic regardless of actor count — and a single-graph system can be
   neuro-symbolic too (this one: symbolic control, gates, fusion, citation mapping, and the
   numeric pre-pass, with LLMs only where semantics genuinely require them). This ADR chooses the
   neuro-symbolic character **and** the single-graph topology, independently. The topology choice
   is argued below; "multi-agent" is not rejected here because it conflicts with neuro-symbolism —
   it's rejected for this system on cost, determinism, and reviewer-translation grounds. **Both
   parts are agentic workflows** — LangGraph's own docs use "agent" for exactly this idiom
   (durable state graphs with tool-calling nodes), and the brief's "agentic workflows" requirement
   is satisfied by the graph either way.
2. **Home for the deterministic predicates.** New module `src/chains/checks.py` holds the pure
   deterministic rules that were previously scattered as prose across docs: the follow-up
   heuristic (pronoun/ellipsis detection), the tau gate decision, the fan-out cap, the
   context-budget packer, the citation marker↔block correspondence validator, and the new
   numeric claim checker below. Pure functions, unit-tested, zero network — the symbolic layer
   becomes a first-class, testable surface instead of "whatever the nodes do".
3. **A deterministic numeric pre-pass inside verification (the enrichment).** Today
   `groundedness_check` is 100% LLM — the weakest symbolic share in the design, sitting on the
   exam's #1 ranked failure mode (wrong-year figures, `docs/STRESS-TEST-RAG.md` row 1). The
   pre-pass runs before the LLM verifier:
   - Parse each candidate claim's figure and its cited `[n]` block.
   - Normalise both sides the way the documents print them (`$ 211,915` ↔ `$211,915 million` ↔
     `211.9B`, thousands separators, parenthesised negatives, `FY2023` vs `fiscal year 2023`).
   - **Reject deterministically** any claim whose figure string does not occur in its cited span,
     or whose period label in the span disagrees with the period in the claim.
   Trim rejected claims before the LLM verifier runs, and record the rejection as an event.
4. **The vocabulary decision (decided, settled, not to be re-litigated).** The system is called:
   **an agentic workflow** and **a deterministic neuro-symbolic pipeline**. It is *not* called a
   *multi-agent system*, and no node, file, module, or prompt is renamed to agent vocabulary
   (`analyze_question`, not `analyzer_agent`). Reasons, in order of weight:
   1. **The brief asks for "agentic workflows,"** which this satisfies exactly under LangGraph's
      own usage of the term. "Multi-agent" buys nothing on the grading checklist.
   2. **The word sets an expectation the reviewer audits against** — agent identity, autonomy,
      coordination protocol, message passing — and a fixed 7-node DAG would then read as an
      overclaim. Precision reads as *more* sophisticated, not less, to anyone who knows the
      taxonomy.
   3. **147 node-name references exist across this doc set**; renaming them for zero information
      gain is pure churn (`synthesize_answer` is self-documenting already).
   4. **The thesis-scale system this design was derived from is genuinely multi-agent** — 19
      classes, parallel spawning, conditional fan-out. Using one word for both flattens the
      distinction that makes that system genuinely good.
   The axis confusion that prompted this was resolved in item 1: multi-agent and neuro-symbolic
   are orthogonal. Rejected label, accepted labels, and the single deliberate exception (the ADR
   titles/index line that state the position) are recorded here so a future reader does not
   re-open the question from a clean slate.

## Why not a pure-LLM agent loop (the honest comparison)

| | Pure-LLM agent (ReAct-style tool loop) | This pipeline |
|---|---|---|
| Control | model chooses each next step | fixed edges on typed state |
| Cost | unbounded until the model stops | bounded by construction (≤ 4 fan-out, 1 retry, budget cap) |
| Testability offline | none — the graph is the model's mood | full suite runs on `MOCK_LLM=1` |
| A wrong number | plausible, fluently asserted | figure absent from the cited span → rejected without a model call |
| Reviewer answer to "why that path?" | read the trace of a stochastic run | read the routing table |

The middle ground that was considered and rejected: a *router LLM* that picks between
deterministic strategies. It costs a call per question, reintroduces nondeterministic control at
the exact place determinism matters, and duplicates what `analyze_question` already returns as a
structured plan. The plan is data; routing on it is symbolic.

## And why not an AOSE-style multi-agent restructuring (the fair version of the question)

Agent-Oriented Software Engineering — Gaia/Tropos/Prometheus methodologies, BDI agents, FIPA-ACL
message passing, agent platforms with roles and interaction protocols — is a real discipline, and
the question "make it a neuro-symbolic multi-agent DAG through AOSE" is not naive. Here is why it
loses *for this system*, with the conditions under which it would win.

| | AOSE multi-agent restructure | This DAG |
|---|---|---|
| Who decides what happens next | each agent decides whether/when to act and whom to address | fixed edges on typed state; no node chooses anything |
| Cost bound | emergent, post-hoc | by construction (≤4 fan-out, 1 retry, budget cap, recursion limit) |
| Reproducibility | agenda-dependent | identical inputs → identical path, verified by `MOCK_LLM=1` |
| Offline testability | needs agent lifecycle harnessed | every rule is a pure function in `checks.py` |
| Reviewer translation cost | nodes→agents, edges→messages, state→shared blackboard/ACL — a reviewer must translate back to the LangGraph idiom the brief names | reads as the LangGraph idiom the brief asks for |
| What it fits | many independent actors, local knowledge, negotiation, a shared environment, long-running autonomy | one 130-page document, one question at a time, a bounded fan-out that merges back |

Four reasons, in order of weight:

1. **"Deterministic" and "multi-agent" pull against each other.** The instant agents decide their
   own next hop or whom to answer, the deterministic bound that protects this build disappears —
   and that bound is what makes quota, latency, and the wrong-year failure visible and testable.
   The one *guaranteed* worst case for a demo is an agent that loops politely.
2. **The problem shape has nothing to negotiate.** There is one document, one question at a time,
   and a ≤4 fan-out that merges back into a single answer. That is a DAG edge, not an
   organization. AOSE pays off where actors have *conflicting local knowledge and goals* — supply
   chains, market simulation, distributed sensor fleets. None of that exists here.
3. **Methodology overhead lands on the clock, not on the artifact.** Agent framework, message
   bus, lifecycle management, role/protocol modeling, then a translation layer to Python — for
   zero observable difference in what the reviewer sees. The lifecycle verdict on this submission
   was *Extend* (`docs/LIFECYCLE-PLAYS.md`): supply-side pressure says ceremony that produces no
   observable improvement is the wrong spend.
4. **It breaks the graded mapping.** ADR-001's routing table is how a reviewer checks each exam
   capability (decompose, plan, route, cite, verify) against a named node. Agents that choose their
   own successors make "why that path?" answerable only by reading a stochastic trace — the exact
   opposite of the traceability this design is built on.

**What AOSE has that this design already took (the useful 20%):** explicit roles with owned
responsibilities and frozen contracts between them. `docs/CAPABILITY-MAP.md` *is* that — seven
module ids, dependency direction, interfaces frozen in `docs/SPEC.md` §9. The design borrows
agent-oriented thinking at the architecture level and refuses it at the runtime level, where it
would buy ceremony and cost determinism.

**External witness (public, verifiable):** the mature open-source coding-agent harness *omp*
(Stencil Labs, ~35k GitHub stars) ships exactly this pattern — its `task` tool is a
**parent-orchestrated deterministic fan-out**: the parent defines the workers and the budget,
each worker runs in an isolated context and returns a **schema-validated object** ("no prose to
parse, no merge conflicts between siblings, no orphaned edits"), and `workflowz` is documented as
a *deterministic* multi-subagent workflow. The state of the art is not a fleet of agents that
negotiate their own next hop; it is deterministic orchestration with tool-calling workers — which
is precisely what this ADR keeps, and what ADR-001 already specifies.

**Where a real AOSE build would be the right answer** (stated so the rejection is honest, not
convenient): a fleet of documents each parsed by its own long-running extractor/verifier agents
competing over a shared environment; a multi-user product where background indexing runs for
hours and must be resumable and schedulable; a market of many independent information sources.
That is a different product than a 46-hour exam chatbot — and if this ever became that product,
the migration is *additive*, not a rewrite: the graph already sits behind
`orchestration.invoke()` behind `chat-store`, so the two things AOSE would add (worker processes,
resumable scheduling) attach without touching the pipeline's contracts.

## Consequences

- The symbolic/neural split is now a **reviewable interface**: any `import streamlit`-style
  boundary check can ask "is this rule in `checks.py`?" and a rule that exists nowhere is a gap.
- Numeric verification no longer depends on a model deciding to be careful. The LLM verifier
  keeps the claims it is actually good at — computed figures (margins, growth rates), scope
  matching, and qualitative claims — while exact printed figures are checked symbolically.
  `docs/prompts/summarization-and-verification.md` is updated so P5 does not duplicate work the
  pre-pass already did.
- New failure-mode coverage with a deterministic test: a wrong-year answer (FY2024's $245,122M
  in place of $211,915M) must be **rejected by `checks.py` without any model involved** —
  a fail-without-the-change test in the hermetic suite (`docs/TESTING-AND-EVAL.md`).
- Honest limits of the checker: it validates figures **as printed**. It cannot arbitrate
  computed values (it can check that a stated margin is arithmetically consistent with the
  cited revenue/operating-income spans — that part stays LLM judgement or is deferred), and it
  says nothing about qualitative claims. Those remain the neural verifier's job. The design is
  hybrid on purpose; the checker removes the class of error a model is worst at, not every error.
- Cost: one small pure module, no new dependency, no new LLM call — it *reduces* tokens by
  trimming before verification.

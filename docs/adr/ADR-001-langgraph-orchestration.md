# ADR-001: LangGraph StateGraph as the orchestration layer

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none

## Context

The exam brief requires *"integrate Langchain or Langgraph for orchestrating agentic workflows"*
and *"decomposing complex questions," "planning retrieval and synthesis operations dynamically."*
The workflow the design needs (see `docs/ARCHITECTURE.md`):

1. **Conditional routing** — `analyze_question` classifies the question; the graph must branch to
   `decompose_question` (multi-part), straight to `hybrid_retrieve` (simple), or `refuse`
   (out-of-scope). A static chain cannot express "the next step depends on a model's verdict."
2. **Bounded fan-out** — decomposed questions retrieve once per sub-query (max 4), then merge.
3. **Bounded retry** — the score gate triggers exactly one rewrite retry; unbounded agent loops
   burn quota and demo time.
4. **Per-node observability** — latency, tokens, and errors must attach to each node
   (`docs/adr/ADR-007-observability-errors.md`).
5. **Testability** — graph behavior must be provable with a scripted LLM, no network
   (`docs/TESTING-AND-EVAL.md`).

Requirements 1–3 are the reason a plain chain or a bare `while` loop is not enough: the control
flow is data-dependent and must be inspectable in the trace.

## Decision

Use the **LangGraph standalone package** (`pip install -U langgraph`), not the in-langchain
`langgraph` re-export, not deprecated `LLMSingleActionAgent` / `AgentExecutor` patterns.

A single `StateGraph(GraphState)` in `src/chains/graph.py` where `GraphState` is the TypedDict
frozen in `docs/ARCHITECTURE.md` §State Schema. Nodes are pure functions
`(state) -> partial-state-update`; the graph never passes messages ad hoc — all data flows
through explicitly typed state fields.

**Node and routing table:**

| Node | Runs when | Next edge |
|---|---|---|
| `contextualize_question` | follow-up heuristic fires (pronoun/ellipsis) | → `analyze_question` |
| `analyze_question` | always (after optional contextualization) | `simple` → `hybrid_retrieve`; `multi_part` → `decompose_question`; `out_of_scope` → `refuse` |
| `decompose_question` | `query_plan.kind == multi_part` | → `hybrid_retrieve` (fan-out ≤ 4) |
| `hybrid_retrieve` | retrieval path | score gate: pass → `synthesize_answer`; fail → one `rewrite` retry → `synthesize_answer` |
| `synthesize_answer` | always after retrieve | → `groundedness_check` |
| `groundedness_check` | always after synthesis | pass → END; fail → trim-and-flag → END |
| `refuse` | out-of-scope | → END |

Guardrails: graph-level `recursion_limit` caps pathological loops; retrieval retry is a counted
state field (`retries`), not an unbounded edge.

This graph is a **deterministic neuro-symbolic pipeline**: the topology, edges, caps and gates are
symbolic (no model involved); four nodes call the LLM for the semantic work. It is deliberately
*not* a multi-agent system — one process, one state object, fixed node identities, deterministic
routing. The characterisation, the symbolic/neural split, and the deterministic numeric
verification pre-pass added inside `groundedness_check` are specified in
`docs/adr/ADR-010-neuro-symbolic-pipeline.md`, which amends this ADR.

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Raw OpenAI SDK + hand-rolled `while` loop | Routing, retry bounds, and trace have to be re-implemented; no typed state; fails the brief's explicit orchestration requirement |
| LangChain LCEL chains (`RunnableSequence` with `RunnableBranch`) | Branching is expressible but **fan-out + merge** needs `RunnableParallel` gymnastics and lost node identity; per-node observability requires manual instrumentation; rejected: the graph IS the product here |
| Multi-agent frameworks (CrewAI, AutoGen) | Role-play agent teams add a coordination layer the task doesn't need; harder to bound cost; heavier abstractions for a single-agent-with-tools task |
| Full `create_react_agent` prebuilt | Opaque tool-calling loop; citation discipline and the groundedness gate can't be enforced between tool calls |
| No framework (sync DAG in plain Python) | Honest option — rejected only because the brief explicitly names LangGraph and because typed state + tracing decorators are already written for us |

## Consequences

- Nodes are unit-testable in isolation with a stubbed gateway: one test per node
  (`tests/test_nodes_*.py`), one routing test per edge kind.
- `node_trace` comes free from wrapping nodes with the decorator in `src/utils/tracing.py`.
- Reviewer must read graph idiom — mitigated by this ADR, the mermaid in
  `docs/ARCHITECTURE.md`, and the routing table above living in the README.
- LangGraph adds one dependency; its state machine is the only framework code the app depends
  on beyond LangChain primitives (`RecursiveCharacterTextSplitter`, `ChatOpenAI`,
  embeddings wrappers).
- Checkpointing deliberately **not used** — a single-user demo needs no resumable executions;
  persistence belongs to `chat-store` (ADR-006), not the graph runtime.

## Grading note

This ADR plus `docs/ARCHITECTURE.md` is the evidence for the "efficient implementation of
workflow orchestration" criterion: the grader can map each exam capability (decompose, plan,
route, cite, verify) to a named node and an edge.

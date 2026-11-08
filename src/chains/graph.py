"""The LangGraph StateGraph skeleton: topology, routing, and the typed-error spine.

This module (task T9) builds the whole graph and wires every edge the finished pipeline
needs, with the seven node bodies registered as **stubs that raise** instead of
returning empty updates. T10-T15 fill those bodies in; the topology, entry point,
conditional routing and ``recursion_limit`` defined here are the contract those bodies
land on, and they are correct before a single node has any logic.

Why the stubs raise instead of returning empty results
------------------------------------------------------
ADR-007's hard rule is "no failure path yields an empty answer". A stub returning
``{}`` — or an ``answer_md`` of ``""`` — would look exactly like a completed run that
produced nothing: the trace would show seven green events, the debug panel would show
no error, and the UI would render a blank chat bubble, which the ADR names as the one
outcome the app must never produce. Raising instead makes the gap loud at the point
where it exists (the node body) and keeps the invariant that a graph run either
returns a real :class:`~src.chains.state.AnswerEnvelope` or raises a typed error the
UI renders as a banner. The tests assert this: an unimplemented node raises
``GraphNodeError`` with ``node_trace[-1].error`` set, never a ``KeyError``.

Reducer convention (per ``src/chains/state.py``)
-----------------------------------------------
``GraphState`` is a ``TypedDict(total=False)`` with **no ``Annotated[..., operator.add]``
reducers**: every key is a ``last_value`` channel, so a node's returned update
overwrites the previous value. ``node_trace`` is the only accreting field and is owned
by ``@trace_node``, which appends to the live list rather than returning a value for
the key. Two alternatives were considered and rejected: (a) ``Annotated[list,
operator.add]`` on ``node_trace`` — then the decorator would *have* to return
``{"node_trace": [event]}`` on every node, double-appending whenever a node also
mutated the state in place, and any node migrating from the ``graph.invoke`` input
shape would re-emit the seed events; (b) returning nothing and mutating alone — loses
the events entirely on the final write, because an unreturned key is not persisted.
The decorator does both (append in place *and* rewrite a wrongly-returned key), so
nodes need do neither.

Node contract (docs/ARCHITECTURE.md Node Contract Table)
-------------------------------------------------------
| Node | Reads | Writes | LLM |
|---|---|---|---|
| ``contextualize_question`` | ``question``, ``history`` | ``standalone_question`` | yes, cheap |
| ``analyze_question`` | ``standalone_question``, ``doc_catalog`` | ``query_plan`` | yes, JSON |
| ``decompose_question`` | ``query_plan`` | ``query_plan.sub_queries`` | no |
| ``hybrid_retrieve`` | ``sub_queries`` | ``retrieved``, ``retries`` | no |
| ``synthesize_answer`` | ``retrieved``, ``standalone_question`` | ``answer_md`` + ``cites`` | yes |
| ``groundedness_check`` | ``answer_md``, ``citations`` | ``grounded``, ``refusal_reason`` | yes |
| ``refuse`` | ``query_plan``, ``doc_catalog`` | ``answer_md``, ``refusal_reason`` | no |


The conditional entry point on ``contextualize_question`` is decided by the follow-up
heuristic that ships with T14; until then the graph always enters at
``contextualize_question``, which is also the only node that reads ``history`` — the
rewritten question, not the raw history, reaches synthesis (P2's purpose statement).
The router functions here are real and testable now; only the node bodies are stubs.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import Any, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.chains.nodes.analyze import analyze_question
from src.chains.nodes.contextualize import contextualize_question
from src.chains.nodes.decompose import decompose_question
from src.chains.nodes.refuse import refuse
from src.chains.nodes.retrieve import hybrid_retrieve
from src.chains.nodes.synthesize import synthesize_answer
from src.chains.nodes.verify import groundedness_check
from src.chains.state import (
    AnswerEnvelope,
    DocSummary,
    GraphState,
    QueryPlan,
    Turn,
    final_envelope,
)
from src.utils.llm import LLMParams
from src.utils.tracing import trace_node

__all__ = [
    "NODE_NAMES",
    "RECURSION_LIMIT",
    "GraphError",
    "GraphNodeError",
    "GraphStateError",
    "build_graph",
    "initial_state",
    "invoke",
    "route_after_analyze",
    "route_after_retrieve",
]

#: Node names, in pipeline order. The order is the documented contract; the graph's
#: traversal order depends on routing, not on this list.
NODE_NAMES: tuple[str, ...] = (
    "contextualize_question",
    "analyze_question",
    "decompose_question",
    "hybrid_retrieve",
    "synthesize_answer",
    "groundedness_check",
    "refuse",
)

#: LangGraph's ``recursion_limit`` is a super-step budget, not a Python recursion cap;
#: it counts graph traversals, and langgraph's own default (10007, overridable by
#: ``LANGGRAPH_DEFAULT_RECURSION_LIMIT``) is sized for unbounded agent loops. This
#: graph is bounded by design — ADR-001: "the score gate triggers exactly one query
#: rewrite retry, never an unbounded loop" — so the budget is set explicitly far below
#: the default: seven nodes' worth of hops, plus the single bounded retry, leaves room
#: to spare while turning a wiring mistake that makes the graph loop into a loud
#: ``GraphRecursionError`` instead of a long run that burns quota.
RECURSION_LIMIT = 25


# --- typed errors (ADR-007 taxonomy's graph-side errors) --------------------------


class GraphError(RuntimeError):
    """Base class for every orchestration error that reaches the UI.

    ``node`` names the node that failed (``None`` for a pre-graph input error) and
    ``error`` the original typed error's message, so the banner can show both without
    the app re-i18ning exception text.
    """

    def __init__(
        self,
        message: str,
        *,
        node: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.node = node
        self.error = error


class GraphNodeError(GraphError, NotImplementedError):
    """A node whose implementation task has not landed yet.

    ``NotImplementedError`` subclass so callers catching "not implemented" still
    catch it; ``GraphError`` so the app's banner path treats it as a node failure
    rather than a framework one. Never caught inside the graph: it propagates through
    ``@trace_node`` (which records it) and out of ``invoke()``.
    """


class GraphStateError(GraphError, ValueError):
    """Invalid or missing state at graph entry — a caller bug, never a silent default.

    Subclasses ``ValueError`` because the two failures it covers are argument errors:
    a non-string question, and a question that is empty or whitespace-only (there is
    nothing to retrieve; ``invoke("")`` must not produce a plausible empty answer).
    """


# --- routing (ADR-001's table, implemented and testable now) -----------------------

#: ``query_plan.kind`` values the ``analyze_question`` router switches on.
KIND_SIMPLE = "simple"
KIND_MULTI_PART = "multi_part"
KIND_OUT_OF_SCOPE = "out_of_scope"

#: The routing table as ADR-001 writes it: kind → destination node. These are the
#: ``path_map`` handed to ``add_conditional_edges``, so they must be exactly what
#: :func:`route_after_analyze` returns as keys and node names as values. Keyed
#: ``Hashable`` because that is langgraph's own declared key type for a ``path_map``.
ANALYZE_ROUTES: dict[Hashable, str] = {
    KIND_SIMPLE: "hybrid_retrieve",
    KIND_MULTI_PART: "decompose_question",
    KIND_OUT_OF_SCOPE: "refuse",
}

#: The score-gate table for :func:`route_after_retrieve`: a strong retrieval goes
#: straight to synthesis; a weak one takes the single rewrite retry, which lands back
#: on ``hybrid_retrieve`` (the self-loop the ``retries`` counter bounds).
RETRIEVE_PASS = "synthesize"
RETRIEVE_REWRITE = "rewrite"
RETRIEVE_ROUTES: dict[Hashable, str] = {
    RETRIEVE_PASS: "synthesize_answer",
    RETRIEVE_REWRITE: "hybrid_retrieve",
}


def _plan_of(state: Any) -> QueryPlan | None:
    """The ``query_plan`` in ``state``, or ``None`` when it is absent or wrongly typed."""

    if isinstance(state, Mapping):
        plan = state.get("query_plan")
        return plan if isinstance(plan, QueryPlan) else None
    plan = getattr(state, "query_plan", None)
    return plan if isinstance(plan, QueryPlan) else None


def route_after_analyze(state: Any) -> str:
    """The ``analyze_question`` router, keyed on ``query_plan.kind`` (ADR-001).

    A missing or unrecognised plan routes to ``refuse``: ADR-007's error table makes a
    refusal a first-class, user-visible outcome ("an explicit refusal with reason"),
    and P1's own failure handling degrades a schema-violating planner reply into
    ``out_of_scope``. Returning the destination (rather than ``None``, or raising) is
    what keeps the trace honest — langgraph turns an unmapped destination into an
    opaque internal error instead of a routed refusal, and the acceptance criterion
    demands a typed failure rather than a crash.

    Returns the plan's ``kind`` string — the token ``ANALYZE_ROUTES`` is keyed by, and
    what ``add_conditional_edges``'s ``path_map`` maps to a node name. Returning the
    token rather than the node name keeps the router independent of the node naming:
    a retry edge or a renamed node is a change in the table above, not in this function.
    """

    plan = _plan_of(state)
    kind = plan.kind if plan is not None else ""
    if kind in ANALYZE_ROUTES:
        return kind
    # An unrecognised plan (a schema-violating planner reply that T10 degrades per
    # P1's failure handling) routes to refusal rather than crashing or defaulting to
    # retrieval, which would produce a plausible-but-ungrounded answer.
    return KIND_OUT_OF_SCOPE


def route_after_retrieve(state: Any, weak: bool | None = None) -> str:
    """The ``hybrid_retrieve`` router: the one bounded score-gate retry (ADR-001, ADR-002).

    Returns one of the :data:`RETRIEVE_ROUTES` tokens. Two things are settled here and
    one is deliberately deferred:

    *Settled — the bound.* At most one rewrite per run, expressed in the counted state
    field ``retries`` (ADR-001: "retrieval retry is a counted state field, not an
    unbounded edge"). ``hybrid_retrieve`` sets ``retries`` to the number of rewrites
    it has already performed, and the router reads the state **after** that update has
    been applied — langgraph composes the node's output into the branch's input, so
    ``retries`` is the already-incremented count, not the pre-update one (verified on
    langgraph 1.2.14). ``retries >= 1`` therefore always means "the one allowed
    rewrite has already been spent" and routes to synthesis, so a weak-recall answer is
    *produced and marked* by ``groundedness_check`` (trim-and-flag) rather than
    looping forever.

    *Deferred — the gate condition.* Comparing the best fused score against ``tau``
    (default 0.02 RRF) belongs to ``hybrid_retrieve``, which is what decides what
    retrieval success means; that is T11's task. So the condition arrives as the
    ``weak`` argument, which langgraph does not pass (it calls the router with the
    state only) and tests do: a routing test states the gate outcome explicitly, and
    T11 supplies the value. Left unset, the router falls back to "no ``retrieved``
    chunks is weak", which keeps the graph runnable today and is a conservative
    default — it asks for the rewrite instead of synthesising from nothing.

    Args:
        state: The graph state after ``hybrid_retrieve``'s update.
        weak: Whether the retrieval is below the gate. ``None`` derives it from
            ``retrieved``; ``True``/``False`` states it outright.
    """

    retries = state.get("retries") if isinstance(state, Mapping) else getattr(state, "retries", 0)
    try:
        spent = int(retries) if retries is not None else 0
    except (TypeError, ValueError):
        spent = 0
    below_gate = (not _has_chunks(state)) if weak is None else bool(weak)
    if below_gate and spent == 0:
        return RETRIEVE_REWRITE
    return RETRIEVE_PASS


def _has_chunks(state: Any) -> bool:
    if isinstance(state, Mapping):
        raw = state.get("retrieved")
    else:
        raw = getattr(state, "retrieved", None)
    return bool(isinstance(raw, list) and raw)


# --- node stubs (T10-T15) ----------------------------------------------------------


def _stub(node: str, task: str) -> Callable[[GraphState], dict[str, Any]]:
    """A traced node that raises, describing the task that implements it.

    The body raises rather than returning a partial update because an empty answer is
    the one outcome ADR-007 forbids; see the module docstring. The declared return type
    is the node contract — a partial state update — so when T10-T15 replace this stub
    with the real body the signature stays identical and no call site changes.
    """

    @trace_node(node)
    def _node(state: GraphState) -> dict[str, Any]:
        raise GraphNodeError(
            f"{node} is not implemented yet — implemented in {task}",
            node=node,
        )

    return _node


#: One stub per node name. Built here rather than inline in ``add_node`` so a later
#: task replaces a stub by name in exactly one place.
_STUBS: dict[str, Any] = {
    "contextualize_question": _stub("contextualize_question", "T14"),
    "analyze_question": _stub("analyze_question", "T10"),
    "decompose_question": _stub("decompose_question", "T10b"),
    "hybrid_retrieve": _stub("hybrid_retrieve", "T11"),
    "synthesize_answer": _stub("synthesize_answer", "T12"),
    "groundedness_check": _stub("groundedness_check", "T13"),
    "refuse": _stub("refuse", "T15"),
}

#: Nodes whose implementation task has landed, overriding the stub for that name.
#: ``add_node`` refuses a duplicate, so a landed node is swapped in *here* rather than
#: registered after the stub loop. The stub entry stays so the dict still satisfies the
#: loop's expectation of one entry per name.
_NODES: dict[str, Any] = {
    "contextualize_question": contextualize_question,
    "analyze_question": analyze_question,
    "decompose_question": decompose_question,
    "hybrid_retrieve": hybrid_retrieve,
    "synthesize_answer": synthesize_answer,
    "groundedness_check": groundedness_check,
    "refuse": refuse,
}


# --- graph construction ------------------------------------------------------------


def build_graph() -> CompiledStateGraph[GraphState]:
    """Build the complete graph, all seven nodes registered and every edge wired.

    Returns an uncompiled ``StateGraph[GraphState]``, so a caller (or a test) can
    attach extra instrumentation or replace a stub before ``compile()``. Use
    :func:`invoke` for the normal path; this function exists for inspection.

    Topology (ADR-001's node/routing table):

    * entry → ``contextualize_question`` → ``analyze_question``
    * ``analyze_question`` → ``hybrid_retrieve`` | ``decompose_question`` | ``refuse`` (routed)
    * ``decompose_question`` → ``hybrid_retrieve``
    * ``hybrid_retrieve`` → ``synthesize_answer`` (routed; the retry edge lands with T11)
    * ``synthesize_answer`` → ``groundedness_check``
    * ``groundedness_check`` → END, ``refuse`` → END
    """

    graph = StateGraph(GraphState)

    for name in NODE_NAMES:
        # A landed implementation (``_NODES``) wins over its stub, so the graph routes
        # through real logic instead of ``GraphNodeError``. Names not yet landed keep
        # the raising stub until their own task arrives.
        graph.add_node(name, _NODES.get(name, _STUBS[name]))

    graph.set_entry_point("contextualize_question")
    graph.add_edge("contextualize_question", "analyze_question")
    graph.add_conditional_edges(
        "analyze_question",
        route_after_analyze,
        ANALYZE_ROUTES,
    )
    graph.add_edge("decompose_question", "hybrid_retrieve")
    graph.add_conditional_edges(
        "hybrid_retrieve",
        route_after_retrieve,
        RETRIEVE_ROUTES,
    )
    graph.add_edge("synthesize_answer", "groundedness_check")
    graph.add_edge("groundedness_check", END)
    graph.add_edge("refuse", END)

    # Compiled eagerly: a malformed topology is a build-time error, not a first-run
    # surprise, and ``validate()`` runs inside ``compile()``.
    return graph.compile()


def initial_state(
    question: str,
    history: list[Turn] | None = None,
    params: LLMParams | None = None,
    doc_catalog: list[DocSummary] | None = None,
) -> GraphState:
    """Build the initial graph state, validating the inputs at the graph boundary.

    ``question`` must be a non-empty string: an empty question has nothing to retrieve
    and nothing to answer, so ``invoke("")`` raises ``GraphStateError`` here instead of
    running seven nodes and producing an empty answer (ADR-007: "no failure path yields
    an empty answer"). ``history``/``doc_catalog`` default to empty lists rather than
    ``None`` so nodes never need a ``None`` branch; ``params`` defaults to an empty
    :class:`~src.utils.llm.LLMParams` ("provider defaults") because the schema is
    total-optional by design.

    ``node_trace`` is seeded as an empty list rather than left absent, even though the
    decorator creates it on demand: the input list is the *same object* langgraph
    threads through the run (verified on 1.2.14), so seeding it here is what makes a
    trace complete even when ``invoke`` raises before returning state.
    """

    if not isinstance(question, str):
        raise GraphStateError(
            f"question must be a string, got {type(question).__name__}",
        )
    if not question.strip():
        raise GraphStateError("question is empty — there is nothing to retrieve or answer")

    state: GraphState = {
        "question": question.strip(),
        "history": list(history or []),
        "params": params if params is not None else LLMParams(),
        "doc_catalog": list(doc_catalog or []),
        "retries": 0,
        "grounded": False,
        "node_trace": [],
    }
    return state


def invoke(
    question: str,
    history: list[Any] | None = None,
    params: LLMParams | None = None,
    doc_catalog: list[Any] | None = None,
) -> AnswerEnvelope:
    """Run one question end to end and return the frozen answer envelope (docs/SPEC.md §9).

    Builds the initial state, invokes the compiled graph with an explicit
    ``recursion_limit``, and assembles the :class:`~src.chains.state.AnswerEnvelope`.

    With the T9 stubs this raises ``GraphNodeError`` from the first node — which is the
    expected, asserted behaviour of the skeleton, not a defect. Errors are typed and
    unwrapped: langgraph wraps node failures, and the wrapper is what reaches the UI
    when a node raises (see ADR-007's error table, ``GraphNodeError`` row).
    """

    state = initial_state(question, history=history, params=params, doc_catalog=doc_catalog)
    graph = build_graph()
    final = graph.invoke(state, {"recursion_limit": RECURSION_LIMIT})
    if not isinstance(final, Mapping):
        raise GraphError(
            f"graph returned {type(final).__name__}, expected a state mapping",
            node=None,
        )
    return final_envelope(cast(GraphState, final))

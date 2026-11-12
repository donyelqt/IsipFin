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

NODE_NAMES: tuple[str, ...] = (
    "contextualize_question",
    "analyze_question",
    "decompose_question",
    "hybrid_retrieve",
    "synthesize_answer",
    "groundedness_check",
    "refuse",
)

RECURSION_LIMIT = 25


class GraphError(RuntimeError):

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
    pass


class GraphStateError(GraphError, ValueError):
    pass


KIND_SIMPLE = "simple"
KIND_MULTI_PART = "multi_part"
KIND_OUT_OF_SCOPE = "out_of_scope"

ANALYZE_ROUTES: dict[Hashable, str] = {
    KIND_SIMPLE: "hybrid_retrieve",
    KIND_MULTI_PART: "decompose_question",
    KIND_OUT_OF_SCOPE: "refuse",
}

RETRIEVE_PASS = "synthesize"
RETRIEVE_REWRITE = "rewrite"
RETRIEVE_ROUTES: dict[Hashable, str] = {
    RETRIEVE_PASS: "synthesize_answer",
    RETRIEVE_REWRITE: "hybrid_retrieve",
}


def _plan_of(state: Any) -> QueryPlan | None:

    if isinstance(state, Mapping):
        plan = state.get("query_plan")
        return plan if isinstance(plan, QueryPlan) else None
    plan = getattr(state, "query_plan", None)
    return plan if isinstance(plan, QueryPlan) else None


def route_after_analyze(state: Any) -> str:

    plan = _plan_of(state)
    kind = plan.kind if plan is not None else ""
    if kind in ANALYZE_ROUTES:
        return kind
    return KIND_OUT_OF_SCOPE


def route_after_retrieve(state: Any, weak: bool | None = None) -> str:

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


def _stub(node: str, task: str) -> Callable[[GraphState], dict[str, Any]]:

    @trace_node(node)
    def _node(state: GraphState) -> dict[str, Any]:
        raise GraphNodeError(
            f"{node} is not implemented yet — implemented in {task}",
            node=node,
        )

    return _node


_STUBS: dict[str, Any] = {
    "contextualize_question": _stub("contextualize_question", "T14"),
    "analyze_question": _stub("analyze_question", "T10"),
    "decompose_question": _stub("decompose_question", "T10b"),
    "hybrid_retrieve": _stub("hybrid_retrieve", "T11"),
    "synthesize_answer": _stub("synthesize_answer", "T12"),
    "groundedness_check": _stub("groundedness_check", "T13"),
    "refuse": _stub("refuse", "T15"),
}

_NODES: dict[str, Any] = {
    "contextualize_question": contextualize_question,
    "analyze_question": analyze_question,
    "decompose_question": decompose_question,
    "hybrid_retrieve": hybrid_retrieve,
    "synthesize_answer": synthesize_answer,
    "groundedness_check": groundedness_check,
    "refuse": refuse,
}


def build_graph() -> CompiledStateGraph[GraphState]:

    graph = StateGraph(GraphState)

    for name in NODE_NAMES:
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

    return graph.compile()


def initial_state(
    question: str,
    history: list[Turn] | None = None,
    params: LLMParams | None = None,
    doc_catalog: list[DocSummary] | None = None,
) -> GraphState:

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

    state = initial_state(question, history=history, params=params, doc_catalog=doc_catalog)
    graph = build_graph()
    final = graph.invoke(state, {"recursion_limit": RECURSION_LIMIT})
    if not isinstance(final, Mapping):
        raise GraphError(
            f"graph returned {type(final).__name__}, expected a state mapping",
            node=None,
        )
    return final_envelope(cast(GraphState, final))

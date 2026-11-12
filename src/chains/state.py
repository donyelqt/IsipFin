from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from src.core.retrieval import ScoredChunk
from src.utils.llm import LLMParams

__all__ = [
    "AnswerEnvelope",
    "Citation",
    "DocSummary",
    "GraphState",
    "NodeEvent",
    "QueryPlan",
    "Turn",
    "final_envelope",
    "replace_event",
    "section_paths",
]

SECTION_PATH_SEPARATOR = " > "


def section_paths(value: Any) -> tuple[tuple[str, ...], ...]:

    if value is None:
        return ()
    items: list[Any]
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return ()
    paths: list[tuple[str, ...]] = []
    for item in items:
        if isinstance(item, str):
            levels = tuple(level.strip() for level in item.split(SECTION_PATH_SEPARATOR))
        elif isinstance(item, (list, tuple)):
            levels = tuple(str(level).strip() for level in item)
        else:
            continue
        levels = tuple(level for level in levels if level)
        if levels:
            paths.append(levels)
    return tuple(paths)


class GraphState(TypedDict, total=False):

    question: str
    history: list[Turn]
    params: LLMParams
    doc_catalog: list[DocSummary]
    standalone_question: str
    query_plan: QueryPlan
    retrieved: list[ScoredChunk]
    planner_degraded: bool
    retries: int
    answer_md: str
    citations: list[Citation]
    grounded: bool
    refusal_reason: str | None
    node_trace: list[NodeEvent]


@dataclass(frozen=True)
class Turn:

    role: str
    content: str
    cited_pages: tuple[dict[str, Any], ...] = ()
    entities: tuple[str, ...] = ()
    id: int | None = None
    session_id: str | None = None
    seq: int | None = None
    created_at: str | None = None

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def is_assistant(self) -> bool:
        return self.role == "assistant"


@dataclass(frozen=True)
class DocSummary:

    doc_id: str
    filename: str
    sections: tuple[tuple[str, ...], ...] = ()
    page_count: int | None = None


@dataclass(frozen=True)
class QueryPlan:

    kind: str
    sub_queries: list[str] = field(default_factory=list)
    required_sections: list[str] = field(default_factory=list)
    fiscal_period: str | None = None
    missing_aspect: str | None = None

    @property
    def is_simple(self) -> bool:
        return self.kind == "simple"

    @property
    def is_multi_part(self) -> bool:
        return self.kind == "multi_part"

    @property
    def is_out_of_scope(self) -> bool:
        return self.kind == "out_of_scope"


@dataclass(frozen=True)
class Citation:

    ref_index: int
    doc_id: str
    page: int
    section_path: tuple[str, ...] = ()
    snippet: str = ""


@dataclass(frozen=True)
class NodeEvent:

    node: str
    started_at: float
    duration_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class AnswerEnvelope:

    answer_md: str
    citations: list[Citation]
    grounded: bool
    refusal_reason: str | None
    plan: QueryPlan
    retrieval_debug: list[ScoredChunk]
    node_trace: list[NodeEvent]
    planner_degraded: bool = False

    def as_dict(self) -> dict[str, Any]:

        return asdict(self)


def replace_event(event: NodeEvent, **changes: Any) -> NodeEvent:

    return dataclasses.replace(event, **changes)


def final_envelope(state: GraphState) -> AnswerEnvelope:

    plan = state.get("query_plan") or QueryPlan(kind="", sub_queries=[], required_sections=[])
    citations = [c for c in state.get("citations") or [] if isinstance(c, Citation)]
    retrieval = [c for c in state.get("retrieved") or [] if isinstance(c, ScoredChunk)]
    trace = [e for e in state.get("node_trace") or [] if isinstance(e, NodeEvent)]
    return AnswerEnvelope(
        answer_md=state.get("answer_md") or "",
        citations=citations,
        grounded=bool(state.get("grounded")),
        refusal_reason=state.get("refusal_reason"),
        plan=plan,
        retrieval_debug=retrieval,
        node_trace=trace,
        planner_degraded=bool(state.get("planner_degraded")),
    )

"""Graph state contract: the typed spine every Phase-3 orchestration node reads and writes.

This module is the single source of truth for the state shape frozen by
``docs/ARCHITECTURE.md`` §State Schema and the ``AnswerEnvelope`` frozen by
``docs/SPEC.md`` §9. Later nodes (``src/chains/nodes/*.py``) import the dataclasses
here and never re-declare a field name, so a rename is a one-line change.

Redux convention (why these are plain values, not langgraph reducers)
---------------------------------------------------------------------
``GraphState`` is a ``TypedDict(total=False)`` with **no field annotated with
``Annotated[..., operator.add]``**: every key is a ``last_value`` channel, so a node's
returned update *overwrites* the previous value instead of merging into it. That is the
correct default for every field except the observability list, and for that one the
accretion is done explicitly by the ``@trace_node`` decorator, which appends to the
live list object (see ``src/utils/tracing.py`` for the measured behaviour of the
alternatives). ``add_messages`` is deliberately not used: this graph does not move
messages through state, and the fields are typed domain records (``LLMParams``,
``ScoredChunk``) that ``add_messages`` would reject or silently reinterpret.

Sooner or later some node will want to append to a list — the state has three list
fields (``retrieved``, ``citations``, ``node_trace``). The rule is: a node returns the
**complete** value it wants the field to hold, not a delta. ``node_trace`` is the single
exception and the decorator owns it; a node must never include ``node_trace`` in its
partial update (the decorator rewrites such an entry with the accumulated list so it
cannot clobber sibling events).

Token convention
----------------
Token counts are *per node*, in ``NodeEvent.tokens_in``/``tokens_out``. They are extracted
from the :class:`~src.utils.llm.ChatResult` a node's gateway call produced, never summed
by the state module: ``answer_md``-producing nodes may issue several calls with different
sampling parameters, and only the node knows which call's accounting belongs in its event.
A node that made no LLM call leaves both fields ``None`` — ``None`` ("not measured") is
distinct from ``0`` ("measured as zero", which is what an endpoint omitting ``usage``
reports).

"""

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
]

#: The state schema (docs/ARCHITECTURE.md §State Schema) follows the code block on that
#: section verbatim: same field names, same order, same ``total=False`` semantics.


# --- the state schema (docs/ARCHITECTURE.md §State Schema) ------------------------


class GraphState(TypedDict, total=False):
    """The single state object the seven nodes read from and write to.

    ``total=False`` mirrors the architecture doc: every key is optional, so a node can
    build a partial update containing only the fields it owns, and a state that missed
    a node (a stub run, a graph rebuilt with a different topology) is still a legal
    state rather than a crash. Nodes that read a field use ``state.get(...)``, never
    ``state[...]``, so a missing field is an absent value rather than a ``KeyError``
    — the acceptance criterion "``invoke("")`` raises a typed error, never a KeyError"
    is the visible case of that same rule.

    Annotations are evaluated by langgraph at ``StateGraph(...)`` construction time, so
    every referenced type is imported **at runtime** (not under ``TYPE_CHECKING``): the
    schema construction calls ``get_type_hints`` and a deferred name raises
    ``NameError``.

    No reducer annotations are used anywhere in this class; see the module docstring
    for why accretion is the decorator's job instead.
    """

    # inputs
    question: str
    history: list[Turn]
    params: LLMParams
    doc_catalog: list[DocSummary]
    # intermediate
    standalone_question: str
    query_plan: QueryPlan
    retrieved: list[ScoredChunk]
    #: ``True`` when ``analyze_question`` degraded to an honest refusal because the
    #: planner call failed (schema violation / transport error). Declared so it
    #: survives the merge — langgraph drops undeclared keys, so an undeclared flag
    #: is invisible to the debug panel while a direct caller still sees it. The
    #: debug panel renders it as "planner degraded", never as a routing fact.
    planner_degraded: bool
    retries: int
    # outputs
    answer_md: str
    citations: list[Citation]
    grounded: bool
    refusal_reason: str | None
    # observability — appended to by ``@trace_node``, read by nothing else
    node_trace: list[NodeEvent]


@dataclass(frozen=True)
class Turn:
    """One conversation turn handed to the graph as ``history``.

    The four ``id``/``session_id``/``seq``/``created_at`` fields mirror the ``turns``
    row so the state can carry the store's rows verbatim instead of a second,
    lossier projection. ``cited_pages`` and ``entities`` are the disambiguation
    dictators of the STRESS-TEST-RAG #3 pronoun-drift fix
    (``docs/prompts/follow-up-contextualization.md`` v1.0.0): a follow-up like "which
    segment drove that?" resolves against what the *previous answer* was grounded in,
    not against the raw user text alone. Both are empty by default — older turns that
    predate the enrichment simply contribute no disambiguation.
    """

    role: str  # 'user' | 'assistant'
    content: str
    cited_pages: tuple[dict[str, Any], ...] = ()
    entities: tuple[str, ...] = ()
    id: int | None = None
    session_id: str | None = None
    seq: int | None = None
    created_at: str | None = None

    @property
    def is_user(self) -> bool:
        """True for a user turn (a question), False for an assistant turn."""
        return self.role == "user"

    @property
    def is_assistant(self) -> bool:
        """True for an assistant turn (an answer), False for a user turn."""
        return self.role == "assistant"


@dataclass(frozen=True)
class DocSummary:
    """One catalog entry for the document catalog built at ingest time.

    ``sections`` is the section-path listing ``analyze_question`` reasons over: it is
    what makes an out-of-scope classification decidable *before* any retrieval call
    (ADR-001: "out_of_scope skips retrieval entirely — no LLM call"). ``page_count``
    is carried for the debug panel, not for routing.
    """

    doc_id: str
    filename: str
    sections: tuple[tuple[str, ...], ...] = ()
    page_count: int | None = None


@dataclass(frozen=True)
class QueryPlan:
    """The ``analyze_question`` verdict: what kind of question this is and how to serve it.

    ``kind`` is one of the three literals the router switches on (``"simple"``,
    ``"multi_part"``, ``"out_of_scope"``); it is typed ``str`` rather than a
    ``Literal`` so a schema-violating model reply can still be recorded and routed to
    ``refuse`` instead of raising before the router exists (P1's failure handling:
    "then classify out_of_scope with missing_aspect: null"). ``missing_aspect``
    distinguishes time-period gaps from topic and entity gaps because the refusal copy
    differs (STRESS-TEST-RAG #7).
    """

    kind: str
    sub_queries: list[str] = field(default_factory=list)
    required_sections: list[str] = field(default_factory=list)
    fiscal_period: str | None = None
    missing_aspect: str | None = None

    @property
    def is_simple(self) -> bool:
        """True when the plan routes straight to hybrid retrieval."""
        return self.kind == "simple"

    @property
    def is_multi_part(self) -> bool:
        """True when the plan routes through decomposition."""
        return self.kind == "multi_part"

    @property
    def is_out_of_scope(self) -> bool:
        """True when the plan routes to a refusal."""
        return self.kind == "out_of_scope"


@dataclass(frozen=True)
class Citation:
    """One numbered citation: the ``[n]`` marker in the answer traced back to its source.

    ``ref_index`` is the marker's number as rendered in the markdown; it is what the
    store persists in ``citations.ref_index`` and what makes ``[3]`` a stable
    reference rather than a positional accident. ``section_path`` is a tuple to match
    ``ScoredChunk``; the store encodes it as JSON (see ``src/store/sessions.py``).
    """

    ref_index: int
    doc_id: str
    page: int
    section_path: tuple[str, ...] = ()
    snippet: str = ""


@dataclass(frozen=True)
class NodeEvent:
    """One node's trace entry: timing, token accounting, typed error (ADR-007).

    ``started_at`` is a ``time.time()`` epoch stamp (the wall clock, for the debug
    panel); ``duration_ms`` is measured with ``time.perf_counter()`` (monotonic, so
    an NTP step mid-run cannot make a duration negative). ``tokens_in``/``tokens_out``
    are ``None`` for a node that made no LLM call — see the module docstring's token
    convention. ``error`` holds the typed error's class name plus message, and is
    ``None`` for a success.
    """

    node: str
    started_at: float
    duration_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None


# --- the frozen envelope (docs/SPEC.md §9) ----------------------------------------


@dataclass(frozen=True)
class AnswerEnvelope:
    """The complete, typed answer the UI renders — the only thing that crosses the boundary.

    Every field is present on every path: a refusal yields ``answer_md`` copy and a
    populated ``refusal_reason`` with ``grounded=True`` — a refusal is a *correct*
    outcome, not an ungrounded one, and ``grounded=False`` is reserved for answers
    whose claims failed verification (it drives the UI's "ungrounded" badge, SPEC
    criterion 7). An error path never yields this object at all, because ``invoke()``
    re-raises and the UI renders a banner (ADR-007's hard rule: no code path renders an
    empty answer).
    """

    answer_md: str
    citations: list[Citation]
    grounded: bool
    refusal_reason: str | None
    plan: QueryPlan
    retrieval_debug: list[ScoredChunk]
    node_trace: list[NodeEvent]

    def as_dict(self) -> dict[str, Any]:
        """Shallow-encode this envelope as plain JSON-able types.

        Used by the debug panel and by the golden-QA harness: the dataclass instances
        belong to the graph, but every consumer downstream renders JSON.
        """

        return asdict(self)


def replace_event(event: NodeEvent, **changes: Any) -> NodeEvent:
    """Rebuild a :class:`NodeEvent` with fields replaced, leaving the original intact.

    Exists for the tracing module: ``NodeEvent`` is frozen on purpose — a completed
    trace is evidence, not a work in progress, so no code should be able to edit an
    event that another pane or the store may already hold. The decorator must still be
    able to attach an error string to an event it has already timed, which is what
    this does by construction (``dataclasses.replace``) rather than by mutation.
    """

    return dataclasses.replace(event, **changes)


def final_envelope(state: GraphState) -> AnswerEnvelope:
    """Assemble the answer envelope from a finished (or failed) graph run.

    Tolerates a partially-populated state because it must also be able to describe a
    run that raised mid-graph: the caller catches the typed error and still wants the
    trace. Missing keys read as the honest empty value — no fabricated citations, no
    fabricated plan. A run that never reached ``analyze_question`` (a stub raising, a
    rebuild gone wrong) therefore yields an empty plan, and ``answer_md == ""`` plus
    ``plan.kind == ""`` makes the gap visible in the debug panel instead of hiding it
    behind a confident-looking refusal.

    ``refusal_reason`` is ``None`` unless a node set it, so "answered" and "refused"
    stay distinguishable: a refusal is a decision, not the absence of one.
    """

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
    )

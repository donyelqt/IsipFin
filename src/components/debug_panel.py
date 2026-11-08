"""The in-app debug panel: why an answer came out the way it did (task T20).

Boundary (ADR-009)
------------------
This module is a UI leaf, exactly like :mod:`src.components.chat`. Its only imports
are :mod:`streamlit` and the token layer in :mod:`src.components.theme`; the contract
types it renders (``AnswerEnvelope``, ``Message``, ``NodeEvent``, ``ScoredChunk``)
are named under ``TYPE_CHECKING`` and read by duck typing, so the panel never imports
the graph nodes, the gateway, the retriever, the prompt builders, or the store.
Everything it shows is handed in by the caller, which is what keeps it renderable
with no index, no API key, and no database.

Two callers, one renderer
-------------------------
A **live** answer arrives as the frozen ``AnswerEnvelope`` the graph returned: the
only shape that carries ``plan``, ``retrieval_debug``, and a ``node_trace`` of
``NodeEvent`` instances. A **historical** answer arrives as the stored ``Message``:
its ``node_trace`` was flattened to plain dicts when the row was written, and it
carries neither a plan nor retrieval scores. Both are normalized into one view here,
and a section a stored answer cannot fill says "not recorded for this answer" rather
than drawing an empty block — the panel never implies data it was not given.

What is rendered (``docs/OBSERVABILITY.md``'s debug-panel contract)
------------------------------------------------------------------
1. **Plan** — ``query_plan``: kind, sub-queries, required sections, fiscal period,
   missing aspect.
2. **Path** — the ordered node list with per-node duration and tokens, the route the
   trace shows (simple / decomposed / refused), and any node whose ``error`` is set.
3. **Retrieval** — the selected chunks in fused-rank order: dense, sparse, and fused
   RRF scores, page, section path, and a truncated evidence snippet.
4. **Grounding** — the grounded flag, the refusal reason, and the citation count.
   When ``planner_degraded`` is set the section also carries the "planner degraded"
   badge: the refusal was a planner failure degraded into a refusal, not a judgement.
   A stored message never carries the flag, so the section says it was not recorded
   rather than implying the planner was healthy.

Deliberately absent: the deterministic pre-pass rejections and the per-claim verdicts
(ADR-007). Those live in the ``events`` table and in the answer text, not on the
answer object this panel is handed, and the Grounding section says so rather than
inventing verdict data.

Cost
----
Collapsed by default, and cheap even so: the body formats at most seven node events
and the fused top-k chunks it was handed, builds a handful of strings, and reads
nothing — no store, no index, no model call. Whether a closed expander's body is
skipped at all is a Streamlit-version detail, so this module does not depend on it
being skipped.
"""

from __future__ import annotations

import html as _html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.components.theme import badge, caption_note, figure, panel

if TYPE_CHECKING:  # contract types only — no runtime import, so the module stays leaf-level
    from src.chains.state import AnswerEnvelope
    from src.store.sessions import Message

__all__ = ["render_debug_panel"]


# --- constants ---------------------------------------------------------------------

#: The expander's label: the panel's whole footprint while collapsed. Short because the
#: rail it lives in is 260px wide.
_EXPANDER_LABEL = "Debug panel"

#: The section titles, spelled exactly as the debug-panel contract in
#: ``docs/OBSERVABILITY.md`` names them. They are rendered through
#: :func:`src.components.theme.panel`, whose ``title`` argument draws them with the
#: token layer's uppercase monospace label voice.
_PLAN_TITLE = "Plan"
_PATH_TITLE = "Path"
_RETRIEVAL_TITLE = "Retrieval"
_GROUNDING_TITLE = "Grounding"

#: The explicit dash for every absent value. ``str(None)`` would print the literal
#: "None" in the panel — the rendered-null bug the theme's own ``_escape`` exists to
#: avoid — so "no value here" is always this glyph and never a Python repr.
_DASH = "—"

#: Node names the route is inferred from. Matched as strings rather than imported from
#: ``src.chains.graph``: the panel stays a leaf, and a renamed node degrades to
#: "unrecorded" instead of failing at import time.
_REFUSE_NODE = "refuse"
_DECOMPOSE_NODE = "decompose_question"
_RETRIEVE_NODE = "hybrid_retrieve"

#: The route labels, one per path ADR-001 defines plus the honest fallback.
_ROUTE_SIMPLE = "simple"
_ROUTE_DECOMPOSED = "decomposed"
_ROUTE_REFUSED = "refused"
_ROUTE_UNRECORDED = "unrecorded"

#: Longest snippet rendered per chunk, ellipsis included. A chunk's window is what the
#: three scores explain; 200 characters is enough to recognize the line item without
#: turning the rail into a document viewer.
_SNIPPET_CHARS = 200

#: Longest document id rendered before it is cut. A ``doc_id`` is a sha256 — stable and
#: unique, but not something a reader can scan — so an unnamed document gets a prefix.
_SHORT_ID_CHARS = 16

#: The failure badge on a node whose ``error`` is set.
_FAILED_LABEL = "failed"

#: Grounding verdict labels. ``_UNKNOWN_LABEL`` covers a stored row written before the
#: flag was persisted (``grounded`` is ``None``), which is not the same as ``False``.
_GROUNDED_LABEL = "grounded"
_UNGROUNDED_LABEL = "ungrounded"
_UNKNOWN_LABEL = "not recorded"

#: The degraded-planner marker and the two lines it renders. ``PLANNER_DEGRADED_COPY``
#: is the single word-source for the marker: ``src.components.chat`` renders the same
#: badge and the same sentence under a degraded refusal in the transcript, so the two
#: surfaces cannot drift into different claims about the same turn.
PLANNER_DEGRADED_LABEL = "planner degraded"
PLANNER_DEGRADED_COPY = (
    "This refusal was not a judgement: the query planner failed, so the question was "
    "classified out of scope instead of being reasoned about."
)
PLANNER_DEGRADED_NOT_RECORDED_COPY = (
    "Not recorded for this answer — the planner-degraded marker is carried only on a "
    "live answer, so a stored row says neither way."
)

#: Copy for the states the objects themselves make visible.
_NO_ANSWER_COPY = "No answer selected: pass a live envelope or a stored message."
_PLAN_NOT_RECORDED_COPY = (
    "Not recorded for this answer — only a live answer carries the query plan."
)
_RETRIEVAL_NOT_RECORDED_COPY = (
    "Not recorded for this answer — retrieval scores are not persisted with a stored answer."
)
_NO_RETRIEVAL_COPY = "No chunks were retrieved for this answer (a refusal, or an empty index)."
_NO_TRACE_COPY = "No node trace was recorded for this answer."
_RRF_NOTE = (
    "The fused RRF score itself is not carried on the retrieved-chunk record: the chip shows the "
    "0-based rank that score produced."
)
_VERDICT_NOTE = (
    "Deterministic pre-pass rejections and per-claim verdicts are not carried on the answer "
    "record; they surface only through the node events and the answer text, so this panel shows "
    "no verdict counts."
)

#: Layout styles. Inline only where a layout choice is per-row; every color, size, and
#: radius still comes from the token layer's classes (never a second scale).
_ROW_STYLE = "display:flex;flex-wrap:wrap;gap:0.3rem 0.6rem;align-items:baseline;margin:0.15rem 0"
_PAIR_STYLE = (
    "display:flex;flex-wrap:wrap;justify-content:space-between;gap:0.5rem;"
    "align-items:baseline;margin:0.12rem 0"
)
_LABEL_STYLE = "margin:0"
_ERROR_STYLE = "margin:0 0 0.45rem 0"

#: ``panel``'s modifier for a group that must not draw a second card: the expander
#: already paints one, so a nested ``isip-panel`` box would read as a hole in it.
_FLUSH = "isip-flush"


# --- the normalized view -----------------------------------------------------------


@dataclass(frozen=True)
class _NodeRow:
    """One node's trace entry, read off either a ``NodeEvent`` or its stored dict."""

    node: str
    duration_ms: float | None
    tokens_in: int | None
    tokens_out: int | None
    error: str | None


@dataclass(frozen=True)
class _PanelView:
    """One answer's renderable facts, normalized from whichever argument was passed.

    ``refusal_reason`` and ``node`` names are already display text (the dash when the
    record carried nothing), so no renderer has to decide what an absent value looks
    like twice.
    """

    live: bool
    plan: Any | None
    nodes: tuple[_NodeRow, ...]
    chunks: tuple[Any, ...]
    grounded: bool | None
    refusal_reason: str
    citation_count: str
    planner_degraded: bool | None
    """The degraded-planner flag, or ``None`` when the answer never carried one.

    ``None`` (the stored-message shape) is *not* ``False``: the store has no column for
    the flag, so a historical answer is "not recorded", never "not degraded". That is
    the same distinction :data:`_UNKNOWN_LABEL` draws for ``grounded``.
    """


# --- the public renderer -----------------------------------------------------------


def render_debug_panel(
    *,
    envelope: AnswerEnvelope | None = None,
    message: Message | None = None,
    doc_names: Mapping[str, str] | None = None,
) -> None:
    """Render the observability panel for one answer: plan, path, retrieval, grounding.

    A sidebar section, collapsed by default, following the debug-panel contract in
    ``docs/OBSERVABILITY.md``. Exactly one of ``envelope`` / ``message`` carries the
    answer: a live answer arrives as the frozen ``AnswerEnvelope`` the graph returned
    (the only shape carrying ``plan`` and ``retrieval_debug``), a historical one as the
    stored ``Message`` read back from the chat store. The sections a stored answer
    cannot fill state that they were not recorded instead of rendering empty — including
    the planner-degraded flag, which only a live envelope carries.

    Args:
        envelope: The live answer to render. Wins when both arguments are given.
        message: The stored answer to render when there is no live envelope.
        doc_names: ``doc_id`` → display filename map for the retrieval rows. ``None``
            falls back to a truncated id, which is what a caller with no catalog (or a
            document since removed from the index) can render honestly.

    Returns:
        ``None``. The panel draws into the current container.
    """

    with st.expander(_EXPANDER_LABEL, expanded=False):
        view = _answer_view(envelope=envelope, message=message)
        if view is None:
            caption_note(_NO_ANSWER_COPY)
            return
        _render_plan(view)
        _render_path(view)
        _render_retrieval(view, doc_names)
        _render_grounding(view)


# --- normalization -----------------------------------------------------------------


def _answer_view(
    *,
    envelope: AnswerEnvelope | None,
    message: Message | None,
) -> _PanelView | None:
    """Normalize whichever answer shape the caller passed, or ``None`` for neither.

    A stored message has no plan and no retrieval scores by construction — the store
    persists the plan nowhere and the chunks nowhere — so its view carries ``None`` and
    ``()``, and the sections read that as "not recorded" rather than as "empty".
    """

    if envelope is not None:
        return _PanelView(
            live=True,
            plan=_attr(envelope, "plan"),
            nodes=_node_rows(_attr(envelope, "node_trace")),
            chunks=_sorted_chunks(_attr(envelope, "retrieval_debug")),
            grounded=_flag(_attr(envelope, "grounded")),
            refusal_reason=_as_text(_attr(envelope, "refusal_reason")),
            citation_count=_count_text(_attr(envelope, "citations")),
            planner_degraded=_flag(_attr(envelope, "planner_degraded")),
        )
    if message is not None:
        return _PanelView(
            live=False,
            plan=None,
            nodes=_node_rows(_attr(message, "node_trace")),
            chunks=(),
            grounded=_flag(_attr(message, "grounded")),
            refusal_reason=_as_text(_attr(message, "refusal_reason")),
            citation_count=_count_text(_attr(message, "citations")),
            # A stored row has no planner-degraded column (see ``final_envelope``'s
            # persistence note), so history reads as never recorded rather than clear.
            planner_degraded=None,
        )
    return None


def _node_rows(records: Any) -> tuple[_NodeRow, ...]:
    """The trace entries as rows, in the order the graph recorded them.

    ``node_trace`` is a list of ``NodeEvent`` instances on a live envelope and a
    JSON-decoded list of dicts on a stored message, so entries are read through
    :func:`_attr`. An empty list (a row written before tracing landed) and a ``None``
    trace both yield ``()`` rather than raising.
    """

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return ()
    return tuple(_node_row(record) for record in records)


def _node_row(record: Any) -> _NodeRow:
    """One trace entry read off a ``NodeEvent`` or its stored dict form."""

    return _NodeRow(
        node=_as_text(_attr(record, "node")),
        duration_ms=_number(_attr(record, "duration_ms")),
        tokens_in=_int_or_none(_attr(record, "tokens_in")),
        tokens_out=_int_or_none(_attr(record, "tokens_out")),
        error=_optional_text(_attr(record, "error")),
    )


def _sorted_chunks(records: Any) -> tuple[Any, ...]:
    """The selected chunks ordered by ``fused_rank`` — the order the fused score made.

    ``ScoredChunk`` persists the rank rather than the RRF score it came from (the score
    is a function of both retrievers' ranks, which are not stored), so the rank is the
    sortable fact. A record with no rank sorts after the ranked ones while keeping the
    caller's order — ``sorted`` is stable — because inventing a position for it would
    misstate the retrieval.
    """

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return ()
    return tuple(sorted(records, key=_rank_key))


def _rank_key(chunk: Any) -> tuple[int, int]:
    rank = _rank_of(chunk)
    return (1, 0) if rank is None else (0, rank)


# --- sections ----------------------------------------------------------------------


def _render_plan(view: _PanelView) -> None:
    """Render section 1: the ``query_plan`` the router switched on."""

    with panel(title=_PLAN_TITLE, extra_classes=_FLUSH):
        plan = view.plan
        if plan is None:
            caption_note(_PLAN_NOT_RECORDED_COPY)
            return
        st.html(
            _field_row("kind", _as_text(_attr(plan, "kind")))
            + _field_row("sub-queries", _joined(_attr(plan, "sub_queries")))
            + _field_row("sections", _joined(_attr(plan, "required_sections")))
            + _field_row("fiscal period", _as_text(_attr(plan, "fiscal_period")))
            + _field_row("missing aspect", _as_text(_attr(plan, "missing_aspect")))
        )


def _render_path(view: _PanelView) -> None:
    """Render section 2: the ordered node list, the route taken, and any failure."""

    with panel(title=_PATH_TITLE, extra_classes=_FLUSH):
        if not view.nodes:
            caption_note(_NO_TRACE_COPY)
            return
        route = _route_of(view.nodes)
        badge(route, kind="neutral" if route == _ROUTE_UNRECORDED else "accent")
        st.html("".join(_node_row_html(row) for row in view.nodes))


def _route_of(nodes: Sequence[_NodeRow]) -> str:
    """The route the trace shows, inferred from which nodes ran (ADR-001's three paths).

    Order matters: a decomposed run also passes through ``hybrid_retrieve`` and a
    refused one stops at ``refuse`` before retrieval, so the more specific marker is
    tested first.
    """

    names = {row.node for row in nodes}
    if _REFUSE_NODE in names:
        return _ROUTE_REFUSED
    if _DECOMPOSE_NODE in names:
        return _ROUTE_DECOMPOSED
    if _RETRIEVE_NODE in names:
        return _ROUTE_SIMPLE
    return _ROUTE_UNRECORDED


def _render_retrieval(view: _PanelView, doc_names: Mapping[str, str] | None) -> None:
    """Render section 3: the selected chunks, in fused-rank order."""

    with panel(title=_RETRIEVAL_TITLE, extra_classes=_FLUSH):
        if not view.chunks:
            caption_note(_NO_RETRIEVAL_COPY if view.live else _RETRIEVAL_NOT_RECORDED_COPY)
            return
        st.html("".join(_chunk_card(chunk, doc_names) for chunk in view.chunks))
        if any(_number(_attr(chunk, "fused_score")) is None for chunk in view.chunks):
            caption_note(_RRF_NOTE)


def _render_grounding(view: _PanelView) -> None:
    """Render section 4: the grounded flag, the refusal reason, and the citation count."""

    with panel(title=_GROUNDING_TITLE, extra_classes=_FLUSH):
        if view.grounded is None:
            badge(_UNKNOWN_LABEL, kind="neutral")
        else:
            badge(
                _GROUNDED_LABEL if view.grounded else _UNGROUNDED_LABEL,
                kind="success" if view.grounded else "warn",
            )
        if view.planner_degraded is True:
            badge(PLANNER_DEGRADED_LABEL, kind="error")
            caption_note(PLANNER_DEGRADED_COPY)
        elif view.planner_degraded is None:
            caption_note(PLANNER_DEGRADED_NOT_RECORDED_COPY)
        figure(view.citation_count, prefix="citations ")
        st.html(_field_row("refusal", view.refusal_reason))
        caption_note(_VERDICT_NOTE)


# --- markup ------------------------------------------------------------------------


def _node_row_html(row: _NodeRow) -> str:
    duration = _escape(_format_ms(row.duration_ms))
    cells = [
        f'<span class="isip-figure">{_escape(row.node)}</span>',
        f'<span class="isip-figure isip-figure-muted">{duration}</span>',
        f'<span class="isip-figure isip-figure-muted">{_escape(_format_tokens(row))}</span>',
    ]
    if row.error is not None:
        cells.append(f'<span class="isip-badge isip-badge-error">{_escape(_FAILED_LABEL)}</span>')
    markup = f'<div style="{_ROW_STYLE}">{"".join(cells)}</div>'
    if row.error is not None:
        markup += (
            f'<div class="isip-figure isip-figure-muted" style="{_ERROR_STYLE}">'
            f"{_escape(row.error)}</div>"
        )
    return markup


def _chunk_card(chunk: Any, doc_names: Mapping[str, str] | None) -> str:
    rank = _rank_of(chunk)
    page = _int_or_none(_attr(chunk, "page"))
    doc_label = _doc_label(_as_text(_attr(chunk, "doc_id")), doc_names)
    head = (
        f'<span class="isip-chip isip-chip-accent">'
        f"{_escape(_DASH if rank is None else f'#{rank}')}</span>"
        f'<span class="isip-figure">{_escape(doc_label)}</span>'
        f'<span class="isip-figure isip-figure-muted">'
        f"{_escape(_DASH if page is None else f'page {page}')}</span>"
    )
    return (
        f'<div class="isip-panel">'
        f'<div style="{_ROW_STYLE}">{head}</div>'
        f'<div class="isip-figure isip-figure-muted">{_escape(_scores_line(chunk))}</div>'
        f'<div class="isip-note">{_escape(_join_path(_attr(chunk, "section_path")))}</div>'
        f'<div class="isip-figure isip-figure-muted">{_escape(_snippet(chunk))}</div>'
        f"</div>"
    )


def _field_row(key: str, value: str) -> str:
    return (
        f'<div style="{_PAIR_STYLE}">'
        f'<span class="isip-label" style="{_LABEL_STYLE}">{_escape(key)}</span>'
        f'<span class="isip-figure">{_escape(value)}</span>'
        f"</div>"
    )


def _scores_line(chunk: Any) -> str:
    """``dense +0.4898 · sparse 8.359 · fused 0.0328``, each absent score as a dash."""

    return " · ".join(
        (
            f"dense {_format_score(_attr(chunk, 'dense_score'), digits=4, signed=True)}",
            f"sparse {_format_score(_attr(chunk, 'sparse_score'), digits=3)}",
            f"fused {_format_score(_attr(chunk, 'fused_score'), digits=4)}",
        )
    )


# --- field readers -----------------------------------------------------------------


def _attr(record: Any, name: str) -> Any:
    """Read ``name`` off a record in either shape: an object or a stored dict.

    The two callers supply the same records in two forms — a live envelope carries
    ``NodeEvent``/``ScoredChunk`` instances, a stored message carries the JSON dicts
    they were flattened into — and this is the single place that difference is handled,
    so no renderer needs to know which caller it is serving. ``getattr`` with a default
    also covers an older record that predates a field.
    """

    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _number(value: Any) -> float | None:
    """``value`` as a finite float, or ``None``. Bools are not numbers here."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    """``value`` as an int, or ``None`` when it is absent or not a number."""

    number = _number(value)
    return None if number is None else int(number)


def _flag(value: Any) -> bool | None:
    """The grounding flag as a bool, or ``None`` when the record never carried one."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_text(value: Any) -> str:
    return _optional_text(value) or _DASH


def _joined(value: Any) -> str:
    """A list field (sub-queries, required sections) as one middot-joined line."""

    if isinstance(value, str):
        return _as_text(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return _DASH
    items = []
    for item in value:
        text = _optional_text(item)
        if text is not None:
            items.append(text)
    return " · ".join(items) if items else _DASH


def _join_path(value: Any) -> str:
    parts = _path_parts(value)
    return " > ".join(parts) if parts else _DASH


def _path_parts(value: Any) -> list[str]:
    """The non-empty elements of a ``section_path`` (a list, or a single string)."""

    if isinstance(value, str):
        return [_as_text(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [text for part in value if (text := _optional_text(part)) is not None]


def _count_text(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return f"{len(value):,}"
    return _DASH


def _rank_of(chunk: Any) -> int | None:
    rank = _int_or_none(_attr(chunk, "fused_rank"))
    return rank if rank is not None and rank >= 0 else None


# --- formatting --------------------------------------------------------------------


def _format_ms(value: float | None) -> str:
    number = _number(value)
    return _DASH if number is None else f"{number:,.1f} ms"


def _format_tokens(row: _NodeRow) -> str:
    """``tokens 5,117 → 214``, or the dash when the node reported no usage at all.

    ``None`` on either side means "not measured" (the node made no LLM call), which is
    the token convention ``src/utils/tracing.py`` documents — never a zero.
    """

    if row.tokens_in is None and row.tokens_out is None:
        return _DASH
    return f"tokens {_token_count(row.tokens_in)} → {_token_count(row.tokens_out)}"


def _token_count(value: int | None) -> str:
    return _DASH if value is None else f"{value:,}"


def _format_score(value: Any, *, digits: int, signed: bool = False) -> str:
    """A retrieval score at a fixed precision — never a raw float repr."""

    number = _number(value)
    if number is None:
        return _DASH
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _truncate(text: str) -> str:
    """Collapse whitespace and cut to :data:`_SNIPPET_CHARS` including the ellipsis.

    Collapsing first is what makes a PDF snippet readable: the extracted text arrives
    with hard line breaks at the original page's column width.
    """

    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_CHARS:
        return collapsed
    return collapsed[: _SNIPPET_CHARS - 1].rstrip() + "…"


def _doc_label(doc_id: str, doc_names: Mapping[str, str] | None) -> str:
    if doc_id == _DASH:
        return _DASH
    if doc_names is not None:
        name = doc_names.get(doc_id)
        if name:
            return str(name)
    if len(doc_id) <= _SHORT_ID_CHARS:
        return doc_id
    return f"{doc_id[: _SHORT_ID_CHARS - 1]}…"


def _snippet(chunk: Any) -> str:
    """The chunk's evidence text, whitespace-collapsed and capped for the rail.

    The child window is what was scored — ranking happens at child granularity — so it
    is the text the three scores explain, while the citation cards already carry the
    parent expansion. A record that does carry its own ``snippet`` wins, so a future
    shape renders without a change here.
    """

    for name in ("snippet", "child_text", "parent_text"):
        text = _optional_text(_attr(chunk, name))
        if text is not None:
            return _truncate(text)
    return _DASH


def _escape(text: str) -> str:
    """Escape document-derived text (ids, section paths, snippets) for HTML.

    ``st.html`` sanitizes with DOMPurify as well, but every corpus-derived string is
    escaped here first: the panel renders retrieved text, and retrieved text is the one
    input in this app that a user did not type.
    """

    return _html.escape(text, quote=True)

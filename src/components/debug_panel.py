from __future__ import annotations

import html as _html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.components.theme import badge, caption_note, figure, panel

if TYPE_CHECKING:
    from src.chains.state import AnswerEnvelope
    from src.store.sessions import Message

__all__ = ["render_debug_panel"]


_EXPANDER_LABEL = "Debug panel"

_PLAN_TITLE = "Plan"
_PATH_TITLE = "Path"
_RETRIEVAL_TITLE = "Retrieval"
_GROUNDING_TITLE = "Grounding"

_DASH = "—"

_REFUSE_NODE = "refuse"
_DECOMPOSE_NODE = "decompose_question"
_RETRIEVE_NODE = "hybrid_retrieve"

_ROUTE_SIMPLE = "simple"
_ROUTE_DECOMPOSED = "decomposed"
_ROUTE_REFUSED = "refused"
_ROUTE_UNRECORDED = "unrecorded"

_SNIPPET_CHARS = 200

_SHORT_ID_CHARS = 16

_FAILED_LABEL = "failed"

_GROUNDED_LABEL = "grounded"
_UNGROUNDED_LABEL = "ungrounded"
_UNKNOWN_LABEL = "not recorded"

PLANNER_DEGRADED_LABEL = "planner degraded"
PLANNER_DEGRADED_COPY = (
    "This refusal was not a judgement: the query planner failed, so the question was "
    "classified out of scope instead of being reasoned about."
)
PLANNER_DEGRADED_NOT_RECORDED_COPY = (
    "Not recorded for this answer — the planner-degraded marker is carried only on a "
    "live answer, so a stored row says neither way."
)

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

_ROW_STYLE = "display:flex;flex-wrap:wrap;gap:0.3rem 0.6rem;align-items:baseline;margin:0.15rem 0"
_PAIR_STYLE = (
    "display:flex;flex-wrap:wrap;justify-content:space-between;gap:0.5rem;"
    "align-items:baseline;margin:0.12rem 0"
)
_LABEL_STYLE = "margin:0"
_ERROR_STYLE = "margin:0 0 0.45rem 0"

_FLUSH = "isip-flush"


@dataclass(frozen=True)
class _NodeRow:

    node: str
    duration_ms: float | None
    tokens_in: int | None
    tokens_out: int | None
    error: str | None


@dataclass(frozen=True)
class _PanelView:

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


def render_debug_panel(
    *,
    envelope: AnswerEnvelope | None = None,
    message: Message | None = None,
    doc_names: Mapping[str, str] | None = None,
) -> None:

    with st.expander(_EXPANDER_LABEL, expanded=False):
        view = _answer_view(envelope=envelope, message=message)
        if view is None:
            caption_note(_NO_ANSWER_COPY)
            return
        _render_plan(view)
        _render_path(view)
        _render_retrieval(view, doc_names)
        _render_grounding(view)


def _answer_view(
    *,
    envelope: AnswerEnvelope | None,
    message: Message | None,
) -> _PanelView | None:

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
            planner_degraded=None,
        )
    return None


def _node_rows(records: Any) -> tuple[_NodeRow, ...]:

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return ()
    return tuple(_node_row(record) for record in records)


def _node_row(record: Any) -> _NodeRow:

    return _NodeRow(
        node=_as_text(_attr(record, "node")),
        duration_ms=_number(_attr(record, "duration_ms")),
        tokens_in=_int_or_none(_attr(record, "tokens_in")),
        tokens_out=_int_or_none(_attr(record, "tokens_out")),
        error=_optional_text(_attr(record, "error")),
    )


def _sorted_chunks(records: Any) -> tuple[Any, ...]:

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return ()
    return tuple(sorted(records, key=_rank_key))


def _rank_key(chunk: Any) -> tuple[int, int]:
    rank = _rank_of(chunk)
    return (1, 0) if rank is None else (0, rank)


def _render_plan(view: _PanelView) -> None:

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

    with panel(title=_PATH_TITLE, extra_classes=_FLUSH):
        if not view.nodes:
            caption_note(_NO_TRACE_COPY)
            return
        route = _route_of(view.nodes)
        badge(route, kind="neutral" if route == _ROUTE_UNRECORDED else "accent")
        st.html("".join(_node_row_html(row) for row in view.nodes))


def _route_of(nodes: Sequence[_NodeRow]) -> str:

    names = {row.node for row in nodes}
    if _REFUSE_NODE in names:
        return _ROUTE_REFUSED
    if _DECOMPOSE_NODE in names:
        return _ROUTE_DECOMPOSED
    if _RETRIEVE_NODE in names:
        return _ROUTE_SIMPLE
    return _ROUTE_UNRECORDED


def _render_retrieval(view: _PanelView, doc_names: Mapping[str, str] | None) -> None:

    with panel(title=_RETRIEVAL_TITLE, extra_classes=_FLUSH):
        if not view.chunks:
            caption_note(_NO_RETRIEVAL_COPY if view.live else _RETRIEVAL_NOT_RECORDED_COPY)
            return
        st.html("".join(_chunk_card(chunk, doc_names) for chunk in view.chunks))
        if any(_number(_attr(chunk, "fused_score")) is None for chunk in view.chunks):
            caption_note(_RRF_NOTE)


def _render_grounding(view: _PanelView) -> None:

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

    return " · ".join(
        (
            f"dense {_format_score(_attr(chunk, 'dense_score'), digits=4, signed=True)}",
            f"sparse {_format_score(_attr(chunk, 'sparse_score'), digits=3)}",
            f"fused {_format_score(_attr(chunk, 'fused_score'), digits=4)}",
        )
    )


def _attr(record: Any, name: str) -> Any:

    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _number(value: Any) -> float | None:

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:

    number = _number(value)
    return None if number is None else int(number)


def _flag(value: Any) -> bool | None:

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


def _format_ms(value: float | None) -> str:
    number = _number(value)
    return _DASH if number is None else f"{number:,.1f} ms"


def _format_tokens(row: _NodeRow) -> str:

    if row.tokens_in is None and row.tokens_out is None:
        return _DASH
    return f"tokens {_token_count(row.tokens_in)} → {_token_count(row.tokens_out)}"


def _token_count(value: int | None) -> str:
    return _DASH if value is None else f"{value:,}"


def _format_score(value: Any, *, digits: int, signed: bool = False) -> str:

    number = _number(value)
    if number is None:
        return _DASH
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _truncate(text: str) -> str:

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

    for name in ("snippet", "child_text", "parent_text"):
        text = _optional_text(_attr(chunk, name))
        if text is not None:
            return _truncate(text)
    return _DASH


def _escape(text: str) -> str:

    return _html.escape(text, quote=True)

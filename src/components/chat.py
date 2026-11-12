from __future__ import annotations

import html as _html
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from src.components import debug_panel
from src.components.theme import badge as theme_badge
from src.components.theme import caption_note as theme_caption_note
from src.components.theme import eyebrow as theme_eyebrow
from src.components.theme import footer_note as theme_footer_note
from src.components.theme import hero_mark as theme_hero_mark
from src.components.theme import hero_sub as theme_hero_sub
from src.components.theme import hero_title as theme_hero_title

if TYPE_CHECKING:
    from src.chains.state import AnswerEnvelope

__all__ = [
    "SUGGESTED_QUESTIONS",
    "render_answer",
    "render_answer_error",
    "render_citation_card",
    "render_composer",
    "render_empty_state",
    "render_footer_note",
    "render_pending_answer",
    "render_stored_answer",
    "render_transcript",
]

_ANSWER_KEY = "isip-answer"
_QUESTION_KEY = "isip-question"

_KEY_COUNTER_KEY = "_isip_chat_key_counter"

_CLASS_PANEL = "isip-panel"
_CLASS_CHIP = "isip-chip"
_CLASS_CHIP_ACCENT = "isip-chip-accent"
_CLASS_FIGURE = "isip-figure"
_CLASS_FIGURE_MUTED = "isip-figure-muted"

HERO_GREETING = "Ask your documents anything."

HERO_ACCENT = "documents"

HERO_SUB = (
    "Answers come only from the documents in this workspace, and every claim carries its citation."
)

HERO_EYEBROW = "TRY A DEMO"

HERO_NO_DOCS = "No document indexed yet — upload a PDF in the sidebar, then ask."

_SUGGESTION_CHIPS: tuple[tuple[str, str], ...] = (
    ("What was revenue in FY2023?", ":material/show_chart:"),
    ("Compare operating income FY2023 vs FY2022", ":material/compare_arrows:"),
    ("Which segment had the highest revenue?", ":material/donut_small:"),
    ("What was net income and diluted EPS?", ":material/payments:"),
    ("Summarize FY2023 key results", ":material/summarize:"),
    ("What was the margin trend?", ":material/trending_up:"),
)

SUGGESTED_QUESTIONS: tuple[str, ...] = tuple(question for question, _ in _SUGGESTION_CHIPS)

_SUGGESTION_COLUMNS = 3

COMPOSER_PLACEHOLDER = "Ask a question about your documents…"

_REFUSAL_REASON_COPY: dict[str, str] = {
    "time_period": "The question asks about a period the indexed documents do not cover.",
    "topic": "The topic is outside what the indexed documents cover.",
    "entity": "The entity is not one the indexed documents discuss.",
    "no_documents": "No document is indexed in this session yet.",
    "unspecified": "No passage in the indexed documents supports this question.",
}

_UNGROUNDED_COPY = (
    "Not every claim in this answer is supported by the retrieved passages — treat it "
    "as unverified and check the citations below."
)

PLANNER_DEGRADED_LABEL = debug_panel.PLANNER_DEGRADED_LABEL
PLANNER_DEGRADED_COPY = debug_panel.PLANNER_DEGRADED_COPY

PENDING_COPY = "Searching the documents, then checking the answer against them."

_SUGGESTION_KEY = "isip-suggestion"

_COMPOSER_MAX_CHARS = 4000


@dataclass(frozen=True)
class _AnswerView:

    answer_md: str
    grounded: bool | None
    refusal_reason: str | None
    citations: tuple[object, ...]
    planner_degraded: bool | None


def render_empty_state(*, doc_labels: Sequence[str] | None = None) -> str | None:

    theme_hero_mark()
    theme_hero_title(HERO_GREETING, accent=HERO_ACCENT)
    theme_hero_sub(_hero_subtext(doc_labels))
    theme_eyebrow(HERO_EYEBROW)
    for row_start in range(0, len(_SUGGESTION_CHIPS), _SUGGESTION_COLUMNS):
        columns = st.columns(_SUGGESTION_COLUMNS)
        row = _SUGGESTION_CHIPS[row_start : row_start + _SUGGESTION_COLUMNS]
        for offset, (question, icon) in enumerate(row):
            with columns[offset]:
                if st.button(
                    question,
                    key=f"{_SUGGESTION_KEY}-{row_start + offset}",
                    icon=icon,
                    use_container_width=True,
                ):
                    return question
    return None


def _hero_subtext(doc_labels: Sequence[str] | None) -> str:
    if doc_labels:
        return f"Indexed and searchable: {', '.join(doc_labels)}. {HERO_SUB}"
    return f"{HERO_NO_DOCS} {HERO_SUB}"


def render_footer_note(text: str) -> None:

    theme_footer_note(text)


def render_transcript(
    turns: Sequence[object],
    messages: Sequence[object] = (),
    *,
    doc_names: Mapping[str, str] | None = None,
) -> None:

    answers = _answers_by_turn_id(messages)
    for turn in turns:
        if _is_user_turn(turn):
            with _question_container(_turn_key(turn)):
                st.markdown(str(_turn_content(turn)))
            continue
        message = answers.get(_turn_id(turn))
        if message is None:
            with st.chat_message(name="assistant"):
                st.markdown(str(_turn_content(turn)))
            continue
        render_stored_answer(message, doc_names=doc_names, role_key=_turn_key(turn))


def render_answer(
    answer: AnswerEnvelope,
    *,
    doc_names: Mapping[str, str] | None = None,
    role_key: str | None = None,
) -> None:

    view = _AnswerView(
        answer_md=answer.answer_md or "",
        grounded=answer.grounded,
        refusal_reason=answer.refusal_reason or None,
        citations=tuple(answer.citations or ()),
        planner_degraded=bool(answer.planner_degraded),
    )
    _render_answer_view(view, doc_names=doc_names, role_key=role_key)


def render_stored_answer(
    message: object,
    *,
    doc_names: Mapping[str, str] | None = None,
    role_key: str | None = None,
) -> None:

    view = _AnswerView(
        answer_md=str(getattr(message, "answer_md", "") or ""),
        grounded=getattr(message, "grounded", None),
        refusal_reason=(getattr(message, "refusal_reason", None) or None),
        citations=tuple(getattr(message, "citations", ()) or ()),
        planner_degraded=getattr(message, "planner_degraded", None),
    )
    _render_answer_view(view, doc_names=doc_names, role_key=role_key)


def render_citation_card(citation: object, *, doc_names: Mapping[str, str] | None = None) -> None:

    ref_index, doc_id, page, section_path, snippet = _citation_fields(citation)
    doc_label = (doc_names or {}).get(doc_id) or _short_doc_label(doc_id)
    location = f"page {page}" if page is not None else "page not recorded"
    with st.expander(f"[{ref_index}]  {doc_label} · {location}"):
        _emit_html(
            f'<div class="{_CLASS_PANEL}">'
            f'<span class="{_CLASS_CHIP} {_CLASS_CHIP_ACCENT}">[{ref_index}]</span>'
            f' <span class="{_CLASS_FIGURE} {_CLASS_FIGURE_MUTED}">'
            f"{_escape(location)}</span></div>"
        )
        if section_path:
            st.markdown(f"**Section:** {' > '.join(section_path)}")
        else:
            theme_caption_note("Section not recorded")
        if snippet.strip():
            st.markdown(snippet)
        else:
            theme_caption_note("No snippet recorded for this citation.")
        theme_caption_note(f"Document id: {doc_id}")


def render_pending_answer(label: str = PENDING_COPY) -> None:

    with st.chat_message(name="assistant"), st.spinner(label):
        st.caption("Retrieving, then synthesising, then verifying against sources.")


def render_answer_error(message: str, *, details: str | None = None, retry: bool = False) -> bool:

    st.error(message)
    if details:
        st.caption(details)
    if retry:
        return bool(st.button("Retry question", key="isip-chat-retry"))
    return False


def _emit_html(markup: str) -> None:

    st.html(markup)


def _escape(text: object) -> str:
    return _html.escape(str(text), quote=True)


def render_composer(
    *,
    key: str = "isip-composer",
    placeholder: str = COMPOSER_PLACEHOLDER,
    disabled: bool = False,
    max_chars: int | None = _COMPOSER_MAX_CHARS,
) -> str | None:

    submitted = st.chat_input(placeholder, key=key, disabled=disabled, max_chars=max_chars)
    if submitted is None:
        return None
    return str(submitted).strip()


def _render_answer_view(
    view: _AnswerView,
    *,
    doc_names: Mapping[str, str] | None,
    role_key: str | None = None,
) -> None:

    with st.chat_message(name="assistant"):
        with st.container(border=False, key=_keyed(_ANSWER_KEY, role_key)):
            if view.answer_md.strip():
                st.markdown(view.answer_md)
            elif not view.refusal_reason:
                theme_caption_note("This turn recorded no answer text.")
        _render_markers(view)
        for citation in view.citations:
            render_citation_card(citation, doc_names=doc_names)


def _render_markers(view: _AnswerView) -> None:

    if view.planner_degraded is True:
        theme_badge(PLANNER_DEGRADED_LABEL, kind="error")
        theme_caption_note(PLANNER_DEGRADED_COPY)
        return
    if view.refusal_reason:
        theme_badge("Refusal", kind="neutral")
        reason = _REFUSAL_REASON_COPY.get(view.refusal_reason, _REFUSAL_REASON_COPY["unspecified"])
        theme_caption_note(reason)
        return
    if view.grounded is False or (view.grounded is None and not view.citations):
        theme_badge("Ungrounded", kind="warn")
        theme_caption_note(_UNGROUNDED_COPY)


def _keyed(role_key: str, suffix: str | None) -> str:

    if suffix is None:
        counter = int(st.session_state.get(_KEY_COUNTER_KEY, 0))
        st.session_state[_KEY_COUNTER_KEY] = counter + 1
        return f"{role_key}-{counter}"
    return f"{role_key}-{suffix}"


def _turn_key(turn: object) -> str | None:
    turn_id = _turn_id(turn)
    return None if turn_id is None else str(turn_id)


@contextmanager
def _question_container(role_key: str | None = None) -> Iterator[st.delta_generator.DeltaGenerator]:

    with (
        st.chat_message(name="user"),
        st.container(border=False, key=_keyed(_QUESTION_KEY, role_key)),
    ):
        yield st._main


def _answers_by_turn_id(messages: Sequence[object]) -> dict[int, object]:

    indexed: dict[int, object] = {}
    for message in messages:
        turn = getattr(message, "turn", None)
        turn_id = getattr(turn, "id", None) if turn is not None else None
        if isinstance(turn_id, int):
            indexed[turn_id] = message
    return indexed


def _is_user_turn(turn: object) -> bool:
    return getattr(turn, "role", "") == "user"


def _turn_content(turn: object) -> str:
    return str(getattr(turn, "content", "") or "")


def _turn_id(turn: object) -> int | None:
    turn_id = getattr(turn, "id", None)
    return turn_id if isinstance(turn_id, int) else None


def _citation_fields(
    citation: object,
) -> tuple[int, str, int | None, tuple[str, ...], str]:

    raw_index = getattr(citation, "ref_index", None)
    ref_index = raw_index if isinstance(raw_index, int) else 0
    doc_id = str(getattr(citation, "doc_id", "") or "")
    page = getattr(citation, "page", None)
    page_value = page if isinstance(page, int) else None
    raw_path = getattr(citation, "section_path", ()) or ()
    section_path = tuple(part for part in raw_path if isinstance(part, str))
    snippet = str(getattr(citation, "snippet", "") or "")
    return ref_index, doc_id, page_value, section_path, snippet


def _short_doc_label(doc_id: str) -> str:

    if len(doc_id) <= 12:
        return doc_id or "unknown document"
    return f"{doc_id[:12]}…"

"""The chat canvas: transcript renderer, citation cards, and the composer (task T18).

ADR-009's boundary lives here: this module **renders** the fields the pipeline already
decided. It reads :class:`~src.chains.state.AnswerEnvelope` (and the same fields off the
store's :class:`~src.store.sessions.Message` rows, which is how history survives a
reload) and calls nothing — no retrieval, no prompt, no gateway, no
:class:`~src.chains.graph` import. Every state the demo script exercises is present:
the hero greeting for an empty session, the transcript for a populated one, a waiting
bubble while the answer is produced, citation cards under every cited answer, an
explicit ungrounded marker when the verifier did not pass a claim through, and an
error block with a retry action for a failed turn.

Public surface
--------------
``render_empty_state``      hero greeting plus suggestion buttons (empty session)
``render_transcript``       a session's full transcript, oldest turn first
``render_answer``           one live :class:`AnswerEnvelope`
``render_stored_answer``    one stored assistant answer
``render_citation_card``    one ``[n]`` → document · page · section · snippet card
``render_pending_answer``   the waiting bubble while the graph runs
``render_answer_error``     a failed turn, with an optional retry action
``render_composer``         the docked question box; returns the submitted text

The canvas is theme-aware rather than theme-agnostic: it is built from Streamlit-native
primitives (``st.chat_message``, ``st.chat_input``, ``st.expander``) and marks its own
role containers with the widget keys the theme's stylesheet targets
(``st-key-isip-answer`` for the assistant's accent left rule, ``st-key-isip-question``
for the dimmed question). Citation cards are one complete ``st.html`` fragment per card
using the theme's ``isip-panel``/``isip-chip``/``isip-figure`` classes, with every
document-derived value HTML-escaped, so a filename or snippet cannot inject markup.
"""

from __future__ import annotations

import html as _html
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from src.components.theme import badge as theme_badge
from src.components.theme import caption_note as theme_caption_note

if TYPE_CHECKING:  # contract types only — no runtime import, so the module stays leaf-level
    from src.chains.state import AnswerEnvelope

__all__ = [
    "SUGGESTED_QUESTIONS",
    "render_answer",
    "render_answer_error",
    "render_citation_card",
    "render_composer",
    "render_empty_state",
    "render_pending_answer",
    "render_stored_answer",
    "render_transcript",
]

#: Widget keys for the role containers inside each chat bubble. The theme's stylesheet
#: targets the frontend-derived ``st-key-<key>`` classes on these containers to draw
#: the answer's accent rule and dim the question (``st.chat_message`` takes no
#: ``key`` itself, and separate ``st.html`` calls are sibling blocks — verified — so
#: a keyed container is the only hook that really wraps the content). The theme
#: matches these as **prefix** selectors, so these two prefixes are reserved: no
#: other widget key in this app may start with ``isip-answer-`` or
#: ``isip-question-``, or it inherits the role's border/padding.
_ANSWER_KEY = "isip-answer"
_QUESTION_KEY = "isip-question"

#: Session-state counter backing :func:`_keyed`'s generated suffixes. Not a widget
#: state and not durable app data — it only has to be distinct within one run.
_KEY_COUNTER_KEY = "_isip_chat_key_counter"

#: CSS class names :mod:`src.components.theme`'s stylesheet defines, used by this
#: module's own markup. Listed here rather than referenced through ``CSS_CLASSES`` so
#: a typo fails lint/read-time review instead of silently emitting an unstyled div.
_CLASS_PANEL = "isip-panel"
_CLASS_CHIP = "isip-chip"
_CLASS_CHIP_ACCENT = "isip-chip-accent"
_CLASS_FIGURE = "isip-figure"
_CLASS_FIGURE_MUTED = "isip-figure-muted"

#: The greeting shown on a session that has never been asked anything.
HERO_GREETING = "Ask your documents anything."

#: Shown on the hero when no document has been indexed yet — the honest empty state,
#: so an empty canvas reads as "upload first" rather than as a broken page.
HERO_NO_DOCS = "No document indexed yet — upload a PDF in the sidebar, then ask."

#: The three questions the hero offers as starting points: one that returns a cited
#: figure, one that fans out, one that is a follow-up. They mirror the demo script.
SUGGESTED_QUESTIONS: tuple[str, ...] = (
    "What was revenue in FY2023?",
    "Compare operating income FY2023 vs FY2022.",
    "Which segment drove that?",
)

#: Placeholder in the docked composer.
COMPOSER_PLACEHOLDER = "Ask a question about your documents…"

#: Caption on ``refusal_reason``: the store and the graph both record one of the five
#: reasons ``src.chains.nodes.refuse`` writes, and this maps each to the user-facing
#: reading. The answer's own markdown already carries the refusal sentence; this line
#: is the short marker a reviewer can scan down the transcript.
_REFUSAL_REASON_COPY: dict[str, str] = {
    "time_period": "The question asks about a period the indexed documents do not cover.",
    "topic": "The topic is outside what the indexed documents cover.",
    "entity": "The entity is not one the indexed documents discuss.",
    "no_documents": "No document is indexed in this session yet.",
    "unspecified": "No passage in the indexed documents supports this question.",
}

#: The explanation attached to the ungrounded marker — why a claim reaching the user
#: is not presented as a verified one (docs/SPEC.md §10.7).
_UNGROUNDED_COPY = (
    "Not every claim in this answer is supported by the retrieved passages — treat it "
    "as unverified and check the citations below."
)

#: Waiting bubble copy while the graph runs.
PENDING_COPY = "Searching the documents, then checking the answer against them."

#: Prefix for the hero suggestion buttons' widget keys, so the same suggestion on two
#: sessions never collides.
_SUGGESTION_KEY = "isip-suggestion"

#: Longest question the composer accepts; 4000 characters is a paste of several
#: questions, and a wall of text rarely survives analysis anyway.
_COMPOSER_MAX_CHARS = 4000


@dataclass(frozen=True)
class _AnswerView:
    """The one field set the canvas renders, read off either shape it arrives on.

    :class:`AnswerEnvelope` and the store's ``Message`` row carry the same four
    renderable fields; normalising to this view is what keeps one renderer instead of
    a live path and a history path that drift. The citation objects themselves stay
    duck-typed: the two packages each define their own ``Citation`` class with the
    same five attributes.
    """

    answer_md: str
    grounded: bool | None
    refusal_reason: str | None
    citations: tuple[object, ...]


def render_empty_state(
    *,
    suggestions: Sequence[str] = SUGGESTED_QUESTIONS,
    doc_labels: Sequence[str] = (),
) -> str | None:
    """Render the hero greeting for a session with no turns; return a clicked suggestion.

    Args:
        suggestions: Question prompts offered as buttons. Each is a real, labeled
            Streamlit button — an unlabeled icon button is not an accessibility-safe
            control (docs/CONSTRAINTS.md).
        doc_labels: Display names of the indexed documents, when any exist. Passed so
            the greeting can name what is searchable without this module reading the
            catalog itself.

    Returns:
        The text of the clicked suggestion, or ``None`` when nothing was clicked. The
        caller feeds the returned string into the composer's question path, so picking
        a suggestion behaves exactly like typing it.
    """

    st.markdown(f"### {HERO_GREETING}")
    if doc_labels:
        st.caption(f"Indexed: {', '.join(doc_labels)}")
    else:
        st.caption(HERO_NO_DOCS)
    for index, suggestion in enumerate(suggestions):
        if st.button(suggestion, key=f"{_SUGGESTION_KEY}-{index}"):
            return suggestion
    return None


def render_transcript(
    turns: Sequence[object],
    messages: Sequence[object] = (),
    *,
    doc_names: Mapping[str, str] | None = None,
) -> None:
    """Render a session's stored transcript, oldest turn first.

    Args:
        turns: Every role's turns in order — the ``SessionHistory.turns`` list. A
            user turn renders as the question; an assistant turn renders as its answer.
        messages: Assistant answers with their citation blocks
            (``SessionHistory.messages``). Each is paired to its turn by turn id; an
            assistant turn with no message row still renders, just without citations,
            so a half-written exchange stays visible instead of silently vanishing.
        doc_names: ``doc_id`` → display name, built by the caller from the document
            catalog. Without it a citation shows a shortened document id.
    """

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
    """Render one live answer returned by ``src.chains.graph.invoke``.

    Args:
        answer: The frozen envelope: ``answer_md``, ``citations``, ``grounded`` and
            ``refusal_reason``. Nothing else on the envelope reaches the user from this
            module — ``plan``, ``retrieval_debug`` and ``node_trace`` belong to the
            debug panel, which is a separate component.
        doc_names: ``doc_id`` → display name for the citation cards.
        role_key: Uniquifier for the role container's widget key. Streamlit refuses a
            duplicate key on one page, so a transcript with several answers needs one
            distinct value per answer; ``render_transcript`` derives it from the turn.
    """

    view = _AnswerView(
        answer_md=answer.answer_md or "",
        grounded=answer.grounded,
        refusal_reason=answer.refusal_reason or None,
        citations=tuple(answer.citations or ()),
    )
    _render_answer_view(view, doc_names=doc_names, role_key=role_key)


def render_stored_answer(
    message: object,
    *,
    doc_names: Mapping[str, str] | None = None,
    role_key: str | None = None,
) -> None:
    """Render one assistant answer read back from the chat store.

    The question it answered is rendered as the user turn that precedes it, so this
    covers the answer alone: the markdown, the grounding marker, the refusal reason
    and the citation cards — all fields of the stored ``Message`` row.

    Args:
        message: A :class:`~src.store.sessions.Message` with its ``citations``.
        doc_names: ``doc_id`` → display name for the citation cards.
        role_key: Uniquifier for the role container's widget key; see
            :func:`render_answer`.
    """

    view = _AnswerView(
        answer_md=str(getattr(message, "answer_md", "") or ""),
        grounded=getattr(message, "grounded", None),
        refusal_reason=(getattr(message, "refusal_reason", None) or None),
        citations=tuple(getattr(message, "citations", ()) or ()),
    )
    _render_answer_view(view, doc_names=doc_names, role_key=role_key)


def render_citation_card(citation: object, *, doc_names: Mapping[str, str] | None = None) -> None:
    """Render one citation as an expandable card: ``[n]`` → document · page · snippet.

    Args:
        citation: Either citation class — the graph's and the store's carry the same
            ``ref_index``/``doc_id``/``page``/``section_path``/``snippet`` attributes,
            and ``page`` is optional in both.
        doc_names: ``doc_id`` → display name; without it the shortened document id is
            shown, because a raw sha256 is not something a reader can follow.
    """

    ref_index, doc_id, page, section_path, snippet = _citation_fields(citation)
    doc_label = (doc_names or {}).get(doc_id) or _short_doc_label(doc_id)
    location = f"page {page}" if page is not None else "page not recorded"
    with st.expander(f"[{ref_index}]  {doc_label} · {location}"):
        # The accent chip carries the [n] marker; the rest of the metadata row is a
        # monospace figure so page numbers and document ids line up across cards.
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
    """Draw the assistant bubble the real answer replaces once the graph returns.

    Args:
        label: The status text shown while the answer is being produced.
    """

    with st.chat_message(name="assistant"), st.spinner(label):
        st.caption("Retrieving, then synthesising, then verifying against sources.")


def render_answer_error(message: str, *, details: str | None = None, retry: bool = False) -> bool:
    """Render a failed turn: the error banner plus an optional retry action.

    The rule this exists to keep (ADR-007): a failed turn is never an empty answer —
    the user is told what failed and offered the question again.

    Args:
        message: One-line, user-readable description of the failure.
        details: The typed error's class name and message, when the caller has one.
        retry: When ``True``, a labeled retry button is rendered.

    Returns:
        ``True`` exactly once, on the rerun in which the retry button was clicked.
    """

    st.error(message)
    if details:
        st.caption(details)
    if retry:
        return bool(st.button("Retry question", key="isip-chat-retry"))
    return False


def _emit_html(markup: str) -> None:
    """Write one raw HTML fragment, sanitized by ``st.html`` (DOMPurify).

    ``st.html`` is the only primitive that controls the DOM — ``st.markdown(``,
    ``unsafe_allow_html=True``) would re-parse the fragment as markdown — so every
    document-derived value this module interpolates is passed through
    :func:`_escape` first.
    """

    st.html(markup)


def _escape(text: object) -> str:
    """Escape document-derived text for HTML."""

    return _html.escape(str(text), quote=True)


def render_composer(
    *,
    key: str = "isip-composer",
    placeholder: str = COMPOSER_PLACEHOLDER,
    disabled: bool = False,
    max_chars: int | None = _COMPOSER_MAX_CHARS,
) -> str | None:
    """Render the docked composer and return the question the user submitted.

    The returned value is non-``None`` for exactly one rerun — the one triggered by
    the submit — which is what makes "invoke the graph once per question" a matter of
    the caller reading the return value instead of counting widget state.

    Args:
        key: Widget key. Distinct keys keep two composers on one page independent.
        placeholder: The greyed prompt inside the box.
        disabled: Grey the box while an answer is in flight, so a second submit cannot
            interleave with the first.
        max_chars: Upper bound on the question length, or ``None`` for no bound.

    Returns:
        The submitted question with surrounding whitespace stripped, or ``None``.
    """

    submitted = st.chat_input(placeholder, key=key, disabled=disabled, max_chars=max_chars)
    if submitted is None:
        return None
    return str(submitted).strip()


# --- answer rendering --------------------------------------------------------------


def _render_answer_view(
    view: _AnswerView,
    *,
    doc_names: Mapping[str, str] | None,
    role_key: str | None = None,
) -> None:
    """Render the answer body, its markers, and its citation cards.

    A refusal renders its copy and its reason marker but no "ungrounded" line: a
    refusal is a correct outcome (``grounded=True`` on that path by contract), so
    warning about it would be false.
    """

    with st.chat_message(name="assistant"):
        with st.container(border=False, key=_keyed(_ANSWER_KEY, role_key)):
            if view.answer_md.strip():
                st.markdown(view.answer_md)
            elif not view.refusal_reason:
                # The contract forbids an empty answer, so reaching here means a stored
                # row lost its body; showing the gap beats rendering silence.
                theme_caption_note("This turn recorded no answer text.")
        _render_markers(view)
        for citation in view.citations:
            render_citation_card(citation, doc_names=doc_names)


def _render_markers(view: _AnswerView) -> None:
    """Render the grounding badge/caption block under an answer."""

    if view.refusal_reason:
        theme_badge("Refusal", kind="neutral")
        reason = _REFUSAL_REASON_COPY.get(view.refusal_reason, _REFUSAL_REASON_COPY["unspecified"])
        theme_caption_note(reason)
        return
    if view.grounded is False or (view.grounded is None and not view.citations):
        theme_badge("Ungrounded", kind="warn")
        theme_caption_note(_UNGROUNDED_COPY)


def _keyed(role_key: str, suffix: str | None) -> str:
    """One unique widget key per rendered turn, rooted at the theme's role key.

    The theme's selectors match the frontend-derived ``st-key-<key>`` class, so the
    prefix must stay ``isip-answer`` / ``isip-question`` exactly. The suffix exists
    because Streamlit rejects a duplicate widget key on one page — three answers on
    the transcript would die on the third — and it is *generated* rather than trusted
    to the caller, so the failure cannot reach the app from a caller that forgets.
    A per-render counter in ``st.session_state`` keeps the keys distinct for one
    script run, which is the lifetime Streamlit itself gives them.
    """

    if suffix is None:
        counter = int(st.session_state.get(_KEY_COUNTER_KEY, 0))
        st.session_state[_KEY_COUNTER_KEY] = counter + 1
        return f"{role_key}-{counter}"
    return f"{role_key}-{suffix}"


def _turn_key(turn: object) -> str | None:
    """A stored turn's row id as a widget-key suffix, or ``None`` when unreadable."""

    turn_id = _turn_id(turn)
    return None if turn_id is None else str(turn_id)


@contextmanager
def _question_container(role_key: str | None = None) -> Iterator[st.delta_generator.DeltaGenerator]:
    """The user bubble: ``st.chat_message`` carrying the question role class.

    The theme's ``isip-question`` rule dims the question's own markdown and drops the
    accent rule, which is what makes the answer's orange edge the loudest mark on the
    canvas. ``st.chat_message`` takes no ``key`` (verified: it is not in its
    signature), so the class hook is a keyed ``st.container`` inside the bubble — the
    frontend derives ``st-key-<key>`` and renders the container's children inside
    that element, which a stray open/close ``st.html`` pair cannot do.
    """

    with (
        st.chat_message(name="user"),
        st.container(border=False, key=_keyed(_QUESTION_KEY, role_key)),
    ):
        yield st._main


# --- transcript plumbing ------------------------------------------------------------


def _answers_by_turn_id(messages: Sequence[object]) -> dict[int, object]:
    """Index stored answers by the turn id they belong to.

    ``Message.turn`` is the authoritative link; anything without a turn is dropped
    here because it cannot be placed in the transcript.
    """

    indexed: dict[int, object] = {}
    for message in messages:
        turn = getattr(message, "turn", None)
        turn_id = getattr(turn, "id", None) if turn is not None else None
        if isinstance(turn_id, int):
            indexed[turn_id] = message
    return indexed


def _is_user_turn(turn: object) -> bool:
    """True when a stored turn is a question (``role == 'user'``)."""

    return getattr(turn, "role", "") == "user"


def _turn_content(turn: object) -> str:
    """The stored turn's text, or ``""`` when the row carries none."""

    return str(getattr(turn, "content", "") or "")


def _turn_id(turn: object) -> int | None:
    """The stored turn's row id, or ``None`` — used to find its answer row."""

    turn_id = getattr(turn, "id", None)
    return turn_id if isinstance(turn_id, int) else None


# --- citation plumbing --------------------------------------------------------------


def _citation_fields(
    citation: object,
) -> tuple[int, str, int | None, tuple[str, ...], str]:
    """Read the five renderable citation fields off either citation class.

    Both citation classes agree on the field names, so this is a get-and-coerce rather
    than a converter: a field that is absent or wrongly typed degrades to the neutral
    value (``0``, ``""``, ``None``) that renders as "not recorded" rather than raising
    on one malformed row.
    """

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
    """A citation's document label when no display name is available.

    ``doc_id`` is the sha256 of the uploaded file — stable and unique, but not something
    a reader can scan — so a long one is shown truncated and the full id stays on the
    card's footer line.
    """

    if len(doc_id) <= 12:
        return doc_id or "unknown document"
    return f"{doc_id[:12]}…"

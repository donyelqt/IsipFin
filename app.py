"""IsipFin — Streamlit entrypoint.

The UI shell (ADR-009). This module wires the component modules together and owns
exactly four responsibilities, none of them business logic:

1. **Boundary** — nothing under ``src/components/`` (or this module) imports
   retrieval, prompt, or gateway internals. The pipeline is reached only through
   ``src.chains.graph.invoke()``, whose ``AnswerEnvelope`` is the single object
   that crosses into the UI.
2. **Environment** — ``settings.load_env()`` runs first, because the docs tell a
   reader to ``cp .env.example .env`` and nothing used to read that file.
3. **View routing** — the rail's nav writes ``st.session_state["isip-view"]`` and
   the canvas renders the matching surface. This is the information architecture
   the rail used to lack: previously every control lived in the rail at once.
4. **Turn lifecycle** — session state, the graph call, and persistence.

Run: ``streamlit run app.py`` (see docs/RUNBOOK.md).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import streamlit as st
from src.chains.graph import invoke
from src.chains.state import AnswerEnvelope, DocSummary, Turn
from src.components import chat, errors, sidebar, theme, upload
from src.components.debug_panel import render_debug_panel
from src.store.db import StoreError
from src.store.sessions import Message, SessionStore
from src.utils.settings import load_env

logger = logging.getLogger(__name__)

#: Session-state keys this module owns.
_QUESTION_KEY = "isip-question"
_ANSWER_KEY = "isip-answer"
_LAST_KEY = "isip-last-answer"
_LAST_SESSION_KEY = "isip-last-answer-session"
_VIEW_KEY = "isip-view"

#: The Diagnostics picker's widget key: the answer whose panel is drawn. A widget key
#: rather than a state key — the selection belongs to the run that made it, and nothing
#: else reads it.
_ANSWER_PICKER_KEY = "isip-diagnostics-answer"
_ANSWER_PICKER_LABEL = "Answer to inspect"

#: The picker's row for the live envelope, offered first when there is one: the live
#: answer is the only shape carrying the query plan and the retrieval scores.
_LIVE_OPTION_LABEL = "Current answer (live)"

#: Longest question a picker row renders before it is cut, ellipsis included.
_ANSWER_LABEL_CHARS = 72

_RAIL_WIDTH = 260  # the docs confirm an int means "auto behaviour at this width" (200-600)


def _open_store() -> SessionStore | None:
    """Open the chat store, or return ``None`` when it cannot be opened.

    A missing or unreadable database is a degraded UI (the rail's status card says
    so), never a crash: questions can still be asked and answered without
    persistence.
    """
    if "isip-store" not in st.session_state:
        try:
            st.session_state["isip-store"] = SessionStore()
        except Exception as exc:  # sqlite failures surface as several concrete types
            logger.warning("chat store unavailable: %s", exc)
            st.session_state["isip-store"] = None
    store: SessionStore | None = st.session_state["isip-store"]
    return store


def _doc_catalog() -> list[DocSummary]:
    """The document catalog the graph routes over, built from the on-disk index."""
    catalog: dict[str, dict[str, Any]] = upload.read_catalog()
    entries: list[DocSummary] = []
    for doc_id, entry in sorted(catalog.items()):
        sections = entry.get("sections") or []
        if isinstance(sections, str):
            sections = [sections]
        entries.append(
            DocSummary(
                doc_id=str(entry.get("doc_id", doc_id)),
                filename=str(entry.get("filename", "")),
                sections=[str(s) for s in sections],
                page_count=int(entry.get("page_count") or 0),
            )
        )
    return entries


def _history(store: SessionStore | None, session_id: str | None) -> list[Turn]:
    """The last six turns of the active session as graph history.

    Only ``contextualize_question`` reads this (ADR-006), so the window is sliced
    here once rather than passed down whole.
    """
    if store is None or not session_id:
        return []
    try:
        return list(store.last_n_turns(session_id))
    except Exception as exc:
        logger.warning("history read failed for %s: %s", session_id, exc)
        return []


def _trace_payload(envelope: AnswerEnvelope) -> list[dict[str, Any]]:
    """The node trace as plain dicts, because the store JSON-encodes it verbatim.

    ``NodeEvent`` is a dataclass and the store's encoder rejects anything that is
    not JSON-native, so the trace is flattened here rather than at the store
    boundary. Shaped to the same fields the debug panel reads back.
    """
    return [
        {
            "node": event.node,
            "started_at": event.started_at,
            "duration_ms": event.duration_ms,
            "tokens_in": event.tokens_in,
            "tokens_out": event.tokens_out,
            "error": event.error,
        }
        for event in envelope.node_trace
    ]


def _ask(question: str, *, params: Any, history: list[Turn], k: int) -> AnswerEnvelope:
    """Run one question through the graph and return its envelope.

    ``src.chains.nodes.retrieve`` reads a module constant rather than a parameter,
    so the rail's k slider is applied here for the duration of one call and
    restored in ``finally`` — the frozen graph signature is not forked for it.
    """
    from src.chains.nodes import retrieve as retrieve_node

    previous = retrieve_node.RETRIEVE_K
    retrieve_node.RETRIEVE_K = max(1, int(k))
    try:
        return invoke(question, history=history, params=params, doc_catalog=_doc_catalog())
    finally:
        retrieve_node.RETRIEVE_K = previous


def _record(
    store: SessionStore | None,
    session_id: str | None,
    question: str,
    envelope: AnswerEnvelope,
) -> tuple[bool, StoreError | None]:
    """Persist the question, the answer, its citations and its node trace.

    Returns ``(question_landed, failure)``. ``question_landed`` is ``True`` when the
    user's question reached the ``turns`` table — a fact the caller needs, because
    ``record_message`` writes its turn and its message in one transaction, so a store
    can absolutely commit the question and then refuse the answer; the transcript would
    then already show that question, and drawing it again would show one ask as two.

    ``failure`` is ``None`` when the whole turn landed, otherwise the store failure to
    tell the user about. Persistence is best-effort by contract — a read-only database,
    a locked file, a corrupt page — so a failing store is a *degraded* app, never a
    crashed one, and the failure is reported rather than raised into the render.
    Reporting it is what lets the caller keep the answer it already produced: losing a
    *turn* must never cost the *answer* that was computed (ADR-007's "no code path
    renders an empty answer", reached from the direction where the answer existed and
    was discarded).

    ``store is None`` or a missing session counts as "nothing to persist", not a
    failure: that is the no-store path the caller already handles by carrying the
    envelope itself.
    """

    if store is None or not session_id:
        return False, None
    try:
        store.append_turns(session_id, [{"role": "user", "content": question}])
    except Exception as exc:
        logger.warning("persisting question failed: %s", exc)
        return False, _as_store_failure(exc)
    try:
        store.record_message(
            session_id,
            question,
            envelope.answer_md,
            grounded=envelope.grounded,
            refusal_reason=envelope.refusal_reason,
            node_trace=_trace_payload(envelope),
            citations=[
                {
                    "ref_index": c.ref_index,
                    "doc_id": c.doc_id,
                    "page": c.page,
                    "section_path": list(c.section_path),
                    "snippet": c.snippet,
                }
                for c in envelope.citations
            ],
        )
        return True, None
    except Exception as exc:
        logger.warning("recording answer failed: %s", exc)
        return True, _as_store_failure(exc)


def _as_store_failure(exc: Exception) -> StoreError:
    """A store failure the taxonomy recognises, whatever the driver raised.

    ``sqlite3``'s own types (``OperationalError`` on a read-only file or a locked
    database, ``DatabaseError`` on a corrupt page) are rebound to
    :class:`~src.store.db.StoreError`, the base class the classifier maps to the
    store's own banner. The original is kept as the cause, so the log still carries
    the driver's own message and the banner never has to.
    """

    if isinstance(exc, StoreError):
        return exc
    failure = StoreError(str(exc))
    failure.__cause__ = exc
    return failure


def _footer_text(rail_ready: bool, docs: int) -> str:
    """One honest status line: what the pipeline is, and whether it is offline."""
    parts = ["Hybrid RAG · FAISS + BM25 + RRF", "deterministic numeric pre-pass"]
    parts.append(
        f"{docs} document{'' if docs == 1 else 's'} indexed"
        if rail_ready
        else "no index yet — add a PDF under Documents"
    )
    if os.environ.get("MOCK_LLM", "").strip() == "1":
        parts.append("offline mock mode")
    return " · ".join(parts)


def _live_answer(session_id: str | None) -> AnswerEnvelope | None:
    """The live answer *for this conversation*, or ``None`` when there is none.

    ``_LAST_KEY`` is written when an answer is produced and is not cleared when the
    rail switches conversations, so it counts as this conversation's answer only when
    the session recorded beside it in :data:`_LAST_SESSION_KEY` is the active one.
    Returning it for any other session would put one conversation's live panel under
    another's name — the silently-wrong surface the panel exists to rule out.
    """

    if st.session_state.get(_LAST_SESSION_KEY) != session_id:
        return None
    last: AnswerEnvelope | None = st.session_state.get(_LAST_KEY)
    return last


def _stored_answers(store: SessionStore | None, session_id: str | None) -> list[Message]:
    """The active conversation's stored answers, newest first, or ``[]``.

    Newest first because the picker leads with the most recent answer; the store
    returns them oldest first, so the order is turned around here rather than at the
    widget. A store that cannot be read yields ``[]`` — the view then shows its
    "ask a question first" placeholder, the honest reading of "no answers to inspect"
    — and the failure is logged, never raised into the render.
    """

    if store is None or not session_id:
        return []
    try:
        return list(reversed(store.list_messages(session_id)))
    except Exception as exc:  # sqlite failures surface as several concrete types
        logger.warning("stored answers read failed for %s: %s", session_id, exc)
        return []


def _answer_option_label(message: Message) -> str:
    """One picker row: a stored answer's question, whitespace-collapsed and shortened."""

    question = " ".join(str(message.question or "").split())
    if not question:
        return f"Answer {message.id}"
    if len(question) > _ANSWER_LABEL_CHARS:
        return question[: _ANSWER_LABEL_CHARS - 1].rstrip() + "…"
    return question


def _render_chat_view(rail: Any, store: SessionStore | None) -> None:
    session_id: str | None = rail.session_id
    doc_labels = rail.doc_labels

    # Whether the conversation already holds anything. The hero is an *empty state*, so it
    # may only render when this is false — otherwise it reappears underneath the transcript
    # after every answer, which is what it did before this flag existed.
    has_content = False

    if store is not None and session_id:
        try:
            # The public reader, not the store's private turn accessors: get_history
            # returns turns and messages together and documents that messages are loaded
            # whole so a citation block is never sliced away from its answer.
            history = store.get_history(session_id)
            if history.turns:
                has_content = True
                chat.render_transcript(history.turns, history.messages, doc_names=doc_labels)
        except Exception as exc:
            logger.warning("transcript read failed: %s", exc)
            errors.render_error(exc)

    answer: AnswerEnvelope | None = st.session_state[_ANSWER_KEY]
    if answer is not None:
        has_content = True
        chat.render_answer(answer, doc_names=doc_labels)
        st.session_state[_ANSWER_KEY] = None

    pending: str | None = chat.render_composer(disabled=not rail.index_ready)
    question = pending or st.session_state[_QUESTION_KEY]

    if not question:
        if has_content:
            chat.render_footer_note(_footer_text(rail.index_ready, rail.doc_count))
            return
        # The empty state returns the suggestion the user clicked; discarding that
        # value was the bug that made a suggestion click do nothing at all.
        suggestion = chat.render_empty_state(doc_labels=list(doc_labels.values()) or None)
        if not suggestion:
            chat.render_footer_note(_footer_text(rail.index_ready, rail.doc_count))
            return
        question = suggestion

    k = int(st.session_state.get("isip-param-k", 8))
    chat.render_pending_answer()
    try:
        envelope = _ask(question, params=rail.params, history=_history(store, session_id), k=k)
    except Exception as exc:
        logger.warning("question failed: %s", exc, exc_info=True)
        # ADR-007's taxonomy, not a catch-all: the classifier names the failure class
        # (missing key, rate limit, over-long context, …) and its own Retry verdict, so
        # a missing key is a banner with the env var to set, not one that offers a
        # retry that cannot help.
        view = errors.classify_error(exc)
        errors.render_error(exc, on_retry=view.retryable)
        return

    # Persist first. With a store the transcript renders the turn from the database, so
    # the rerun is what puts it on the canvas; without a store the envelope is carried
    # for one rerun and drawn from state.
    st.session_state[_QUESTION_KEY] = ""
    st.session_state[_LAST_KEY] = envelope
    st.session_state[_LAST_SESSION_KEY] = session_id
    question_landed = False
    store_failure = None
    if store is not None and session_id:
        question_landed, store_failure = _record(store, session_id, question, envelope)
    else:
        st.session_state[_ANSWER_KEY] = envelope

    if store_failure is not None:
        # The store refused the turn, so the rerun has no row to re-read and would
        # render an empty canvas — the defect this path exists to close. The turn is
        # drawn now, from the envelope the graph produced, and the banner says
        # persistence failed, so the gap in history is named rather than discovered
        # later by a user wondering where their question went.
        logger.warning("turn not persisted for %s: %s", session_id, store_failure)
        if not question_landed:
            # A committed question is a transcript row, so it is only drawn from here
            # for a store that took neither row — otherwise one ask is two questions.
            with st.chat_message(name="user"):
                st.markdown(question)
        chat.render_answer(envelope, doc_names=doc_labels)
        errors.render_error(store_failure)
        chat.render_footer_note(_footer_text(rail.index_ready, rail.doc_count))
        return

    st.rerun()


def _render_view(rail: Any, store: SessionStore | None) -> None:
    view = getattr(rail, "view", "chat")

    if view == "documents":
        upload.render_upload_panel(store=store)
        upload.render_document_catalog()
        return

    if view == "retrieval":
        upload.render_index_status(upload.index_status())
        st.caption(
            "Retrieval is hybrid: dense FAISS vectors and sparse BM25 keyword ranks, "
            "fused with Reciprocal Rank Fusion over child chunks, then expanded to the "
            "parent passage that carries the full table or section. Per-answer scores "
            "are on Diagnostics."
        )
        return

    if view == "parameters":
        st.subheader("Model parameters")
        sidebar.render_model_params()
        st.caption(
            "These values travel with every question. Temperature and top-p apply to "
            "sampling; retrieval k sets how many fused children are considered per query."
        )
        return

    if view == "diagnostics":
        _render_diagnostics_view(rail, store)
        return

    _render_chat_view(rail, store)


def _render_diagnostics_view(rail: Any, store: SessionStore | None) -> None:
    """The Diagnostics surface: the plan, path, retrieval and grounding of an answer.

    ``docs/OBSERVABILITY.md``'s contract is that the panel renders *the selected*
    answer — "for any message the user selects, not just the latest". The live answer
    is the current one, offered first because it is the only shape carrying the query
    plan and the retrieval scores; the active conversation's stored answers are the
    historical ones, each rendering the ``node_trace`` read back from the store, with
    the sections a stored row cannot fill saying so instead of drawing an empty block.
    """

    st.subheader("Diagnostics")
    live = _live_answer(rail.session_id)
    stored = _stored_answers(store, rail.session_id)

    # The live marker is the ``None`` key and a stored answer's key is its row id —
    # both hashable, unlike a ``Message``, whose ``node_trace`` is a list. Keying by id
    # also keeps the picker's selection fixed to one answer as the list grows.
    answers_by_id: dict[int, Message] = {message.id: message for message in stored}
    labels: dict[int | None, str] = {}
    if live is not None:
        labels[None] = _LIVE_OPTION_LABEL
    for message in stored:
        labels[message.id] = _answer_option_label(message)

    options = list(labels)
    if not options:
        st.caption(
            "Ask a question first — the debug panel renders the plan, the node "
            "path with per-node latency and tokens, and the retrieval scores for "
            "that answer."
        )
        return

    # A lone answer needs no control: the picker appears only when there is a choice.
    choice: int | None = options[0]
    if len(options) > 1:
        choice = st.selectbox(
            _ANSWER_PICKER_LABEL,
            options,
            format_func=labels.__getitem__,
            key=_ANSWER_PICKER_KEY,
        )
    if choice is None:
        render_debug_panel(envelope=live, doc_names=rail.doc_labels)
    else:
        render_debug_panel(message=answers_by_id[choice], doc_names=rail.doc_labels)


def main() -> None:
    st.set_page_config(
        page_title="IsipFin",
        page_icon=":material/finance:",
        layout="wide",
        # The rail width. Streamlit applies this as an inline style, so no CSS rule
        # can set it (see src/components/theme.py); 260 sits inside its 200-600 range.
        initial_sidebar_state=_RAIL_WIDTH,
    )
    # The docs tell a reader to `cp .env.example .env`; this is what reads it.
    load_env()
    theme.inject_theme()

    store = _open_store()
    rail = sidebar.render_sidebar(store)

    st.session_state.setdefault(_QUESTION_KEY, "")
    st.session_state.setdefault(_ANSWER_KEY, None)
    st.session_state.setdefault(_VIEW_KEY, "chat")

    _render_view(rail, store)


if __name__ == "__main__":
    main()

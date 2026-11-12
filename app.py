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

_QUESTION_KEY = "isip-question"
_ANSWER_KEY = "isip-answer"
_LAST_KEY = "isip-last-answer"
_LAST_SESSION_KEY = "isip-last-answer-session"
_VIEW_KEY = "isip-view"

_ANSWER_PICKER_KEY = "isip-diagnostics-answer"
_ANSWER_PICKER_LABEL = "Answer to inspect"

_LIVE_OPTION_LABEL = "Current answer (live)"

_ANSWER_LABEL_CHARS = 72

_RAIL_WIDTH = 260


def _open_store() -> SessionStore | None:
    if "isip-store" not in st.session_state:
        try:
            st.session_state["isip-store"] = SessionStore()
        except Exception as exc:
            logger.warning("chat store unavailable: %s", exc)
            st.session_state["isip-store"] = None
    store: SessionStore | None = st.session_state["isip-store"]
    return store


def _doc_catalog() -> list[DocSummary]:
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
    if store is None or not session_id:
        return []
    try:
        return list(store.last_n_turns(session_id))
    except Exception as exc:
        logger.warning("history read failed for %s: %s", session_id, exc)
        return []


def _trace_payload(envelope: AnswerEnvelope) -> list[dict[str, Any]]:
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

    if isinstance(exc, StoreError):
        return exc
    failure = StoreError(str(exc))
    failure.__cause__ = exc
    return failure


def _footer_text(rail_ready: bool, docs: int) -> str:
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

    if st.session_state.get(_LAST_SESSION_KEY) != session_id:
        return None
    last: AnswerEnvelope | None = st.session_state.get(_LAST_KEY)
    return last


def _stored_answers(store: SessionStore | None, session_id: str | None) -> list[Message]:

    if store is None or not session_id:
        return []
    try:
        return list(reversed(store.list_messages(session_id)))
    except Exception as exc:
        logger.warning("stored answers read failed for %s: %s", session_id, exc)
        return []


def _answer_option_label(message: Message) -> str:
    question = " ".join(str(message.question or "").split())
    if not question:
        return f"Answer {message.id}"
    if len(question) > _ANSWER_LABEL_CHARS:
        return question[: _ANSWER_LABEL_CHARS - 1].rstrip() + "…"
    return question


def _render_chat_view(rail: Any, store: SessionStore | None) -> None:
    session_id: str | None = rail.session_id
    doc_labels = rail.doc_labels

    has_content = False

    if store is not None and session_id:
        try:
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
        view = errors.classify_error(exc)
        errors.render_error(exc, on_retry=view.retryable)
        return

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
        logger.warning("turn not persisted for %s: %s", session_id, store_failure)
        if not question_landed:
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

    st.subheader("Diagnostics")
    live = _live_answer(rail.session_id)
    stored = _stored_answers(store, rail.session_id)

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
        initial_sidebar_state=_RAIL_WIDTH,
    )
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

"""IsipFin — Streamlit entrypoint.

The UI shell (ADR-009). This module wires the four component modules together and
owns exactly three responsibilities, none of them business logic:

1. **Boundary** — nothing under ``src/components/`` (or this module) imports
   retrieval, prompt, or gateway internals. The pipeline is reached only through
   ``src.chains.graph.invoke()``, whose ``AnswerEnvelope`` is the single object
   that crosses into the UI.
2. **Session state** — which session is active, and the transcript rendered for it.
3. **Plumbing the sliders actually control** — the retrieval ``k`` slider is passed
   to the retriever the same way an explicit argument would be, so a change shows
   up in the next answer (GRADING-ALIGNMENT row 16). The node reads a module
   constant, so this module sets it for the duration of the call rather than
   changing the frozen graph signature.

Run: ``streamlit run app.py`` (see docs/RUNBOOK.md).
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st
from src.chains.graph import invoke
from src.chains.state import AnswerEnvelope, DocSummary, Turn
from src.components import chat, sidebar, theme, upload
from src.store.sessions import SessionStore

logger = logging.getLogger(__name__)

#: Session-state keys this module owns. Components own theirs under the same prefix.
_QUESTION_KEY = "isip-question"
_ANSWER_KEY = "isip-answer"
_ERROR_KEY = "isip-error"


def _open_store() -> SessionStore | None:
    """Open the chat store, or return ``None`` when it cannot be opened.

    A missing or unreadable database is a degraded UI (the session rail says so),
    never a crash: questions can still be asked and answered without persistence.
    """
    if "isip-store" not in st.session_state:
        try:
            st.session_state["isip-store"] = SessionStore()
        except Exception as exc:  # sqlite errors surface as several concrete types
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


def _ask(question: str, *, params: Any, history: list[Turn], k: int) -> AnswerEnvelope:
    """Run one question through the graph and return its envelope."""
    from src.chains.nodes import retrieve as retrieve_node

    previous = retrieve_node.RETRIEVE_K
    retrieve_node.RETRIEVE_K = max(1, int(k))
    try:
        return invoke(
            question,
            history=history,
            params=params,
            doc_catalog=_doc_catalog(),
        )
    finally:
        retrieve_node.RETRIEVE_K = previous


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


def _record(
    store: SessionStore | None,
    session_id: str | None,
    question: str,
    envelope: AnswerEnvelope,
) -> None:
    """Persist the question, the answer, its citations and its node trace."""
    if store is None or not session_id:
        return
    try:
        store.append_turns(
            session_id,
            [{"role": "user", "content": question}],
        )
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
    except Exception as exc:
        logger.warning("persisting turn failed: %s", exc)


def main() -> None:
    """Render the shell: theme, rail, canvas, composer, and one question's turn."""
    st.set_page_config(
        page_title="IsipFin",
        page_icon="IPF",
        layout="wide",
        # The rail width. Streamlit applies this as an inline style, so no CSS rule
        # can set it (see src/components/theme.py); 260 sits inside its 200-600 range.
        initial_sidebar_state=260,
    )
    theme.inject_theme()

    store = _open_store()

    rail = sidebar.render_sidebar(
        store,
        upload_panel=upload.render_upload_panel,
    )

    st.session_state.setdefault(_QUESTION_KEY, "")
    st.session_state.setdefault(_ANSWER_KEY, None)
    st.session_state.setdefault(_ERROR_KEY, "")

    session_id: str | None = rail.session_id
    history = _history(store, session_id)
    doc_labels = rail.doc_labels

    # --- the canvas: stored transcript, then the live answer for this turn ------
    if store is not None and session_id:
        try:
            turns = list(store._turns(session_id))
            messages = list(store._messages_for(session_id))
            chat.render_transcript(turns, messages, doc_names=doc_labels)
        except Exception as exc:
            logger.warning("transcript read failed: %s", exc)
            chat.render_answer_error(
                "The saved conversation could not be loaded.",
                details=str(exc),
            )

    # --- the live answer from THIS turn, rendered before the composer ----------
    answer: AnswerEnvelope | None = st.session_state[_ANSWER_KEY]
    if answer is not None:
        chat.render_answer(answer, doc_names=doc_labels)
        st.session_state[_ANSWER_KEY] = None

    # --- the composer: where the next question comes from ---------------------
    pending: str | None = chat.render_composer(disabled=not rail.index_ready)
    question = pending or st.session_state[_QUESTION_KEY]

    if not question:
        # The empty state returns the suggestion the user clicked; discarding that
        # value was the bug that made a suggestion click do nothing at all.
        suggestion = chat.render_empty_state(
            doc_labels=list(doc_labels.values()) or None,
        )
        if not suggestion:
            return
        question = suggestion

    k = int(st.session_state.get("isip-param-k", 8))
    chat.render_pending_answer()
    try:
        envelope = _ask(question, params=rail.params, history=history, k=k)
    except Exception as exc:
        logger.warning("question failed: %s", exc, exc_info=True)
        chat.render_answer_error(
            "The answer could not be produced.",
            details=f"{type(exc).__name__}: {exc}",
            retry=True,
        )
        return

    # Persist first, then rerun. With a store the transcript renders the turn from
    # the database, so holding the envelope too would draw it twice; without a
    # store there is nothing to re-read, so the envelope carries it for one rerun.
    st.session_state[_QUESTION_KEY] = ""
    if store is not None and session_id:
        _record(store, session_id, question, envelope)
    else:
        st.session_state[_ANSWER_KEY] = envelope
    st.rerun()


if __name__ == "__main__":
    main()

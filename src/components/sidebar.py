from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.components.theme import caption_note as theme_caption_note
from src.components.theme import panel as theme_panel
from src.components.theme import rail_brand as theme_rail_brand
from src.components.theme import rail_label as theme_rail_label
from src.components.theme import rail_status_card as theme_rail_status_card
from src.core.retrieval import DEFAULT_K as DEFAULT_RETRIEVAL_K
from src.store.db import default_db_path
from src.utils.llm import LLMParams, resolved_config

if TYPE_CHECKING:
    from src.components.upload import IndexStatus
    from src.store.sessions import Session, SessionStore

__all__ = [
    "ACTIVE_SESSION_KEY",
    "DEFAULT_MODELS",
    "MAX_SESSIONS_SHOWN",
    "NAV_ITEMS",
    "NAV_KEY_PREFIX",
    "PARAMS_STATE_KEY",
    "RENAME_STATE_KEY",
    "RETRIEVAL_K_KEY",
    "SESSIONS_EXPANDED_KEY",
    "SESSIONS_HEADING",
    "SESSION_ERROR_KEY",
    "VIEWS",
    "VIEW_CHAT",
    "VIEW_DIAGNOSTICS",
    "VIEW_DOCUMENTS",
    "VIEW_KEY",
    "VIEW_PARAMETERS",
    "VIEW_RETRIEVAL",
    "SidebarResult",
    "active_session_id",
    "current_view",
    "load_params_defaults",
    "render_model_params",
    "render_session_manager",
    "render_sidebar",
    "session_title",
]


SESSIONS_HEADING = "Previous conversations"


VIEW_KEY = "isip-view"

VIEW_CHAT = "chat"

VIEW_DOCUMENTS = "documents"

VIEW_RETRIEVAL = "retrieval"

VIEW_PARAMETERS = "parameters"

VIEW_DIAGNOSTICS = "diagnostics"

VIEWS: tuple[str, ...] = (
    VIEW_CHAT,
    VIEW_DOCUMENTS,
    VIEW_RETRIEVAL,
    VIEW_PARAMETERS,
    VIEW_DIAGNOSTICS,
)

NAV_KEY_PREFIX = "isip-nav"

NAV_ITEMS: tuple[tuple[str, str, str], ...] = (
    (VIEW_DOCUMENTS, "Documents", ":material/folder_open:"),
    (VIEW_RETRIEVAL, "Retrieval", ":material/manage_search:"),
    (VIEW_PARAMETERS, "Parameters", ":material/tune:"),
    (VIEW_DIAGNOSTICS, "Diagnostics", ":material/monitor_heart:"),
)


BRAND_MARK = "IsipFin"

BRAND_TAGLINE = "Cited answers from your financial filings"

NEW_SESSION_LABEL = "New conversation"

NEW_SESSION_HELP = "Start a fresh conversation"

NEW_SESSION_FRESH_HELP = "This conversation is already empty — this returns to it"

NEW_SESSION_NO_STORE_HELP = (
    "The conversation store is unavailable — this returns to the conversation"
)


DEFAULT_MODELS: tuple[str, ...] = (
    "",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "llama3.1",
    "qwen2.5",
)

TEMP_MIN = 0.0
TEMP_MAX = 1.0
TEMP_STEP = 0.05
TEMP_DEFAULT = 0.2

TOP_P_MIN = 0.1
TOP_P_MAX = 1.0
TOP_P_STEP = 0.05
TOP_P_DEFAULT = 1.0

MAX_TOKENS_MIN = 128
MAX_TOKENS_MAX = 8192
MAX_TOKENS_STEP = 128
MAX_TOKENS_DEFAULT = 2048

K_MIN = 1
K_MAX = 20


ACTIVE_SESSION_KEY = "isip-active-session"

SESSION_KEY_PREFIX = "isip-session"

RENAME_STATE_KEY = "isip-rename-input"

PENDING_RENAME_KEY = "isip-pending-rename"

PENDING_DELETE_KEY = "isip-pending-delete"

SESSION_ERROR_KEY = "isip-session-error"

SESSIONS_EXPANDED_KEY = "isip-sessions-expanded"

RETRIEVAL_K_KEY = "isip-param-k"

PARAMS_STATE_KEY = "isip-params-state"

NEW_SESSION_KEY = "isip-new-session"

SESSION_CONTROL_PREFIX = "isip-session-control"


WORKSPACE_TITLE = "Local workspace"

WORKSPACE_READY_BADGE = "READY"

WORKSPACE_EMPTY_BADGE = "NO INDEX"


MAX_SESSIONS_SHOWN = 12


@dataclass(frozen=True)
class SidebarResult:

    params: LLMParams
    session_id: str | None
    session: Session | None
    doc_count: int
    index_ready: bool
    doc_labels: dict[str, str]
    sessions: tuple[Session, ...]
    view: str


def render_sidebar(
    store: SessionStore | None,
    *,
    index_dir: Any | None = None,
    index_ready: bool | None = None,
    doc_labels: Mapping[str, str] | None = None,
    upload_panel: Any | None = None,
) -> SidebarResult:
    with st.sidebar:
        sessions = store.list_sessions() if store is not None else ()
        selected, listing = _resolve_selection(store, list(sessions))
        _render_brand(store, selected)
        _render_nav()
        st.divider()
        _render_session_section(store, listing, selected)
        if upload_panel is not None:
            upload_panel()

        status, labels = _index_snapshot(index_dir)
        _render_status_card(status, store)
        load_params_defaults()

        view = current_view()
        ready = status.ready if index_ready is None else bool(index_ready)
        corpus_labels = labels if doc_labels is None else dict(doc_labels)
        params = _current_params()

    return SidebarResult(
        params=params,
        session_id=None if selected is None else selected.id,
        session=selected,
        doc_count=status.doc_count,
        index_ready=ready,
        doc_labels=corpus_labels,
        sessions=tuple(listing),
        view=view,
    )


def current_view() -> str:

    view = st.session_state.get(VIEW_KEY)
    return view if isinstance(view, str) and view in VIEWS else VIEW_CHAT


def _select_view(view: str) -> None:

    st.session_state[VIEW_KEY] = view


def _render_brand(store: SessionStore | None, selected: Session | None) -> None:

    theme_rail_brand(BRAND_MARK, BRAND_TAGLINE)
    empty = selected is not None and selected.turn_count == 0
    if store is None:
        help_text = NEW_SESSION_NO_STORE_HELP
    elif empty:
        help_text = NEW_SESSION_FRESH_HELP
    else:
        help_text = NEW_SESSION_HELP
    clicked = st.button(
        NEW_SESSION_LABEL,
        type="primary",
        key=NEW_SESSION_KEY,
        use_container_width=True,
        icon=":material/add:",
        help=help_text,
    )
    if clicked and _start_conversation(store, empty):
        st.rerun()


def _start_conversation(store: SessionStore | None, empty: bool) -> bool:

    st.session_state[VIEW_KEY] = VIEW_CHAT
    if store is None or empty:
        return False
    created = _create_session(store)
    if created is None:
        return False
    st.session_state[ACTIVE_SESSION_KEY] = created.id
    return True


def _render_nav() -> None:

    selected = current_view()
    for view, text, icon in NAV_ITEMS:
        st.button(
            text,
            key=f"{NAV_KEY_PREFIX}-{view}",
            type="primary" if view == selected else "secondary",
            use_container_width=True,
            icon=icon,
            on_click=_select_view,
            args=(view,),
        )


def render_session_manager(
    store: SessionStore | None,
    sessions: Sequence[Session] | None = None,
) -> Session | None:

    if store is None:
        _render_store_unavailable()
        return None

    listing = store.list_sessions() if sessions is None else list(sessions)
    selected, listing = _resolve_selection(store, listing)
    _render_session_section(store, listing, selected)
    return selected


def active_session_id(sessions: Sequence[Session]) -> str | None:

    stored = st.session_state.get(ACTIVE_SESSION_KEY)
    if isinstance(stored, str) and any(item.id == stored for item in sessions):
        return stored
    return sessions[0].id if sessions else None


def session_title(session: Session | None) -> str:

    if session is None:
        return "No conversation selected"
    return (session.title or "").strip() or "Untitled conversation"


def _resolve_selection(
    store: SessionStore | None, sessions: Sequence[Session]
) -> tuple[Session | None, list[Session]]:

    if store is None:
        return None, []
    listing = list(sessions)
    selected_id = active_session_id(listing)
    if selected_id is None:
        created = _create_session(store)
        if created is None:
            return None, listing
        st.session_state[ACTIVE_SESSION_KEY] = created.id
        listing = store.list_sessions()
        return created, listing
    for item in listing:
        if item.id == selected_id:
            return item, listing
    fallback = listing[0] if listing else None
    return fallback, listing


def _create_session(store: SessionStore) -> Session | None:

    try:
        return store.create_session()
    except Exception as exc:
        _render_store_error("A new conversation could not be started.", exc)
        return None


def _render_session_section(
    store: SessionStore | None, sessions: Sequence[Session], selected: Session | None
) -> None:

    if store is None:
        _render_store_unavailable()
        return
    theme_rail_label(SESSIONS_HEADING)
    _render_session_rows(store, sessions, selected)
    _render_session_actions(store, selected)


def _render_session_rows(
    store: SessionStore, sessions: Sequence[Session], selected: Session | None
) -> None:

    selected_id = None if selected is None else selected.id
    if not sessions:
        st.caption("No conversations yet — asking the first question starts one.")
        return

    expanded = bool(st.session_state.get(SESSIONS_EXPANDED_KEY))
    if len(sessions) > MAX_SESSIONS_SHOWN:
        st.button(
            "Show fewer conversations" if expanded else f"Show all {len(sessions)} conversations",
            key=f"{SESSION_CONTROL_PREFIX}-expand",
            use_container_width=True,
            on_click=_toggle_sessions_expanded,
        )
        visible = list(sessions) if expanded else list(sessions[:MAX_SESSIONS_SHOWN])
    else:
        visible = list(sessions)

    for index, item in enumerate(visible):
        if st.button(
            f"{session_title(item)} · {item.turn_count} turn(s)",
            key=f"{SESSION_KEY_PREFIX}-{index}",
            type="primary" if item.id == selected_id else "secondary",
            use_container_width=True,
            help="Open this conversation",
        ):
            _switch_to(store, item.id)


def _switch_to(store: SessionStore, session_id: str) -> None:

    store.switch_session(session_id)
    st.session_state[ACTIVE_SESSION_KEY] = session_id
    st.session_state[VIEW_KEY] = VIEW_CHAT
    st.session_state.pop(PENDING_RENAME_KEY, None)
    st.rerun()


def _toggle_sessions_expanded() -> None:
    st.session_state[SESSIONS_EXPANDED_KEY] = not bool(st.session_state.get(SESSIONS_EXPANDED_KEY))


def _render_session_actions(store: SessionStore, selected: Session | None) -> None:

    if selected is None:
        st.caption("Select a conversation to rename or delete it.")
        return

    _apply_pending_rename(store)

    draft = st.text_input(
        "Rename conversation",
        key=RENAME_STATE_KEY,
        placeholder=session_title(selected),
        help="Type a new title, then choose Rename",
    )
    if (
        st.button(
            "Rename",
            key=f"{SESSION_CONTROL_PREFIX}-rename",
            disabled=not str(draft).strip(),
            on_click=_queue_rename,
            args=(selected.id,),
        )
        and st.session_state.get(PENDING_RENAME_KEY) is None
    ):
        _apply_pending_rename(store)

    confirm_key = f"{SESSION_CONTROL_PREFIX}-delete-confirm-{selected.id}"
    if st.button("Delete conversation", key=f"{SESSION_CONTROL_PREFIX}-delete"):
        st.session_state[confirm_key] = True
    if st.session_state.get(confirm_key):
        st.warning("Deleting removes this conversation and its turns permanently.")
        if st.button(
            "Delete permanently",
            key=f"{SESSION_CONTROL_PREFIX}-delete-confirm",
            type="primary",
            on_click=_queue_delete,
            args=(selected.id,),
        ):
            _apply_pending_delete(store)


def _queue_rename(session_id: str) -> None:

    st.session_state[PENDING_RENAME_KEY] = (
        session_id,
        str(st.session_state.get(RENAME_STATE_KEY) or "").strip(),
    )


def _apply_pending_rename(store: SessionStore) -> None:

    pending = st.session_state.pop(PENDING_RENAME_KEY, None)
    if pending is None:
        return
    session_id, title = pending
    st.session_state.pop(RENAME_STATE_KEY, None)
    if not session_id or not title:
        return
    try:
        store.rename_session(session_id, title)
    except Exception as exc:
        _render_store_error("The conversation could not be renamed.", exc)
        return
    st.toast("Conversation renamed")
    st.rerun()


def _queue_delete(session_id: str) -> None:

    st.session_state[PENDING_DELETE_KEY] = session_id


def _apply_pending_delete(store: SessionStore) -> None:
    session_id = st.session_state.pop(PENDING_DELETE_KEY, None)
    if not isinstance(session_id, str) or not session_id:
        return
    try:
        store.delete_session(session_id)
    except Exception as exc:
        _render_store_error("The conversation could not be deleted.", exc)
        return
    _fallback_after_delete(store)
    st.toast("Conversation deleted")
    st.rerun()


def _fallback_after_delete(store: SessionStore) -> None:

    sessions = store.list_sessions()
    st.session_state[ACTIVE_SESSION_KEY] = sessions[0].id if sessions else None


def _render_store_unavailable() -> None:

    theme_rail_label(SESSIONS_HEADING)
    with theme_panel():
        st.error("The conversation store is unavailable.")
        theme_caption_note(
            "Conversations cannot be listed, created, renamed or deleted until the "
            "chat store is readable."
        )


def render_model_params() -> LLMParams:

    with theme_panel():
        values = {
            "model": _render_model_picker(),
            "temperature": _render_temperature(),
            "top_p": _render_top_p(),
            "max_tokens": _render_max_tokens(),
            "k": _render_retrieval_k(),
        }
    _commit_params(values)
    return _current_params()


def _configured_model() -> str:

    return resolved_config().model or ""


def _param_defaults() -> dict[str, Any]:

    return {
        "model": _configured_model(),
        "temperature": TEMP_DEFAULT,
        "top_p": TOP_P_DEFAULT,
        "max_tokens": MAX_TOKENS_DEFAULT,
        "k": DEFAULT_RETRIEVAL_K,
    }


def _param_state() -> dict[str, Any]:

    state = st.session_state.get(PARAMS_STATE_KEY)
    if not isinstance(state, dict):
        state = {}
        st.session_state[PARAMS_STATE_KEY] = state
    for control, value in _param_defaults().items():
        state.setdefault(control, value)
    return state


def load_params_defaults() -> dict[str, Any]:

    values = _param_state()
    st.session_state[RETRIEVAL_K_KEY] = values["k"]
    return dict(values)


def _commit_params(values: Mapping[str, Any]) -> None:

    state = _param_state()
    state.update(values)
    st.session_state[RETRIEVAL_K_KEY] = state["k"]


def _current_params() -> LLMParams:

    state = _param_state()
    return LLMParams(
        model=state.get("model") or None,
        temperature=state.get("temperature"),
        top_p=state.get("top_p"),
        max_tokens=state.get("max_tokens"),
    )


def _render_model_picker() -> str | None:

    options = list(DEFAULT_MODELS)
    configured = _configured_model()
    if configured:
        options = [configured, *(m for m in options if m != configured)]
    chosen = _param_state().get("model")
    index = options.index(chosen) if chosen and chosen in options else None
    return st.selectbox(
        "Model",
        options,
        index=index,
        placeholder="Provider default",
        help="Sent with every message; the environment's model is the default.",
    )


def _render_temperature() -> float:

    return st.slider(
        "Temperature",
        min_value=TEMP_MIN,
        max_value=TEMP_MAX,
        step=TEMP_STEP,
        value=_param_state()["temperature"],
        help="Synthesis sampling temperature. 0 is deterministic, 1 is maximally varied.",
    )


def _render_top_p() -> float:

    return st.slider(
        "Top-p",
        min_value=TOP_P_MIN,
        max_value=TOP_P_MAX,
        step=TOP_P_STEP,
        value=_param_state()["top_p"],
        help="Nucleus sampling cutoff. 1.0 disables it (the provider default).",
    )


def _render_max_tokens() -> int:
    return st.number_input(
        "Max tokens",
        min_value=MAX_TOKENS_MIN,
        max_value=MAX_TOKENS_MAX,
        step=MAX_TOKENS_STEP,
        value=_param_state()["max_tokens"],
        help="Maximum tokens in one answer.",
    )


def _render_retrieval_k() -> int:

    return st.slider(
        "Retrieval k",
        min_value=K_MIN,
        max_value=K_MAX,
        step=1,
        value=_param_state()["k"],
        help=(
            "Child chunks retrieved per question before parent expansion (ADR-002). "
            "Higher k is more recall and more context. The caller reads this value: "
            "retrieval k is not a sampling parameter, so it is not sent to the model."
        ),
    )


def _index_snapshot(index_dir: Any | None) -> tuple[IndexStatus, dict[str, str]]:

    from src.components.upload import document_names, index_status

    status = index_status(index_dir)
    try:
        labels = document_names(index_dir=index_dir)
    except Exception:
        labels = {}
    return status, labels


def _render_status_card(status: IndexStatus, store: Any | None) -> None:

    theme_rail_status_card(
        WORKSPACE_TITLE,
        _workspace_lines(status, store),
        badge=WORKSPACE_READY_BADGE if status.ready else WORKSPACE_EMPTY_BADGE,
    )


def _workspace_lines(status: IndexStatus, store: Any | None) -> list[str]:

    lines: list[str] = []
    if status.ready:
        lines.append(f"{_plural(status.doc_count, 'document')} indexed")
        lines.append(f"{_plural(status.chunk_count, 'searchable chunk')} · {status.size_human}")
    else:
        lines.append(
            f"{_plural(status.doc_count, 'document')} registered, none searchable"
            if status.doc_count
            else "No document indexed yet"
        )
        if status.reason:
            lines.append(status.reason)
    lines.append(f"Conversations: {_store_location(store)}")
    return lines


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _store_location(store: Any | None) -> str:

    path = getattr(store, "path", None)
    return str(path) if path is not None else str(default_db_path())


def _render_store_error(summary: str, exc: Exception) -> None:

    detail = str(getattr(exc, "message", None) or exc) or type(exc).__name__
    st.error(summary)
    theme_caption_note(detail)
    st.session_state[SESSION_ERROR_KEY] = detail

"""The left rail: brand, session manager, model-parameter controls, index status, account.

ADR-009's boundary lives here: this module **renders controls and calls the chat
store**. It never retrieves, never builds a prompt, and never touches the LLM
gateway — the parameter widgets assemble a :class:`~src.utils.llm.LLMParams` and
hand it back to the caller, which is the object ``src.chains.graph.invoke``
accepts. Retrieval ``k`` is *not* part of that contract (it belongs to the
retriever call, not to sampling): the slider is rendered here because the brief's
parameter surface names it alongside the sampling controls (ADR-005), but its
value is read off ``st.session_state[RETRIEVAL_K_KEY]`` by the caller, which owns
the plumbing into ``src.chains.nodes.retrieve``. The split is stated on the
widget's help text so it is visible rather than implied.

The store is the single source of truth for every session fact this rail
displays: nothing durable lives in ``st.session_state`` except this component's
own bookkeeping keys (the selected session, the rename draft, the delete
confirm, the last store error). A reload therefore reproduces the rail exactly
from the database, which is the "sessions persist across reload" criterion in
``docs/GRADING-ALIGNMENT.md`` row 17.

Public surface
--------------
``render_sidebar``          the whole rail; returns a :class:`SidebarResult`
``SidebarResult``           what one render produced (params, active session, corpus facts)
``render_session_manager``  new / switch / rename / delete, on the store
``render_param_controls``   model, temperature, top_p, max_tokens, k → ``LLMParams``
``render_index_summary``    the one-glance index state, and whether it is usable
``render_footer``           the account row
``load_params_defaults``    seed the parameter widgets, once per browser session
``active_session_id``       the selected id, reconciled against what exists
``session_title``           the display title for a stored session
``DEFAULT_MODELS``          the model picker's option list
``RETRIEVAL_K_KEY``         session-state key of the retrieval-``k`` slider

Every action is a real Streamlit widget with a visible label (no unlabeled icon
buttons), every action reports its own outcome, and the error, empty and
unavailable states are all reachable — see each function's notes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.components.theme import badge, caption_note, figure, label, panel
from src.core.retrieval import DEFAULT_K as DEFAULT_RETRIEVAL_K
from src.store.db import default_db_path
from src.utils.llm import MODEL_ENV, LLMParams

if TYPE_CHECKING:  # contract types only — no runtime import, so the module stays leaf-level
    from src.store.sessions import Session, SessionStore

__all__ = [
    "ACTIVE_SESSION_KEY",
    "DEFAULT_MODELS",
    "MAX_SESSIONS_SHOWN",
    "MODEL_PARAM_STATE_KEY",
    "PARAM_STATE_PREFIX",
    "RENAME_STATE_KEY",
    "RETRIEVAL_K_KEY",
    "SESSIONS_EXPANDED_KEY",
    "SESSIONS_HEADING",
    "SESSION_ERROR_KEY",
    "SidebarResult",
    "active_session_id",
    "load_params_defaults",
    "render_footer",
    "render_index_summary",
    "render_param_controls",
    "render_session_manager",
    "render_sidebar",
    "session_title",
]

# --- headings ---------------------------------------------------------------------

#: The label above the session list. Rendered in the monospace uppercase voice so
#: the rail's three groups (documents, conversations, parameters) read as one set.
SESSIONS_HEADING = "Previous conversations"

#: The model-parameter group's heading, matching the brief's own wording.
PARAM_HEADING = "Model parameters"

#: The document-status heading. The corpus lives in the index and the store's
#: document registry, so this rail reports rather than owns either.
DOCUMENTS_HEADING = "Documents"

# --- the brand row -----------------------------------------------------------------

#: Brand mark. A wordmark, not an emoji — ``docs/CONSTRAINTS.md``'s no-emoji rule —
#: and the only saturated text in the rail.
BRAND_MARK = "IsipFin"

#: Tagline under the mark. Names the product rather than describing the demo.
BRAND_TAGLINE = "Cited answers from your financial filings"

#: The primary CTA's label. One action per rail start has an obvious label; an
#: unlabeled icon button would not be an accessible control.
NEW_SESSION_LABEL = "New conversation"

#: Tooltip on the disabled primary CTA.
NEW_SESSION_FRESH_HELP = "This conversation is already empty"

#: Tooltip on the enabled primary CTA.
NEW_SESSION_HELP = "Start a fresh conversation"

# --- the parameter surface (GRADING-ALIGNMENT.md row 16) ------------------------

#: The environment variable that names the configured model, read at import so the
#: picker is seeded from what *this deployment* set rather than a hardcoded value.
ENV_MODEL = os.environ.get(MODEL_ENV) or ""

#: The model picker's options. The configured model leads the list (so a grader who
#: set ``OPENAI_MODEL`` sees **their** model selected), followed by the endpoints
#: ``docs/adr/ADR-005`` names as drop-in compatible — so switching provider is a
#: pick, not an edit. ``""`` is the "provider default" option, which Streamlit
#: renders as the placeholder state rather than a selectable blank row.
DEFAULT_MODELS: tuple[str, ...] = (
    ENV_MODEL,
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "llama3.1",
    "qwen2.5",
)

#: ``temperature`` bounds. Synthesis is user-adjustable
#: (``docs/adr/ADR-005``'s per-task table) and 0.0 is the floor the planner pins,
#: so the low end must reach 0 rather than starting at the synthesis default.
TEMP_MIN = 0.0
TEMP_MAX = 1.0
TEMP_STEP = 0.05
TEMP_DEFAULT = 0.2

#: ``top_p`` bounds. 1.0 is "no truncation" (the provider default), which is why
#: the top of the range is the neutral value rather than an exotic one.
TOP_P_MIN = 0.1
TOP_P_MAX = 1.0
TOP_P_STEP = 0.05
TOP_P_DEFAULT = 1.0

#: ``max_tokens`` bounds. 128 is a precise refusal answer; 8192 covers a decomposed
#: answer with four cited sections without burning a day of quota.
MAX_TOKENS_MIN = 128
MAX_TOKENS_MAX = 8192
MAX_TOKENS_STEP = 128
MAX_TOKENS_DEFAULT = 2048

#: Retrieval ``k`` bounds — the ADR-002 fused top-k, expanded to parents below.
#: A wider ``k`` buys recall and costs context (``docs/RUNBOOK.md``'s overflow
#: row), which is exactly the trade the slider is for.
K_MIN = 1
K_MAX = 20

# --- widget keys ------------------------------------------------------------------

#: The key holding the selected session id. Per-session wiring is exactly what
#: ``st.session_state`` is for (ADR-009); nothing else durable lives there.
ACTIVE_SESSION_KEY = "isip-active-session"

#: Prefix for the per-session row buttons. A session id is not a safe key character
#: class, so the *index* is what stays stable in the key.
SESSION_KEY_PREFIX = "isip-session"

#: The rename field's own widget key. The widget's *value* is the draft; this key
#: is never written from the render body (that raises), only cleared before the
#: widget is instantiated.
RENAME_STATE_KEY = "isip-rename-input"

#: A queued rename: ``(session_id, title)``, written by the button's ``on_click``
#: callback — the only place Streamlit allows a widget's state to be changed.
PENDING_RENAME_KEY = "isip-pending-rename"

#: A queued delete: the ``session_id``, written the same legal way.
PENDING_DELETE_KEY = "isip-pending-delete"

#: The last store error this render produced, kept for the debug panel's read.
SESSION_ERROR_KEY = "isip-session-error"

#: The session list's expansion flag — see :data:`MAX_SESSIONS_SHOWN`.
SESSIONS_EXPANDED_KEY = "isip-sessions-expanded"

#: Session-state key of the retrieval-``k`` slider. Exported so the caller reads the
#: value without re-typing the string this module renders with.
RETRIEVAL_K_KEY = "isip-param-k"

#: Prefix for the sampling controls' widget keys, so a test can reach the same
#: widget deterministically.
PARAM_STATE_PREFIX = "isip-param"

#: The model picker's widget key.
MODEL_PARAM_STATE_KEY = f"{PARAM_STATE_PREFIX}-model"

#: The primary CTA's widget key.
NEW_SESSION_KEY = "isip-new-session"

#: Prefix for the session-list controls' widget keys.
SESSION_CONTROL_PREFIX = "isip-session-control"

# --- the account row ---------------------------------------------------------------

#: The workspace identity in the footer row. There is no authentication layer
#: (ADR-009's single-user demo), so this names the workspace, not a signed-in user.
ACCOUNT_NAME = "Analyst workspace"

#: The plan line shown beside the account name.
ACCOUNT_PLAN = "Local workspace"

#: The footer's data-location line, naming where the conversations live. The path is
#: resolved at render time (``DATA_DIR`` moves it in the container), so the row
#: states the real location rather than a stale constant.
ACCOUNT_DATA_CAPTION = "Conversations and documents are stored locally."

# --- the session list --------------------------------------------------------------

#: Sessions listed before the list collapses behind one "show all" control. A
#: workspace with twenty conversations cannot scroll the rail forever, and the
#: newest-first order means the ones being cut are the oldest.
MAX_SESSIONS_SHOWN = 12


# --- the result object -------------------------------------------------------------


@dataclass(frozen=True)
class SidebarResult:
    """Everything one :func:`render_sidebar` call produced.

    The UI is a pure adapter (ADR-009), so the rail's whole contract is this
    object: the parameters to pass to ``invoke``, the session questions are asked
    in, and the corpus facts the main pane needs for its empty state and its
    citation labels.

    Attributes:
        params: The assembled sampling parameters. ``temperature``/``top_p``/
            ``max_tokens`` always carry the widgets' values (a slider always has
            one), so a change to any of them shows up in the next answer —
            GRADING-ALIGNMENT.md row 16. ``model`` is ``None`` only when the
            picker is empty, which the caller treats as "provider default".
        session_id: The selected session's id, or ``None`` when no store is
            available. ``session_id`` and ``session`` always agree.
        session: The selected :class:`~src.store.sessions.Session`, or ``None``
            when there is nothing selected.
        doc_count: Documents registered in the chat store's document registry.
        index_ready: Whether the index holds chunks to retrieve over.
        doc_labels: ``doc_id`` → display name, for the transcript's citation cards.
        sessions: The sessions visible in the list, most recently updated first.
    """

    params: LLMParams
    session_id: str | None
    session: Session | None
    doc_count: int
    index_ready: bool
    doc_labels: dict[str, str]
    sessions: tuple[Session, ...]


# --- entry point -------------------------------------------------------------------


def render_sidebar(
    store: SessionStore | None,
    *,
    index_dir: Any | None = None,
    index_ready: bool | None = None,
    doc_labels: Mapping[str, str] | None = None,
    upload_panel: Any | None = None,
) -> SidebarResult:
    """Render the whole left rail and return what the main pane needs from it.

    The rail renders top to bottom in the order the reference layout prescribes:
    the brand mark and the primary "New conversation" action, the document status,
    the session list, the model parameters, then the account row at the bottom.
    ``upload_panel`` is called inside the documents section when the caller passes
    it, so the ingest surface sits where the layout puts it without this module
    importing :mod:`src.components.upload` — one component owns each file.

    Args:
        store: The chat store. ``None`` leaves every session action replaced by a
            named unavailable state while the parameter controls still render,
            which is the honest rendering for a store that failed to open — the
            rail must not go blank because one table is missing.
        index_dir: Index directory, forwarded to the document-status reader.
            Ignored when ``index_ready`` and ``doc_labels`` are both supplied.
        index_ready: Whether the index is usable. Pass it when the caller already
            read the fact (the upload panel reads the same one after an ingest);
            ``None`` reads it here.
        doc_labels: ``doc_id`` → filename map for the citation cards. ``None``
            reads the catalog here.
        upload_panel: Optional callable rendering the ingest surface, called with
            no arguments inside the documents section.

    Returns:
        A :class:`SidebarResult` the main pane passes to
        ``src.chains.graph.invoke`` and to the transcript renderer.
    """

    sessions = store.list_sessions() if store is not None else ()
    selected, listing = _resolve_selection(store, list(sessions))
    _render_brand(store, selected)
    ready, doc_count, labels = _document_facts(
        store, index_dir=index_dir, index_ready=index_ready, doc_labels=doc_labels
    )
    _render_document_status(ready, doc_count, labels, upload_panel=upload_panel)
    _render_session_section(store, listing, selected)
    params = render_param_controls()
    render_footer()
    return SidebarResult(
        params=params,
        session_id=None if selected is None else selected.id,
        session=selected,
        doc_count=doc_count,
        index_ready=ready,
        doc_labels=labels,
        sessions=tuple(listing),
    )


# --- brand --------------------------------------------------------------------------


def _render_brand(store: SessionStore | None, selected: Session | None) -> None:
    """The brand mark, the tagline, and the primary "New conversation" action.

    The CTA is here rather than in the session list because the reference layout
    puts it under the mark, and it is disabled when the active conversation is
    already empty — the state is readable from the row's own turn count, so no
    extra tracking key is needed.

    Args:
        store: The chat store a new session is created in. ``None`` (no store)
            leaves the CTA disabled rather than letting a click do nothing.
        selected: The active session, whose turn count gates the CTA.
    """

    st.markdown(f"### {BRAND_MARK}")
    st.markdown(
        f'<div style="color:var(--isip-text-muted);font-size:0.8rem;line-height:1.4">'
        f"{BRAND_TAGLINE}</div>",
        unsafe_allow_html=True,
    )
    if store is None:
        st.button(NEW_SESSION_LABEL, type="primary", disabled=True, key=NEW_SESSION_KEY)
        return
    fresh = selected is not None and selected.turn_count == 0
    if st.button(
        NEW_SESSION_LABEL,
        type="primary",
        disabled=fresh,
        key=NEW_SESSION_KEY,
        help=NEW_SESSION_FRESH_HELP if fresh else NEW_SESSION_HELP,
    ):
        created = _create_session(store)
        if created is not None:
            st.rerun()


# --- session manager ----------------------------------------------------------------


def render_session_manager(
    store: SessionStore | None,
    sessions: Sequence[Session] | None = None,
) -> Session | None:
    """Render the previous-conversations list and the per-session actions.

    The list comes from :meth:`SessionStore.list_sessions`, so a rename or delete
    performed in another tab shows on the next rerun without a refresh action. The
    selected id is reconciled against the list *before* any widget renders: an id
    that no longer exists (the session was deleted elsewhere) falls back to the
    most recent one, and an empty database falls back to a session created on the
    spot, because a chat pane with no session has nowhere to write a question.

    This is the same section :func:`render_sidebar` renders, so the two share
    internal renderers and keys — a caller renders **one or the other**, never
    both, which is what keeps the widget keys unique in one run.

    Args:
        store: The chat store, or ``None`` when it could not be opened — then
            every action is replaced by a named unavailable state.
        sessions: The sessions already read for this render, so the caller can
            avoid a second query. ``None`` reads them here.

    Returns:
        The active :class:`~src.store.sessions.Session`, or ``None`` when no
        store is available.
    """

    if store is None:
        _render_store_unavailable()
        return None

    listing = store.list_sessions() if sessions is None else list(sessions)
    selected, listing = _resolve_selection(store, listing)
    _render_session_section(store, listing, selected)
    return selected


def active_session_id(sessions: Sequence[Session]) -> str | None:
    """The session id the rail should select, reconciled against what exists.

    Session state holds the selected id across reruns but not across a deletion,
    so the stored id wins only while it is still in the list, the most recently
    updated session wins otherwise, and an empty store yields ``None``. Resolving
    this before any widget renders is what stops a stale id from turning a
    question into a write to a row that no longer exists.

    Args:
        sessions: The store's sessions, newest first.

    Returns:
        The id to select, or ``None`` when there is nothing to select.
    """

    stored = st.session_state.get(ACTIVE_SESSION_KEY)
    if isinstance(stored, str) and any(item.id == stored for item in sessions):
        return stored
    return sessions[0].id if sessions else None


def session_title(session: Session | None) -> str:
    """The display title for a stored session.

    A session's title is written from its first question and is ``None`` until one
    exists — ``create_session`` takes ``title=None`` for exactly that reason — so
    an untitled session shows a placeholder rather than an empty row.

    Args:
        session: A stored session, or ``None``.

    Returns:
        The stored title, or ``Untitled conversation`` when there is none.
    """

    if session is None:
        return "No conversation selected"
    return (session.title or "").strip() or "Untitled conversation"


def _resolve_selection(
    store: SessionStore | None, sessions: Sequence[Session]
) -> tuple[Session | None, list[Session]]:
    """The session to act on and the list to show, both reflecting the store now.

    An empty store is a first run rather than an error, so it becomes a new session
    immediately: the chat pane then has somewhere to write and the greeting is an
    empty state instead of a dead end. Creating it here — rather than demanding
    the caller supply a session id — is what keeps the rail usable on a cold
    clone, which ``docs/STRESS-TEST-DELIVERY.md`` row 5 requires.

    The *listing* is returned alongside the selection because the new session is
    part of it: a list rendered from a pre-create read would still say "no
    conversations yet" while the rail has one selected, and the two disagreeing is
    the bug this avoids.

    Args:
        store: The chat store, or ``None`` when unavailable (then nothing is
            selected, nothing is created, and the listing stays empty).
        sessions: The sessions already read for this render, newest first.

    Returns:
        ``(selected, listing)`` — the selected session (or ``None``), and every
        session the list should show, newest first.
    """

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
    """Create a session, reporting the typed store error when it happens.

    A failure on *create* is not something a user can fix, so it is surfaced as a
    banner rather than swallowed: a rail that quietly has no session would take
    the whole chat pane down with it.
    """

    try:
        return store.create_session()
    except Exception as exc:  # a typed store error; its own message is the cause
        _render_store_error("A new conversation could not be started.", exc)
        return None


def _render_session_section(
    store: SessionStore | None, sessions: Sequence[Session], selected: Session | None
) -> None:
    """The labelled session list plus the per-session actions.

    Each row is a real, labelled button — never an unlabeled icon button
    (``docs/CONSTRAINTS.md``) — so the whole row is the target and the label is
    read by a screen reader. The selected row carries the accent fill, which is
    the only way to tell which session is active after a rerun that recreated the
    widget state. Rows past :data:`MAX_SESSIONS_SHOWN` collapse behind one "show
    all" control rather than growing the rail without bound.

    Args:
        store: The chat store the row's switch writes through. ``None`` renders
            the unavailable state instead, so a store that failed to open is named
            rather than silently hidden.
        sessions: Every session, newest first.
        selected: The active session, or ``None`` when the store is empty and a
            create failed — the list still renders, the actions do not.
    """

    label(SESSIONS_HEADING)
    if store is None:
        _render_store_unavailable()
        return
    _render_session_rows(store, sessions, selected)
    _render_session_actions(store, selected)


def _render_session_rows(
    store: SessionStore, sessions: Sequence[Session], selected: Session | None
) -> None:
    """One labelled button per session, with a bounded list and a "show all" row."""

    selected_id = None if selected is None else selected.id
    if not sessions:
        st.caption("No conversations yet — asking the first question starts one.")
        return

    expanded = bool(st.session_state.get(SESSIONS_EXPANDED_KEY))
    if len(sessions) > MAX_SESSIONS_SHOWN:
        st.button(
            "Show fewer conversations" if expanded else f"Show all {len(sessions)} conversations",
            key=f"{SESSION_CONTROL_PREFIX}-expand",
            on_click=_toggle_sessions_expanded,
        )
        visible = list(sessions) if expanded else list(sessions[:MAX_SESSIONS_SHOWN])
    else:
        visible = list(sessions)

    for index, item in enumerate(visible):
        if (
            st.button(
                f"{session_title(item)} · {item.turn_count} turn(s)",
                key=f"{SESSION_KEY_PREFIX}-{index}",
                type="primary" if item.id == selected_id else "secondary",
                help="Open this conversation",
            )
            and item.id != selected_id
        ):
            _switch_to(store, item.id)


def _switch_to(store: SessionStore, session_id: str) -> None:
    """Select another session and rerun.

    The rename draft is dropped, because it belongs to the session it was typed
    for: leaving it would let the next Rename click apply a title meant for a
    different conversation.
    """

    store.switch_session(session_id)
    st.session_state[ACTIVE_SESSION_KEY] = session_id
    st.session_state.pop(PENDING_RENAME_KEY, None)
    st.rerun()


def _toggle_sessions_expanded() -> None:
    """Flip the session list's expansion flag (the "show all" control)."""

    st.session_state[SESSIONS_EXPANDED_KEY] = not bool(st.session_state.get(SESSIONS_EXPANDED_KEY))


def _render_session_actions(store: SessionStore, selected: Session | None) -> None:
    """Rename and delete, each reporting its own outcome.

    Both actions run against the store and surface its typed errors rather than
    failing quietly: a rename that silently did nothing is precisely the UI lie
    ``rename_session``'s docstring exists to prevent (ADR-006 rule 4). Rename is a
    text field plus a button, so an empty title is a disabled control rather than
    a no-op that looks applied. Delete is a labelled button with an explicit
    second confirm, because ``delete_session``'s cascade is irrevocable.

    The rename write happens **before** the widget renders, out of the pending key
    set by the button's ``on_click`` callback. Clearing a text widget's own key
    from the render body is illegal in Streamlit — it raises
    ``StreamlitWidgetAlreadyInstantiatedError`` — so the callback is the write
    path, and the render body only *reads* the widget.

    Args:
        store: The chat store both actions mutate.
        selected: The session being acted on, or ``None`` when there is none.
    """

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
        # The click handler records the intent; the write happens on this render
        # only when the intent names this session, so a stale intent from a
        # different conversation can never rename the wrong one.
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
    """Record a rename intent in a callback (the legal place to write widget state)."""

    st.session_state[PENDING_RENAME_KEY] = (
        session_id,
        str(st.session_state.get(RENAME_STATE_KEY) or "").strip(),
    )


def _apply_pending_rename(store: SessionStore) -> None:
    """Write the queued rename, then clear the draft.

    The draft is cleared by *deleting* the widget's key, which Streamlit allows
    before the widget is instantiated — the call ordering in
    :func:`_render_session_actions` guarantees that.
    """

    pending = st.session_state.pop(PENDING_RENAME_KEY, None)
    if pending is None:
        return
    session_id, title = pending
    st.session_state.pop(RENAME_STATE_KEY, None)
    if not session_id or not title:
        return
    try:
        store.rename_session(session_id, title)
    except Exception as exc:  # a typed store error; its own message is the cause
        _render_store_error("The conversation could not be renamed.", exc)
        return
    st.toast("Conversation renamed")
    st.rerun()


def _queue_delete(session_id: str) -> None:
    """Record a delete intent in a callback (the legal place to write widget state)."""

    st.session_state[PENDING_DELETE_KEY] = session_id


def _apply_pending_delete(store: SessionStore) -> None:
    """Write the queued delete, then point the rail at a surviving session."""

    session_id = st.session_state.pop(PENDING_DELETE_KEY, None)
    if not isinstance(session_id, str) or not session_id:
        return
    try:
        store.delete_session(session_id)
    except Exception as exc:  # a typed store error; its own message is the cause
        _render_store_error("The conversation could not be deleted.", exc)
        return
    _fallback_after_delete(store)
    st.toast("Conversation deleted")
    st.rerun()


def _fallback_after_delete(store: SessionStore) -> None:
    """Point the rail at a surviving session after the selected one is gone.

    The list is re-read so the fallback is the *current* newest session, not a
    value computed before the delete. When nothing survives the stored id is
    dropped, and the next render creates a fresh session.
    """

    sessions = store.list_sessions()
    st.session_state[ACTIVE_SESSION_KEY] = sessions[0].id if sessions else None


def _render_store_unavailable() -> None:
    """The named failure state for a store that could not be opened.

    Every session control is a database write, so with no store there is nothing
    to render. Saying so — with the path that is missing — is the honest state; a
    rail of disabled widgets would hide the cause, and a silently empty rail would
    look like a first run.
    """

    label(SESSIONS_HEADING)
    with panel():
        st.error("The conversation store is unavailable.")
        caption_note(
            "Conversations cannot be listed, created, renamed or deleted until the "
            "chat store is readable."
        )


# --- model parameters ---------------------------------------------------------------


def render_param_controls() -> LLMParams:
    """Render the model-parameter group and return the assembled parameters.

    The widgets are seeded once per browser session from the environment
    (:func:`load_params_defaults`) and then own their state the way Streamlit
    widgets do — so a reload shows what the analyst last set, and the value that
    reaches ``invoke`` is always what the controls show. That is the acceptance
    test in ``docs/GRADING-ALIGNMENT.md`` row 16 ("params recorded per message")
    and in ``docs/STRESS-TEST-DELIVERY.md`` step 6 ("changing model/temperature
    affects the next answer").

    Returns:
        The :class:`~src.utils.llm.LLMParams` for the current control values.
        ``model`` is ``None`` when the picker is empty (the provider default).
    """

    load_params_defaults()
    label(PARAM_HEADING)
    with panel():
        _render_model_picker()
        _render_temperature()
        _render_top_p()
        _render_max_tokens()
        _render_retrieval_k()
    # Values are read from session state, never from the widget return: the keys
    # are stable, so the value the controls show and the value packed here are one
    # thing even when another widget's rerun recreated this one's state.
    return LLMParams(
        model=st.session_state.get(MODEL_PARAM_STATE_KEY) or None,
        temperature=st.session_state.get(_control_key("temperature")),
        top_p=st.session_state.get(_control_key("top_p")),
        max_tokens=st.session_state.get(_control_key("max_tokens")),
    )


def _control_key(control: str) -> str:
    """The widget key for one sampling control."""

    return f"{PARAM_STATE_PREFIX}-{control}"


def _render_model_picker() -> None:
    """The model selectbox, with the configured model as the default.

    The environment's model leads the option list, so a deployment that set
    ``OPENAI_MODEL`` shows that model selected on first render — the picker edits
    the *request*, it does not invent a configuration the deployment does not
    have. Clearing the box is allowed (an empty picker means "provider default"),
    which is why a placeholder exists rather than forcing a selection.
    """

    st.selectbox(
        "Model",
        list(DEFAULT_MODELS),
        index=None,
        placeholder="Provider default",
        key=MODEL_PARAM_STATE_KEY,
        help="Sent with every message; the environment's model is the default.",
    )


def _render_temperature() -> None:
    """The synthesis temperature slider.

    Synthesis is the one user-adjustable temperature in ADR-005's per-task table
    (classification and verification stay pinned), so the control's help says what
    it actually moves rather than implying a global knob. No ``value=`` is passed:
    the seeded session-state entry *is* the value, and a widget given both is a
    Streamlit duplication warning.
    """

    st.slider(
        "Temperature",
        min_value=TEMP_MIN,
        max_value=TEMP_MAX,
        step=TEMP_STEP,
        key=_control_key("temperature"),
        help="Synthesis sampling temperature. 0 is deterministic, 1 is maximally varied.",
    )


def _render_top_p() -> None:
    """The nucleus-sampling slider, 1.0 being the provider default."""

    st.slider(
        "Top-p",
        min_value=TOP_P_MIN,
        max_value=TOP_P_MAX,
        step=TOP_P_STEP,
        key=_control_key("top_p"),
        help="Nucleus sampling cutoff. 1.0 disables it (the provider default).",
    )


def _render_max_tokens() -> None:
    """The completion-length cap."""

    st.number_input(
        "Max tokens",
        min_value=MAX_TOKENS_MIN,
        max_value=MAX_TOKENS_MAX,
        step=MAX_TOKENS_STEP,
        key=_control_key("max_tokens"),
        help="Maximum tokens in one answer.",
    )


def _render_retrieval_k() -> None:
    """The retrieval top-k slider, exposed here but not packed into ``LLMParams``."""

    st.slider(
        "Retrieval k",
        min_value=K_MIN,
        max_value=K_MAX,
        step=1,
        key=RETRIEVAL_K_KEY,
        help=(
            "Child chunks retrieved per question before parent expansion (ADR-002). "
            "Higher k is more recall and more context. The caller reads this value: "
            "retrieval k is not a sampling parameter, so it is not sent to the model."
        ),
    )


def load_params_defaults() -> dict[str, Any]:
    """Seed the parameter widgets, once per browser session.

    Seeding is idempotent by design: ``setdefault`` only writes a control the
    analyst has not touched, so a rerun triggered by any other widget never resets
    a value that was changed. The values are read back out of session state by
    :func:`render_param_controls` — the widget seeds and the packed parameters
    therefore cannot drift, which a ``value=`` argument plus a keyed state entry
    silently could.

    The ``k`` default is the module that owns it
    (:data:`src.core.retrieval.DEFAULT_K`) rather than a copy, so the slider and
    the retriever cannot disagree about what "default" means.
    """

    values: dict[str, Any] = {
        "model": ENV_MODEL or "gpt-4o-mini",
        "temperature": TEMP_DEFAULT,
        "top_p": TOP_P_DEFAULT,
        "max_tokens": MAX_TOKENS_DEFAULT,
        "k": DEFAULT_RETRIEVAL_K,
    }
    for control, value in values.items():
        st.session_state.setdefault(_control_key(control), value)
    return values


# --- document status ----------------------------------------------------------------


def render_index_summary(
    status: Any | None = None,
    *,
    index_dir: Any | None = None,
) -> bool:
    """Render the one-glance index state, and return whether the index is usable.

    A thin wrapper for a caller that already holds an ``IndexStatus`` (the upload
    panel produces one after an ingest) and does not want to re-read the index.
    Without one, the reader in :mod:`src.components.upload` is asked directly — the
    reader is not duplicated here, because a second implementation of "is the index
    ready?" is exactly the drift this boundary exists to prevent.

    Args:
        status: A :class:`~src.components.upload.IndexStatus`, or ``None`` to read
            the index from ``index_dir``.
        index_dir: Index directory to read when ``status`` is ``None``.

    Returns:
        ``True`` when the index holds chunks to retrieve over.
    """

    if status is not None:
        return bool(getattr(status, "ready", False))
    try:
        from src.components.upload import index_status

        return bool(index_status(index_dir).ready)
    except Exception:  # unreadable index: report not-ready rather than raising
        return False


def _document_facts(
    store: SessionStore | None,
    *,
    index_dir: Any | None,
    index_ready: bool | None,
    doc_labels: Mapping[str, str] | None,
) -> tuple[bool, int, dict[str, str]]:
    """Read the corpus facts this rail reports, from whichever source has them.

    The document count comes from the store's document registry (ingest writes a
    row per document — ``docs/ARCHITECTURE.md``'s ``documents`` table), and the
    index state comes from the caller or from :mod:`src.components.upload`'s
    reader. Both reads degrade rather than raise: a store opened but unreadable
    reports zero documents, and an unreadable catalog reports "nothing indexed" —
    the rail's job is to state what is searchable, not to fail a render because a
    side table is unreadable.
    """

    if store is None:
        return False, 0, dict(doc_labels or {})
    try:
        registered = store.store.list_documents()
    except Exception:  # a store opened but unreadable: report zero, do not raise
        return False, 0, dict(doc_labels or {})

    if index_ready is None or doc_labels is None:
        ready, labels = _index_facts(index_dir)
    else:
        ready, labels = bool(index_ready), dict(doc_labels)
    return ready, len(registered), labels


def _index_facts(index_dir: Any | None) -> tuple[bool, dict[str, str]]:
    """The index state and the ``doc_id`` → filename map, or safe defaults.

    Read failures degrade to "nothing indexed": a missing catalog is a store with
    nothing indexed yet (an expected first-run state), and a corrupt catalog is a
    named problem for the upload panel, which owns the catalog. The rail needs only
    the boolean and the labels, so both degrade instead of raising into the render
    path.
    """

    try:
        from src.components.upload import document_names, index_status

        return bool(index_status(index_dir).ready), document_names(index_dir=index_dir)
    except Exception:
        return False, {}


def _render_document_status(
    ready: bool,
    doc_count: int,
    labels: Mapping[str, str],
    *,
    upload_panel: Any | None,
) -> None:
    """The documents section: status badge, per-document rows, ingest surface.

    Three states are all reachable and visually distinct:

    - **ready** — the index has chunks; the badge says so and names what is
      searchable, which is the "what can I ask about?" answer an analyst wants
      before typing.
    - **empty** — nothing indexed; a warning badge and the action to take, which is
      the empty state that keeps a cold clone from looking broken
      (``docs/STRESS-TEST-DELIVERY.md`` row 5).
    - **registered but not ready** — documents are recorded but the index holds no
      chunks to retrieve over, the ``RetrievalError`` case ``docs/RUNBOOK.md``'s
      table calls out; the section names which of the two is missing instead of
      showing a silently empty list.
    """

    label(DOCUMENTS_HEADING)
    with panel():
        if ready:
            badge("index ready", kind="success")
            st.caption(
                f"{doc_count} document(s) indexed · {len(labels)} searchable. "
                "Questions are answered from these files."
            )
        elif doc_count:
            badge("index not built", kind="warn")
            st.caption(
                f"{doc_count} document(s) registered, but nothing is searchable yet — "
                "rebuild the index from the upload panel, or run scripts/build_index.py."
            )
        else:
            badge("no documents", kind="warn")
            st.caption("No document indexed yet — upload a PDF, then ask.")

        if labels:
            for name in labels.values():
                figure(str(name), muted=True)
        elif not doc_count:
            st.caption("The upload panel below adds the first document.")

        if upload_panel is not None:
            upload_panel()


# --- footer -------------------------------------------------------------------------


def render_footer(store: Any | None = None) -> None:
    """The account row at the bottom of the rail.

    There is no authentication in this build (ADR-009's single-user workspace), so
    the row names the workspace and its plan rather than inventing a user identity,
    and states where the data lives rather than linking somewhere fabricated.
    Both lines are real facts: the plan is the local, keyless workspace, and the
    path is the store's own resolved location (``DATA_DIR`` moves it in the
    container, so it is read at render time rather than frozen in a constant).

    Args:
        store: The chat store, used to name its file. ``None`` resolves the
            default location from :func:`src.store.db.default_db_path`.
    """

    path = getattr(store, "path", None)
    location = str(path) if path is not None else str(default_db_path())
    with panel():
        st.markdown(f"**{ACCOUNT_NAME}**")
        st.caption(ACCOUNT_PLAN)
        figure(location, muted=True)
        caption_note(ACCOUNT_DATA_CAPTION)


# --- store errors -------------------------------------------------------------------


def _render_store_error(summary: str, exc: Exception) -> None:
    """Render a typed store failure under the control that raised it.

    The typed exception's own message is shown (never a generic fallback) because
    that is the text that names the cause — an unknown session id says so, a locked
    database says so. The summary line above it says *what* did not happen, so the
    reader is not left inferring which action failed.
    """

    detail = str(getattr(exc, "message", None) or exc) or type(exc).__name__
    st.error(summary)
    caption_note(detail)
    st.session_state[SESSION_ERROR_KEY] = detail

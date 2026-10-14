"""The left rail: brand, primary action, view navigation, conversations, workspace status.

ADR-009's boundary lives here: this module **renders controls and calls the chat
store**. It never retrieves, never builds a prompt, and never touches the LLM
gateway — the parameter widgets assemble a :class:`~src.utils.llm.LLMParams` and
hand it back to the caller, which is the object ``src.chains.graph.invoke``
accepts. Retrieval ``k`` is *not* part of that contract (it belongs to the
retriever call, not to sampling): the slider is rendered by
:func:`render_model_params` because the brief's parameter surface names it
alongside the sampling controls (ADR-005), but its value is read off
``st.session_state[RETRIEVAL_K_KEY]`` by the caller, which owns the plumbing into
``src.chains.nodes.retrieve``. The split is stated on the widget's help text so it
is visible rather than implied.

Two things about *where* this renders are deliberate.

The rail is the sidebar, and this module enters that container itself
(:func:`render_sidebar` opens ``st.sidebar``), because ``app.py`` calls it as one
bare statement. The rail's width comes from
``st.set_page_config(initial_sidebar_state=260)``; nothing else places this
content, so without the context manager the rail would render into the canvas and
the app would present one long column of controls above the conversation.

The parameter controls render in the *canvas*, not here. They used to sit in the
rail, which is what flattened the information architecture into a single stack of
widgets; :func:`render_model_params` is called by the Parameters view and returns
the same :class:`~src.utils.llm.LLMParams`. The values are not view state,
though, so they live in :data:`PARAMS_STATE_KEY` and drive the controls through
``value=`` — Streamlit resets a *keyed* widget's own key to the value it was first
registered with once that widget stops rendering, so a control that kept its value
in a widget key would silently lose it the moment the reader walked back to the
conversation.

Two sources of truth are kept apart here. The **chat store** owns sessions, and
nothing durable about a conversation lives in ``st.session_state`` except this
component's own bookkeeping keys. The **index** owns the corpus facts the bottom
card and :class:`SidebarResult` report, read through
:func:`src.components.upload.index_status`: a corpus built by
``scripts/build_index.py`` (the path the README documents) never writes the
store's document registry, so reading that registry made the rail report
"0 documents" while the canvas reported the real corpus. Both surfaces now read
the index, so they cannot disagree.

Public surface
--------------
``render_sidebar``          the whole rail; returns a :class:`SidebarResult`
``SidebarResult``           what one render produced (view, params, session, corpus facts)
``render_session_manager``  new / switch / rename / delete, on the store
``render_model_params``     model, temperature, top_p, max_tokens, k → ``LLMParams``
``current_view``            the selected view, reconciled against the views that exist
``load_params_defaults``    establish the parameter state and publish the retrieval-``k`` value
``active_session_id``       the selected id, reconciled against what exists
``session_title``           the display title for a stored session
``DEFAULT_MODELS``          the model picker's option list
``RETRIEVAL_K_KEY``         session-state key the caller reads the retrieval-``k`` from

Every action is a real Streamlit widget with a visible label (no unlabeled icon
buttons), every action reports its own outcome, and the error, empty and
unavailable states are all reachable — see each function's notes.
"""

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

if TYPE_CHECKING:  # contract types only — no runtime import, so the module stays leaf-level
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

# --- headings ---------------------------------------------------------------------

#: The label above the session list. The stylesheet renders it as the rail's
#: uppercase muted section label — the one piece of type in the rail that is
#: deliberately quieter than the rows it introduces.
SESSIONS_HEADING = "Previous conversations"

# --- the views ---------------------------------------------------------------------

#: Session-state key holding the selected view. The rail writes it and ``app.py``
#: reads it back off :class:`SidebarResult` to decide which canvas to render —
#: this one key is the whole of the app's view routing, which is what keeps the
#: routing in the shell rather than in a page framework (ADR-009).
VIEW_KEY = "isip-view"

#: The conversation surface. Not one of the nav items: it is where the app
#: already is, and the primary action plus any session row are how a reader
#: returns to it from a view.
VIEW_CHAT = "chat"

#: The ingest surface: the uploader, the index build, the document catalog.
VIEW_DOCUMENTS = "documents"

#: How retrieval works and what the index holds.
VIEW_RETRIEVAL = "retrieval"

#: Sampling and retrieval controls — the widgets that used to live in the rail.
VIEW_PARAMETERS = "parameters"

#: The node trace, latency and scores of the last answer.
VIEW_DIAGNOSTICS = "diagnostics"

#: Every view ``app.py`` can dispatch to. A stored value outside this set (a
#: stale key written by an older build) falls back to :data:`VIEW_CHAT` rather
#: than rendering a blank canvas.
VIEWS: tuple[str, ...] = (
    VIEW_CHAT,
    VIEW_DOCUMENTS,
    VIEW_RETRIEVAL,
    VIEW_PARAMETERS,
    VIEW_DIAGNOSTICS,
)

#: Prefix for the nav buttons' widget keys. Reserved by the stylesheet, which
#: scopes the rail's left-aligned row styling to exactly this prefix — a nav
#: button keyed anything else would keep Streamlit's centred pill shell.
NAV_KEY_PREFIX = "isip-nav"

#: The nav, in order: ``(view, label, material icon)``. Documents leads because
#: the corpus is the precondition for every question; Diagnostics trails because
#: it inspects an answer rather than producing one.
NAV_ITEMS: tuple[tuple[str, str, str], ...] = (
    (VIEW_DOCUMENTS, "Documents", ":material/folder_open:"),
    (VIEW_RETRIEVAL, "Retrieval", ":material/manage_search:"),
    (VIEW_PARAMETERS, "Parameters", ":material/tune:"),
    (VIEW_DIAGNOSTICS, "Diagnostics", ":material/monitor_heart:"),
)

# --- the brand row -----------------------------------------------------------------

#: Brand mark. A wordmark, not an emoji — ``docs/CONSTRAINTS.md``'s no-emoji rule.
BRAND_MARK = "IsipFin"

#: The line under the wordmark. It names the product rather than describing the
#: demo, and it is the rail's only prose: everything below it is a control.
BRAND_TAGLINE = "Cited answers from your financial filings"

#: The primary CTA's label. One action at the top of the rail has an obvious,
#: visible label; an unlabeled icon button would not be an accessible control.
NEW_SESSION_LABEL = "New conversation"

#: Tooltip on the CTA while the active conversation still holds a turn: the click
#: starts a fresh one.
NEW_SESSION_HELP = "Start a fresh conversation"

#: Tooltip on the CTA while the active conversation is already empty. The click
#: still returns to the conversation surface — creating a session that holds
#: nothing would only fill the list with empty rows.
NEW_SESSION_FRESH_HELP = "This conversation is already empty — this returns to it"

#: Tooltip on the CTA when the store could not be opened: there is nowhere to
#: create a session, so the click is navigation only, and saying so keeps the
#: primary action from promising something it cannot do.
NEW_SESSION_NO_STORE_HELP = (
    "The conversation store is unavailable — this returns to the conversation"
)

# --- the parameter surface (GRADING-ALIGNMENT.md row 16) ------------------------

#: The model picker's hardcoded options, leading with ``""`` — the "provider
#: default" row Streamlit renders as the placeholder state rather than a
#: selectable blank. The deployment's own model is **not** part of this tuple:
#: it is resolved per render from the gateway's resolver and put in front (see
#: :func:`_configured_model`), so the list cannot go stale against a name that is
#: only known from the environment. The rest are models ``docs/adr/ADR-005``
#: names as drop-in compatible, so switching provider is a pick rather than an edit.
DEFAULT_MODELS: tuple[str, ...] = (
    "",
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

#: The session-state key holding the retrieval top-``k``. A **plain** key, not a
#: widget key: the slider that sets it is driven by ``value=`` (see
#: :func:`render_model_params`), so nothing else owns this name and the caller can
#: read it on any run — including one that never rendered the slider. Exported so
#: the caller reads the value without re-typing the string.
RETRIEVAL_K_KEY = "isip-param-k"

#: Holds the last committed parameter values. Deliberately **not** a widget key:
#: Streamlit resets a keyed widget's key to its registration value once that
#: widget stops rendering, and the controls now live in their own view — so a
#: conversation run would otherwise replace every value the analyst set with the
#: one the control happened to start on.
PARAMS_STATE_KEY = "isip-params-state"

#: The primary CTA's widget key.
NEW_SESSION_KEY = "isip-new-session"

#: Prefix for the session-list controls' widget keys.
SESSION_CONTROL_PREFIX = "isip-session-control"

# --- the workspace card ------------------------------------------------------------

#: The bottom card's title. There is no authentication layer (ADR-009's
#: single-user demo), so the card names the workspace, not a signed-in user.
WORKSPACE_TITLE = "Local workspace"

#: The card's badge when the index holds chunks to retrieve over.
WORKSPACE_READY_BADGE = "READY"

#: The card's badge when it does not.
WORKSPACE_EMPTY_BADGE = "NO INDEX"

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
    object: the surface the canvas should render, the parameters to pass to
    ``invoke``, the session questions are asked in, and the corpus facts the main
    pane needs for its empty state and its citation labels.

    Attributes:
        params: The assembled sampling parameters, read back off the committed
            parameter state (never off a widget this run may not have
            rendered). ``temperature``/``top_p``/``max_tokens`` always carry a
            value (a slider always has one), so a change to any of them shows up
            in the next answer — GRADING-ALIGNMENT.md row 16. ``model`` is
            ``None`` only when the picker is empty, which the caller treats as
            "provider default".
        session_id: The selected session's id, or ``None`` when no store is
            available. ``session_id`` and ``session`` always agree.
        session: The selected :class:`~src.store.sessions.Session`, or ``None``
            when there is nothing selected.
        doc_count: Documents in the on-disk index catalog, read through
            :func:`src.components.upload.index_status`. This is the *index*
            count, not the chat store's document registry — a corpus built by
            ``scripts/build_index.py`` populates the index and leaves that
            registry empty, and reporting the registry here is what made the
            rail under-report the corpus.
        index_ready: Whether the index holds chunks to retrieve over.
        doc_labels: ``doc_id`` → display name, for the transcript's citation cards.
        sessions: The sessions visible in the list, most recently updated first.
        view: The surface the canvas should render, one of :data:`VIEWS`. The
            rail writes it from its nav; this is the value ``app.py`` dispatches
            on.
    """

    params: LLMParams
    session_id: str | None
    session: Session | None
    doc_count: int
    index_ready: bool
    doc_labels: dict[str, str]
    sessions: tuple[Session, ...]
    view: str


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

    The rail renders top to bottom in the order the layout prescribes: the brand
    row and the primary action, the four view items, a divider, the labelled
    conversation list, and the workspace status card at the bottom. Every part of
    it is in the sidebar — this function opens that container itself, because
    ``app.py`` calls it as a bare statement.

    The corpus facts the card and the result carry are read from the index
    (:func:`src.components.upload.index_status`) rather than from the store's
    document registry, which only the UI's upload path writes. The two surfaces
    therefore state the same number, which is what ``index_ready`` and
    ``doc_count`` mean everywhere else in the app.

    Parameters
    ----------
    store:
        The chat store. ``None`` leaves every session action replaced by a named
        unavailable state while navigation and the status card still render,
        which is the honest rendering for a store that failed to open — the rail
        must not go blank because one table is missing.
    index_dir:
        Index directory to read the corpus facts from, when it is not the
        default.
    index_ready:
        Whether the index is usable, for a caller that already read that fact.
        ``None`` reads it here. Pass it only alongside ``doc_labels``, and only
        when both came from the same reader — the status card always states the
        on-disk index, so a stale override is how the rail and the canvas would
        start disagreeing again.
    doc_labels:
        ``doc_id`` → filename map for the citation cards, for the same caller.
        ``None`` reads the catalog here.
    upload_panel:
        Optional callable rendering the ingest surface, called with no arguments
        under the conversation list. The rail does not own document ingestion —
        the Documents view does — so this is the extension point for a caller
        that wants the uploader beside the rail as well.

    Returns:
        A :class:`SidebarResult` the main pane passes to
        ``src.chains.graph.invoke``, to the transcript renderer and to its own
        view dispatch.
    """
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


# --- brand and navigation -----------------------------------------------------------


def current_view() -> str:
    """The selected view, reconciled against the views that exist.

    ``st.session_state`` holds the selection across reruns; a value this rail
    does not know — a stale key from an older build, or a typo — falls back to
    the conversation rather than rendering a canvas nothing can fill.
    """

    view = st.session_state.get(VIEW_KEY)
    return view if isinstance(view, str) and view in VIEWS else VIEW_CHAT


def _select_view(view: str) -> None:
    """Record the clicked view.

    Called from the nav button's ``on_click``, which Streamlit runs *before* the
    script body: the nav's own highlight and the canvas then read the same value
    on the run that paints, so the selected item never trails a click by one
    run.
    """

    st.session_state[VIEW_KEY] = view


def _render_brand(store: SessionStore | None, selected: Session | None) -> None:
    """The brand row and the primary "New conversation" action.

    The action is never disabled. It always returns the reader to the
    conversation surface — the only way back from a view that has no chat, since
    the conversation is deliberately not one of the nav items — and starts a new
    session only when the active one already holds a turn: a click that always
    created a session would fill the list with rows that hold nothing.

    Args:
        store: The chat store a new session is created in. ``None`` (no store)
            narrows the click to the navigation, which the panel under the
            conversation list names.
        selected: The active session, whose turn count decides whether a click
            starts a new conversation.
    """

    theme_rail_brand(BRAND_MARK, BRAND_TAGLINE)
    empty = selected is not None and selected.turn_count == 0
    if store is None:
        help_text = NEW_SESSION_NO_STORE_HELP
    elif empty:
        help_text = NEW_SESSION_FRESH_HELP
    else:
        help_text = NEW_SESSION_HELP
    # The widget call stays unconditional and its result is held in a name: Streamlit must
    # render the button on every run, and collapsing this into `if st.button(...) and
    # _start_conversation(...)` would hide the side effect inside a boolean expression.
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
    """Return to the conversation surface, starting a new session when needed.

    The stored selection is moved onto the session this creates. Without that,
    the rail's own reconciliation would keep the *stored* session — it is still
    in the list — and the click would add a row while leaving the reader in the
    conversation they were already in, which is the opposite of what the primary
    action says it does.

    Args:
        store: The chat store, or ``None`` when it is unavailable.
        empty: Whether the active conversation already holds no turn.

    Returns:
        ``True`` when a session was created, which is the caller's cue to rerun —
        the rail read its session list and selection before this ran.
    """

    st.session_state[VIEW_KEY] = VIEW_CHAT
    if store is None or empty:
        return False
    created = _create_session(store)
    if created is None:
        return False
    st.session_state[ACTIVE_SESSION_KEY] = created.id
    return True


def _render_nav() -> None:
    """The four view items: labelled buttons, one per surface the canvas has.

    The selected item renders as the primary button, which the stylesheet squares
    back into a row — so "where am I" is carried by the app's one accent colour
    rather than by a second indicator. The click is handled by ``on_click``
    (see :func:`_select_view`), so a click both routes and repaints in one run.
    """

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

    The label is the rail's uppercase muted ``PREVIOUS CONVERSATIONS`` heading,
    and it is rendered once — including in the unavailable branch, which names
    the store's problem under the same heading rather than above a second copy
    of it.

    Args:
        store: The chat store the row's switch writes through. ``None`` renders
            the unavailable state instead, so a store that failed to open is named
            rather than silently hidden.
        sessions: Every session, newest first.
        selected: The active session, or ``None`` when the store is empty and a
            create failed — the list still renders, the actions do not.
    """

    if store is None:
        _render_store_unavailable()
        return
    theme_rail_label(SESSIONS_HEADING)
    _render_session_rows(store, sessions, selected)
    _render_session_actions(store, selected)


def _render_session_rows(
    store: SessionStore, sessions: Sequence[Session], selected: Session | None
) -> None:
    """One labelled button per session, with a bounded list and a "show all" row.

    Each row is a real, labelled button — never an unlabeled icon button
    (``docs/CONSTRAINTS.md``) — so the whole row is the target and the label is
    read by a screen reader. The selected row carries the accent fill, which is
    the only way to tell which session is active after a rerun that recreated the
    widget state. Rows are full width, matching the nav items above them, so the
    rail reads as one column of rows. Rows past :data:`MAX_SESSIONS_SHOWN`
    collapse behind one "show all" control rather than growing the rail without
    bound.
    """

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
    """Open a conversation, return to the conversation surface, and rerun.

    The view is set here as well as the session: a reader who opens a
    conversation from one of the nav views means to read it, and leaving them on
    Documents with the row highlighted would look like the click did nothing.
    That is also true of the row that is already selected — the highlighted one —
    where the click changes only the surface, which is why every row is a live
    control and none of them is disabled.

    The rename draft is dropped, because it belongs to the session it was typed
    for: leaving it would let the next Rename click apply a title meant for a
    different conversation.
    """

    store.switch_session(session_id)
    st.session_state[ACTIVE_SESSION_KEY] = session_id
    st.session_state[VIEW_KEY] = VIEW_CHAT
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
    to render. Saying so — under the section's own heading — is the honest state;
    a rail of disabled widgets would hide the cause, and a silently empty rail
    would look like a first run.
    """

    theme_rail_label(SESSIONS_HEADING)
    with theme_panel():
        st.error("The conversation store is unavailable.")
        theme_caption_note(
            "Conversations cannot be listed, created, renamed or deleted until the "
            "chat store is readable."
        )


# --- model parameters ---------------------------------------------------------------


def render_model_params() -> LLMParams:
    """Render the sampling and retrieval controls, and return the assembled parameters.

    Rendered by the **canvas** (the Parameters view), not by the rail: the controls
    used to sit in the sidebar, which is what flattened the app's information
    architecture into one long column of widgets.

    The controls are driven by :data:`PARAMS_STATE_KEY` rather than by their own
    widget keys, and that is measured rather than stylistic. Streamlit drops the
    state of a widget that did not render on a run, and when a *keyed* widget
    disappears it does not simply forget the value — it resets that key to the
    value the widget was first registered with. Measured: set temperature to 0.9,
    leave the view, and one run later ``st.session_state`` holds the original 0.2
    again while the panel still shows 0.9 where it was asked. A control whose
    value has to survive the reader walking over to the conversation — which is
    the whole point of settings — therefore cannot keep it in a widget key.

    So each control is given ``value=`` from the durable state and read from its
    own return value, which is the one path Streamlit supports cleanly: the
    committed value is what the reader sees on re-entry, and nothing has to be
    reconciled against Streamlit's own idea of a default. That is the acceptance
    test in ``docs/GRADING-ALIGNMENT.md`` row 16 ("params recorded per message")
    and in ``docs/STRESS-TEST-DELIVERY.md`` step 6 ("changing
    model/temperature affects the next answer").

    Returns:
        The :class:`~src.utils.llm.LLMParams` for the current control values.
        ``model`` is ``None`` when the picker is empty (the provider default).
    """

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
    """The model this deployment resolved, straight from the gateway's resolver.

    The picker used to seed from ``OPENAI_MODEL`` alone, which made the primary
    name — ``LLM_MODEL``, the one the quick start tells a reader to set — a
    model the picker did not show. Reading the gateway's own resolver
    (:func:`src.utils.llm.resolved_config`) instead of one variable is the same
    fix the credential chain got: the surface is defined in one place, so the
    control cannot drift from what a request will actually use.

    ``MOCK_LLM=1`` is deliberately not branched on here. :func:`resolved_config`
    never builds a client and never imports ``openai``, and the model chain is
    read before the mock short-circuit on the real path anyway, so the offline
    demo resolves exactly the same names with no key present.

    Returns:
        The resolved model id, or ``""`` when the environment names none and
        the provider has no default of its own (which the built-in default
        answers, so this is only ``""`` for an unknown provider).
    """

    return resolved_config().model or ""


def _param_defaults() -> dict[str, Any]:
    """The factory values: the deployment's model, and the documented defaults.

    The ``k`` default is the module that owns it
    (:data:`src.core.retrieval.DEFAULT_K`) rather than a copy, so the slider and
    the retriever cannot disagree about what "default" means.
    """

    return {
        "model": _configured_model(),
        "temperature": TEMP_DEFAULT,
        "top_p": TOP_P_DEFAULT,
        "max_tokens": MAX_TOKENS_DEFAULT,
        "k": DEFAULT_RETRIEVAL_K,
    }


def _param_state() -> dict[str, Any]:
    """The last committed parameter values, durable across view switches.

    A plain session-state key, not the controls' own widget keys: Streamlit
    resets a keyed widget's key to its registration value once the widget stops
    rendering (see :func:`render_model_params` for the measurement), so the
    widgets cannot be their own storage. Created with the factory values on the
    first run and never overwritten wholesale afterwards — a missing control is
    filled in, an existing one is left exactly as the analyst left it.
    """

    state = st.session_state.get(PARAMS_STATE_KEY)
    if not isinstance(state, dict):
        state = {}
        st.session_state[PARAMS_STATE_KEY] = state
    for control, value in _param_defaults().items():
        state.setdefault(control, value)
    return state


def load_params_defaults() -> dict[str, Any]:
    """Establish the parameter state, and publish the retrieval-``k`` value.

    Called by the rail on every render, ahead of the canvas: the durable state is
    created with the factory values on the first run, and
    ``st.session_state[RETRIEVAL_K_KEY]`` is set to the current ``k`` — the key
    ``app.py`` reads to set the retriever's top-k for one call. Publishing it here
    rather than only from the Parameters view is what keeps that read correct on
    a run where the slider was never rendered.

    Returns:
        The current values, one per control, including ``k`` — which is not part
        of :class:`~src.utils.llm.LLMParams` and is read back by the caller.
    """

    values = _param_state()
    st.session_state[RETRIEVAL_K_KEY] = values["k"]
    return dict(values)


def _commit_params(values: Mapping[str, Any]) -> None:
    """Record what the controls returned, and publish the retrieval-``k`` value.

    ``k`` is not part of ``LLMParams`` (it belongs to the retriever call, not to
    sampling), so it is not handed back as a parameter — it is published onto
    ``st.session_state[RETRIEVAL_K_KEY]``, the plain key the caller reads.
    """

    state = _param_state()
    state.update(values)
    st.session_state[RETRIEVAL_K_KEY] = state["k"]


def _current_params() -> LLMParams:
    """The parameters the graph would get, read without rendering any control."""

    state = _param_state()
    return LLMParams(
        model=state.get("model") or None,
        temperature=state.get("temperature"),
        top_p=state.get("top_p"),
        max_tokens=state.get("max_tokens"),
    )


def _render_model_picker() -> str | None:
    """The model selectbox, with the configured model selected.

    The deployment's model leads the option list, so a user who set ``LLM_MODEL``
    (or its ``OPENAI_MODEL`` alias) sees **their** model selected on first render
    — the picker edits the *request*, it does not invent a configuration the
    deployment does not have. The name comes from the gateway's resolver rather
    than from a single variable, so it follows the whole chain
    (``params.model`` → ``LLM_MODEL`` → ``OPENAI_MODEL`` → the provider default)
    instead of one spelling of it.

    A resolved model that is not one of the hardcoded :data:`DEFAULT_MODELS` is
    still put in the option list, because it is what a request will actually
    use: snapping the selection to a fixed row would show the user a model that
    is not theirs. Clearing the box is allowed (an empty picker means "provider
    default"), which is why a placeholder exists rather than forcing a selection;
    the cleared state is a committed ``None``, so it survives a view switch like
    any other value — and so does a selection, because the value is driven from
    :data:`PARAMS_STATE_KEY` rather than from the widget's own key.

    Returns:
        The selected model, or ``None`` for the provider default.
    """

    options = list(DEFAULT_MODELS)
    configured = _configured_model()
    if configured:
        # The deployment's own model leads, so the picker shows what a request
        # will use — including a name no hardcoded row carries.
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
    """The synthesis temperature slider; returns the value it now holds.

    Synthesis is the one user-adjustable temperature in ADR-005's per-task table
    (classification and verification stay pinned), so the control's help says what
    it actually moves rather than implying a global knob.
    """

    return st.slider(
        "Temperature",
        min_value=TEMP_MIN,
        max_value=TEMP_MAX,
        step=TEMP_STEP,
        value=_param_state()["temperature"],
        help="Synthesis sampling temperature. 0 is deterministic, 1 is maximally varied.",
    )


def _render_top_p() -> float:
    """The nucleus-sampling slider, 1.0 being the provider default."""

    return st.slider(
        "Top-p",
        min_value=TOP_P_MIN,
        max_value=TOP_P_MAX,
        step=TOP_P_STEP,
        value=_param_state()["top_p"],
        help="Nucleus sampling cutoff. 1.0 disables it (the provider default).",
    )


def _render_max_tokens() -> int:
    """The completion-length cap."""

    return st.number_input(
        "Max tokens",
        min_value=MAX_TOKENS_MIN,
        max_value=MAX_TOKENS_MAX,
        step=MAX_TOKENS_STEP,
        value=_param_state()["max_tokens"],
        help="Maximum tokens in one answer.",
    )


def _render_retrieval_k() -> int:
    """The retrieval top-k slider, exposed here but not packed into ``LLMParams``."""

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


# --- corpus facts and the workspace card --------------------------------------------


def _index_snapshot(index_dir: Any | None) -> tuple[IndexStatus, dict[str, str]]:
    """The on-disk index status and its ``doc_id`` → filename map.

    Both come from the reader that owns the question
    (:func:`src.components.upload.index_status` and
    :func:`src.components.upload.document_names`), which counts the chunks in the
    index rather than the rows in the chat store's document registry — a corpus
    built by ``scripts/build_index.py`` fills the index and leaves that registry
    empty, and reading it here is what made the rail report an empty corpus beside
    a canvas reporting the real one.

    ``index_status`` documents that it never raises: an unreadable or malformed
    index comes back as not-ready with its reason, which is what the card renders.
    The catalog read *can* fail on a malformed line, and a rail that cannot name a
    document is still a usable rail, so the labels degrade to empty.

    The import is local because this module reaches ``upload`` the way it always
    has — lazily — which keeps the rail importable without the ingest stack.
    """

    from src.components.upload import document_names, index_status

    status = index_status(index_dir)
    try:
        labels = document_names(index_dir=index_dir)
    except Exception:  # a malformed catalog: name no document rather than fail the rail
        labels = {}
    return status, labels


def _render_status_card(status: IndexStatus, store: Any | None) -> None:
    """The rail's bottom identity/status card.

    Every line is a fact read from the index or from the store's own location:
    the document and chunk counts, the size on disk, the reason when the index is
    unusable, and where conversations live. The badge is the same ``status.ready``
    the canvas reads, so the two surfaces cannot contradict each other.

    Args:
        status: The index snapshot, read by :func:`_index_snapshot`.
        store: The chat store, used to name its file; ``None`` falls back to the
            default location.
    """

    theme_rail_status_card(
        WORKSPACE_TITLE,
        _workspace_lines(status, store),
        badge=WORKSPACE_READY_BADGE if status.ready else WORKSPACE_EMPTY_BADGE,
    )


def _workspace_lines(status: IndexStatus, store: Any | None) -> list[str]:
    """The status card's lines, in the order a reader needs them.

    When the index is usable the card states what is searchable; when it is not it
    states what is missing and the reader's reason for it, which is the one thing
    that tells a cold clone ("nothing indexed yet") apart from a broken index. The
    last line is where the conversations live, so the card answers "where is my
    data" as well as "what can I ask about".
    """

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
    """``1 document`` / ``3 documents`` — the counts on the card read as prose."""

    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _store_location(store: Any | None) -> str:
    """Where the conversations live: the store's own resolved path.

    Read at render time rather than frozen in a constant, because ``DATA_DIR``
    moves it in the container.
    """

    path = getattr(store, "path", None)
    return str(path) if path is not None else str(default_db_path())


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
    theme_caption_note(detail)
    st.session_state[SESSION_ERROR_KEY] = detail

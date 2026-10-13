"""User-facing error banners: one classifier for every failure, one renderer (task T21).

The app never shows a stack trace and never shows nothing. This module is the single place
that turns an exception into the banner a person can act on, and the single place that draws
it, so a failed turn, a failed upload and a store that will not open all end up saying the
same kind of thing: a short title, what happened in plain language, and the next step.

The taxonomy is the one frozen in ``docs/adr/ADR-007-observability-errors.md``; the wording
follows its table and docs/OBSERVABILITY.md's "Failure -> user mapping". The wording rules
this module keeps:

* the title never carries a class name — exception names belong in the log, not in the UI;
* the hint always names a concrete next step (an env var, ``MOCK_LLM=1``, a lower k, an
  upload, a rebuild), and is ``None`` only when there is genuinely nothing to do;
* a failure whose type this module does not know is still rendered, not swallowed: the
  generic view keeps ``type(exc).__name__`` and ``str(exc)`` in the message;
* a secret is never echoed — the auth view names ``OPENAI_API_KEY`` and never a key value.

Boundary (ADR-009)
------------------
This module renders; it decides nothing about the pipeline. It runs no retrieval, builds no
prompt, and **never imports the model gateway and never constructs a client**. That is why
the gateway's typed errors are the one branch matched structurally, by the module that
declares the class (:data:`_GATEWAY_MODULE`), instead of with an ``isinstance`` check: the
import that would make the check possible is the import this module must not have.

The same structural rule covers the graph's typed-error spine and the local embedder.
``src.chains.graph`` imports langgraph and every node, and ``src.core.local_embedder``
imports ``fastembed``; a component module importing either would drag the pipeline (or a
heavy optional dependency) in at import time. Both families are therefore matched by the
class's own ``__module__`` and ``__name__`` — :func:`_is_graph_error` on the spine's base
class, so subclasses inherit one view instead of each getting a copy, and
:func:`_is_local_embedder_error` by name. The contract types are named under
``TYPE_CHECKING``, for annotations only.

Public surface
--------------
``ErrorView``       the frozen banner description (title, message, hint, retryable)
``classify_error``  any exception -> an :class:`ErrorView`; every exception gets one
``render_error``    draw the banner, with an optional Retry action
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from src.components.theme import caption_note, label
from src.core.pdf_parser import ParseError
from src.core.retrieval import RetrievalError
from src.store.db import SessionNotFoundError, StoreError

if TYPE_CHECKING:  # contract types only — no runtime import, so the module stays leaf-level
    from src.chains.graph import GraphError
    from src.core.local_embedder import LocalEmbeddingError

__all__ = [
    "ErrorView",
    "classify_error",
    "render_error",
]

#: The gateway module, used as a match key only — it is deliberately **not** imported.
_GATEWAY_MODULE = "src.utils.llm"

#: The graph module and the base class of its typed-error spine, both used as match keys
#: only. ``src.chains.graph`` imports langgraph and every node, so a component module that
#: imported it would pull the whole pipeline in at import time (ADR-009's leaf rule); the
#: family is therefore matched structurally, exactly like the gateway's errors below.
_GRAPH_MODULE = "src.chains.graph"
_GRAPH_BASE = "GraphError"

#: The Retry action's widget key. Fixed, like the chat canvas's ``isip-chat-retry``: one
#: error banner with an action is rendered per run, so a clash would mean the call site drew
#: a second banner, not that the key needed to be dynamic.
_RETRY_KEY = "isip-error-retry"
_RETRY_LABEL = "Retry"

#: Attributes a :class:`~src.core.pdf_parser.ParseError` may carry the failing page on, and
#: the fallback: the parser's own messages name the page ("pdfplumber failed on page 3").
_PAGE_ATTRS = ("page", "page_number")
_PAGE_IN_MESSAGE = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ErrorView:
    """One failure, described for the person looking at the screen.

    Attributes:
        title: A short headline with no exception or class names ("Model is busy").
        message: What happened, in plain language.
        hint: The actionable next step, or ``None`` when there is nothing to do.
        retryable: Whether asking again could plausibly succeed. The caller forwards this
            to :func:`render_error` as ``on_retry`` when it has a retry path.
    """

    title: str
    message: str
    hint: str | None
    retryable: bool


# --- the views (ADR-007's taxonomy, worded for a user) ---------------------------


_AUTH_VIEW = ErrorView(
    title="Model access denied",
    message=(
        "The model provider refused the request: the API key is missing, wrong, or no longer "
        "valid."
    ),
    hint=(
        "Set OPENAI_API_KEY to a valid key and restart the app, or set MOCK_LLM=1 to keep "
        "working offline with the mock model."
    ),
    retryable=False,
)

_RATE_LIMITED_VIEW = ErrorView(
    title="Model is busy",
    message=(
        "The model provider is rate limiting requests right now — the quota is spent, or "
        "the service is briefly overloaded."
    ),
    hint=(
        "Wait a few seconds and retry. A narrower question is also less likely to hit the "
        "limit again."
    ),
    retryable=True,
)

_CONTEXT_LENGTH_VIEW = ErrorView(
    title="Context is too large",
    message=(
        "This question needs more retrieved context than the model can hold, so the provider "
        "rejected the request."
    ),
    hint="Lower the retrieval k in the sidebar, or ask a narrower question.",
    retryable=True,
)

_TIMEOUT_VIEW = ErrorView(
    title="Model timed out",
    message="The model did not answer within the time limit.",
    hint="Retry the question; a shorter or narrower question is less likely to time out.",
    retryable=True,
)

_SCHEMA_VIOLATION_VIEW = ErrorView(
    title="Model reply could not be used",
    message=(
        "The model's reply did not match the structure the answer step requires, so it could "
        "not be validated."
    ),
    hint=(
        "Retry the question — the gateway asks the model once more with a stricter " "instruction."
    ),
    retryable=True,
)

_RETRIEVAL_VIEW = ErrorView(
    title="Search index unavailable",
    message=(
        "The index is missing or could not be read, so there is nothing to retrieve from for "
        "this question."
    ),
    hint="Upload a PDF to build the index, or run scripts/build_index.py.",
    retryable=False,
)

_STORE_VIEW = ErrorView(
    title="Conversation store unavailable",
    message="The conversation store could not be read or written, so this turn was not saved.",
    hint=(
        "Keep asking questions — the answers themselves are unaffected; history stays "
        "incomplete until the store recovers."
    ),
    retryable=False,
)

_SESSION_MISSING_VIEW = ErrorView(
    title="Conversation not found",
    message="That conversation no longer exists — it may have been deleted in another tab.",
    hint="Start a new conversation from the sidebar and ask again.",
    retryable=False,
)

#: Every :class:`~src.chains.graph.GraphError`, matched through its base class so the whole
#: family — ``GraphNodeError``, ``GraphStateError`` and any subclass a later task adds —
#: lands on one view rather than three copies of it.
_GRAPH_VIEW = ErrorView(
    title="The answer could not be produced",
    message=(
        "Answering the question was refused at a step the pipeline controls, so no answer "
        "was produced for this turn."
    ),
    hint=(
        "Change the question — an empty or whitespace-only one, or a non-text input, is not "
        "answerable — and ask again. The same failure will come back while the input is "
        "unchanged, so retrying as-is will not help."
    ),
    retryable=False,
)

#: A :class:`~src.core.local_embedder.LocalEmbeddingError` that escaped outside
#: ``src.core.retrieval``'s wrapper, so it was never rebound to a ``RetrievalError``. The
#: same view as that wrapper's rebind: a dense leg that cannot embed leaves the index
#: unusable, whether the embedder is this module's or the runtime one.
_LOCAL_EMBEDDER_VIEW = _RETRIEVAL_VIEW

#: The provider-mapped gateway errors ADR-007 names, keyed by class name. Every value is
#: reached through :func:`_gateway_error_name`, never through an import.
_GATEWAY_VIEWS: dict[str, ErrorView] = {
    "Auth": _AUTH_VIEW,
    "ContextLength": _CONTEXT_LENGTH_VIEW,
    "RateLimited": _RATE_LIMITED_VIEW,
    "SchemaViolationError": _SCHEMA_VIOLATION_VIEW,
    "Timeout": _TIMEOUT_VIEW,
}


# --- the public API --------------------------------------------------------------


def classify_error(exc: BaseException) -> ErrorView:
    """Turn any exception into the banner a user should see.

    The branches are ADR-007's table in order: the store first (so its subclass wins),
    then retrieval, then the parser (whose view is built from the exception, to name the
    page), then the gateway's typed errors, then the graph's spine and the local embedder
    (both matched structurally, see the module docstring), and finally the generic view —
    which is the branch that keeps a failure nobody planned for from becoming a blank
    screen.

    A typed view never carries the exception's own text: its message and hint are fixed
    strings, so a class name and a provider message reach the log and stop there. That is
    the whole point of the branches — a view that leaked ``GraphStateError: question is
    empty`` at the user would be telling them nothing they can act on.

    Args:
        exc: The exception that ended the turn or the upload.

    Returns:
        A frozen :class:`ErrorView`. Every exception gets one: nothing here raises, and
        the generic view carries ``type(exc).__name__`` and ``str(exc)`` so the detail is
        never dropped.

    """

    if isinstance(exc, SessionNotFoundError):
        return _SESSION_MISSING_VIEW
    if isinstance(exc, StoreError):
        return _STORE_VIEW
    if isinstance(exc, RetrievalError):
        return _RETRIEVAL_VIEW
    if isinstance(exc, ParseError):
        return _parse_view(exc)

    name = _gateway_error_name(exc)
    if name is not None:
        gateway_view = _GATEWAY_VIEWS.get(name)
        if gateway_view is not None:
            return gateway_view

    if _is_graph_error(exc):
        return _GRAPH_VIEW
    if _is_local_embedder_error(exc):
        return _LOCAL_EMBEDDER_VIEW

    return _generic_view(exc)


def render_error(exc: BaseException, *, on_retry: bool = False) -> bool:
    """Draw the banner for ``exc``; ``True`` on the rerun in which Retry was clicked.

    Built from the theme's own helpers — :func:`~src.components.theme.label` for the
    title and :func:`~src.components.theme.caption_note` for the hint — plus
    ``st.error`` and ``st.button``, so the banner inherits the app's tokens, radius and
    alert skin rather than declaring a second palette. Nothing from the exception is
    interpolated raw: the message is escaped here and the theme helpers escape their own
    text.

    Args:
        exc: The exception to describe.
        on_retry: Whether this call site has a retry path, i.e. whether the banner offers
            the action. Forward the view's own answer when it does::

                view = classify_error(exc)
                if render_error(exc, on_retry=view.retryable):
                    ...

    Returns:
        ``True`` on the rerun in which the Retry button was clicked, ``False`` otherwise.

    """

    view = classify_error(exc)
    label(view.title)
    st.error(_escape(view.message))
    if view.hint:
        caption_note(view.hint)
    if on_retry:
        return bool(st.button(_RETRY_LABEL, key=_RETRY_KEY))
    return False


# --- helpers ---------------------------------------------------------------------


def _parse_view(exc: ParseError) -> ErrorView:
    """The parse-failure banner, naming the failing page when the exception carries one."""

    page = _page_number(exc)
    page_note = f" Page {page} could not be read." if page is not None else ""
    return ErrorView(
        title="Could not read the PDF",
        message=(
            "The document could not be parsed — it may be a scanned (image-only) PDF, or "
            "the file may be corrupt." + page_note
        ),
        hint="Upload a text-based PDF, or OCR the scanned file first and upload it again.",
        retryable=False,
    )


def _generic_view(exc: BaseException) -> ErrorView:
    """The fallback banner: the failure's own name and message, never swallowed."""

    detail = str(exc).strip()
    message = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return ErrorView(
        title="Unexpected error",
        message=message,
        hint=(
            "Retry the action; if it fails again, the app logs carry the details for this " "turn."
        ),
        retryable=True,
    )


def _gateway_error_name(exc: BaseException) -> str | None:
    """The gateway error's class name, matched without importing the gateway module.

    The method-resolution order is walked rather than reading ``type(exc)`` so that a
    subclass declared anywhere (a test double, a future wrapper around a gateway error)
    still classifies as the provider-mapped base ADR-007 names.

    Returns:
        The gateway class name (``"RateLimited"``), or ``None`` when no class in the MRO
        was declared by :data:`_GATEWAY_MODULE`.

    """

    for cls in type(exc).__mro__:
        if cls.__module__ == _GATEWAY_MODULE:
            return cls.__name__
    return None


def _is_graph_error(exc: GraphError | BaseException) -> bool:
    """Whether ``exc`` is one of the graph's typed errors (ADR-007's graph-side rows).

    The graph module declares ``GraphError`` as the base of the whole spine —
    ``GraphNodeError`` and ``GraphStateError`` both inherit from it, and any subclass a
    later task adds will too — so one check on that base covers the family. The
    module-plus-name test mirrors :func:`_gateway_error_name`: it reads the class's own
    attributes and never imports ``src.chains.graph``, which is what keeps this module
    leaf-level (ADR-009). The parameter is annotated with the base so the signature says
    what the guard matches; ``BaseException`` stays in the union because the classifier
    hands over whatever ended the turn, known or not.

    Returns:
        ``True`` for :class:`~src.chains.graph.GraphError` and its descendants.

    """

    for cls in type(exc).__mro__:
        if cls.__name__ == _GRAPH_BASE and cls.__module__ == _GRAPH_MODULE:
            return True
    return False


def _is_local_embedder_error(exc: LocalEmbeddingError | BaseException) -> bool:
    """Whether ``exc`` is a :class:`~src.core.local_embedder.LocalEmbeddingError`.

    Matched by class name rather than with ``isinstance`` for the reason the graph family
    above is: the embedder module imports ``fastembed`` and ``os``, and this component
    module must stay importable without either.

    Returns:
        ``True`` for a local-embedder failure, wherever in the pipeline it was raised.

    """

    for cls in type(exc).__mro__:
        if cls.__name__ == "LocalEmbeddingError" and cls.__module__ == "src.core.local_embedder":
            return True
    return False


def _page_number(exc: BaseException) -> int | None:
    """The page a parse failure names, from an attribute or from its own message."""

    for name in _PAGE_ATTRS:
        value = getattr(exc, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    match = _PAGE_IN_MESSAGE.search(str(exc))
    return int(match.group(1)) if match is not None else None


def _escape(text: object) -> str:
    """Escape one value for the alert body; the theme helpers escape their own text."""

    return html.escape(str(text), quote=True)

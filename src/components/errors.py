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

if TYPE_CHECKING:
    from src.chains.graph import GraphError
    from src.core.local_embedder import LocalEmbeddingError

__all__ = [
    "ErrorView",
    "classify_error",
    "render_error",
]

_GATEWAY_MODULE = "src.utils.llm"

_GRAPH_MODULE = "src.chains.graph"
_GRAPH_BASE = "GraphError"

_RETRY_KEY = "isip-error-retry"
_RETRY_LABEL = "Retry"

_PAGE_ATTRS = ("page", "page_number")
_PAGE_IN_MESSAGE = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ErrorView:

    title: str
    message: str
    hint: str | None
    retryable: bool


# --- the views (ADR-007's taxonomy, worded for a user) ---------------------------


_AUTH_VIEW = ErrorView(
    title="Model access denied",
    message=(
        "The model provider refused the request: the API key is missing, wrong, or no longer valid."
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
    hint=("Retry the question — the gateway asks the model once more with a stricter instruction."),
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

_LOCAL_EMBEDDER_VIEW = _RETRIEVAL_VIEW

_GATEWAY_VIEWS: dict[str, ErrorView] = {
    "Auth": _AUTH_VIEW,
    "ContextLength": _CONTEXT_LENGTH_VIEW,
    "RateLimited": _RATE_LIMITED_VIEW,
    "SchemaViolationError": _SCHEMA_VIOLATION_VIEW,
    "Timeout": _TIMEOUT_VIEW,
}


# --- the public API --------------------------------------------------------------


def classify_error(exc: BaseException) -> ErrorView:

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

    view = classify_error(exc)
    label(view.title)
    st.error(_escape(view.message))
    if view.hint:
        caption_note(view.hint)
    if on_retry:
        return bool(st.button(_RETRY_LABEL, key=_RETRY_KEY))
    return False


def _parse_view(exc: ParseError) -> ErrorView:
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

    detail = str(exc).strip()
    message = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return ErrorView(
        title="Unexpected error",
        message=message,
        hint=("Retry the action; if it fails again, the app logs carry the details for this turn."),
        retryable=True,
    )


def _gateway_error_name(exc: BaseException) -> str | None:

    for cls in type(exc).__mro__:
        if cls.__module__ == _GATEWAY_MODULE:
            return cls.__name__
    return None


def _is_graph_error(exc: GraphError | BaseException) -> bool:

    for cls in type(exc).__mro__:
        if cls.__name__ == _GRAPH_BASE and cls.__module__ == _GRAPH_MODULE:
            return True
    return False


def _is_local_embedder_error(exc: LocalEmbeddingError | BaseException) -> bool:

    for cls in type(exc).__mro__:
        if cls.__name__ == "LocalEmbeddingError" and cls.__module__ == "src.core.local_embedder":
            return True
    return False


def _page_number(exc: BaseException) -> int | None:
    for name in _PAGE_ATTRS:
        value = getattr(exc, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    match = _PAGE_IN_MESSAGE.search(str(exc))
    return int(match.group(1)) if match is not None else None


def _escape(text: object) -> str:

    return html.escape(str(text), quote=True)

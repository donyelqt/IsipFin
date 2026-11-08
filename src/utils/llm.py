"""The one LLM gateway: OpenAI-compatible chat, typed outputs, bounded retries (ADR-005).

Every node reaches the model through this module and no other path: ``build_llm(params)``
returns a client and the node calls ``await client.chat(messages, params, schema)``.
Configuration, retries, token accounting and latency measurement live here once, so a node
is testable with a stub and a provider change is a ``base_url`` env change — OpenAI, Azure
OpenAI, Ollama (``http://localhost:11434/v1``), vLLM or Groq
(``https://api.groq.com/openai/v1``) all work without a code edit.

**Credential resolution.** Four settings decide where a request goes, and each has one
primary name: ``LLM_PROVIDER`` (which endpoint family — ``openai`` is the default, plus
``groq``, ``nvidia``, ``ollama``, ``vllm``, ``azure``), ``LLM_API_KEY`` (the key),
``LLM_BASE_URL`` (an endpoint override, for a self-hosted or Azure deployment this module
does not know a host for) and ``LLM_MODEL`` (the model id). The chain is:

* provider: ``LLM_PROVIDER`` → ``groq`` when ``GROQ_API_KEY`` supplied the key → ``openai``;
* key: ``LLM_API_KEY`` → ``OPENAI_API_KEY`` → ``GROQ_API_KEY``;
* base URL: ``LLM_BASE_URL`` → the provider's own endpoint when ``LLM_PROVIDER`` named one
  → ``OPENAI_BASE_URL`` → Groq's URL (when the key came from ``GROQ_API_KEY``) → the
  provider's default endpoint. An explicitly named provider outranks the ``OPENAI_BASE_URL``
  alias because naming the vendor is a deliberate act, and an ambient export is not;
* model: ``params.model`` → ``LLM_MODEL`` → ``OPENAI_MODEL`` → the provider's default model.

The ``OPENAI_*`` spellings are aliases at lower precedence, because that is what hosts, CI
jobs and tutorials export and dropping them would break working setups; the names describe
the wire protocol, not a vendor. ``GROQ_API_KEY`` stays as a key alias and implies
``LLM_PROVIDER=groq`` — the convenience that one name already promised. The ``LLM_*``
overrides outrank both because ``.env`` is loaded with ``override=False``: a real
environment variable beats the file, so on a host that already exports ``OPENAI_API_KEY``
and ``OPENAI_BASE_URL`` a credential written into ``.env`` under that name is
silently ignored and the ambient one is sent instead — which is the failure
:func:`resolved_config` exists to make visible, with the key redacted.

**Retry policy (ADR-005).** HTTP 429 and HTTP 5xx are retried with capped exponential
backoff plus jitter — ``min(1s * 2**(attempt-1), 30s)`` with half-fixed/half-random
jitter — for at most ``MAX_ATTEMPTS = 3`` attempts (one call plus two retries); a third
consecutive 429 raises :class:`LLMError.RateLimited`. When a 429 instead names how long
to wait — a ``Retry-After`` response header, or the seconds written into the body the way
Groq states a tokens-per-minute window — that hint is used verbatim rather than the
computed backoff, capped by :data:`RETRY_AFTER_CAP_S`: the rolling window the provider
measured is what clears, so a retry on our own 1s/2s schedule lands back inside it. Every
other retryable failure keeps the jittered backoff unchanged. Auth (401/403),
context-length and timeout failures are **not** retried: a second identical attempt
cannot succeed, so they surface immediately as their own typed errors, and each node's
failure table in ADR-005 decides what happens next. The SDK's transport-level retries are
disabled (``max_retries=0``) so this policy is the only retry authority — otherwise a 429
would be attempted six times before the node ever saw it. Sleeps and jitter are injectable
(constructor arguments, with module-level defaults that monkeypatching can replace), so a
retry test never spends three real seconds.

**Owned client lifetime (ADR-005).** The gateway owns the provider client it builds. That
ownership is discharged on the gateway's own terms — :meth:`ChatLLM.aclose` — instead of
being left to the interpreter's garbage collector, which would fire the SDK's ``__del__``
whenever the collector happened to get to the object. The SDK's finalizer answers
``asyncio.get_running_loop().create_task(self.aclose())``, so a client collected **after**
its loop closed schedules a coroutine that can never run and asyncio reports it as
``Task exception was never retrieved`` — the scary traceback a fully successful keyed run
prints to stderr. Closing deterministically while the loop is still alive closes the real
connection pool, so nothing is left for a finalizer to complain about. No warning is
filtered and no global handler is installed: the requirement is that the pool is closed,
not that the symptom is silenced.

**Offline-mock contract.** ``MOCK_LLM=1`` makes :func:`build_llm` return
:class:`src.utils.mock_llm.MockLLM`: deterministic, keyed by a prompt fingerprint, and
performing no network operation of any kind. That branch imports neither ``openai`` nor any
HTTP client, so the demo and the entire graph suite run with no API key, no quota and no
network — a missing key can never produce a broken ``docker compose up`` demo (ADR-005).

Typed outputs: when a caller passes a ``schema``, the gateway requests
``response_format={"type": "json_schema", ...}``, parses the returned JSON object and
validates it against the schema. A violation retries exactly once with a stricter system
nudge appended, then raises :class:`SchemaViolationError`. With no schema the raw text is
returned as-is and ``parsed`` is ``None`` (docs/SPEC.md section 9).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "GROQ_BASE_URL",
    "LLM_PROVIDER_ENV",
    "MAX_ATTEMPTS",
    "MODEL_ENV",
    "PROVIDER_DEFAULTS",
    "REDACTED_KEY",
    "REQUEST_TIMEOUT_S",
    "RETRY_AFTER_CAP_S",
    "Auth",
    "ChatClient",
    "ChatLLM",
    "ChatResult",
    "ContextLength",
    "LLMError",
    "LLMParams",
    "RateLimited",
    "ResolvedConfig",
    "SchemaViolationError",
    "Timeout",
    "backoff_delay",
    "build_llm",
    "default_jitter",
    "default_sleep",
    "resolved_config",
    "resolved_provider",
    "retry_after_seconds",
    "validate_json_schema",
]

# --- configuration (read from the environment by build_llm, once per client) -------
# These names are duplicated from scripts/build_index.py on purpose: utils must not import
# from scripts (which is not a package), and the two modules must agree on the env surface.
MOCK_ENV = "MOCK_LLM"

# --- one name per setting, and the precedence that decides them ----------------------
#
# THE PRECEDENCE CHAIN, IN ONE BLOCK — read this before editing anything below.
#
#   provider: LLM_PROVIDER -> groq, implied by GROQ_API_KEY -> openai (the default)
#   key:      LLM_API_KEY  -> OPENAI_API_KEY -> GROQ_API_KEY
#   base URL: LLM_BASE_URL -> the provider's endpoint when LLM_PROVIDER named one ->
#             OPENAI_BASE_URL -> Groq's URL (only when the key came from GROQ_API_KEY) ->
#             the provider's own default endpoint
#   model:    params.model -> LLM_MODEL -> OPENAI_MODEL -> the provider's default model
#
# Each setting has exactly one PRIMARY name — the LLM_* one. The OPENAI_* spellings are
# aliases that stay accepted at LOWER precedence, and GROQ_API_KEY is a key alias that also
# implies the provider; neither ever wins over its LLM_* counterpart.
#
# WHY THE LLM_* NAMES EXIST: src/utils/settings.py loads .env with override=False, so a real
# environment variable always beats the file. This host exports OPENAI_API_KEY and
# OPENAI_BASE_URL, so a credential written into .env under those names is silently ignored
# and the ambient pair is used instead — a Groq key sent to an NVIDIA endpoint, for
# instance. The LLM_* names cover the settings an ambient environment demonstrably hijacks.
#
# WHY THE ALIASES STAY: hosts, CI jobs and tutorials export OPENAI_API_KEY /
# OPENAI_BASE_URL / OPENAI_MODEL by default, and those names describe the OpenAI-compatible
# WIRE PROTOCOL, not a vendor — one spelling already addresses OpenAI, Groq, NVIDIA, Ollama
# and vLLM. LLM_PROVIDER exists to say the vendor out loud when the protocol cannot infer
# it (an unknown endpoint, or a default model to pick).
#
# WHY GROQ_API_KEY STAYS: that is the name a user reaches for, and it implies the Groq
# provider, so setting that one variable alone is a working Groq configuration.
#
# MOCK_LLM is checked first in build_llm and short-circuits every name below: nothing in
# this block is read on the offline path.
LLM_PROVIDER_ENV = "LLM_PROVIDER"
LLM_KEY_ENV = "LLM_API_KEY"
LLM_BASE_URL_ENV = "LLM_BASE_URL"
LLM_MODEL_ENV = "LLM_MODEL"
API_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"
MODEL_ENV = "OPENAI_MODEL"
GROQ_KEY_ENV = "GROQ_API_KEY"

#: Groq's OpenAI-compatible chat endpoint: the fallback that makes ``GROQ_API_KEY`` alone work.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
#: The provider a bare configuration means, and the one an unrecognised ``LLM_PROVIDER``
#: falls back to rather than failing: a typo in a vendor name is a configuration slip, and
#: the request still has to go to the OpenAI-compatible protocol the user asked for.
DEFAULT_PROVIDER = "openai"

#: The providers this module can aim at without being told an endpoint, and the default
#: model each one stands for. ``azure`` deliberately has no endpoint: an Azure deployment
#: lives at a per-deployment URL, so it selects no default and expects ``LLM_BASE_URL`` —
#: the override exists for exactly that case. ``None`` as a model means "no opinion": the
#: deployment names its own, and a wrong guess here would fail the request.
PROVIDER_DEFAULTS: dict[str, tuple[str | None, str | None]] = {
    "openai": (DEFAULT_BASE_URL, DEFAULT_MODEL),
    "groq": (GROQ_BASE_URL, "openai/gpt-oss-120b"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "meta/llama-3.1-8b-instruct"),
    "ollama": ("http://localhost:11434/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
    "azure": (None, None),
}

#: Fixed-length stand-in for the credential in diagnostics: no prefix, no length and no
#: character of the key, so a resolved configuration is always safe to log or assert on.
REDACTED_KEY = "***"

#: Hosts reported as ``local`` by :func:`resolved_provider` (Ollama, vLLM, in-box gateways).
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})

#: Maximum attempts per request: one call plus two retries (ADR-005).
MAX_ATTEMPTS = 3
#: First backoff delay in seconds; each retry doubles it (ADR-005 base 1s).
BACKOFF_BASE_S = 1.0
#: Ceiling for the exponential growth, so a long outage cannot stall a turn for minutes.
BACKOFF_CAP_S = 30.0
#: Ceiling for a provider's own retry hint on a 429, and the reason it is not unbounded:
#: a hint is a scheduling promise, not a licence to hold a turn. It matches
#: :data:`BACKOFF_CAP_S` because the gateway's whole latency promise is already "one chat
#: call plus bounded waits" — a hundred-second hint would outlast the turn the user is
#: waiting on. TPM/RPM windows are short and *rolling*, so half a minute of waiting has
#: already moved the request into a fresh window most of the time, and the attempt budget
#: (two retries) keeps the worst case at ``2 * 30s = 60s`` of waiting, comfortably inside
#: :data:`REQUEST_TIMEOUT_S`.
RETRY_AFTER_CAP_S = 30.0
#: Per-request timeout handed to the OpenAI client (the gateway owns one client per call).
REQUEST_TIMEOUT_S = 120.0

_SCHEMA_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DEFAULT_SCHEMA_NAME = "structured_response"

# Appended once when a schema request came back unusable (ADR-005: "retry once with a
# stricter system nudge"). It is a *user* message, not a second system message: several
# OpenAI-compatible servers reject conversations with more than one system turn.
_STRICT_NUDGE = (
    "Your previous reply was not a single valid JSON object matching the requested schema. "
    "Reply with valid JSON matching the schema exactly — no markdown fences, no prose before "
    "or after the object, every required field present."
)


# --- the frozen contracts (docs/SPEC.md section 9) --------------------------------


@dataclass(frozen=True)
class LLMParams:
    """Per-message sampling parameters, flowing from the sidebar into ``GraphState.params``.

    Every field is optional: ``None`` means "provider default", so a node that does not set
    temperature sends nothing and the endpoint's own default applies. ``model`` overrides
    the client's configured model for a single call (the sidebar's model picker). Retrieval
    ``k`` is deliberately absent — it belongs to the retriever call, not to sampling.
    """

    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class ChatResult:
    """One completed chat call with its accounting (docs/SPEC.md section 9).

    ``parsed`` is the validated JSON object when a schema was requested, and ``None``
    otherwise. Token counts come from the response's usage field (``0`` when an endpoint
    omits it, which local servers often do), and ``latency_ms`` covers the whole ``chat``
    call including retries — it is what the node trace and the debug panel report.
    """

    text: str
    parsed: dict[str, Any] | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


@runtime_checkable
class ChatClient(Protocol):
    """The gateway surface every node depends on: one ``chat`` coroutine.

    Both :class:`ChatLLM` (real provider) and :class:`MockLLM` (``MOCK_LLM=1``) satisfy
    it, which is what makes a node testable with either without an import-time branch.
    """

    async def chat(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        schema: dict[str, Any] | None = None,
    ) -> ChatResult: ...


# --- typed errors (ADR-007 taxonomy, ADR-005 mapping) -----------------------------
#
# The four provider-mapped subclasses are defined at module level and then bound onto
# LLMError as ``LLMError.RateLimited`` etc. A nested class cannot inherit from the class it
# is nested in — the name is not bound yet inside its own body — so the frozen contract's
# ``LLMError.X`` spelling is an alias of the real class rather than the definition site.


class LLMError(Exception):
    """Base class for every gateway failure.

    ``status`` is the HTTP status when the provider gave one (``None`` for transport and
    decoding failures) and ``attempts`` is how many attempts were spent. Callers switch on
    the subclass — never on ``status`` — because that is what ADR-005's per-task failure
    tables are keyed on.
    """

    if False:  # pragma: no cover - declarations only; the classes are bound below
        RateLimited: type[LLMError]
        ContextLength: type[LLMError]
        Auth: type[LLMError]
        Timeout: type[LLMError]

    #: Retried by the gateway only for 429 and 5xx (ADR-005); see ``retryable`` below.
    retryable: bool = False

    #: The provider's response headers when the failure carried them — ``Retry-After``,
    #: the hint :func:`retry_after_seconds` reads, lives here — and the response object
    #: they came off. Both are ``None`` for a transport or decoding failure, which has no
    #: HTTP response at all; the mapper fills them in from the SDK exception it wrapped.
    headers: dict[str, str] | None
    response: Any

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        attempts: int | None = None,
        headers: dict[str, str] | None = None,
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.attempts = attempts
        self.headers = headers
        self.response = response


class RateLimited(LLMError):
    """HTTP 429 from the provider: quota, rate limit or a shared-key collision."""

    retryable = True


class ContextLength(LLMError):
    """The prompt is too long for the model's context window.

    Never retried — only a smaller ``k``, a narrower question or a bigger-context model
    fixes it (ADR-005's context-budget packing is the upstream prevention).
    """


class Auth(LLMError):
    """HTTP 401/403: missing, wrong or revoked chat credential.

    The credential is whichever of ``LLM_API_KEY``, ``OPENAI_API_KEY`` or ``GROQ_API_KEY``
    resolved first; the message names the variables a user can set and never the key
    itself, so it is safe to render (ADR-007: "never leaks the key").
    """


class Timeout(LLMError):
    """The provider did not answer within the timeout.

    Surfaces immediately rather than being retried here: ADR-005's retry clause covers
    429/5xx only, and a doubled timeout budget is a configuration decision, not a retry.
    """


class SchemaViolationError(Exception):
    """A schema-requested response was not a JSON object matching the schema.

    Standalone by contract (docs/SPEC.md section 9): the JSON nodes handle it beside
    ``LLMError``, because a malformed answer is a *content* failure, not a transport one,
    and each node's ADR-005 failure table prescribes its own recovery (pessimistic default,
    ``out_of_scope`` classification, trimmed claims).
    """


LLMError.RateLimited = RateLimited
LLMError.ContextLength = ContextLength
LLMError.Auth = Auth
LLMError.Timeout = Timeout


# --- JSON-schema validation ------------------------------------------------------
# A minimal validator over the subset of JSON Schema the project's hand-authored schemas
# use (type, properties, required, items, enum, additionalProperties, numeric/string/length
# bounds). It exists so the gateway can enforce the shape itself instead of trusting the
# provider — which is what keeps local servers that ignore response_format supported, and
# what makes MOCK_LLM=1 provably schema-conformant. `jsonschema` is deliberately not a
# dependency: the validator is ~40 lines and the schema language it needs is frozen.

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, int | float) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def _json_type(value: Any) -> str:
    """The JSON type name of ``value`` (used in validation messages)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def validate_json_schema(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Validate ``payload`` against ``schema``; return one message per violation.

    An empty list means the payload matches. Messages carry the JSON path
    (``$.claims[0].verdict``) so a retry can name the offending field.
    """
    errors: list[str] = []
    _validate(payload, schema, "$", errors)
    return errors


def _validate(payload: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    if isinstance(expected, str):
        expected = [expected]
    if (
        isinstance(expected, list)
        and expected
        and not any(_TYPE_CHECKS.get(name, lambda _: True)(payload) for name in expected)
    ):
        errors.append(f"{path}: expected {' | '.join(expected)}, got {_json_type(payload)}")
        return  # deeper checks would only report the same mismatch in worse terms
    allowed = schema.get("enum")
    if isinstance(allowed, list) and payload not in allowed:
        errors.append(f"{path}: {payload!r} is not one of {allowed}")
    if isinstance(payload, dict):
        _validate_object(payload, schema, path, errors)
    elif isinstance(payload, list):
        _validate_array(payload, schema, path, errors)
    elif isinstance(payload, str):
        _validate_string(payload, schema, path, errors)
    elif isinstance(payload, int | float) and not isinstance(payload, bool):
        _validate_number(payload, schema, path, errors)


def _validate_object(
    payload: dict[str, Any], schema: dict[str, Any], path: str, errors: list[str]
) -> None:
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get("required")
    if isinstance(required, list):
        for key in required:
            if key not in payload:
                errors.append(f"{path}: missing required property '{key}'")
    extras = schema.get("additionalProperties", True)
    for key, value in payload.items():
        child = f"{path}.{key}"
        if key in properties:
            _validate(value, properties[key], child, errors)
        elif extras is False:
            errors.append(f"{child}: unexpected property")
        elif isinstance(extras, dict):
            _validate(value, extras, child, errors)


def _validate_array(
    payload: list[Any], schema: dict[str, Any], path: str, errors: list[str]
) -> None:
    items = schema.get("items")
    if isinstance(items, dict):
        for index, value in enumerate(payload):
            _validate(value, items, f"{path}[{index}]", errors)
    if isinstance(minimum := schema.get("minItems"), int) and len(payload) < minimum:
        errors.append(f"{path}: expected at least {minimum} items, got {len(payload)}")
    if isinstance(maximum := schema.get("maxItems"), int) and len(payload) > maximum:
        errors.append(f"{path}: expected at most {maximum} items, got {len(payload)}")


def _validate_string(payload: str, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Length bounds."""
    if isinstance(minimum := schema.get("minLength"), int) and len(payload) < minimum:
        errors.append(f"{path}: shorter than {minimum} characters")
    if isinstance(maximum := schema.get("maxLength"), int) and len(payload) > maximum:
        errors.append(f"{path}: longer than {maximum} characters")


def _validate_number(payload: float, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Numeric range bounds."""
    if isinstance(minimum := schema.get("minimum"), int | float) and payload < minimum:
        errors.append(f"{path}: below the minimum {minimum}")
    if isinstance(maximum := schema.get("maximum"), int | float) and payload > maximum:
        errors.append(f"{path}: above the maximum {maximum}")


def _decode_json_object(text: str) -> dict[str, Any] | None:
    """Decode ``text`` as one JSON object, or ``None`` when it is not one.

    A single surrounding ```json fence is tolerated and stripped: Ollama and vLLM honor
    ``messages`` but frequently ignore ``response_format``, and rejecting a perfectly
    good payload for a fence would burn the stricter retry on a formatting artifact.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        without_fence = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", without_fence).strip()
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_structured(text: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Decode and validate a structured response.

    Returns the payload and an empty reason on success, or ``(None, reason)`` where the
    reason is what the stricter retry and, failing that, the :class:`SchemaViolationError`
    report.
    """
    payload = _decode_json_object(text)
    if payload is None:
        return None, f"response is not a JSON object: {text[:120]!r}"
    errors = validate_json_schema(payload, schema)
    if errors:
        return None, "; ".join(errors[:3])
    return payload, ""


# --- provider error mapping ------------------------------------------------------


def _provider_error_classes(*names: str) -> tuple[type[BaseException], ...]:
    """The named ``openai`` exception classes, resolved lazily.

    The import lives here rather than at module scope so ``MOCK_LLM=1`` never imports the
    OpenAI package at all: the offline path must work on a machine where the paid
    dependency was never installed (ADR-005's "no key at all" demo, and CI without it).
    """
    try:
        import openai
    except ImportError:  # pragma: no cover - pinned in requirements.txt
        return ()
    return tuple(
        candidate for name in names if isinstance(candidate := getattr(openai, name, None), type)
    )


def _status_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status


def _response_headers(exc: BaseException) -> dict[str, str] | None:
    """The response headers ``exc`` carries, as a plain ``str`` map, or ``None``.

    Both spellings are read — the SDK's own exceptions set ``.headers``, and ``httpx``
    responses expose ``.headers`` on the object the error carries as ``.response``. The
    mapping is copied into a plain ``dict`` because a case-insensitive header mapping is
    not a ``dict`` and the declared type is one; anything else (a stub with no response at
    all, or a header value that is not text) yields ``None``, which leaves the hint to be
    read from the message body.
    """
    response = getattr(exc, "response", None)
    headers = getattr(exc, "headers", None)
    if headers is None:
        headers = getattr(response, "headers", None)
    if headers is None:
        return None
    if not isinstance(headers, dict):
        try:
            headers = dict(headers)
        except (TypeError, ValueError):
            return None
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items()):
        return None
    return headers


def _response_kwargs(exc: BaseException) -> dict[str, Any]:
    """The provider's ``headers`` and ``response``, read off ``exc`` once.

    Every mapped error carries what the provider failure did, so the retry-hint path and
    any later diagnostic read the same object rather than reaching back through the
    ``__cause__`` chain. The status is *not* here: each taxonomy row states its own.
    """
    return {
        "headers": _response_headers(exc),
        "response": getattr(exc, "response", None),
    }


def _map_provider_error(exc: BaseException) -> LLMError:
    """Translate a provider/transport exception into the typed taxonomy (ADR-005).

    Status codes lead and exception classes are the fallback, so a stub that raises any
    object carrying ``status_code`` (or the real SDK's exceptions) maps identically. The
    provider's ``headers``/``response`` travel with the mapped error, which is where the
    429 retry hint is read from.
    """
    status = _status_of(exc)
    detail = str(exc)
    lowered = detail.lower()
    provider = _response_kwargs(exc)
    timeout_classes = _provider_error_classes("APITimeoutError")
    auth_classes = _provider_error_classes("AuthenticationError", "PermissionDeniedError")
    rate_limit_classes = _provider_error_classes("RateLimitError")
    bad_request_classes = _provider_error_classes("BadRequestError")

    if isinstance(exc, (*timeout_classes, TimeoutError)):
        return LLMError.Timeout(
            f"model request timed out ({detail or type(exc).__name__})", status=status
        )
    if status in (401, 403) or isinstance(exc, auth_classes):
        source = resolved_config().key_source
        where = (
            f"the credential in use came from {source}"
            if source is not None
            else "no credential resolved"
        )
        hint = (
            f"the chat key ({LLM_KEY_ENV}, {API_KEY_ENV} or {GROQ_KEY_ENV}) is missing, wrong "
            f"or revoked: {where}; or run with {MOCK_ENV}=1 for the offline demo path."
        )
        return LLMError.Auth(f"model authentication failed: {hint}", status=status, **provider)
    if status == 429 or isinstance(exc, rate_limit_classes):
        return LLMError.RateLimited(
            f"model rate limited the request ({detail or type(exc).__name__})",
            status=429,
            **provider,
        )
    if status in (400, 413) or isinstance(exc, bad_request_classes):
        context_markers = (
            "context length",
            "context window",
            "maximum context",
            "too long",
            "too many tokens",
            "reduce the length",
        )
        if any(marker in lowered for marker in context_markers):
            return LLMError.ContextLength(
                f"the prompt exceeds the model's context window: {detail}", status=status
            )
        return LLMError(
            f"the model rejected the request as invalid: {detail}", status=status, **provider
        )
    if status is not None and 500 <= status < 600:
        # Retried by ``_is_retryable`` like a 429, but not reported as a rate limit: a
        # 500/502/503 is the endpoint being unavailable, which the node's failure table
        # treats differently ("try again" beats "you are being throttled").
        return LLMError(
            f"the model endpoint returned a server error: {detail}", status=status, **provider
        )
    return LLMError(
        f"model request failed ({type(exc).__name__}: {detail})", status=status, **provider
    )


# --- the OpenAI-compatible client ------------------------------------------------


async def default_sleep(seconds: float) -> None:
    """The real backoff sleep; replaceable so retry tests never wait for wall-clock time."""
    await asyncio.sleep(seconds)


def backoff_delay(attempt: int, base: float = BACKOFF_BASE_S, cap: float = BACKOFF_CAP_S) -> float:
    """The deterministic half of attempt ``attempt``'s delay: ``base * 2**(attempt-1)``, capped.

    The exponent is zero-based (the first failure waits one ``base``), which is why a
    three-attempt run sleeps roughly 1s then 2s, never 3s.
    """
    delay: float = base * float(2 ** max(0, attempt - 1))
    return min(delay, cap)


def default_jitter(delay: float) -> float:
    """Equal jitter: half the computed delay, plus a random half — thundering-herd safe.

    Randomness here widens the *wait*, never the outcome, so injecting a deterministic
    jitter function makes retries fully reproducible in tests.
    """
    return delay / 2.0 + random.random() * (delay / 2.0)


# --- the provider's own retry hint -------------------------------------------------
#
# A 429 that names how long to wait is not a generic failure: Groq's answers an exact
# window ("Please try again in 4.0955s" for a tokens-per-minute limit, in seconds with a
# fraction; OpenAI/Azure use `Retry-After: 3`), and the rolling window it names is what
# actually unblocks the request. Retrying on our own 1s/2s schedule instead means the
# retry lands inside the very window the provider just measured, which is how one 429
# becomes three consecutive ones and a degraded turn.
#
# Two sources, checked in this order — a header is the standard, the body is what this
# project actually observes:
#   1. the HTTP response the raised exception carries (``response.headers``, or an
#      httpx-style ``response`` on SDK exceptions);
#   2. the seconds written into the message body (Groq's ``... try again in 4.0955s``).
# Anything unparseable, zero or negative is *no hint at all* and leaves the exponential
# backoff untouched.


def _retry_after_header(exc: BaseException) -> float | None:
    """The seconds a ``Retry-After`` response header on ``exc`` names, if any.

    The SDK's rate-limit errors carry the ``httpx.Response`` they came from, so the header
    is read off that first; a stub raising anything else simply has no response and falls
    through to the message body. It may also be an HTTP-date rather than a delay, which is
    not a duration and is deliberately ignored — parsing one would need a clock.
    """
    candidates = (exc, getattr(exc, "response", None))
    for holder in candidates:
        if holder is None:
            continue
        headers = getattr(holder, "headers", None)
        if not isinstance(headers, dict):
            continue
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if isinstance(raw, str):
            return _retry_after_seconds(raw)
    return None


def _retry_after_message(exc: BaseException) -> float | None:
    """The seconds ``exc``'s message names, as Groq writes it: ``try again in 4.0955s``.

    The pattern is anchored on "in" plus a number plus an "s" so it also matches the
    minutes form some providers use (``try again in 2m``) — converted, not ignored, since
    the number and the unit are both there.
    """
    match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s|m|h)?", str(exc), flags=re.IGNORECASE
    )
    if match is None:
        return None
    return _retry_after_seconds(match.group(1), match.group(2))


def _retry_after_seconds(raw: str, unit: str | None = None) -> float | None:
    """``raw`` as a positive number of seconds, or ``None`` when it is not one.

    A unitless value is seconds (that is what both the HTTP header and Groq mean); a named
    unit is converted. Anything that is not a plain positive number — a date, a mangled
    string — yields ``None`` so the caller keeps the backoff it can reason about.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value <= 0.0:  # NaN, zero and negatives are not a wait
        return None
    multiplier = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}.get((unit or "s").lower(), 1.0)
    return value * multiplier


def retry_after_seconds(exc: BaseException) -> float | None:
    """How long the provider asked us to wait before retrying, in seconds, or ``None``.

    Header first, then the message body: a header is the interoperable form, and Groq's
    body hint is the one this project observes on a tokens-per-minute limit. The value is
    returned as the provider stated it; clamping to :data:`RETRY_AFTER_CAP_S` is the
    caller's policy decision, not a parsing one.
    """
    return _retry_after_header(exc) or _retry_after_message(exc)


# --- closing the client the gateway owns ------------------------------------------
#
# The gateway wraps an ``openai.AsyncOpenAI``, whose async close is spelled ``close()``;
# the HTTP client one level down spells it ``aclose()``. Both spellings are accepted, in
# that order, rather than hard-coding one name — the rename the SDK made between the two
# objects is the kind that happens again. A stub with neither spelling is simply never
# closed: a test stand-in that opened no socket has nothing to release, and that is the
# state the existing stub machinery has always been in.
_CLOSE_SPELLINGS = ("close", "aclose")


def _resolve_close(client: Any) -> Callable[[], Awaitable[None]] | None:
    for spelling in _CLOSE_SPELLINGS:
        candidate = getattr(client, spelling, None)
        if asyncio.iscoroutinefunction(candidate):
            return candidate
    return None


class ChatLLM:
    """One OpenAI-compatible chat client with the gateway's retries and accounting.

    ``client`` is anything exposing ``chat.completions.create(**kwargs) -> awaitable`` and
    returning a completion with ``choices[0].message.content`` (plus the optional
    ``choices[0].message.refusal``, ``usage.prompt_tokens``/``usage.completion_tokens`` and
    ``model``). In production it is ``openai.AsyncOpenAI``; in tests it is a stub with that
    surface — the seam is deliberately the SDK object itself so no second indirection has
    to stay in sync with a provider upgrade.

    Retries (429/5xx, ``max_attempts = 3``) and the schema-retry are both bounded; the
    class holds no per-client state beyond its configuration, so one client per
    ``build_llm`` call is trivially reproducible. A 429 that names its own wait (a
    ``Retry-After`` header, or Groq's body hint) sleeps that long instead of the backoff,
    up to ``retry_after_cap_s``.

    The client this wraps is *owned*: :func:`build_llm` constructs it and hands it over
    with nobody else holding a reference, so the gateway is the one place that can close
    it. Callers discharge that by driving the chat through :meth:`aclose` — directly, or
    via ``async with build_llm(...) as client`` — so the SDK's connection pool is closed
    while the loop is still alive. Left to the collector instead, the SDK's ``__del__``
    schedules ``aclose`` onto whatever loop happens to be running, or none, and a client
    finalised after its loop closed prints ``Task exception was never retrieved`` for a
    run that in fact succeeded. :meth:`aclose` is idempotent and closing is not something
    a caller has to worry about having already happened.
    Because the client is closed by the call, one gateway serves exactly one chat turn:
    that is the shape every node already builds in, and it is what keeps one connection
    pool from outliving the ``asyncio.run`` loop that opened it.
    """

    def __init__(
        self,
        client: Any,
        model: str | None = None,
        *,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_base_s: float = BACKOFF_BASE_S,
        backoff_cap_s: float = BACKOFF_CAP_S,
        retry_after_cap_s: float = RETRY_AFTER_CAP_S,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._client = client
        self._model = model
        # Set by aclose once the underlying client has been closed; every later close is
        # a no-op, so a caller never has to know whether someone already closed this one.
        self._closed = False
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s
        self._retry_after_cap_s = retry_after_cap_s
        self._sleep = sleep
        self._jitter = jitter

    @property
    def client(self) -> Any:
        """The underlying transport object (the stubbing seam)."""
        return self._client

    @property
    def model(self) -> str | None:
        """The model requested when a call's ``params.model`` is ``None``."""
        return self._model

    @property
    def closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        """Close the underlying provider client, exactly once.

        This is what keeps the SDK's garbage-collector finalizer out of the picture. The
        SDK's ``__del__`` schedules ``aclose()`` on the *currently running* loop, so a
        client that is collected once that loop has closed schedules a coroutine which can
        never run, and asyncio reports the abandoned task as
        ``Task exception was never retrieved`` — a stack trace printed over a run that
        succeeded. Closing here, while the caller's loop is still alive, closes the real
        connection pool instead, leaving nothing for a finalizer to complain about.

        The close is idempotent, and a second (or concurrent) call is a no-op rather than
        a second close on the SDK object nor a raise: a caller that already closed, or two
        tasks closing one client, is not an error state.

        The relationship being closed is not a *transaction*: nothing is being offered to
        the provider, so a failure to release the pool must not replace the answer the
        caller already has, or the error they are already handling. An exception from the
        close is therefore logged and dropped, never propagated.
        """
        if self._closed:
            return
        # Claimed before the await so a concurrent closer sees this one as issued. The
        # SDK's own close is itself idempotent, so even that window is harmless.
        self._closed = True
        close = _resolve_close(self._client)
        if close is None:
            return
        try:
            await close()
        except Exception as exc:  # the pool failed to release; the result still stands
            logging.getLogger(__name__).warning(
                "closing the provider client failed: %s: %s", type(exc).__name__, exc
            )

    async def __aenter__(self) -> ChatLLM:
        """Enter the ``async with`` scope; the client it returns closes on exit."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Complete one chat call with retries, typed errors and accounting.

        With ``schema`` the response is requested as JSON, parsed and validated; a
        violation retries once with a stricter nudge appended and then raises
        :class:`SchemaViolationError`. Without it the raw text is returned and ``parsed``
        stays ``None``. ``latency_ms`` covers every attempt, retries included.

        The owned client is closed as this call finishes, success or failure alike. A node
        calls ``build_llm()`` inline and holds nothing after the call, so without this the
        SDK's finalizer would be the only thing that ever closes the pool — and it
        schedules that close on whatever loop happens to be running, which by then is not
        the one the pool belongs to, so asyncio reports
        ``Task exception was never retrieved``. Closing here is the same close, performed
        by the code that owns the object.
        """
        if not messages:
            # No request was built and no pool was used: nothing to close, and a rejected
            # call must not raise a second error from the close path on top of this one.
            raise LLMError("chat requires at least one message")
        request = self._build_request(messages, params, schema)
        started = time.perf_counter()

        try:
            result = self._to_result(await self._send_with_retries(request))
        except Exception as error:
            # Close before the raise, on the loop that used the pool, so the error a
            # caller is about to handle never leaves a live connection pool behind.
            await self.aclose()
            raise error
        if schema is None:
            return await self._finished(self._timed(result, started))

        parsed, reason = _parse_structured(result.text, schema)
        if parsed is None:
            # ADR-005: exactly one stricter retry, then the typed failure. The nudge rides
            # along as an extra user turn so single-system-message servers accept it.
            retry_messages = [*messages, {"role": "user", "content": _STRICT_NUDGE}]
            retry_request = self._build_request(retry_messages, params, schema)
            try:
                retry_result = self._to_result(await self._send_with_retries(retry_request))
            except Exception as error:
                await self.aclose()
                raise error
            parsed, reason = _parse_structured(retry_result.text, schema)
            if parsed is None:
                violation = SchemaViolationError(
                    f"model response did not match the requested schema after one stricter "
                    f"retry: {reason}"
                )
                await self.aclose()
                raise violation
            result = retry_result
        return await self._finished(replace(self._timed(result, started), parsed=parsed))

    async def _finished(self, result: ChatResult) -> ChatResult:
        """Close the owned client, then hand ``result`` back.

        Closing happens on the success path too, because a node that got its answer holds
        no reference to the client afterwards and the pool would otherwise wait for the
        collector. The await is ordered *before* the return so the close is already done
        when the node's ``asyncio.run`` closes its loop.
        """
        await self.aclose()
        return result

    # --- internals ---------------------------------------------------------------

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """The chat-completions kwargs: only the fields the caller actually set are sent.

        Omitting unset sampling parameters (rather than sending ``None``) is what keeps
        the same call valid against endpoints that reject explicit nulls.
        """
        request: dict[str, Any] = {
            "model": self._resolve_model(params),
            "messages": messages,
        }
        if params.temperature is not None:
            request["temperature"] = params.temperature
        if params.top_p is not None:
            request["top_p"] = params.top_p
        if params.max_tokens is not None:
            request["max_tokens"] = params.max_tokens
        if schema is not None:
            request["response_format"] = _response_format(schema)
        return request

    def _resolve_model(self, params: LLMParams) -> str:
        model = params.model or self._model
        if not model:
            raise LLMError(
                f"no model configured: set {LLM_MODEL_ENV} or {MODEL_ENV} or pass "
                "params.model (the sidebar's model picker writes it)"
            )
        return model

    def _resolve_sleep(self) -> Callable[[float], Awaitable[None]]:
        return self._sleep if self._sleep is not None else default_sleep

    def _retry_delay(
        self, attempt: int, exc: BaseException, error: LLMError, jitter: Callable[[float], float]
    ) -> float:
        """The wait before ``attempt``'s retry: the provider's 429 hint, else the backoff.

        A 429 that says how long to wait gets exactly that (capped), because the rolling
        window it names is the thing that actually clears. Everything else — 5xx, and 429s
        with no parseable hint — keeps the jittered exponential schedule unchanged, which
        is the whole point of having two policies: the backoff stays the default, and the
        hint only overrides it when the provider was specific.
        """
        if isinstance(error, RateLimited):
            hint = retry_after_seconds(exc)
            if hint is not None:
                return min(hint, self._retry_after_cap_s)
        return jitter(backoff_delay(attempt, self._backoff_base_s, self._backoff_cap_s))

    def _resolve_jitter(self) -> Callable[[float], float]:
        return self._jitter if self._jitter is not None else default_jitter

    async def _send_with_retries(self, request: dict[str, Any]) -> Any:
        """Send ``request``, retrying 429/5xx with backoff; raise the typed error otherwise."""
        sleeper = self._resolve_sleep()
        jitter = self._resolve_jitter()
        attempts = 0
        while True:
            attempts += 1
            try:
                return await self._client.chat.completions.create(**request)
            except Exception as exc:
                error = _map_provider_error(exc)
                error.attempts = attempts
                if not _is_retryable(error) or attempts >= self._max_attempts:
                    raise error from exc
                await sleeper(self._retry_delay(attempts, exc, error, jitter))

    def _to_result(self, completion: Any) -> ChatResult:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMError("the model returned no choices")
        message = choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise LLMError(f"the model refused the request: {refusal}")
        text = getattr(message, "content", None)
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        usage = getattr(completion, "usage", None)
        model = str(getattr(completion, "model", None) or self._model or DEFAULT_MODEL)
        return ChatResult(
            text=text,
            parsed=None,
            prompt_tokens=_usage_field(usage, "prompt_tokens"),
            completion_tokens=_usage_field(usage, "completion_tokens"),
            latency_ms=0.0,
            model=model,
        )

    def _timed(self, result: ChatResult, started: float) -> ChatResult:
        """``result`` with ``latency_ms`` measured from ``started`` (ms, three decimals)."""
        return replace(result, latency_ms=round((time.perf_counter() - started) * 1000.0, 3))


def _response_format(schema: dict[str, Any]) -> dict[str, Any]:
    """The ``json_schema`` response-format body for one hand-authored schema.

    ``strict`` is deliberately omitted: strict structured-outputs mode only accepts a
    constrained subset (every property required, no extra keywords), while these schemas
    are authored for readability. Enforcement happens on this side of the wire, in
    :func:`validate_json_schema`, which is also what keeps endpoints that ignore
    ``response_format`` inside the supported set.
    """
    name = schema.get("title")
    if not isinstance(name, str) or not _SCHEMA_NAME.fullmatch(name):
        name = _DEFAULT_SCHEMA_NAME
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def _usage_field(usage: Any, field: str) -> int:
    """One usage counter, ``0`` when the endpoint omits usage (common for local servers)."""
    value = getattr(usage, field, None)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _is_retryable(error: LLMError) -> bool:
    """True for the two classes ADR-005 retries: HTTP 429 and HTTP 5xx.

    The check reads ``status`` rather than the subclass because a 5xx arrives as a generic
    ``LLMError`` (mapped in :func:`_map_provider_error`) and is still worth one more
    attempt, while a 400, an auth failure or a timeout never is.
    """
    status = error.status
    return error.retryable or (status is not None and 500 <= status < 600)


# --- the one factory: credential resolution, then the client -----------------------
#
# Resolution is a pure function of the environment so that build_llm and the diagnostic
# helpers below cannot drift: both read the chain through _resolve_credentials and
# _select_model, and a test can assert on what was chosen without constructing a client.


@dataclass(frozen=True)
class _Credentials:
    """The raw credential set — the key here is real, so it never leaves this module."""

    api_key: str
    key_source: str | None
    base_url: str
    base_url_source: str | None
    provider: str


@dataclass(frozen=True)
class ResolvedConfig:
    """What the gateway resolved, with the credential replaced by :data:`REDACTED_KEY`.

    ``provider`` is derived from the endpoint's host rather than from which variable
    supplied the key, because the host is where the request actually goes — reporting
    ``groq`` for a Groq URL is what makes a key/endpoint mismatch visible. ``provider`` is
    ``None`` exactly when no key resolved at all, i.e. when :func:`build_llm` would raise
    :class:`LLMError.Auth`; ``api_key`` is :data:`REDACTED_KEY` or ``""``, never a key and
    never a prefix or length of one.

    ``configured_provider`` is the same question answered from the *settings* instead of
    the wire: what ``LLM_PROVIDER`` named (or ``GROQ_API_KEY`` implied, else
    :data:`DEFAULT_PROVIDER`). It is what picks the default endpoint and default model, and
    keeping it beside the host-derived ``provider`` is what makes a mismatch legible — a
    ``configured_provider`` of ``groq`` over ``provider`` of ``openai-compatible`` is a Groq
    key pointed at an NVIDIA endpoint, which is the failure this diagnostic exists to show.
    """

    provider: str | None
    configured_provider: str | None
    api_key: str
    base_url: str
    model: str
    key_source: str | None
    base_url_source: str | None


def _first_env(*names: str) -> tuple[str, str | None]:
    """The first non-blank ``name=value`` among ``names``, with the name that supplied it.

    A variable set to whitespace counts as unset, exactly as the single-name lookups did
    before the chain existed, so ``OPENAI_API_KEY=`` cannot shadow a later ``GROQ_API_KEY``.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value, name
    return "", None


def _select_provider() -> tuple[str, str | None]:
    """The provider whose defaults apply, and the variable that named it.

    ``LLM_PROVIDER`` wins when set; otherwise ``GROQ_API_KEY`` implies ``groq`` — the
    convenience that one name already promised. A provider this module does not know
    resolves to :data:`DEFAULT_PROVIDER` rather than raising: a typo in a vendor name is a
    configuration slip, and the OpenAI-compatible protocol was still what was asked for.
    The second element tells the caller whether the vendor was named *explicitly*, which
    is what lets an explicit provider outrank an ambient ``OPENAI_BASE_URL``.
    """
    explicit, source = _first_env(LLM_PROVIDER_ENV)
    if explicit:
        name = explicit.lower()
        return (name if name in PROVIDER_DEFAULTS else DEFAULT_PROVIDER), source
    if _first_env(GROQ_KEY_ENV)[0]:
        return "groq", GROQ_KEY_ENV
    return DEFAULT_PROVIDER, None


def _resolve_credentials(configured_provider: str, provider_source: str | None) -> _Credentials:
    """The key and endpoint, per the chain documented at the config block.

    The endpoint is the first of, in order:

    1. ``LLM_BASE_URL`` — the override for a deployment this module has no host for
       (self-hosted, Azure);
    2. the provider's own endpoint, **when ``LLM_PROVIDER`` named a provider** — naming the
       vendor is a deliberate act, and it outranks the protocol-named alias a host or a
       CI job happens to export. An Azure deployment signs up for "supply the endpoint
       yourself", which this step makes explicit rather than silently inheriting whatever
       ambient export is around;
    3. ``OPENAI_BASE_URL`` — the alias, which is the historical default resolution;
    4. Groq's URL when the key came from ``GROQ_API_KEY``, the convenience that name
       already promised;
    5. the provider's default endpoint as the last resort.

    ``base_url_source`` is ``None`` whenever the endpoint came from a *provider default*
    rather than a variable, because it names the variable that supplied it — and a default
    is exactly the case no variable did.
    """
    api_key, key_source = _first_env(LLM_KEY_ENV, API_KEY_ENV, GROQ_KEY_ENV)
    base_url, base_url_source = _first_env(LLM_BASE_URL_ENV)
    provider_endpoint, _ = PROVIDER_DEFAULTS.get(configured_provider, (None, None))
    if not base_url and provider_source == LLM_PROVIDER_ENV:
        base_url, base_url_source = provider_endpoint or "", None
    if not base_url:
        base_url, base_url_source = _first_env(BASE_URL_ENV)
    if not base_url and key_source == GROQ_KEY_ENV:
        base_url, base_url_source = GROQ_BASE_URL, GROQ_KEY_ENV
    if not base_url and provider_endpoint:
        base_url, base_url_source = provider_endpoint, None
    return _Credentials(
        api_key=api_key,
        key_source=key_source,
        base_url=base_url,
        base_url_source=base_url_source,
        provider=configured_provider,
    )


def _select_model(params: LLMParams | None, provider: str = DEFAULT_PROVIDER) -> str:
    """The model a build would configure: ``params.model``, ``LLM_MODEL``, ``OPENAI_MODEL``.

    The per-call override keeps winning over every variable — the sidebar's picker writes
    ``params.model``, and that precedence is part of the frozen contract. After the two
    variables, the provider's default model answers; when it has no opinion the built-in
    default is the last word, so a model is always named rather than left blank.
    """
    requested = params.model if params is not None else None
    if requested:
        return requested
    from_env, _ = _first_env(LLM_MODEL_ENV, MODEL_ENV)
    if from_env:
        return from_env
    _, provider_model = PROVIDER_DEFAULTS.get(provider, (None, None))
    return provider_model or DEFAULT_MODEL


def _provider_label(base_url: str) -> str:
    """A short label for where requests go: ``groq``, ``openai``, ``local`` or the rest.

    The label comes from the host, not from the variable that supplied the key, so a Groq
    key aimed at an NVIDIA endpoint reports the endpoint — the mismatch is the point.
    """
    host = (urlsplit(base_url).hostname or "").lower()
    if host.endswith("groq.com"):
        return "groq"
    if host.endswith("openai.com"):
        return "openai"
    if host in _LOOPBACK_HOSTS:
        return "local"
    return "openai-compatible"


def resolved_config(params: LLMParams | None = None) -> ResolvedConfig:
    """The configuration :func:`build_llm` would use, with the key redacted.

    No client is constructed and ``openai`` is never imported, so callers and tests can ask
    *what did this process resolve* — the question this module could not previously answer,
    which is what let an ambient ``OPENAI_API_KEY`` silently replace a key written into
    ``.env``. ``params`` mirrors :func:`build_llm`: when given, its ``model`` is the
    resolved one.
    """
    provider, provider_source = _select_provider()
    credentials = _resolve_credentials(provider, provider_source)
    return ResolvedConfig(
        provider=_provider_label(credentials.base_url) if credentials.api_key else None,
        configured_provider=credentials.provider if credentials.api_key else None,
        api_key=REDACTED_KEY if credentials.api_key else "",
        base_url=credentials.base_url,
        model=_select_model(params, credentials.provider),
        key_source=credentials.key_source,
        base_url_source=credentials.base_url_source,
    )


def resolved_provider() -> str | None:
    """The provider the gateway would talk to, or ``None`` when no key resolved.

    Read from the endpoint's host, so it names where the request would go even when
    ``LLM_PROVIDER`` was never set.
    """
    return resolved_config().provider


def build_llm(params: LLMParams | None = None) -> ChatClient:
    """Build the gateway for this process: the offline mock, or a real-compatible client.

    ``MOCK_LLM=1`` returns :class:`src.utils.mock_llm.MockLLM` — deterministic, offline,
    no key needed; that branch decides before any credential is read, so an ambient
    ``OPENAI_API_KEY`` cannot affect it. Otherwise a real client is built from the chain
    documented at the config block: ``LLM_PROVIDER`` names the endpoint family and supplies
    its default endpoint and default model; the key comes from ``LLM_API_KEY``,
    ``OPENAI_API_KEY`` or ``GROQ_API_KEY`` (the last implying Groq's provider and URL);
    ``LLM_BASE_URL`` or ``OPENAI_BASE_URL`` override the endpoint; and the model is
    ``LLM_MODEL`` / ``OPENAI_MODEL`` overridable per call through ``params.model``. A
    provider with no endpoint of its own (Azure) requires ``LLM_BASE_URL`` rather than
    falling back to a host it does not belong to. No node constructs
    a client itself: this factory is the single place configuration is read, which is what
    keeps the graph testable with a stub client and the demo runnable with no key at all.
    """
    if os.environ.get(MOCK_ENV, "").strip() == "1":
        from src.utils.mock_llm import MockLLM

        return MockLLM()

    provider, provider_source = _select_provider()
    credentials = _resolve_credentials(provider, provider_source)
    if not credentials.api_key:
        raise LLMError.Auth(
            f"chat needs a key: none of {LLM_KEY_ENV}, {API_KEY_ENV} or {GROQ_KEY_ENV} is "
            f"set and {MOCK_ENV} is not '1'. Either export {LLM_KEY_ENV} — or "
            f"{GROQ_KEY_ENV}, whose provider and endpoint follow the key — with "
            f"{LLM_PROVIDER_ENV}, {LLM_BASE_URL_ENV} and {LLM_MODEL_ENV} as needed, or run "
            f"with {MOCK_ENV}=1 for the deterministic offline path that needs no key and no "
            "network."
        )
    if not credentials.base_url:
        # A provider this module has no endpoint for (Azure's per-deployment URL). Sending
        # that key to a default host would be a silent misroute, so it is a stated error.
        raise LLMError.Auth(
            f"provider {provider!r} has no default endpoint: set {LLM_BASE_URL_ENV} to the "
            "deployment's URL (the override exists for a self-hosted or per-deployment "
            "endpoint this module cannot guess), or pick a provider from "
            f"{', '.join(sorted(PROVIDER_DEFAULTS))}"
        )
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - pinned in requirements.txt
        raise LLMError(
            "the OpenAI client is not installed; run `pip install -r requirements.txt` "
            f"or use {MOCK_ENV}=1 for the offline demo path"
        ) from exc

    client = AsyncOpenAI(
        api_key=credentials.api_key,
        base_url=credentials.base_url,
        timeout=REQUEST_TIMEOUT_S,
        # This gateway owns the retry policy; the SDK's own retries are disabled so a 429
        # is attempted three times in total, not three times per SDK-level attempt.
        max_retries=0,
    )
    return ChatLLM(client, model=_select_model(params, credentials.provider))

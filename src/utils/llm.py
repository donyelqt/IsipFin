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

MOCK_ENV = "MOCK_LLM"

LLM_PROVIDER_ENV = "LLM_PROVIDER"
LLM_KEY_ENV = "LLM_API_KEY"
LLM_BASE_URL_ENV = "LLM_BASE_URL"
LLM_MODEL_ENV = "LLM_MODEL"
API_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"
MODEL_ENV = "OPENAI_MODEL"
GROQ_KEY_ENV = "GROQ_API_KEY"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROVIDER = "openai"

PROVIDER_DEFAULTS: dict[str, tuple[str | None, str | None]] = {
    "openai": (DEFAULT_BASE_URL, DEFAULT_MODEL),
    "groq": (GROQ_BASE_URL, "openai/gpt-oss-120b"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "meta/llama-3.1-8b-instruct"),
    "ollama": ("http://localhost:11434/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
    "azure": (None, None),
}

REDACTED_KEY = "***"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
RETRY_AFTER_CAP_S = 30.0
REQUEST_TIMEOUT_S = 120.0

_SCHEMA_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DEFAULT_SCHEMA_NAME = "structured_response"

_STRICT_NUDGE = (
    "Your previous reply was not a single valid JSON object matching the requested schema. "
    "Reply with valid JSON matching the schema exactly — no markdown fences, no prose before "
    "or after the object, every required field present."
)


@dataclass(frozen=True)
class LLMParams:

    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class ChatResult:

    text: str
    parsed: dict[str, Any] | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


@runtime_checkable
class ChatClient(Protocol):

    async def chat(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        schema: dict[str, Any] | None = None,
    ) -> ChatResult: ...


class LLMError(Exception):

    if False:
        RateLimited: type[LLMError]
        ContextLength: type[LLMError]
        Auth: type[LLMError]
        Timeout: type[LLMError]

    retryable: bool = False

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

    retryable = True


class ContextLength(LLMError):
    pass


class Auth(LLMError):
    pass


class Timeout(LLMError):
    pass


class SchemaViolationError(Exception):
    pass


LLMError.RateLimited = RateLimited
LLMError.ContextLength = ContextLength
LLMError.Auth = Auth
LLMError.Timeout = Timeout


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
        return
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
    if isinstance(minimum := schema.get("minLength"), int) and len(payload) < minimum:
        errors.append(f"{path}: shorter than {minimum} characters")
    if isinstance(maximum := schema.get("maxLength"), int) and len(payload) > maximum:
        errors.append(f"{path}: longer than {maximum} characters")


def _validate_number(payload: float, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    if isinstance(minimum := schema.get("minimum"), int | float) and payload < minimum:
        errors.append(f"{path}: below the minimum {minimum}")
    if isinstance(maximum := schema.get("maximum"), int | float) and payload > maximum:
        errors.append(f"{path}: above the maximum {maximum}")


def _decode_json_object(text: str) -> dict[str, Any] | None:
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
    payload = _decode_json_object(text)
    if payload is None:
        return None, f"response is not a JSON object: {text[:120]!r}"
    errors = validate_json_schema(payload, schema)
    if errors:
        return None, "; ".join(errors[:3])
    return payload, ""


def _provider_error_classes(*names: str) -> tuple[type[BaseException], ...]:
    try:
        import openai
    except ImportError:
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
    return {
        "headers": _response_headers(exc),
        "response": getattr(exc, "response", None),
    }


def _map_provider_error(exc: BaseException) -> LLMError:
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
        return LLMError(
            f"the model endpoint returned a server error: {detail}", status=status, **provider
        )
    return LLMError(
        f"model request failed ({type(exc).__name__}: {detail})", status=status, **provider
    )


async def default_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def backoff_delay(attempt: int, base: float = BACKOFF_BASE_S, cap: float = BACKOFF_CAP_S) -> float:
    delay: float = base * float(2 ** max(0, attempt - 1))
    return min(delay, cap)


def default_jitter(delay: float) -> float:
    return delay / 2.0 + random.random() * (delay / 2.0)


def _retry_after_header(exc: BaseException) -> float | None:
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
    match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s|m|h)?", str(exc), flags=re.IGNORECASE
    )
    if match is None:
        return None
    return _retry_after_seconds(match.group(1), match.group(2))


def _retry_after_seconds(raw: str, unit: str | None = None) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value <= 0.0:
        return None
    multiplier = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}.get((unit or "s").lower(), 1.0)
    return value * multiplier


def retry_after_seconds(exc: BaseException) -> float | None:
    return _retry_after_header(exc) or _retry_after_message(exc)


_CLOSE_SPELLINGS = ("close", "aclose")


def _resolve_close(client: Any) -> Callable[[], Awaitable[None]] | None:
    for spelling in _CLOSE_SPELLINGS:
        candidate = getattr(client, spelling, None)
        if asyncio.iscoroutinefunction(candidate):
            return candidate
    return None


class ChatLLM:

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
        self._closed = False
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s
        self._retry_after_cap_s = retry_after_cap_s
        self._sleep = sleep
        self._jitter = jitter

    @property
    def client(self) -> Any:
        return self._client

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = _resolve_close(self._client)
        if close is None:
            return
        try:
            await close()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "closing the provider client failed: %s: %s", type(exc).__name__, exc
            )

    async def __aenter__(self) -> ChatLLM:
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
        if not messages:
            raise LLMError("chat requires at least one message")
        request = self._build_request(messages, params, schema)
        started = time.perf_counter()

        try:
            result = self._to_result(await self._send_with_retries(request))
        except Exception as error:
            await self.aclose()
            raise error
        if schema is None:
            return await self._finished(self._timed(result, started))

        parsed, reason = _parse_structured(result.text, schema)
        if parsed is None:
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
        await self.aclose()
        return result

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
        if isinstance(error, RateLimited):
            hint = retry_after_seconds(exc)
            if hint is not None:
                return min(hint, self._retry_after_cap_s)
        return jitter(backoff_delay(attempt, self._backoff_base_s, self._backoff_cap_s))

    def _resolve_jitter(self) -> Callable[[float], float]:
        return self._jitter if self._jitter is not None else default_jitter

    async def _send_with_retries(self, request: dict[str, Any]) -> Any:
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
        return replace(result, latency_ms=round((time.perf_counter() - started) * 1000.0, 3))


def _response_format(schema: dict[str, Any]) -> dict[str, Any]:
    name = schema.get("title")
    if not isinstance(name, str) or not _SCHEMA_NAME.fullmatch(name):
        name = _DEFAULT_SCHEMA_NAME
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def _usage_field(usage: Any, field: str) -> int:
    value = getattr(usage, field, None)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _is_retryable(error: LLMError) -> bool:
    status = error.status
    return error.retryable or (status is not None and 500 <= status < 600)


@dataclass(frozen=True)
class _Credentials:

    api_key: str
    key_source: str | None
    base_url: str
    base_url_source: str | None
    provider: str


@dataclass(frozen=True)
class ResolvedConfig:

    provider: str | None
    configured_provider: str | None
    api_key: str
    base_url: str
    model: str
    key_source: str | None
    base_url_source: str | None


def _first_env(*names: str) -> tuple[str, str | None]:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value, name
    return "", None


def _select_provider() -> tuple[str, str | None]:
    explicit, source = _first_env(LLM_PROVIDER_ENV)
    if explicit:
        name = explicit.lower()
        return (name if name in PROVIDER_DEFAULTS else DEFAULT_PROVIDER), source
    if _first_env(GROQ_KEY_ENV)[0]:
        return "groq", GROQ_KEY_ENV
    return DEFAULT_PROVIDER, None


def _resolve_credentials(configured_provider: str, provider_source: str | None) -> _Credentials:
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
    requested = params.model if params is not None else None
    if requested:
        return requested
    from_env, _ = _first_env(LLM_MODEL_ENV, MODEL_ENV)
    if from_env:
        return from_env
    _, provider_model = PROVIDER_DEFAULTS.get(provider, (None, None))
    return provider_model or DEFAULT_MODEL


def _provider_label(base_url: str) -> str:
    host = (urlsplit(base_url).hostname or "").lower()
    if host.endswith("groq.com"):
        return "groq"
    if host.endswith("openai.com"):
        return "openai"
    if host in _LOOPBACK_HOSTS:
        return "local"
    return "openai-compatible"


def resolved_config(params: LLMParams | None = None) -> ResolvedConfig:
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
    return resolved_config().provider


def build_llm(params: LLMParams | None = None) -> ChatClient:
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
        raise LLMError.Auth(
            f"provider {provider!r} has no default endpoint: set {LLM_BASE_URL_ENV} to the "
            "deployment's URL (the override exists for a self-hosted or per-deployment "
            "endpoint this module cannot guess), or pick a provider from "
            f"{', '.join(sorted(PROVIDER_DEFAULTS))}"
        )
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise LLMError(
            "the OpenAI client is not installed; run `pip install -r requirements.txt` "
            f"or use {MOCK_ENV}=1 for the offline demo path"
        ) from exc

    client = AsyncOpenAI(
        api_key=credentials.api_key,
        base_url=credentials.base_url,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=0,
    )
    return ChatLLM(client, model=_select_model(params, credentials.provider))

"""llm-gateway invariants (ADR-005, docs/TESTING-AND-EVAL.md).

Every test here is hermetic. The ``MOCK_LLM=1`` path is exercised as a real delivery
path — a missing key and a missing ``openai`` package must not break it — and the
OpenAI-compatible client is driven through the seam its own docstring names, the SDK
object itself (``client.chat.completions.create``), with a stub holding canned payloads
and failures. Retry tests inject the sleeper, so exhausting three attempts costs
milliseconds instead of the three seconds a real backoff would spend.

The invariants pinned are the ones ADR-005 and docs/SPEC.md section 9 freeze:

1. the mock gateway returns the frozen six-field ``ChatResult`` and is deterministic;
2. ``MOCK_LLM=1`` never imports the ``openai`` package and needs no key at all;
3. HTTP 429/5xx retries with exponential backoff, then raises ``LLMError.RateLimited``;
4. HTTP 401 raises ``LLMError.Auth`` immediately, with no retry;
5. a schema-violating response retries exactly once with a stricter nudge, then raises
   ``SchemaViolationError``;
6. token counts come from the response's usage field and ``latency_ms`` is measured;
7. the revenue question returns the exact FY2023 figure through the mock.

An eighth invariant is pinned by the same file: the credential chain resolves ``LLM_API_KEY``
/ ``LLM_BASE_URL`` ahead of the ``OPENAI_*`` names and ahead of ``GROQ_API_KEY`` (whose
endpoint follows the key), and the configuration it resolved is reportable with the key
redacted — the diagnostic that makes an ambient ``OPENAI_API_KEY`` silently replacing a key
written into ``.env`` visible instead of mysterious. The model keeps its single
protocol-named variable: ``OPENAI_MODEL`` names the wire format, not a vendor.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pytest
from src.utils.llm import (
    DEFAULT_BASE_URL,
    GROQ_BASE_URL,
    REDACTED_KEY,
    ChatLLM,
    ChatResult,
    LLMError,
    LLMParams,
    SchemaViolationError,
    backoff_delay,
    build_llm,
    resolved_config,
    resolved_provider,
    validate_json_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- the prompts and schemas the gateway is actually called with -------------------
# The system prompts carry the clauses src/utils/mock_llm.py fingerprints on, so these
# fixtures exercise the production selection path rather than a mock-only shortcut.
SYNTHESIS_SYSTEM_PROMPT = (
    "You are a financial analyst answering questions strictly from the provided document "
    "context. Every factual claim MUST be followed by the [n] marker of the context block "
    "that contains it."
)
QUERY_PLANNER_SYSTEM_PROMPT = (
    "You are the query planner for a financial document QA system. Your job: decide HOW to "
    "answer, not to answer. Respond with JSON matching the QueryPlan schema exactly."
)
GROUNDING_SYSTEM_PROMPT = (
    "You verify that the remaining candidate claims are fully supported by the provided "
    "context. For each factual claim, decide supported, partially or unsupported."
)

# The income-statement rows as the corpus prints them: FY2023 and FY2022 in adjacent
# columns, which is exactly the trap P3's year-aware extraction contract exists for.
INCOME_STATEMENT_CONTEXT = (
    "CONTEXT:\n"
    "[1] msft-fy2023.pdf - pp. 60-61 - Income Statements\n"
    "Revenue $ 211,915 $ 198,270\n"
    "Operating income $ 88,523 $ 83,383\n"
    "Net income $ 72,361 $ 72,738\n"
    "Diluted earnings per share $ 9.68 $ 9.23\n"
)

REVENUE_QUESTION = "What was Microsoft's revenue in FY2023?"
# An adjacent-year figure the corpus never prints for FY2023 revenue: the year-confusion
# canary from STRESS-TEST-RAG #1 (an FY2024 comparative asserted as FY2023).
WRONG_YEAR_FIGURE = "$245,122"

QUERY_PLAN_SCHEMA: dict[str, Any] = {
    "title": "QueryPlan",
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["simple", "multi_part", "out_of_scope"]},
        "sub_queries": {"type": "array", "items": {"type": "string"}},
        "required_sections": {"type": "array", "items": {"type": "string"}},
        "fiscal_period": {"type": ["string", "null"]},
        "missing_aspect": {"type": ["string", "null"]},
    },
    "required": ["kind", "sub_queries", "required_sections", "fiscal_period", "missing_aspect"],
}

GROUNDING_RESULT_SCHEMA: dict[str, Any] = {
    "title": "GroundingResult",
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "object"}},
        "all_supported": {"type": "boolean"},
    },
    "required": ["claims", "all_supported"],
}

# The six fields docs/SPEC.md section 9 freezes for ChatResult.
FROZEN_RESULT_FIELDS = {
    "text",
    "parsed",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "model",
}


def _run(coroutine: Any) -> Any:
    """Drive one coroutine to completion (no async plugin is installed)."""
    return asyncio.run(coroutine)


def _synthesis_messages(question: str = REVENUE_QUESTION) -> list[dict[str, str]]:
    """The P3-shaped payload: the system prompt, then the context plus the question."""
    return [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"{INCOME_STATEMENT_CONTEXT}\nQUESTION:\n{question}"},
    ]


def _planner_messages(question: str = REVENUE_QUESTION) -> list[dict[str, str]]:
    """The P1-shaped payload: catalog plus the question, answered as a QueryPlan."""
    return [
        {"role": "system", "content": QUERY_PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"DOCUMENT CATALOG:\n[]\n\nQUESTION:\n{question}",
        },
    ]


def _grounding_messages() -> list[dict[str, str]]:
    """The P5-shaped payload: candidate answer, then the numbered context blocks."""
    return [
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "CANDIDATE ANSWER:\nMicrosoft's FY2023 revenue was $211,915 million [1].\n\n"
                "CONTEXT:\n[1] Income Statements\nRevenue $ 211,915 million\n"
            ),
        },
    ]


# --- environment isolation --------------------------------------------------------


@pytest.fixture
def mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The offline path with no key present: ``MOCK_LLM=1`` must not need one.

    Every credential name the gateway can read is cleared — including the ``LLM_*`` and
    ``GROQ_*`` ones — so the offline path is proven keyless on a host that exports a key,
    not merely on a host that happens not to.
    """
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def mock_client() -> Any:
    """A fresh offline gateway, built the way the graph builds it."""
    return build_llm(LLMParams())


# --- the stub transport (the seam ChatLLM documents) ------------------------------


class ProviderStatusError(Exception):
    """A provider failure carrying only the status code the gateway reads."""

    def __init__(self, status: int, detail: str = "the provider refused the request") -> None:
        super().__init__(f"{detail} (HTTP {status})")
        self.status_code = status


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list[_Choice]
    usage: _Usage
    model: str = "stub-model"


def _completion(text: str, prompt_tokens: int = 41, completion_tokens: int = 7) -> _Completion:
    """One canned chat completion, shaped like the SDK's response object."""
    return _Completion([_Choice(_Message(text))], _Usage(prompt_tokens, completion_tokens))


class StubTransport:
    """A stand-in for ``openai.AsyncOpenAI`` at the seam ``ChatLLM`` documents.

    Canned responses (or exceptions) are returned in order, the last one repeating once
    the list runs out, and every call is recorded with its exact kwargs so a test can
    count attempts and read what the gateway sent. No socket is ever opened.
    """

    def __init__(self, responses: Sequence[Any] | Any) -> None:
        self._responses = list(responses) if isinstance(responses, Sequence) else [responses]
        self.calls: list[dict[str, Any]] = []

    @property
    def chat(self) -> StubTransport:
        return self

    @property
    def completions(self) -> StubTransport:
        return self

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


class Sleeper:
    """The injected backoff sleeper: records the delays instead of spending them."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _identity(delay: float) -> float:
    """Jitter turned off, so the backoff schedule is exactly what the policy computes."""
    return delay


def _stubbed_client(responses: Sequence[Any] | Any) -> tuple[ChatLLM, StubTransport, Sleeper]:
    """A real gateway over a canned transport, with no wall-clock waiting."""
    transport = StubTransport(responses)
    sleeper = Sleeper()
    client = ChatLLM(transport, model="stub-model", sleep=sleeper, jitter=_identity)
    return client, transport, sleeper


@pytest.fixture
def purged_openai() -> Iterator[None]:
    """Remove the ``openai`` modules a mapped failure lazily imports.

    ``_map_provider_error`` resolves the SDK's exception classes on first failure so the
    real provider's types map identically, which imports ``openai`` even here. That
    mapping is stateless, so unloading the package again afterwards is safe; leaving it
    loaded would leak the paid dependency into a mock-only process, and a sibling test
    (tests/test_retrieval.py) reads ``sys.modules`` as its proof that no network client
    reached retrieval. The cleanup runs as a finalizer, i.e. after the test body.
    """
    yield
    for name in [name for name in sys.modules if name == "openai" or name.startswith("openai.")]:
        del sys.modules[name]


# --- the offline interpreter probes ------------------------------------------------
# These run the gateway in a *fresh* interpreter: "MOCK_LLM=1 imports no openai" is only
# decidable in a process whose sys.modules starts empty, and the fingerprint claim
# ("identical across runs") is only proven across processes, not twice in one of them.
_MOCK_PROBE = """
import asyncio, json, os, sys

from src.utils.llm import LLMParams, build_llm

client = build_llm(LLMParams())
messages = json.loads(os.environ["GATEWAY_PROBE_MESSAGES"])
result = asyncio.run(client.chat(messages, LLMParams()))
print(json.dumps({
    "client": type(client).__name__,
    "text": result.text,
    "model": result.model,
    "prompt_tokens": result.prompt_tokens,
    "completion_tokens": result.completion_tokens,
    "latency_ms": result.latency_ms,
    "openai_imported": any(
        name == "openai" or name.startswith("openai.") for name in sys.modules
    ),
}, sort_keys=True))
"""


def _probe_mock_interpreter(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Run one offline chat in a clean interpreter and return what it reported."""
    env = {**os.environ, "MOCK_LLM": "1", "GATEWAY_PROBE_MESSAGES": json.dumps(messages)}
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_BASE_URL", None)
    completed = subprocess.run(
        [sys.executable, "-c", _MOCK_PROBE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, f"offline probe failed:\n{completed.stderr}"
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


# --- 1. the frozen ChatResult, and determinism ------------------------------------


def test_mock_result_carries_every_frozen_field(mock_mode: None) -> None:
    result = _run(mock_client().chat(_synthesis_messages(), LLMParams()))

    assert isinstance(result, ChatResult)
    assert {field.name for field in fields(ChatResult)} == FROZEN_RESULT_FIELDS
    assert isinstance(result.text, str) and result.text
    assert isinstance(result.parsed, dict | None)
    assert isinstance(result.prompt_tokens, int) and result.prompt_tokens > 0
    assert isinstance(result.completion_tokens, int) and result.completion_tokens > 0
    assert isinstance(result.latency_ms, float) and result.latency_ms > 0
    assert isinstance(result.model, str) and result.model


def test_mock_is_deterministic_across_calls(mock_mode: None) -> None:
    """ADR-005's fingerprint key: the same request must answer the same way twice.

    The sampling parameters differ between the two calls on purpose — the sidebar drives
    them, and in mock mode they must not select a different script.
    """
    client = mock_client()
    messages = _synthesis_messages()

    first = _run(client.chat(messages, LLMParams()))
    second = _run(
        client.chat(messages, LLMParams(model="gpt-4o-mini", temperature=0.0, max_tokens=256))
    )

    assert first == second, "the same prompt produced two different mock results"
    assert first.parsed is None, "a schema-free call must not report a parsed payload"


def test_each_build_llm_call_builds_an_independent_client(mock_mode: None) -> None:
    """No shared mutable state: two clients answer identically but are not one object."""
    first, second = mock_client(), mock_client()
    messages = _synthesis_messages()

    assert first is not second
    assert first.model == second.model
    assert _run(first.chat(messages, LLMParams())) == _run(second.chat(messages, LLMParams()))


def test_both_clients_satisfy_the_node_facing_protocol(mock_mode: None) -> None:
    """One ``chat`` surface: a node can be handed either gateway without a branch."""
    from src.utils.llm import ChatClient

    mock = mock_client()
    client, _, _ = _stubbed_client([_completion("ok")])

    assert isinstance(mock, ChatClient)
    assert isinstance(client, ChatClient)


def test_mock_answers_identically_in_a_fresh_interpreter(mock_mode: None) -> None:
    """sha256 fingerprints are stable across processes, not just within one."""
    first = _probe_mock_interpreter(_synthesis_messages())
    second = _probe_mock_interpreter(_synthesis_messages())

    assert first == second, "the fingerprint is not stable across interpreters"
    assert first["client"] == "MockLLM"


# --- 2. the offline path needs no key and no OpenAI package -----------------------


def test_mock_mode_runs_with_no_key_and_never_imports_openai(mock_mode: None) -> None:
    """ADR-005's keyless demo: a live chat in an interpreter that has neither.

    The key was already removed by the fixture, and the probe runs in a separate
    process, so ``sys.modules`` here reports only what the mock path itself pulled in.
    """
    payload = _probe_mock_interpreter(_synthesis_messages())

    assert (
        payload["openai_imported"] is False
    ), "MOCK_LLM=1 imported the openai package; the offline path is not offline"
    assert payload["prompt_tokens"] > 0 and payload["completion_tokens"] > 0
    assert payload["latency_ms"] > 0
    assert payload["model"] != "stub-model"


def test_missing_api_key_raises_auth_before_any_client_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")

    with pytest.raises(LLMError.Auth) as raised:
        build_llm(LLMParams())

    message = str(raised.value)
    assert (
        "OPENAI_API_KEY" in message and "MOCK_LLM=1" in message
    ), "the setup error must name the missing key and the offline alternative"
    assert raised.value.attempts is None


# --- 3. retries: 429 and 5xx, then a typed error -----------------------------------


def test_rate_limit_exhausts_three_attempts_then_raises_rate_limited(
    mock_mode: None, purged_openai: None
) -> None:
    client, transport, sleeper = _stubbed_client([ProviderStatusError(429, "quota reached")])
    started = time.perf_counter()

    with pytest.raises(LLMError.RateLimited) as raised:
        _run(client.chat(_synthesis_messages(), LLMParams()))
    elapsed = time.perf_counter() - started

    assert transport.calls and len(transport.calls) == 3, "a 429 must be attempted 3 times"
    assert raised.value.status == 429
    assert raised.value.attempts == 3
    assert isinstance(raised.value, LLMError)
    # ADR-005: base 1s, doubling. With jitter off the schedule is exactly that, and with
    # the sleeper injected the whole exhaustion must cost milliseconds, not 3 real seconds.
    assert sleeper.delays == [1.0, 2.0]
    assert elapsed < 1.0, f"the retry path spent {elapsed:.2f}s of real time"


def test_server_error_is_retried_exponentially_then_raises(
    mock_mode: None, purged_openai: None
) -> None:
    """ADR-005 retries 5xx; the raised class is the generic LLMError carrying the status.

    ``_map_provider_error`` labels a 5xx body *RateLimited* internally but the resulting
    error is the base class (ADR-005 only freezes the retryable status set here), so the
    observable contract is: three attempts, doubling sleeps, status preserved.
    """
    client, transport, sleeper = _stubbed_client([ProviderStatusError(500, "backend down")])

    with pytest.raises(LLMError) as raised:
        _run(client.chat(_synthesis_messages(), LLMParams()))

    assert len(transport.calls) == 3, "ADR-005 retries 5xx as well as 429"
    assert sleeper.delays == [1.0, 2.0]
    assert raised.value.status == 500
    assert raised.value.attempts == 3
    assert isinstance(raised.value, LLMError)


def test_backoff_doubles_from_the_base_and_caps() -> None:
    """The exponential policy, asserted as arithmetic rather than as observed sleeps."""
    assert backoff_delay(1) == 1.0
    assert backoff_delay(2) == 2.0
    assert backoff_delay(3) == 4.0
    assert backoff_delay(50, cap=30.0) == 30.0
    assert backoff_delay(0) == 1.0, "a zero-based attempt still waits one base"


def test_a_rate_limit_that_recovers_answers_without_raising(
    mock_mode: None, purged_openai: None
) -> None:
    client, transport, sleeper = _stubbed_client(
        [ProviderStatusError(429, "quota reached"), _completion("recovered")]
    )

    result = _run(client.chat(_synthesis_messages(), LLMParams()))

    assert result.text == "recovered"
    assert len(transport.calls) == 2
    assert sleeper.delays == [1.0]


# --- 4. failures that must never be retried ---------------------------------------


def test_auth_failure_raises_immediately_without_retry(
    mock_mode: None, purged_openai: None
) -> None:
    client, transport, sleeper = _stubbed_client(ProviderStatusError(401, "invalid api key"))

    with pytest.raises(LLMError.Auth) as raised:
        _run(client.chat(_synthesis_messages(), LLMParams()))

    assert len(transport.calls) == 1, "an auth failure is terminal: it must not be retried"
    assert sleeper.delays == []
    assert raised.value.status == 401


def test_context_length_failure_is_not_retried(mock_mode: None, purged_openai: None) -> None:
    oversized = ProviderStatusError(400, "maximum context length exceeded")
    client, transport, sleeper = _stubbed_client([oversized])

    with pytest.raises(LLMError.ContextLength) as raised:
        _run(client.chat(_synthesis_messages(), LLMParams()))

    assert len(transport.calls) == 1, "retrying a too-long prompt cannot shorten it"
    assert sleeper.delays == []
    assert raised.value.status == 400


def test_timeout_is_not_retried(mock_mode: None, purged_openai: None) -> None:
    client, transport, sleeper = _stubbed_client([TimeoutError("read timed out")])

    with pytest.raises(LLMError.Timeout) as raised:
        _run(client.chat(_synthesis_messages(), LLMParams()))

    assert len(transport.calls) == 1, "ADR-005 retries 429/5xx only, never a timeout"
    assert sleeper.delays == []
    assert raised.value.status is None


# --- 5. schema mode: retried once, strictly, then a typed failure ------------------


def test_schema_violation_retries_once_then_raises(mock_mode: None) -> None:
    client, transport, _ = _stubbed_client(
        [_completion("I am afraid I cannot do that."), _completion("still prose")]
    )

    with pytest.raises(SchemaViolationError) as raised:
        _run(client.chat(_synthesis_messages(), LLMParams(), QUERY_PLAN_SCHEMA))

    assert len(transport.calls) == 2, "ADR-005 allows exactly one stricter retry"
    assert len(raised.value.args[0]) > 0
    first_call, second_call = transport.calls
    # The retry carries the stricter nudge as an extra user turn (several servers reject
    # a second system message), and requests JSON through response_format.
    assert len(second_call["messages"]) == len(first_call["messages"]) + 1
    assert second_call["messages"][-1]["role"] == "user"
    assert second_call["response_format"]["type"] == "json_schema"
    assert second_call["response_format"]["json_schema"]["name"] == "QueryPlan"


def test_schema_violation_that_recovers_returns_the_parsed_payload(mock_mode: None) -> None:
    recovered = json.dumps(
        {
            "kind": "simple",
            "sub_queries": [],
            "required_sections": ["Income Statements"],
            "fiscal_period": "2023",
            "missing_aspect": None,
        }
    )
    client, transport, _ = _stubbed_client([_completion("not json"), _completion(recovered)])

    result = _run(client.chat(_synthesis_messages(), LLMParams(), QUERY_PLAN_SCHEMA))

    assert result.parsed == json.loads(recovered), "the retriable payload must be handed back"
    assert json.loads(result.text) == result.parsed
    assert len(transport.calls) == 2


def test_schema_free_call_returns_raw_text_and_no_parsed_payload(mock_mode: None) -> None:
    client, transport, _ = _stubbed_client([_completion("Markdown **body** text.")])

    result = _run(client.chat(_synthesis_messages(), LLMParams()))

    assert result.text == "Markdown **body** text."
    assert result.parsed is None
    assert "response_format" not in transport.calls[0]


# --- 6. accounting: usage, latency, and the request the gateway sends --------------


def test_stubbed_transport_reports_usage_and_measures_latency(mock_mode: None) -> None:
    client, _, _ = _stubbed_client(
        [_completion("cited answer", prompt_tokens=5124, completion_tokens=217)]
    )

    result = _run(client.chat(_synthesis_messages(), LLMParams()))

    assert result.prompt_tokens == 5124, "token counts must come from the response usage"
    assert result.completion_tokens == 217
    assert result.latency_ms > 0, "latency is measured per call, not defaulted to zero"
    assert result.model == "stub-model"


def test_transport_without_usage_still_answers(mock_mode: None) -> None:
    """Local servers (Ollama, vLLM) often omit usage; that is a zero, never a crash."""
    payload = _Completion([_Choice(_Message("local answer"))], None, "llama3")
    client, _, _ = _stubbed_client([payload])

    result = _run(client.chat(_synthesis_messages(), LLMParams()))

    assert result.text == "local answer"
    assert result.prompt_tokens == 0 and result.completion_tokens == 0
    assert result.latency_ms > 0


def test_request_carries_only_the_parameters_the_caller_set(mock_mode: None) -> None:
    client, transport, _ = _stubbed_client([_completion("ok")])
    params = LLMParams(model="per-call-model", temperature=0.2, top_p=0.9, max_tokens=256)

    _run(client.chat(_synthesis_messages(), params))

    sent = transport.calls[0]
    assert set(sent) == {"model", "messages", "temperature", "top_p", "max_tokens"}
    assert sent["model"] == "per-call-model", "params.model overrides the client's model"
    assert sent["temperature"] == 0.2
    # The messages are forwarded verbatim: the gateway is not a prompt rewriter.
    assert sent["messages"] == _synthesis_messages()


def test_chat_without_messages_is_rejected(mock_mode: None) -> None:
    client, transport, _ = _stubbed_client([_completion("ok")])

    with pytest.raises(LLMError):
        _run(client.chat([], LLMParams()))

    assert transport.calls == [], "an empty conversation must never reach the provider"


# --- 7. the revenue figure, exactly, through the mock ------------------------------


def test_revenue_question_returns_the_exact_fy2023_figure(mock_mode: None) -> None:
    result = _run(mock_client().chat(_synthesis_messages(), LLMParams()))

    assert "$211,915 million" in result.text, "the FY2023 revenue figure must be exact"
    assert "FY2023" in result.text
    # Year-confusion canary: an adjacent-period comparative is the failure this guards.
    assert WRONG_YEAR_FIGURE not in result.text
    assert "$ 211,915" in INCOME_STATEMENT_CONTEXT


def test_the_same_figure_survives_every_paraphrase_of_the_question(mock_mode: None) -> None:
    """The script is keyed on the metric, so any honest phrasing of one fact matches."""
    client = mock_client()
    questions = [
        "What was Microsoft's revenue in FY2023?",
        "What was revenue in fiscal year 2023?",
        "Total revenue 2023?",
    ]

    for question in questions:
        result = _run(client.chat(_synthesis_messages(question), LLMParams()))
        assert "$211,915 million" in result.text, f"no exact figure for {question!r}"


def test_structured_mock_answer_matches_the_requested_schema(mock_mode: None) -> None:
    result = _run(
        mock_client().chat(_planner_messages(), LLMParams(temperature=0.0), QUERY_PLAN_SCHEMA)
    )

    assert result.parsed is not None, "a schema call must report its parsed payload"
    assert validate_json_schema(result.parsed, QUERY_PLAN_SCHEMA) == []
    assert json.loads(result.text) == result.parsed
    assert result.parsed["kind"] == "simple"
    assert result.parsed["fiscal_period"] == "2023", "the plan named the wrong period"


def test_grounding_mock_answer_matches_its_schema(mock_mode: None) -> None:
    client = mock_client()
    messages = _grounding_messages()

    result = _run(client.chat(messages, LLMParams(temperature=0.0), GROUNDING_RESULT_SCHEMA))

    assert result.parsed is not None
    assert validate_json_schema(result.parsed, GROUNDING_RESULT_SCHEMA) == []
    assert isinstance(result.parsed["claims"], list) and result.parsed["claims"]
    assert isinstance(result.parsed["all_supported"], bool)
    # The same verdict twice: verification at temperature 0 must be reproducible.
    assert _run(client.chat(messages, LLMParams(), GROUNDING_RESULT_SCHEMA)) == result


# --- the real path: one factory, provider-agnostic by environment -------------------

# The keyed path is probed in a *child interpreter* for a reason mypy cannot see: the
# mock-path tests above must never import ``openai`` in this process, and a sibling test
# (tests/test_retrieval.py) asserts exactly that by reading sys.modules. A construction
# that opens no socket in-process (``AsyncOpenAI`` connects lazily) still pollutes the
# shared module table, so the real path is verified where it cannot leak.
_REAL_PROBE = """
import asyncio, json, os

from src.utils.llm import LLMParams, build_llm, resolved_config

resolved = resolved_config()
try:
    client = build_llm(LLMParams(model=os.environ.get("PROBE_PARAMS_MODEL") or None))
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
else:
    expected_key = os.environ.get("PROBE_EXPECT_KEY")
    print(json.dumps({
        "client": type(client).__name__,
        "model": client.model,
        "base_url": str(client.client.base_url),
        "max_retries": client.client.max_retries,
        "timeout": client.client.timeout,
        "api_key_matches_expected": expected_key is None or client.client.api_key == expected_key,
        "resolved": {
            "provider": resolved.provider,
            "api_key": resolved.api_key,
            "base_url": resolved.base_url,
            "model": resolved.model,
            "key_source": resolved.key_source,
            "base_url_source": resolved.base_url_source,
        },
    }))
"""


#: Every variable the credential chain can read. The probe clears all of them before the
#: test's overrides are applied, so what it reports is the environment the test built and
#: never what the host exported.
_CREDENTIAL_ENV_NAMES = (
    "GROQ_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
)


def _probe_real_interpreter(**overrides: str) -> dict[str, Any]:
    """Build one keyed gateway in a child interpreter and return what it reports.

    The credential variables are cleared first and then the overrides are applied, so an
    override *is* the environment: passing ``OPENAI_API_KEY=""`` leaves that name unset for
    the chain (a blank value counts as unset), which is how a test proves that only
    ``GROQ_API_KEY`` was configured. Without the clearing, this host's exported
    ``OPENAI_API_KEY`` would decide the outcome of tests about precedence.

    ``PROBE_EXPECT_KEY`` is compared inside the child, so a test can prove which key reached
    the client without this process — or the probe's output — ever holding it.
    """
    env = {**os.environ, "MOCK_LLM": "0"}
    for name in _CREDENTIAL_ENV_NAMES:
        env.pop(name, None)
    env["OPENAI_API_KEY"] = "test-key-not-a-secret"
    env.update(overrides)
    completed = subprocess.run(
        [sys.executable, "-c", _REAL_PROBE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, f"real-path probe failed:\n{completed.stderr}"
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434/v1",
        "https://api.groq.com/openai/v1",
        "http://localhost:8000/v1",
    ],
)
def test_real_path_builds_the_client_the_environment_describes(base_url: str) -> None:
    """ADR-005's provider promise: OpenAI, Ollama, vLLM and Groq differ by env only.

    The client is constructed but never called — ``AsyncOpenAI`` opens no connection
    until a request is made, and none is.
    """
    reported = _probe_real_interpreter(OPENAI_BASE_URL=base_url)

    assert "error" not in reported
    assert reported["client"] == "ChatLLM"
    assert base_url.rstrip("/") in reported["base_url"]
    # The gateway owns the retry policy, so the SDK's own retries are off: a 429 must be
    # three attempts in total, not three per SDK-level attempt.
    assert reported["max_retries"] == 0
    assert reported["model"] == "gpt-4o-mini", "the OPENAI_MODEL default was not applied"


def test_the_real_path_takes_the_model_from_the_environment() -> None:
    reported = _probe_real_interpreter(
        OPENAI_BASE_URL="http://localhost:8000/v1", OPENAI_MODEL="meta/llama-3.1-8b-instruct"
    )

    assert reported["model"] == "meta/llama-3.1-8b-instruct"
    assert "error" not in reported


def test_per_call_model_overrides_the_environment() -> None:
    """The sidebar writes params.model; a per-call override must reach the request."""
    reported = _probe_real_interpreter(
        OPENAI_BASE_URL="http://localhost:8000/v1", PROBE_PARAMS_MODEL="gpt-4o"
    )

    assert reported["model"] == "gpt-4o"


# --- 8. credential precedence: LLM_* > OPENAI_* > GROQ_API_KEY ---------------------
#
# The defect these pin: `.env` is loaded with override=False, so a host that exports
# OPENAI_API_KEY (this one does, with an NVIDIA key) silently wins over whatever a user
# wrote into `.env`, and the request goes out with the host's credential and endpoint. The
# LLM_* names outrank it; GROQ_API_KEY is the name a user reaches for and it implies Groq's
# endpoint. Each precedence is proven independently, in a child interpreter, so the key
# itself never reaches this process's assertions — only "which variable supplied it" does.


def test_llm_api_key_beats_a_hostile_openai_key() -> None:
    reported = _probe_real_interpreter(
        LLM_API_KEY="llm-dedicated-key",
        OPENAI_API_KEY="hostile-ambient-key",
        PROBE_EXPECT_KEY="llm-dedicated-key",
    )

    assert "error" not in reported
    assert reported["api_key_matches_expected"] is True, "the ambient OPENAI_API_KEY was used"
    assert reported["resolved"]["key_source"] == "LLM_API_KEY"
    assert reported["resolved"]["api_key"] == REDACTED_KEY


def test_llm_base_url_beats_a_hostile_openai_base_url() -> None:
    reported = _probe_real_interpreter(
        LLM_BASE_URL=GROQ_BASE_URL,
        OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1",
    )

    assert "error" not in reported
    assert reported["resolved"]["base_url_source"] == "LLM_BASE_URL"
    assert reported["base_url"].rstrip("/") == GROQ_BASE_URL
    assert reported["resolved"]["provider"] == "groq"


def test_params_model_still_beats_every_environment_variable() -> None:
    """The sidebar's per-call model keeps winning: that precedence is unchanged."""
    reported = _probe_real_interpreter(
        OPENAI_MODEL="hostile-model",
        PROBE_PARAMS_MODEL="sidebar-chosen-model",
    )

    assert reported["model"] == "sidebar-chosen-model"


def test_groq_key_alone_implies_the_groq_base_url() -> None:
    """One variable is a working Groq configuration, which is what the name implies."""
    reported = _probe_real_interpreter(
        GROQ_API_KEY="gsk-groq-only-key",
        OPENAI_API_KEY="",
        PROBE_EXPECT_KEY="gsk-groq-only-key",
    )

    assert "error" not in reported
    assert reported["api_key_matches_expected"] is True
    assert reported["resolved"]["key_source"] == "GROQ_API_KEY"
    assert reported["resolved"]["base_url_source"] == "GROQ_API_KEY"
    assert reported["base_url"].rstrip("/") == GROQ_BASE_URL
    assert reported["resolved"]["provider"] == "groq"


def test_a_configured_base_url_beats_the_groq_fallback() -> None:
    """The Groq URL follows the key only when no endpoint is configured anywhere."""
    reported = _probe_real_interpreter(
        GROQ_API_KEY="gsk-groq-only-key",
        OPENAI_API_KEY="",
        OPENAI_BASE_URL="http://localhost:8000/v1",
    )

    assert reported["resolved"]["base_url_source"] == "OPENAI_BASE_URL"
    assert "localhost:8000" in reported["base_url"]
    assert reported["resolved"]["provider"] == "local"


def test_the_defaults_are_unchanged_when_no_new_variable_is_set() -> None:
    """With no ``LLM_*``/``GROQ_*`` set, the chain lands exactly where it used to."""
    reported = _probe_real_interpreter()

    assert reported["resolved"]["key_source"] == "OPENAI_API_KEY"
    assert reported["resolved"]["base_url"] == DEFAULT_BASE_URL
    assert reported["base_url"].rstrip("/") == DEFAULT_BASE_URL
    assert reported["resolved"]["provider"] == "openai"
    assert reported["model"] == "gpt-4o-mini"
    assert reported["resolved"]["base_url_source"] is None


def test_missing_key_message_names_every_variable_a_user_can_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primary error a user hits must name the new chain, not just OPENAI_API_KEY."""
    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMError.Auth) as raised:
        build_llm(LLMParams())

    message = str(raised.value)
    assert "LLM_API_KEY" in message
    assert "OPENAI_API_KEY" in message and "GROQ_API_KEY" in message
    assert "MOCK_LLM=1" in message
    assert raised.value.attempts is None


def test_mock_mode_short_circuits_before_any_credential_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MOCK_LLM=1`` returns the mock with keys present — and never resolves them.

    The resolver is replaced with a function that fails, so this is a proof that the chain
    was not read rather than a claim that it did not matter.
    """
    from src.utils import llm as gateway
    from src.utils.mock_llm import MockLLM

    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("LLM_API_KEY", "llm-key-that-must-not-be-read")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-key-that-must-not-be-read")

    def explode() -> Any:
        raise AssertionError("MOCK_LLM=1 must return before any credential is read")

    monkeypatch.setattr(gateway, "_resolve_credentials", explode)

    assert isinstance(build_llm(LLMParams()), MockLLM)


def test_resolved_config_redacts_the_key_and_names_what_won(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diagnostic the gateway lacked: what did this process actually resolve?

    Every four-character window of the real key is searched for, so a leaked prefix fails
    as loudly as printing the key would.
    """
    secret = "gsk_this-prefix-must-not-leak-9f3a71c6"
    monkeypatch.setenv("LLM_API_KEY", secret)
    monkeypatch.setenv("OPENAI_API_KEY", "hostile-ambient-key")
    monkeypatch.setenv("LLM_BASE_URL", GROQ_BASE_URL)
    monkeypatch.setenv("OPENAI_MODEL", "env-chosen-model")

    resolved = resolved_config()

    assert resolved.key_source == "LLM_API_KEY"
    assert resolved.api_key == REDACTED_KEY
    assert resolved.base_url == GROQ_BASE_URL
    assert resolved.base_url_source == "LLM_BASE_URL"
    assert resolved.model == "env-chosen-model"
    assert resolved.provider == "groq"
    assert resolved_provider() == "groq"
    rendered = repr(resolved) + str(resolved)
    assert all(secret[index : index + 4] not in rendered for index in range(len(secret) - 3))
    assert rendered.count(secret) == 0


def test_the_redaction_mask_is_fixed_length_and_independent_of_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mask whose length tracked the key would leak the key's length."""
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "k")
    short_key = resolved_config().api_key
    monkeypatch.setenv("LLM_API_KEY", "gsk_" + "x" * 96)
    long_key = resolved_config().api_key

    assert short_key == long_key == REDACTED_KEY
    assert "gsk_" not in long_key
    assert resolved_provider() == "openai", "an unset endpoint resolves to OpenAI's"


def test_no_key_resolves_to_none_and_an_empty_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to talk to: the diagnostic says so instead of inventing a provider."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    resolved = resolved_config()

    assert resolved.provider is None
    assert resolved_provider() is None
    assert resolved.api_key == "" and resolved.key_source is None

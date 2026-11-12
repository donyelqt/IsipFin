from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from src.chains.state import GraphState, NodeEvent
from src.utils.llm import ChatResult

__all__ = ["TOKEN_KEY", "appended_records", "trace_node"]

TOKEN_KEY = "llm"

TRACE_KEY = "node_trace"

_F = TypeVar("_F", bound=Callable[..., Any])


def appended_records(state: GraphState) -> list[NodeEvent]:

    existing = state.get("node_trace")
    if not isinstance(existing, list):
        existing = []
        state["node_trace"] = existing
    return existing


def _tokens_of(result: Any) -> tuple[int | None, int | None]:

    if isinstance(result, ChatResult):
        return int(result.prompt_tokens), int(result.completion_tokens)
    return None, None


def _finish(state: GraphState, result: Any, event: NodeEvent) -> Any:

    appended_records(state).append(event)
    if isinstance(result, dict) and "node_trace" in result:
        result["node_trace"] = state["node_trace"]
    return result


def _fail(state: GraphState, error: BaseException, event: NodeEvent) -> None:

    from src.chains.state import replace_event

    recorded = replace_event(event, error=f"{type(error).__name__}: {error}")
    appended_records(state).append(recorded)


def _reported(result: Any) -> Any:

    if isinstance(result, dict):
        return result.get(TOKEN_KEY)
    return None


def _event(node: str, started_at: float, duration_ms: float, reported: Any) -> NodeEvent:

    tokens_in, tokens_out = _tokens_of(_reported(reported))
    return NodeEvent(
        node=node,
        started_at=started_at,
        duration_ms=round(duration_ms, 3),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        error=None,
    )


def trace_node(node: str) -> Callable[[_F], _F]:

    def decorator(fn: _F) -> _F:
        def _report(update: Any) -> Any:

            if isinstance(update, dict) and TOKEN_KEY in update:
                del update[TOKEN_KEY]
            return update

        def _elapsed(started: float) -> float:
            return (time.perf_counter() - started) * 1000.0

        def _record(state: GraphState, update: Any, started: float) -> Any:

            event = _event(node, time.time(), _elapsed(started), update)
            return _finish(state, _report(update), event)

        if inspect.iscoroutinefunction(fn):
            _afn = cast("Callable[[GraphState], Awaitable[Any]]", fn)

            async def async_wrapper(state: GraphState) -> Any:
                started = time.perf_counter()
                update: Any = None
                try:
                    update = await _afn(state)
                except BaseException as error:
                    _fail(state, error, _event(node, time.time(), _elapsed(started), None))
                    raise
                return _record(state, update, started)

            return cast("_F", async_wrapper)

        def wrapper(state: GraphState) -> Any:
            started = time.perf_counter()
            update: Any = None
            try:
                update = fn(state)
            except BaseException as error:
                _fail(state, error, _event(node, time.time(), _elapsed(started), None))
                raise
            return _record(state, update, started)

        return cast("_F", wrapper)

    return decorator

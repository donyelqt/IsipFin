"""Every graph node's instrumentation: ``@trace_node`` (ADR-007 §2).

This module wraps the node functions declared in ``src/chains/graph.py``. A traced node
is a drop-in replacement: the wrapper measures the node, records one
:class:`~src.chains.state.NodeEvent`, and returns the node's update unchanged.

Reducer/accretion convention
----------------------------
``GraphState.node_trace`` is a plain (reducer-less) list field, so if a node returned
``node_trace`` it would **overwrite** its siblings' events, and if no node returned it,
the accumulated events would be lost between super-steps. The decorator avoids both by
mutating the live list object that is already inside the state dict — which langgraph
preserves by identity across super-steps (verified on langgraph 1.2.14: the list
handed in as input is the same object the final ``invoke`` returns, for sync runs,
async runs, and mixed ones). It appends the event and, when a node wrongly returns
``node_trace`` in its partial update, replaces that entry with the accumulated list so
one node cannot erase another node's accounting. The state module documents that nodes
must never return ``node_trace``; the decorator enforces it defensively.

Token convention
----------------
Nodes call the LLM gateway (``build_llm(params).chat(...) -> ChatResult``) and get a
:class:`~src.utils.llm.ChatResult` back with ``prompt_tokens``/``completion_tokens``.
Rather than change what a node returns, a node that makes an LLM call reports its
accounting by including the ``ChatResult`` in its partial update under the reserved key
``TOKEN_KEY`` ("llm"); the wrapper lifts the counts into the ``NodeEvent`` and **strips
the key before the update reaches the state**.
tokens_in``/``tokens_out``. They are extracted from the :class:`~src.utils.llm.ChatResult` a
So a token-reporting node looks like::

    @trace_node("analyze_question")
    def analyze_question(state: GraphState) -> dict:
        result: ChatResult = ...
        return {TOKEN_KEY: result, "query_plan": plan}

Nothing about the node's return shape changes for non-LLM nodes, and a node may
return anything at all (see :ref:`non-dict` below). Recording tokens is opt-in: a node
that does not report a ``ChatResult`` contributes ``tokens_in = tokens_out = None``,
meaning "not measured" rather than zero.

.. _non-dict:

Non-dict node returns
---------------------
LangGraph's contract for a state-updating node is a dict of partial updates — a node
returning anything else raises ``langgraph.errors.InvalidUpdateError``. That is the
right behaviour to keep, so the wrapper does **not** attempt to wrap a non-dict into a
state update; a node must return a dict (an empty one when it wrote its results into
the state directly). The one thing the wrapper does guarantee is that tracing happens
*first*: a node whose update shape is wrong still gets a ``NodeEvent``, so the trace
never appears to skip a node that ran.

Exception policy (never swallow)
--------------------------------
``BaseException`` is caught only to record the typed error string in the event; it is
then re-raised unchanged, so ``KeyboardInterrupt``, ``asyncio.CancelledError`` and
langgraph's own control-flow exceptions (``GraphInterrupt`` and friends) propagate as
themselves — they are never converted into a node failure. ADR-007's taxonomy lives in
``src/utils/errors.py``; this module just serialises ``type(exc).__name__: exc`` into
the trace, because a trace row has to be readable after the exception object is long
gone.

"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from src.chains.state import GraphState, NodeEvent
from src.utils.llm import ChatResult

__all__ = ["TOKEN_KEY", "appended_records", "trace_node"]

#: Reserved partial-update key under which a node reports its gateway accounting.
#: The wrapper consumes it (and deletes it), so it never reaches state.
TOKEN_KEY = "llm"

#: The state key the decorator accrets into. Spelled as a literal at the two places it
#: is written, because mypy (and readers) need ``GraphState``'s own key name there —
#: a constant would be checked as an opaque string.
TRACE_KEY = "node_trace"

_F = TypeVar("_F", bound=Callable[..., Any])


def appended_records(state: GraphState) -> list[NodeEvent]:
    """The ``node_trace`` list to append to, creating it when absent.

    This is the one place the decorator depends on a langgraph detail: the list object
    found here is the same object later super-steps see, because the schema has no
    reducer on that key (see the module docstring). When ``state`` is a plain dict —
    which is how unit tests call traced nodes directly, and how a node sees the state
    before the first super-step has written the key — a missing ``node_trace`` is
    created here so the first event is not lost.
    """

    existing = state.get("node_trace")
    if not isinstance(existing, list):
        existing = []
        state["node_trace"] = existing
    return existing


def _tokens_of(result: Any) -> tuple[int | None, int | None]:
    """Read ``(prompt_tokens, completion_tokens)`` off a reported :class:`ChatResult`.

    Only a real :class:`~src.utils.llm.ChatResult` counts: a lookalike with the right
    attribute names would otherwise smuggle unverified numbers into the trace. A node
    reporting nothing yields ``(None, None)``.
    """

    if isinstance(result, ChatResult):
        return int(result.prompt_tokens), int(result.completion_tokens)
    return None, None


def _finish(state: GraphState, result: Any, event: NodeEvent) -> Any:
    """Record the successful node's event, then return the node's update untouched.

    A returned ``node_trace`` entry is replaced with the accumulated list — a node
    returning that key by mistake would otherwise wipe every other node's event.
    """

    appended_records(state).append(event)
    if isinstance(result, dict) and "node_trace" in result:
        result["node_trace"] = state["node_trace"]
    return result


def _fail(state: GraphState, error: BaseException, event: NodeEvent) -> None:
    """Record a failed node's event, then let the caller re-raise the original exception.

    ``NodeEvent`` is frozen (the state contract trades immutability for reuse across
    the debug panel, the store, and golden-QA snapshots), so the error field is set by
    rebuilding the event rather than by mutating it. The duration still measured up to
    the raise — a failure's cost is as worth recording as a success's.
    """

    from src.chains.state import replace_event

    recorded = replace_event(event, error=f"{type(error).__name__}: {error}")
    appended_records(state).append(recorded)


def _reported(result: Any) -> Any:
    """The token report a node's returned update carries, or ``None`` when it has none.

    A node reports its gateway accounting by putting the :class:`ChatResult` under
    :data:`TOKEN_KEY` in its partial update; the update itself stays whatever the node
    wanted to return. Reading the entry here rather than sniffing the whole update
    keeps the convention explicit: a node that returns a ``ChatResult`` under any
    other key contributes no token counts, because another key means the value is
    state, not accounting.
    """

    if isinstance(result, dict):
        return result.get(TOKEN_KEY)
    return None


def _event(node: str, started_at: float, duration_ms: float, reported: Any) -> NodeEvent:
    """Build the finished event, lifting token counts when the node reported them."""

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
    """Wrap a graph node in per-node measurement and trace recording (ADR-007).

    Args:
        node: The node's registered name, recorded in every event and mirrored into
            the ``events`` table. It is passed explicitly rather than derived from
            ``__name__`` so a node can be registered under a different name (a
            shared implementation serving two graph slots) without the trace lying.

    Returns:
        The wrapped node: same signature, same return value, plus one ``NodeEvent``
        per invocation appended to ``state["node_trace"]``. Sync and async nodes are
        both supported (``inspect.iscoroutinefunction`` picks the wrapper), which
        matters because four of the seven nodes await the gateway.
    """

    def decorator(fn: _F) -> _F:
        def _report(update: Any) -> Any:
            """Strip ``TOKEN_KEY`` from a returned update, keeping everything else.

            Mutates the node's own dict in place — the update was built by that node
            and is not shared, so removing the accounting key needs no copy.
            """

            if isinstance(update, dict) and TOKEN_KEY in update:
                del update[TOKEN_KEY]
            return update

        def _elapsed(started: float) -> float:
            return (time.perf_counter() - started) * 1000.0

        def _record(state: GraphState, update: Any, started: float) -> Any:
            """Record the finished event, then return the node's update for the state.

            The event is built from the update *before* ``TOKEN_KEY`` is stripped, so
            the counts are still visible to ``_reported`` when the event is built.
            """

            event = _event(node, time.time(), _elapsed(started), update)
            return _finish(state, _report(update), event)

        if inspect.iscoroutinefunction(fn):
            _afn = cast("Callable[[GraphState], Awaitable[Any]]", fn)

            async def async_wrapper(state: GraphState) -> Any:
                started = time.perf_counter()
                update: Any = None
                try:
                    update = await _afn(state)
                except BaseException as error:  # recorded, then re-raised unchanged
                    _fail(state, error, _event(node, time.time(), _elapsed(started), None))
                    raise
                return _record(state, update, started)

            return cast("_F", async_wrapper)

        def wrapper(state: GraphState) -> Any:
            started = time.perf_counter()
            update: Any = None
            try:
                update = fn(state)
            except BaseException as error:  # recorded, then re-raised unchanged
                _fail(state, error, _event(node, time.time(), _elapsed(started), None))
                raise
            return _record(state, update, started)

        return cast("_F", wrapper)

    return decorator

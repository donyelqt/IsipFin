from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from typing import Any

from src.chains.state import GraphState, QueryPlan
from src.utils.tracing import trace_node

__all__ = ["MAX_SUB_QUERIES", "NODE_NAME", "decompose_question"]

logger = logging.getLogger(__name__)

NODE_NAME = "decompose_question"

MAX_SUB_QUERIES = 4

_ABSENT_KIND = ""


def _sub_query_texts(value: Any) -> list[str]:

    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return []
    return [entry for entry in value if isinstance(entry, str)]


def _deduplicated(entries: Sequence[str]) -> list[str]:

    seen: set[str] = set()
    ordered: list[str] = []
    for entry in entries:
        text = entry.strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


@trace_node(NODE_NAME)
def decompose_question(state: GraphState) -> dict[str, Any]:

    plan = state.get("query_plan")
    deduplicated = _deduplicated(
        _sub_query_texts(plan.sub_queries) if isinstance(plan, QueryPlan) else []
    )
    queries = deduplicated[:MAX_SUB_QUERIES]
    dropped = len(deduplicated) - len(queries)
    if dropped:
        logger.warning(
            "%s dropped %d sub-quer%s of %d to stay under the %d sub-query fan-out cap "
            "(ADR-001's bounded fan-out, STRESS-TEST-RAG #6)",
            NODE_NAME,
            dropped,
            "y" if dropped == 1 else "ies",
            len(deduplicated),
            MAX_SUB_QUERIES,
        )

    decomposed = plan if isinstance(plan, QueryPlan) else QueryPlan(kind=_ABSENT_KIND)
    return {"query_plan": dataclasses.replace(decomposed, sub_queries=queries)}

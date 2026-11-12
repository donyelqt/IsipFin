from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from src.chains.state import GraphState, QueryPlan
from src.core.chunking import estimate_tokens
from src.core.retrieval import RRF_K, ScoredChunk, retrieve
from src.utils.tracing import trace_node

__all__ = [
    "CONTEXT_TOKEN_BUDGET",
    "MAX_RETRIES",
    "NODE_NAME",
    "RETRIEVE_K",
    "SUB_QUERY_LIMIT",
    "TAU",
    "best_fused_score",
    "hybrid_retrieve",
    "merge_passes",
    "pack_passages",
]

logger = logging.getLogger(__name__)

NODE_NAME = "hybrid_retrieve"

TAU = 0.02

MAX_RETRIES = 1

SUB_QUERY_LIMIT = 4

RETRIEVE_K = 8

CONTEXT_TOKEN_BUDGET = 6000

_FILTER_KEY = "section"

_PATH_SEPARATOR = " > "


@dataclasses.dataclass(frozen=True)
class _Passage:

    chunk: ScoredChunk
    score: float
    hits: int


def _parent_text(chunk: ScoredChunk) -> str:

    return chunk.parent_text if chunk.parent_text.strip() else chunk.child_text


def _leg_ranks(leg: Mapping[str, float]) -> dict[str, int]:

    ranked = sorted(
        (chunk_id for chunk_id, score in leg.items() if score > 0.0),
        key=lambda chunk_id: (-leg[chunk_id], chunk_id),
    )
    return {chunk_id: rank for rank, chunk_id in enumerate(ranked, start=1)}


def _fused_scores(hits: Sequence[ScoredChunk]) -> dict[str, float]:

    dense_ranks = _leg_ranks({chunk.chunk_id: chunk.dense_score for chunk in hits})
    sparse_ranks = _leg_ranks({chunk.chunk_id: chunk.sparse_score for chunk in hits})
    scores: dict[str, float] = {}
    for chunk in hits:
        ranks = (dense_ranks.get(chunk.chunk_id), sparse_ranks.get(chunk.chunk_id))
        scores[chunk.chunk_id] = sum(1.0 / (RRF_K + rank) for rank in ranks if rank is not None)
    return scores


def best_fused_score(passes: Sequence[Sequence[ScoredChunk]]) -> float:

    return max(
        (score for hits in passes for score in _fused_scores(hits).values()),
        default=0.0,
    )


def _best_hit(
    current: ScoredChunk, candidate: ScoredChunk, current_score: float, candidate_score: float
) -> ScoredChunk:

    if candidate_score > current_score:
        return candidate
    if candidate_score < current_score:
        return current
    return candidate if candidate.chunk_id < current.chunk_id else current


def merge_passes(passes: Sequence[Sequence[ScoredChunk]]) -> list[ScoredChunk]:

    merged: dict[tuple[str, str], _Passage] = {}
    for hits in passes:
        scores = _fused_scores(hits)
        for chunk in hits:
            key = (chunk.doc_id, _parent_text(chunk))
            score = scores.get(chunk.chunk_id, 0.0)
            passage = merged.get(key)
            if passage is None:
                merged[key] = _Passage(chunk=chunk, score=score, hits=1)
                continue
            merged[key] = _Passage(
                chunk=_best_hit(passage.chunk, chunk, passage.score, score),
                score=max(passage.score, score),
                hits=passage.hits + 1,
            )
    ordered = sorted(
        merged.values(),
        key=lambda passage: (
            -passage.score,
            -passage.hits,
            passage.chunk.doc_id,
            passage.chunk.page,
            passage.chunk.chunk_id,
        ),
    )
    return [
        dataclasses.replace(passage.chunk, fused_rank=rank) for rank, passage in enumerate(ordered)
    ]


def pack_passages(
    chunks: Sequence[ScoredChunk], budget: int = CONTEXT_TOKEN_BUDGET
) -> tuple[list[ScoredChunk], int, int]:

    packed: list[ScoredChunk] = []
    used = 0
    dropped = 0
    for chunk in chunks:
        cost = estimate_tokens(_parent_text(chunk))
        if packed and used + cost > budget:
            dropped += 1
            continue
        packed.append(dataclasses.replace(chunk, fused_rank=len(packed)))
        used += cost
    return packed, dropped, used


def _budget_note(kept: int, dropped: int, used: int, budget: int) -> str:

    if dropped == 0:
        return (
            f"context budget: all {kept} merged parent passage(s) kept "
            f"({used} of {budget} estimated tokens)"
        )
    return (
        f"context budget: kept {kept} of {kept + dropped} merged parent passage(s) "
        f"({used} of {budget} estimated tokens); {dropped} dropped to stay under the cap, "
        "so the context shown may be truncated"
    )


def _question(state: GraphState) -> str:

    for key in ("standalone_question", "question"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _unique(queries: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for query in queries:
        if isinstance(query, str) and query.strip() and query.strip() not in seen:
            seen.add(query.strip())
            ordered.append(query.strip())
    return ordered


def _queries(state: GraphState) -> list[str]:

    plan = state.get("query_plan")
    planned = plan.sub_queries if isinstance(plan, QueryPlan) else []
    queries = _unique(planned)[:SUB_QUERY_LIMIT]
    if not queries:
        single = _question(state)
        if not single:
            raise ValueError(
                "hybrid_retrieve has nothing to retrieve: the plan names no sub-queries "
                "and the state carries no question"
            )
        queries = [single]
    return queries


def _section_filter(state: GraphState) -> dict[str, str] | None:

    plan = state.get("query_plan")
    if not isinstance(plan, QueryPlan):
        return None
    for name in plan.required_sections:
        if not isinstance(name, str):
            continue
        head = name.split(_PATH_SEPARATOR)[0].strip()
        if head:
            return {_FILTER_KEY: head}
    return None


def _retrieve_all(
    queries: Sequence[str], filters: dict[str, str] | None
) -> list[list[ScoredChunk]]:

    return [retrieve(query, k=RETRIEVE_K, filters=filters) for query in queries]


def _rewritten_queries(queries: Sequence[str], state: GraphState) -> list[str]:

    question = _question(state)
    return _unique([question] if question else list(queries))[:SUB_QUERY_LIMIT]


def _retries_spent(state: GraphState) -> int:

    raw = state.get("retries")
    try:
        spent = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        spent = 0
    return max(0, spent)


@trace_node(NODE_NAME)
def hybrid_retrieve(state: GraphState) -> dict[str, Any]:
    queries = _queries(state)
    filters = _section_filter(state)
    passes = _retrieve_all(queries, filters)
    retries = _retries_spent(state)

    if best_fused_score(passes) < TAU and retries < MAX_RETRIES:
        retries += 1
        logger.warning(
            "%s is below the score gate (best fused score %.6f < tau %.2f): running the one "
            "bounded rewrite (ADR-002)",
            NODE_NAME,
            best_fused_score(passes),
            TAU,
            extra={"node": NODE_NAME},
        )
        passes = [*passes, *_retrieve_all(_rewritten_queries(queries, state), None)]

    packed, dropped, used = pack_passages(merge_passes(passes))
    note = _budget_note(kept=len(packed), dropped=dropped, used=used, budget=CONTEXT_TOKEN_BUDGET)
    if dropped:
        logger.warning(
            "%s dropped %d of %d merged parent passage(s) to stay under the %d-token "
            "context budget (STRESS-TEST-RAG #6)",
            NODE_NAME,
            dropped,
            len(packed) + dropped,
            CONTEXT_TOKEN_BUDGET,
            extra={"node": NODE_NAME},
        )
    return {"retrieved": packed, "retries": retries, "retrieval_note": note}

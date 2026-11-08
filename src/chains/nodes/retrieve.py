"""``hybrid_retrieve``: the fan-out, fusion, score-gate and budget node (task T11).

The one retrieval node every answering path runs through. It fans out over the plan's
sub-queries (or the single question when the plan is simple), fuses what comes back into
one ranking, applies ADR-002's score gate ``tau`` with its single bounded rewrite retry,
and packs the merged parents inside STRESS-TEST-RAG #6's global context budget so a wide
multi-part question cannot overflow the model's window.

What the merge has to solve
---------------------------
Each ``retrieve`` call ranks inside its **own filtered candidate set**, and re-ranks from
zero once a section filter applies (``test_section_filter_reranks_from_zero``). So four
sub-queries produce four *unrelated* rank orders: a rank-0 hit from a two-chunk section is
not the rank-0 hit from the whole corpus. The merge therefore rebuilds every hit's RRF
score from the two legs it carries (``_fused_scores``) and re-ranks the union from zero,
which is also what makes ``retrieved`` readable as one ranking in the debug panel.

``retrieve`` reports each hit's two leg scores and its ``fused_rank`` — the ranks are lost,
so the score the ``tau`` gate needs is reconstructed from the legs. Reconstruction is
**conservative and un-exact in a specific, known direction**: a hit's rank in a leg can only
be recovered from the hits that leg returned, so a hit that a leg never surfaced is treated
as unranked and contributes no term for that leg. The reconstruction therefore never
*over*-scores a passage, and the recovered best-hit scores (a rank-1-on-both-legs hit, a
rank-1-on-one-leg hit) match ``retrieve``'s own arithmetic exactly whenever the returned
hits cover the ranked candidates. Ties and near-ties across sub-queries may reorder, which
is harmless: what the gate compares is the best score, not the 7th.

Parents, not children, are the merged unit. Several children of one parent expand to the
same ``(doc_id, parent_text)`` passage, so shipping the passage once per sub-query would
spend the context budget on repeated text — the opposite of what the fan-out is for.
ADR-002 rejects MMR for the same reason ("we *want* redundancy across sub-queries");
redundancy across sub-queries is wanted, duplication *inside* the packed context is not.

The single bounded retry
------------------------
ADR-002's parameter table freezes ``tau = 0.02`` (RRF units) and ``rewrite retries = 1``.
When the best fused score is below the gate the node rewrites **deterministically** — the
whole standalone question as one query, with the section filter dropped — and re-retrieves
once. No LLM call: a retry that costs a model call is a second inference the gate cannot
afford, and the rewrite's job is recall, not phrasing. The rewrite broadens in both
directions at once, because the two ways a first pass misses are a fragmentary sub-query
and a section guess that names something the corpus does not contain.

``retries`` is written as the number of rewrites this run has spent (ADR-001: "a counted
state field, not an unbounded edge"), so the router's self-loop can never fire twice: it
reads the state *after* this update and passes once ``retries >= 1``.

No LLM, no network
------------------
This node is entirely symbolic — it never touches the gateway, so ``MOCK_LLM=1`` needs
nothing from it and no ``openai`` import is reachable from here. The gateway is only ever
reached through ``build_llm``; the index is reached through ``retrieve``.
"""

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

#: The node's registered name, in ``NODE_NAMES``, in every ``NodeEvent``, and the key this
#: module's own log lines use.
NODE_NAME = "hybrid_retrieve"

#: ADR-002's score gate. The best fused score a retrieval must clear, in RRF units
#: (``1 / (RRF_K + rank)`` sums). Below it the one bounded rewrite retry fires.
TAU = 0.02

#: ADR-002's rewrite-retry bound: exactly one, never a loop.
MAX_RETRIES = 1

#: Fan-out cap (STRESS-TEST-RAG #6, ADR-001's "max 4"): sub-queries past this are not run,
#: so a long question cannot quietly become seven retrievals.
SUB_QUERY_LIMIT = 4

#: Children retrieved per sub-query — ADR-002's fused top-k, each already expanded to its
#: parent by ``retrieve``.
RETRIEVE_K = 8

#: Global context budget in estimated tokens (STRESS-TEST-RAG #6: "parents are packed
#: greedily by fused score until a token budget (default ~6k tokens) is hit, remainder
#: dropped with a note"). Strictly the parents: the question, parts and system prompt sit
#: outside it, and ``synthesize_answer`` re-packs whatever ships, so neither layer can
#: overflow the window.
CONTEXT_TOKEN_BUDGET = 6000

#: The only filter key this node ever builds. ``retrieve`` accepts exactly
#: ``("section", "doc_id")`` and raises ``ValueError`` on anything else, so no other key is
#: constructed here — a typo'd filter key is a loud caller bug, never a silently dropped one.
_FILTER_KEY = "section"

#: How ``analyze_question`` renders a nested section path (its `_section_text`): the
#: planner's ``required_sections`` therefore hold ``"MD&A > Results of Operations"``.
#: ``retrieve`` matches an element of a record's section path, so the head is taken.
_PATH_SEPARATOR = " > "

#: Token estimation is ``estimate_tokens`` (``max(1, len(text) // 4)``), the same offline
#: approximation ``chunking`` uses: no tokenizer download, identical on every machine.


@dataclasses.dataclass(frozen=True)
class _Passage:
    """One parent passage seen across the fan-out: its best hit, best score, and hit count.

    ``hits`` counts the sub-queries that retrieved the passage. It orders *ties* only — a
    passage several sub-queries independently found is stronger evidence than one of equal
    score a single sub-query found — so the merge rewards redundancy the way ADR-002 says
    the corpus wants without inflating scores.
    """

    chunk: ScoredChunk
    score: float
    hits: int


# --- leg reconstruction (the fused score ``tau`` is compared against) -----------------


def _parent_text(chunk: ScoredChunk) -> str:
    """The passage a hit contributes: its parent, else the child it came from.

    Expansion is what ADR-002 sends to the model, so the parent is the passage. The child
    fallback keeps a record with an empty parent from costing nothing and vanishing.
    """

    return chunk.parent_text if chunk.parent_text.strip() else chunk.child_text


def _leg_ranks(leg: Mapping[str, float]) -> dict[str, int]:
    """1-based rank per hit for one retrieval leg, descending by score, ``chunk_id`` breaking ties.

    Only a **positive** score earns a rank. ``retrieve`` reports ``0.0`` for a leg this index
    does not have (a BM25-only index's dense leg) as well as for a genuinely unscored hit,
    and the two are indistinguishable here — so the leg is treated as absent for those hits
    rather than inventing a rank. This is what makes the reconstruction conservative: the
    one-hit-per-``chunk_id`` ranking it produces matches ``_fuse``'s arithmetic exactly when
    the returned hits cover the ranked candidates, and under-counts otherwise.
    """

    ranked = sorted(
        (chunk_id for chunk_id, score in leg.items() if score > 0.0),
        key=lambda chunk_id: (-leg[chunk_id], chunk_id),
    )
    return {chunk_id: rank for rank, chunk_id in enumerate(ranked, start=1)}


def _fused_scores(hits: Sequence[ScoredChunk]) -> dict[str, float]:
    """RRF fused score per hit: ``Σ 1 / (RRF_K + rank)`` over the legs that scored it.

    Same arithmetic as :func:`src.core.retrieval._fuse`, evaluated on the hits this node can
    see because ``retrieve`` returns the legs' raw scores but not their ranks. It reproduces
    the frozen score exactly whenever the returned hits cover the ranked candidates, and
    under-counts (never over-counts) when they do not — a leg absent for a hit adds nothing,
    which is the direction that prefers a bounded retry over a confident wrong answer.
    """

    dense_ranks = _leg_ranks({chunk.chunk_id: chunk.dense_score for chunk in hits})
    sparse_ranks = _leg_ranks({chunk.chunk_id: chunk.sparse_score for chunk in hits})
    scores: dict[str, float] = {}
    for chunk in hits:
        ranks = (dense_ranks.get(chunk.chunk_id), sparse_ranks.get(chunk.chunk_id))
        scores[chunk.chunk_id] = sum(1.0 / (RRF_K + rank) for rank in ranks if rank is not None)
    return scores


def best_fused_score(passes: Sequence[Sequence[ScoredChunk]]) -> float:
    """The best fused score across the passes — the value the ``tau`` gate compares."""

    return max(
        (score for hits in passes for score in _fused_scores(hits).values()),
        default=0.0,
    )


# --- the merge -----------------------------------------------------------------------


def _best_hit(
    current: ScoredChunk, candidate: ScoredChunk, current_score: float, candidate_score: float
) -> ScoredChunk:
    """Which of two hits of one passage represents it: higher score, else lower ``chunk_id``.

    Breaking the tie on ``chunk_id`` is what makes the merge total — arrival order across
    sub-queries cannot decide which child stands for its parent.
    """

    if candidate_score > current_score:
        return candidate
    if candidate_score < current_score:
        return current
    return candidate if candidate.chunk_id < current.chunk_id else current


def merge_passes(passes: Sequence[Sequence[ScoredChunk]]) -> list[ScoredChunk]:
    """Fuse the per-sub-query hits into one de-duplicated, deterministically ordered list.

    Every pass is scored on its own terms, the best score a passage earned anywhere is kept
    (a passage several sub-queries retrieved is not demoted by the weakest of them), and the
    union is re-ranked from zero so ``retrieved`` is a single total order rather than four
    unrelated ones. The returned chunks' ``fused_rank`` are 0-based and contiguous, matching
    what ``retrieve`` guarantees for one call.

    Ordering is ``(-score, -hits, doc_id, page, chunk_id)``: score descending, then the
    passages the fan-out agreed on, then identity fields that make the order total.
    """

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


# --- the context budget --------------------------------------------------------------


def pack_passages(
    chunks: Sequence[ScoredChunk], budget: int = CONTEXT_TOKEN_BUDGET
) -> tuple[list[ScoredChunk], int, int]:
    """Greedily pack the merged passages until the global context budget is reached.

    ``chunks`` arrives in fused-score order, so the walk is "take the next passage while it
    fits". A passage that does not fit is **skipped, not terminal**: parents are ~1000 tokens
    (ADR-002), and letting one long parent stop the walk would leave budget unused by shorter
    passages that still fit — which is what keeps a wide multi-part question inside the cap.

    The top-ranked passage always packs, even when it alone exceeds the budget. Retrieval
    that silently returns nothing is the empty-answer path ADR-007 forbids, and ADR-002's
    rule that a table is never cut to fit means an oversized parent cannot be honestly
    shrunk. The caller's budget is what bounds the *candidate* set — ``RETRIEVE_K`` per
    sub-query and the fan-out cap — so this does not become an unbounded escape hatch: a
    passage this node sees is already inside the graph's own bounds, and the over-budget
    case is one parent, not a corpus.

    Returns ``(packed, dropped, used)``: the passages that ship in ``retrieved``, re-ranked
    from zero; how many merged passages did not fit; and the estimated tokens the packed
    passages cost, for the note and the log. ``used`` can exceed ``budget`` only by the top
    passage, which is the deliberate over-budget case above.
    """

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
    """The state's note on what the budget kept and dropped (STRESS-TEST-RAG #6's "note")."""

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


# --- planning inputs -----------------------------------------------------------------


def _question(state: GraphState) -> str:
    """The question retrieval runs on: the standalone rewrite, else the raw question.

    ADR-006 rule 2 — the rewritten question, not raw history, is what downstream nodes see —
    so the standalone rewrite wins when a previous node produced one.
    """

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
    """The queries to fan out over: the plan's sub-queries, else the single question.

    The cap is applied here as well as in ``decompose_question`` because the plan is a
    model-produced artifact and this node is the last thing before the retrieval cost:
    T11 owns the "≤ 4 retrievals per question" bound, so it does not delegate it.

    Raises:
        ValueError: When no query can be built — neither a usable sub-query plan nor a
            non-empty question. There is nothing to retrieve, which is a caller bug
            (``graph.initial_state`` forbids an empty question), not a result.
    """

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
    """The plan's section filter, or ``None`` when it names no usable section.

    ``required_sections`` is a list of section names, while ``retrieve``'s ``section``
    filter matches one name against a record's ``section_path`` — so exactly one is passed:
    the first usable one, which is the name the planner ranked first (P1: "the document
    section names that plausibly contain the answer"). Picking arbitrarily among several
    would make the retrieval depend on an order nobody chose; the gate's bounded retry is
    what recovers a section guess that matches nothing.

    A path arrives as the planner rendered it — ``"MD&A > Results of Operations"`` for a
    nested path (``analyze_question._section_text``, which emits every catalog section that
    way) — and ``retrieve`` matches each *element* of a record's section path, so a joined
    path would match nothing. Splitting on ``" > "`` takes the first element, the coarsest
    heading, which is both a genuine section name and the one most likely to hit: a
    deep-nested path names content the query may not need, and the fan-out already covers
    the rest.

    Entries that are not usable strings are skipped rather than raised on: the plan is a
    schema-validated model reply, and a blank section name is not worth losing the turn over
    (``retrieve`` would raise ``ValueError`` for one). No other key is ever built.
    """

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
    """One retrieval pass per query, in order — the bounded fan-out."""

    return [retrieve(query, k=RETRIEVE_K, filters=filters) for query in queries]


def _rewritten_queries(queries: Sequence[str], state: GraphState) -> list[str]:
    """The one deterministic rewrite: the whole question, as a single query.

    Decomposed fragments are the thing most likely to have under-retrieved, so the rewrite
    goes back to the richest formulation the state has. When there is no question text the
    queries are kept unchanged, so the retry still re-runs rather than becoming a no-op
    policy of its own.
    """

    question = _question(state)
    return _unique([question] if question else list(queries))[:SUB_QUERY_LIMIT]


def _retries_spent(state: GraphState) -> int:
    """How many rewrites the run has already spent, read the way the router reads it."""

    raw = state.get("retries")
    try:
        spent = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        spent = 0
    return max(0, spent)


# --- the node ------------------------------------------------------------------------


@trace_node(NODE_NAME)
def hybrid_retrieve(state: GraphState) -> dict[str, Any]:
    """Fan out, fuse, gate and pack — the node every answering path runs through.

    Reads ``query_plan`` (``sub_queries`` for the fan-out, ``required_sections`` for the
    section filter) and the question text; writes ``retrieved`` and ``retries``.

    The gate compares the best fused score against ``TAU``; below it, one deterministic
    rewrite and re-retrieval happens and ``retries`` becomes 1. A second graph visit with
    ``retries >= 1`` therefore cannot retry again, which is what makes ADR-001's "exactly
    one rewrite, never an unbounded loop" hold by construction rather than by convention.

    Raises:
        ValueError: When the state has no question and no usable sub-queries.
        ValueError: From ``retrieve`` for a filter key outside ``("section", "doc_id")``;
            this node only ever builds ``"section"``, so it is a caller bug, never a
            silently ignored filter.

    Returns:
        ``{"retrieved": [...], "retries": <rewrites spent>, "retrieval_note": <note>}`` —
        the merged, re-ranked, budget-packed passages; the count that bounds the router's
        self-loop; and STRESS-TEST-RAG #6's note on what the budget kept and dropped.

    ``retrieval_note`` is not a ``GraphState`` key — that schema is frozen by
    ``docs/ARCHITECTURE.md`` and owned by T9 — so ``langgraph`` drops the key on the way
    out of a graph run. It is returned for direct callers (unit tests, the debug panel)
    and the same line is logged, which is the channel that survives a full run: a
    silently truncated context must be visible somewhere durable.
    """
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
        # Broadened on both axes at once: the whole question instead of fragments, and no
        # section constraint — a plan-named section the corpus does not contain is the other
        # common way a first pass misses.
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

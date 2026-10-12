"""chunking-index retrieval: hybrid dense + sparse fusion with parent expansion.

Design (ADR-002, docs/ARCHITECTURE.md query pipeline, docs/SPEC.md section 9):

1. **Two retrievers, two scales.** Dense (FAISS ``IndexFlatIP`` over L2-normalized
   vectors, so inner product *is* cosine) catches paraphrases; sparse (``rank_bm25``
   ``BM25Okapi`` over ``\\w+`` tokens) catches exact line items — "operating income",
   "Note 13". They are never blended by weighted sum: cosine is bounded to [0, 1] and
   BM25 is unbounded, so a weight would twitch per query. They are fused by *rank
   instead* — Reciprocal Rank Fusion, free of scale.

2. **RRF.** For a candidate ``d`` retrieved by retriever ``i`` at 1-based rank ``rank_i(d)``:

       fused(d) = Σ_i 1 / (RRF_K + rank_i(d))     with RRF_K = 60

   Only documents a retriever actually *returns* (its top ``k``) receive a term; a
   document found by one retriever still scores on that term alone (a single-list hit
   is never diluted away). A retriever that returns nothing — an empty index, or a
   query with no matching tokens — simply contributes no terms, which is why there is no
   division-by-zero path: every ``1/(60+rank)`` denominator is at least 61.

   **Zero scores still rank.** A query whose BM25 tokens match nothing returns all-zero
   scores, not an error; the descending sort then assigns real ranks to those zeros and
   each document earns ``1/(60+rank)``. That is the ADR-002 formula taken literally over
   the full corpus, and it is deliberately *retained* here rather than filtered: the
   manifestation of a zero-BM25 query is that dense ranks alone decide fusion, which is
   the desired behavior. The dense leg is likewise consulted whenever an embedder is
   available, so a query with no sparse match is still answered by similarity.

3. **Parent expansion AFTER fusion.** Ranking is decided at **child** granularity
   (~250-token windows), then the surviving children are expanded to their parent
   payload. Ranking parents directly would let an enormous parent win on a single lucky
   sentence; ranking children keeps one vote per evidence window. Fusion therefore runs
   to completion first, and the parent's ``text`` is attached afterwards — only for
   chunks that survive to the top ``k``.

4. **Filtering before ranking.** ``filters`` restrict the candidate set (``section`` —
   a substring match against any element of ``section_path`` — and ``doc_id``), then the
   dense/sparse legs rank only what survives, because metadata applied *after* ranking
   would silently shrink the result count and could empty it.

5. **Determinism.** Ties in fused score break by ``chunk_id`` ascending, ``fused_rank``
   is 0-based, and every ordering is a stable sort of that same key — so identical
   indexes yield byte-identical ``ScoredChunk`` lists. No randomness, no clock, no
   iteration-order dependence.

Artifacts (what carries what) — see ``src/core/index.py``:

- ``chunks.jsonl`` is the **source of truth**: texts and provenance.
- FAISS holds **vectors only**; the BM25 pickle holds **token lists only**.
- The embedder is injected and **never persisted**; this module builds no OpenAI client.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.index import ChunkIndex, Embedder
from src.core.pdf_parser import ParsedDocument

# RRF damping constant (ADR-002 parameter table). 60 is the value the ADR freezes:
# large enough that the top and second retrievers' disagreements are dampened, small
# enough that being retrieved at rank 1 beats anywhere on the other list.
RRF_K = 60
# Default fused top-k: 8 children, then expansion to their parents (ADR-002).
DEFAULT_K = 8
# Supported metadata filters; anything else is a caller bug, not silently ignored.
_FILTER_KEYS = ("section", "doc_id")

# Module-level default index location (docs/ARCHITECTURE.md data model).
_INDEX_DIR = Path("data/index")

# Process-wide index, resolved on the first module-level call (built/loaded earlier in
# this process wins; otherwise the disk is read once). Index state is immutable, so the
# singleton only saves a reload per turn.
_INDEX: ChunkIndex | None = None
_INDEX_LOADED = False


# --- frozen contract (docs/SPEC.md section 9) ------------------------------------


@dataclass(frozen=True)
class ScoredChunk:
    """One child hit, annotated with its parent payload and fused rank.

    ``child_text`` is what was scored; ``parent_text`` is the expansion the caller should
    send to the LLM. ``fused_rank`` is 0-based, 0 for the top hit.
    """

    chunk_id: str
    child_text: str
    parent_text: str
    page: int
    section_path: list[str]
    doc_id: str
    dense_score: float
    sparse_score: float
    fused_rank: int


# --- fusion ----------------------------------------------------------------------


def _ranks(scores: np.ndarray | None, size: int | None = None) -> np.ndarray:
    """1-based rank per record for ``scores``, descending, index order breaking ties.

    The result is as long as ``scores`` (or ``size`` when given) and position ``i`` is
    record ``i``, so :func:`_fuse` can index it directly. A record the leg did not
    retrieve — an unavailable leg (``None``), a leg with nothing to rank, or a ``-inf``
    mask from :func:`_top_k` — gets rank ``0``, which :func:`_fuse` skips: that leg
    contributes no term for it, and a document found by one retriever still scores on
    that term alone. Ties keep record order (``kind="stable"``), so ranks are a
    deterministic function of the scores.
    """
    if scores is None:
        return np.zeros(0 if size is None else size, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    length = scores.size if size is None else size
    ranks = np.zeros(length, dtype=np.int64)
    if scores.size == 0 or length == 0:
        return ranks
    retrieved = np.flatnonzero(np.isfinite(scores))
    if retrieved.size == 0:
        return ranks
    order = retrieved[np.argsort(-scores[retrieved], kind="stable")]
    ranks[order] = np.arange(1, order.size + 1, dtype=np.int64)
    return ranks


def _fuse(dense: np.ndarray | None, sparse: np.ndarray | None, size: int) -> np.ndarray:
    """RRF-fused score per record: ``Σ_i 1/(RRF_K + rank_i)``, skipping unretrieved.

    A ``0`` rank means this leg did not retrieve that document, so its term is skipped
    rather than counted as a bogus ``1/RRF_K``. Every counted denominator is therefore
    ``RRF_K + rank >= 61``, which is also why there is no division-by-zero path. A
    document retrieved by a single leg keeps that leg's full term — a single-list hit is
    never diluted away.
    """
    total = np.zeros(size, dtype=np.float64)
    for ranks in (_ranks(dense, size), _ranks(sparse, size)):
        contributed = ranks > 0
        if not contributed.any():
            continue
        total[contributed] += 1.0 / (RRF_K + ranks[contributed].astype(np.float64))
    return total


def _order(fused: np.ndarray, records: Iterable) -> list[int]:
    """Record positions sorted by fused score desc, then ``chunk_id`` ascending.

    The explicit tie-break makes the order total, so ``fused_rank`` is stable even when
    two children earn identical fused scores.
    """
    keyed = [(-float(fused[index]), record.chunk_id, index) for index, record in enumerate(records)]
    keyed.sort()
    return [index for _, _, index in keyed]


# --- filtering -------------------------------------------------------------------


def _filter_positions(index: ChunkIndex, filters: dict | None) -> list[int]:
    """Positions of records surviving ``filters``; unknown keys raise ``ValueError``.

    ``section`` matches when ``query`` is contained in any element of the record's
    ``section_path`` (so ``"Results"`` covers ``["MD&A", "Results of Operations"]``);
    ``doc_id`` is an exact match. An empty ``section``/``doc_id`` string would match
    nearly nothing or everything, so it is rejected rather than honored.
    """
    if not filters:
        return list(range(len(index)))
    unknown = [key for key in filters if key not in _FILTER_KEYS]
    if unknown:
        raise ValueError(
            f"unknown filter key(s) {sorted(unknown)}; supported: {sorted(_FILTER_KEYS)}"
        )
    section = filters.get("section")
    if section is not None and not isinstance(section, str):
        raise ValueError("filter 'section' must be a string")
    doc_id = filters.get("doc_id")
    if doc_id is not None and not isinstance(doc_id, str):
        raise ValueError("filter 'doc_id' must be a string")
    if section is not None and not section.strip():
        raise ValueError("filter 'section' must be a non-empty string")
    if doc_id is not None and not doc_id.strip():
        raise ValueError("filter 'doc_id' must be a non-empty string")

    kept: list[int] = []
    for position, record in enumerate(index.records):
        if section is not None and section not in record.section_path:
            continue
        if doc_id is not None and record.doc_id != doc_id:
            continue
        kept.append(position)
    return kept


def _resolve_k(k: int) -> int:
    """Validate the requested fused top-k (must be a positive integer)."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(f"k must be an integer, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return k


# --- the hybrid retriever --------------------------------------------------------


class HybridRetriever:
    """Hybrid retrieval over one immutable ``ChunkIndex``.

    Both legs pull their own top ``k`` (fetching more of each cannot change the fused
    order's head here, since each leg contributes its top ``k`` candidates by
    construction), the ranks are RRF-fused, and the fused top ``k`` children are
    expanded to parents. The query text is embedded once per call.
    """

    def __init__(self, index: ChunkIndex) -> None:
        self._index = index

    @property
    def index(self) -> ChunkIndex:
        """The underlying chunk index (records, vectors, token lists)."""
        return self._index

    def retrieve(
        self, query: str, k: int = DEFAULT_K, filters: dict | None = None
    ) -> list[ScoredChunk]:
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        limit = _resolve_k(k)
        if not query.strip():
            return []

        # Filter BEFORE ranking: the legs then rank exactly the candidate set.
        positions = _filter_positions(self._index, filters)
        if not positions:
            return []
        candidates = self._index.subset(positions)
        records = candidates.records

        # Raw leg scores are reported verbatim; only the ranking view is truncated to
        # each leg's top k, so a masked position never shows up as a score.
        raw_dense = candidates.dense_scores(query)
        raw_sparse = candidates.sparse_scores(query)
        out_of_band = np.full(len(records), -np.inf, dtype=np.float64)

        # Pull top-k from each leg, at child granularity, then fuse ranks.
        per_leg = min(limit, len(records))
        fused = _fuse(
            _top_k(raw_dense, per_leg, len(records)),
            _top_k(raw_sparse, per_leg, len(records)),
            len(records),
        )
        ordered = _order(fused, records)[:limit]

        # Parent expansion AFTER fusion: only survivors are expanded, so a huge parent
        # can never outrank a small one on its size.
        dense_values = raw_dense if raw_dense is not None else out_of_band
        sparse_values = raw_sparse if raw_sparse is not None else out_of_band
        return [
            self._to_scored(records, position, dense_values, sparse_values, rank)
            for rank, position in enumerate(ordered)
        ]

    @classmethod
    def _to_scored(
        cls,
        records: tuple,
        position: int,
        dense: np.ndarray,
        sparse: np.ndarray,
        rank: int,
    ) -> ScoredChunk:
        """One fused child hit, expanded to its parent payload after fusion.

        An unavailable leg reports ``0.0`` — the absence of a score, not a score of
        zero — so callers cannot mistake a BM25-only index for a genuine match.
        """
        record = records[position]
        dense_score = float(dense[position])
        sparse_score = float(sparse[position])
        return ScoredChunk(
            chunk_id=record.chunk_id,
            child_text=record.child_text,
            parent_text=record.parent_text,
            page=record.page,
            section_path=list(record.section_path),
            doc_id=record.doc_id,
            dense_score=dense_score if dense_score != -np.inf else 0.0,
            sparse_score=sparse_score if sparse_score != -np.inf else 0.0,
            fused_rank=rank,
        )


def _top_k(scores: np.ndarray | None, k: int, size: int) -> np.ndarray | None:
    """Ranking view of ``scores`` where only this leg's top ``k`` remain rankable.

    Positions outside the top ``k`` become ``-inf``, so they sort below every genuinely
    retrieved document — including one whose score is *negative* (real embeddings have
    negative components, so cosine can be negative; masking with ``0.0`` would wrongly
    rank those above an unseen document). ``None`` passes through unchanged (leg
    unavailable); a short list passes through whole (``k >= size``).

    The full-length layout keeps ranks aligned with record positions for :func:`_fuse`.
    Scores are unmodified where they survive: a query matching nothing keeps its genuine
    zeros, which still rank per the "zero scores still rank" policy in the module docstring.
    """
    if scores is None or scores.size == 0:
        return scores
    scores = np.asarray(scores, dtype=np.float64)
    if k >= size:
        return scores
    order = np.argsort(-scores, kind="stable")[:k]
    masked = np.full(size, -np.inf, dtype=np.float64)
    masked[order] = scores[order]
    return masked


# --- index lifecycle -------------------------------------------------------------


def build_index(doc: ParsedDocument, embedder: Embedder | None = None) -> HybridIndexHandle:
    """Index one document into a fresh handle (idempotent by ``doc_id``).

    The document's chunks are chunked by ``src/core/chunking.py`` and embedded through
    the injected ``embedder`` (``None`` builds a BM25-only index). The result stays in
    memory — it becomes this process's current index, and the caller persists it with
    ``handle.save(dir)`` when it wants an on-disk index. Re-indexing the same document
    replaces its rows instead of appending them, so indexing the same PDF twice leaves
    the index identical (idempotent by ``doc_id``). The embedder is kept on the handle
    for query time and never persisted.
    """
    handle = HybridIndexHandle(ChunkIndex(()).replace_document(doc, embedder=embedder))
    _set_index(handle.index)
    return handle


class HybridIndexHandle:
    """A persisted, queryable hybrid index: one retriever plus its on-disk location.

    The handle owns the module's identity — its records and their ``chunk_id``s are the
    single definition of "which chunks exist" for every query served through it.
    """

    def __init__(self, index: ChunkIndex, directory: Path | None = None) -> None:
        self._index = index
        self._retriever = HybridRetriever(index)
        self._directory = Path(directory) if directory is not None else None

    @property
    def index(self) -> ChunkIndex:
        """The chunk index this handle serves: records, vectors, token lists."""
        return self._index

    # --- queries -----------------------------------------------------------------

    def retrieve(
        self, query: str, k: int = DEFAULT_K, filters: dict | None = None
    ) -> list[ScoredChunk]:
        """Hybrid retrieve with parent expansion; see :class:`HybridRetriever`."""
        return self._retriever.retrieve(query, k=k, filters=filters)

    # --- persistence -------------------------------------------------------------

    def save(self, dir: str | Path | None = None) -> None:
        """Write ``chunks.jsonl`` / FAISS / BM25 under ``dir`` (default: the last used)."""
        target = Path(dir) if dir is not None else (self._directory or _INDEX_DIR)
        self._index.save(target)
        self._directory = Path(target)

    def directory(self) -> Path | None:
        """Directory this handle was loaded from or saved to, if any."""
        return self._directory


def load_index(
    dir: str | Path | None = None, embedder: Embedder | None = None
) -> HybridIndexHandle | None:
    """Load a handle from disk, or ``None`` when nothing is persisted.

    A missing directory, a missing ``chunks.jsonl`` (no index ever built), or an empty
    index where retrieval must return ``[]`` are all handled without raising.
    """
    target = Path(dir) if dir is not None else _INDEX_DIR
    loaded = ChunkIndex.load(target, embedder=embedder)
    if loaded is None:
        return None
    _set_index(loaded)
    return HybridIndexHandle(loaded, directory=target)


def _set_index(index: ChunkIndex) -> None:
    """Update the module-level handle used by the schema's loose ``retrieve``."""
    global _INDEX, _INDEX_LOADED
    _INDEX = index
    _INDEX_LOADED = True


def _resolve_index() -> ChunkIndex | None:
    """The module-level index: the built/loaded one, else a lazy disk read (None if absent)."""
    global _INDEX, _INDEX_LOADED
    if not _INDEX_LOADED:
        try:
            _INDEX = ChunkIndex.load(_INDEX_DIR)
        except Exception:
            _INDEX = None
        _INDEX_LOADED = True
    return _INDEX


def retrieve(query: str, k: int = 8, filters: dict | None = None) -> list[ScoredChunk]:
    """Module-level hybrid retrieval over the process-wide index.

    Resolves the index on first use: the handle built/loaded earlier in this process,
    otherwise a lazy read of ``data/index/``. With no index present this returns ``[]``
    (an honest "nothing indexed" answer) instead of raising, so early callers and
    smoke checks stay safe.
    """
    index = _resolve_index()
    if index is None:
        return []
    return HybridRetriever(index).retrieve(query, k=k, filters=filters)

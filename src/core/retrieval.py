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

6. **The dense leg is live on the runtime path.** An index persists vectors only, so
   scoring a *new* query needs an embedder the artifacts do not carry. The runtime
   entries — the module-level :func:`retrieve` (which lazily reads ``data/index/``) and
   :func:`load_index` called with the embedder argument omitted — therefore resolve one
   by the same policy ``scripts/build_index.py`` defines: ``MOCK_LLM=1`` selects the
   deterministic offline embedder, otherwise the OpenAI-compatible client built from
   ``OPENAI_API_KEY`` (with ``OPENAI_BASE_URL`` / ``EMBEDDING_BASE_URL`` /
   ``EMBEDDING_MODEL``). Resolution is lazy, so importing this module builds no client
   and the hermetic test path never reaches the policy.

   **Never silently degrade.** Outside mock mode, with no embedder obtainable — no key,
   no ``MOCK_LLM=1`` — the runtime path raises :class:`RetrievalError` instead of
   answering BM25-only: a dense leg that quietly contributes nothing is the failure this
   module exists to prevent, and an actionable message names the two ways out. The same
   typed error fires when the resolved embedder does not fit the persisted vector space
   (an index built under ``MOCK_LLM=1`` is 64-dimensional and a real embedder is not, so
   the fix is ``scripts/build_index.py``), which would otherwise surface as a bare FAISS
   assertion naming neither number. Callers who *want* the sparse-only seam still have
   it: pass ``embedder=None`` explicitly to :func:`build_index` / :func:`load_index`,
   which keeps ``dense_score`` at ``0.0`` — the documented "absent", never a score of
   zero. The embedder is **never persisted**, in either direction.

Artifacts (what carries what) — see ``src/core/index.py``:

- ``chunks.jsonl`` is the **source of truth**: texts and provenance.
- FAISS holds **vectors only**; the BM25 pickle holds **token lists only**.
- The embedder is query-time wiring: injected by ``build_index``/``load_index`` callers,
  or resolved from the environment on the runtime paths above. It is never persisted.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.core.index import ChunkIndex, ChunkRecord, Embedder
from src.core.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)

# RRF damping constant (ADR-002 parameter table). 60 is the value the ADR freezes:
# large enough that the top and second retrievers' disagreements are dampened, small
# enough that being retrieved at rank 1 beats anywhere on the other list.
RRF_K = 60
# Default fused top-k: 8 children, then expansion to their parents (ADR-002).
DEFAULT_K = 8
# Supported metadata filters; anything else is a caller bug, not silently ignored.
_FILTER_KEYS = ("section", "doc_id")

#: Index directory the runtime path resolves. ``INDEX_DIR`` in the environment overrides it:
#: the mock embedder is 64-dimensional and a real one 384, and a single directory cannot hold
#: both, so `MOCK_LLM=1` after a real build points this at the mock index instead of rebuilding.
_INDEX_DIR = Path(os.environ.get("INDEX_DIR", "").strip() or "data/index")

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


def _order(fused: np.ndarray, records: Iterable[ChunkRecord]) -> list[int]:
    """Record positions sorted by fused score desc, then ``chunk_id`` ascending.

    The explicit tie-break makes the order total, so ``fused_rank`` is stable even when
    two children earn identical fused scores.
    """
    keyed = [(-float(fused[index]), record.chunk_id, index) for index, record in enumerate(records)]
    keyed.sort()
    return [index for _, _, index in keyed]


# --- filtering -------------------------------------------------------------------


def _filter_positions(index: ChunkIndex, filters: dict[str, Any] | None) -> list[int]:
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
        self, query: str, k: int = DEFAULT_K, filters: dict[str, Any] | None = None
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
        records: tuple[ChunkRecord, ...],
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


# --- the runtime embedder policy ---------------------------------------------------


class RetrievalError(Exception):
    """Raised when the hybrid retriever cannot serve a query as configured (ADR-007).

    Three situations raise it, all of them a misconfiguration rather than a bad query:
    no embedder is obtainable outside mock mode (so the dense leg would be silently
    dark), the resolved embedder does not fit the persisted vector space (an index built
    under ``MOCK_LLM=1`` is 64-dimensional, a real one is 1536 — the fix is to rebuild),
    and an index directory that cannot be loaded at all. The graph renders it through
    ``GraphNodeError``; direct callers get an actionable message instead of a BM25-only
    answer that looks fine.
    """


# --- the environment surface the embedder policy reads -----------------------------
# These names duplicate ``scripts/build_index.py`` on purpose: ``src`` must not import
# that script at module scope (it is not a package, and importing it would drag argparse
# and a corpus convention into the app), while the two surfaces must still agree on the
# configuration that decides which embedder runs. The decision itself is *shared*, not
# duplicated: see :func:`_script_build_embedder`.
_EMBEDDING_MODE_ENV = "MOCK_LLM"
_API_KEY_ENV = "OPENAI_API_KEY"
_BASE_URL_ENV = "OPENAI_BASE_URL"
_EMBEDDING_BASE_URL_ENV = "EMBEDDING_BASE_URL"
_EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class _Unset:
    """Type of the :data:`_UNSET` sentinel: "no embedder argument was passed"."""


#: Distinguishes an omitted ``embedder`` argument (the runtime policy applies) from an
#: explicit ``None`` (the caller *wants* the documented sparse-only seam). A plain
#: ``None`` default cannot tell those two intents apart, and the whole point of this fix
#: is that the first must never be reachable by accident.
_UNSET = _Unset()


def _mock_embedder() -> tuple[Embedder, str]:
    """The deterministic offline embedder, identical to ``scripts/build_index``'s.

    ``MOCK_LLM=1`` reuses the script's own embedder through :func:`_script_build_embedder`
    whenever that file is present. This local copy is the fallback when it is not (a wheel
    or a slimmed image), and it reproduces the script's digest-fold bit for bit — sha256
    of the text, folded to one vector, L2-normalized — so an index built one way scores
    identically when loaded the other.
    """
    import hashlib  # imported locally: the real path never pays for it

    dimensions = 64

    def embedder(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # sha256 is 32 bytes; the fold repeats it to reach ``dimensions`` (the same
            # rule the builder uses) and the zero-norm fallback keeps the result a unit
            # vector so a degenerate text can never blow up the inner product.
            vector = [
                float(digest[offset] - 127.5) for offset in range(min(dimensions, len(digest)))
            ]
            if len(vector) < dimensions:
                vector = (vector * ((dimensions // max(1, len(vector))) + 1))[:dimensions]
            norm = float(sum(value * value for value in vector)) ** 0.5
            if norm == 0.0:
                vector = [1.0] + [0.0] * (dimensions - 1)
            else:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors

    return embedder, f"mock embedder, deterministic, {dimensions} dims, offline"


def _script_build_embedder() -> tuple[Embedder, str] | None:
    """``scripts/build_index.build_embedder()``, or ``None`` when that file is absent.

    Loading the script by file location reuses the builder's policy directly — so the
    mock branch, the env names, the key check, the error text and the client construction
    each have exactly one implementation. A ``BuildError`` raised by it propagates
    untouched: that message already names the missing variable and the ``MOCK_LLM=1``
    alternative, which is precisely what the query-time path must show.
    """
    try:
        candidate = Path(__file__).resolve().parents[2] / "scripts" / "build_index.py"
        if not candidate.is_file():
            return None
        spec = importlib.util.spec_from_file_location("isipfin_build_index", candidate)
        if spec is None or spec.loader is None:
            return None
        script = importlib.util.module_from_spec(spec)
        # The module must be registered before it executes: ``@dataclass(frozen=True)``
        # resolves its own annotations through ``sys.modules``, so a module that exists
        # only on the specs and not in the module table breaks at definition time.
        sys.modules[spec.name] = script
        try:
            spec.loader.exec_module(script)
        except Exception:
            sys.modules.pop(spec.name, None)  # a half-executed module must not linger
            raise
        builder = getattr(script, "build_embedder", None)
        if not callable(builder):
            return None
        embedder, description = builder()
        return embedder, str(description)
    except Exception as exc:
        # A ``BuildError`` from the script, or any failure importing that file: either
        # way the operator needs the real cause rather than a bare "no embedder". The
        # script's own types are re-wrapped as *our* typed error, so a caller of this
        # module catches one exception class for every embedder-resolution failure —
        # and the message keeps the script's actionable text (missing key, or
        # ``MOCK_LLM=1``) verbatim, which is the point of reusing the policy.
        raise RetrievalError(str(exc)) from exc


def _owned_embedder(embedder: Embedder, description: str) -> tuple[Embedder, str]:
    """Rebind an embedder's failures to this module's :class:`RetrievalError`.

    The shared policy builds real clients whose own error types (``BuildError`` from the
    builder, SDK transport errors) are internal to the builder. A caller of *this* module
    must be able to catch one class for every retrieval failure, and the dense leg must
    never be swallowed into sparse-only ranking by a mistake in error typing — so the
    adapter converts. The message keeps the underlying cause verbatim; the type is made
    this module's own.
    """

    def embed(texts: list[str]) -> list[list[float]]:
        try:
            return embedder(texts)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"the query-time embedder failed: {exc}") from exc

    return embed, description


def _resolve_embedder() -> tuple[Embedder, str]:
    """The runtime query-time embedder, resolved by the environment (docstring §6).

    ``MOCK_LLM=1`` selects the deterministic offline embedder, key or no key. Otherwise
    an OpenAI-compatible client is built from ``OPENAI_API_KEY`` (with
    ``OPENAI_BASE_URL`` / ``EMBEDDING_BASE_URL`` / ``EMBEDDING_MODEL``), and — exactly as
    the builder fails *before* parsing a corpus — a missing key raises
    :class:`RetrievalError` naming both the variable and the offline alternative. This is
    the decision the builder makes when it writes an index, read the same way here when
    one is queried: an index written by one embedder must be searched by the same one.
    """
    if os.environ.get(_EMBEDDING_MODE_ENV, "").strip() == "1":
        return _owned_embedder(*(_script_build_embedder() or _mock_embedder()))
    from scripts.build_index import build_embedder  # policy reuse: one implementation

    try:
        return _owned_embedder(*build_embedder())
    except Exception as exc:
        # Reuse the builder's policy, own the error type: the caller catches exactly one
        # exception class for every resolution failure of this module.
        raise RetrievalError(str(exc)) from exc


#: The one embedder this process queries with, plus the description logged once.
_EMBEDDER: tuple[Embedder, str] | None = None
_EMBEDDER_RESOLVED = False


def _embedder_for_query() -> Embedder:
    """The process-wide query-time embedder, logged once when it is first resolved.

    Resolution failure raises rather than returning ``None``: the dense leg going dark
    without a word is the exact bug this module was fixed for, so the first query in a
    process either gets a working dense leg or an actionable refusal.
    """
    global _EMBEDDER, _EMBEDDER_RESOLVED
    resolved = _EMBEDDER
    if not _EMBEDDER_RESOLVED or resolved is None:
        resolved = _resolve_embedder()
        _EMBEDDER = resolved
        _EMBEDDER_RESOLVED = True
        logger.info("retrieval embedder: %s", resolved[1])
    return resolved[0]


def _embedder_width(embedder: Embedder) -> int | None:
    """Dimensionality this embedder produces, from one probe call, or ``None`` on doubt.

    The probe is the harness-agnostic way to ask: an embedder is a callable, so the only
    question that matters is what width comes back. A client that refuses the probe — an
    unreachable endpoint, a rate limit, a stub returning a scalar — must not decide a
    whole index's fate, so doubt yields ``None`` and the dense leg meets its own fate
    through the normal scoring path instead.
    """
    try:
        vectors = embedder(["dimensionality probe"])
    except Exception:
        return None
    if not vectors or not isinstance(vectors[0], (list, tuple)):
        return None
    return len(vectors[0])


def _check_dense_dimensions(embedder: Embedder, expected: int, where: Path | str) -> None:
    """Fail early when the embedder's width cannot compare with the index's vectors.

    The classic case is a real ``OPENAI_API_KEY`` against an index built under
    ``MOCK_LLM=1``: deterministic vectors are 64-wide, real ones 1536, and querying anyway
    surfaces as a bare FAISS assertion naming neither number. ``IndexFlatIP`` also
    accepts some mismatched widths and answers with garbage, which is worse than failing
    — so the check runs before any query and the message names both widths plus the
    command that fixes them.
    """
    width = _embedder_width(embedder)
    if width is not None and width != expected:
        raise RetrievalError(
            f"the embedder resolved from the environment is {width}-dimensional but the "
            f"index under {where} holds {expected}-dimensional vectors — they cannot be "
            "compared. Rebuild the index with scripts/build_index.py using the same "
            "embedder (MOCK_LLM=1 selects the deterministic 64-dimensional one)."
        )


def _embedder_argument(embedder: Embedder | _Unset) -> Embedder | None:
    """Normalize a ``load_index`` embedder argument to ``ChunkIndex``'s ``None``-or-callable.

    The sentinel means "no argument was passed", so the runtime policy applies; ``None``
    means the caller explicitly wants the sparse-only seam and is honored as passed.
    """
    if isinstance(embedder, _Unset):
        return _embedder_for_query()
    return embedder


# --- index lifecycle -------------------------------------------------------------

# --- index lifecycle -------------------------------------------------------------


def build_index(doc: ParsedDocument, embedder: Embedder | None = None) -> HybridIndexHandle:
    """Index one document into a fresh handle (idempotent by ``doc_id``).

    The document's chunks are chunked by ``src/core/chunking.py`` and embedded through
    the injected ``embedder`` (an explicit ``None`` builds a BM25-only index, whose
    query-time ``dense_score`` stays at the documented ``0.0`` — that is the
    sparse-only seam a caller asks for on purpose, not the runtime path). The result
    stays in memory — it becomes this process's current index, and the caller persists
    it with ``handle.save(dir)`` when it wants an on-disk index. Re-indexing the same
    document replaces its rows instead of appending them, so indexing the same PDF twice
    leaves the index identical (idempotent by ``doc_id``). The embedder is kept on the
    handle for query time and never persisted.
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
        self, query: str, k: int = DEFAULT_K, filters: dict[str, Any] | None = None
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
    dir: str | Path | None = None, embedder: Embedder | _Unset = _UNSET
) -> HybridIndexHandle | None:
    """Load a handle from disk, or ``None`` when nothing is persisted.

    A missing directory, a missing ``chunks.jsonl`` (no index ever built), or an empty
    index where retrieval must return ``[]`` are all handled without raising.

    With the embedder argument omitted, the runtime policy resolves one (module docstring
    §6): the dense leg is live, and a missing key or a mismatched vector width raises
    :class:`RetrievalError`. Pass ``embedder=None`` explicitly to keep the documented
    sparse-only seam — a deliberate query-time choice, made by a caller that wants one,
    never the accident of an omitted argument.
    """
    target = Path(dir) if dir is not None else _INDEX_DIR
    # "Argument omitted" and "argument explicitly None" are two different intents, and
    # the first must never be reachable by accident, so they use different defaults.
    active = _embedder_argument(embedder)
    loaded = ChunkIndex.load(target, embedder=active)
    if loaded is None:
        return None
    if active is not None and loaded.dim is not None:
        _check_dense_dimensions(active, int(loaded.dim), target)
    _set_index(loaded)
    return HybridIndexHandle(loaded, directory=target)


def _set_index(index: ChunkIndex) -> None:
    """Update the module-level handle used by the schema's loose ``retrieve``."""
    global _INDEX, _INDEX_LOADED
    _INDEX = index
    _INDEX_LOADED = True


def _resolve_index() -> ChunkIndex | None:
    """The module-level index: the built/loaded one, else a lazy disk read (None if absent).

    A handle built here or loaded in this process always wins: it already carries its own
    embedder, so re-reading the directory would discard it.

    The lazy disk read — the production path, reached by a node that imports this module
    without loading an index first — resolves a query-time embedder only when the index
    actually holds vectors, because a disk index persists vectors only and cannot embed a
    new query by itself (module docstring §6). Two falls-through stay honest: a directory
    with nothing persisted reads as ``[]`` (an early caller and a smoke check must both be
    safe), and an index deliberately built BM25-only has no vectors to be found in. The
    vector-holding case is not a fall-through: it resolves the embedder, so a missing key
    raises rather than ranking sparsely in silence.
    """
    global _INDEX, _INDEX_LOADED
    if not _INDEX_LOADED:
        try:
            # Read the index once, vectors only: the embedder is cheap to ask for but
            # the corpus is not cheap to re-read, so the vector-holding decision below
            # is made from this single load rather than loading twice.
            loaded = ChunkIndex.load(_INDEX_DIR)
        except Exception:
            # An unreadable directory is "nothing indexed", not a crash.
            loaded = None
        if loaded is not None and loaded.dim is not None:
            # Vectors on disk with no embedder to compare against: the dense leg must
            # not go dark silently, so the runtime policy is resolved here. The second
            # load exists only because an embedder-less index cannot embed a query —
            # it re-reads the artifacts, never the corpus, and the FAISS rows are the
            # single source of the stored vectors either way.
            embedder = _embedder_for_query()
            _check_dense_dimensions(embedder, int(loaded.dim), Path(_INDEX_DIR))
            with_embedder = ChunkIndex.load(_INDEX_DIR, embedder=embedder)
            loaded = loaded if with_embedder is None else with_embedder
        _INDEX = loaded
        _INDEX_LOADED = True
    return _INDEX


def retrieve(query: str, k: int = 8, filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
    """Module-level hybrid retrieval over the process-wide index.

    Resolves the index on first use: the handle built/loaded earlier in this process,
    otherwise a lazy read of ``data/index/`` under the runtime embedder policy. With no
    index present this returns ``[]`` (an honest "nothing indexed" answer) instead of
    raising, so early callers and smoke checks stay safe. With an index present but no
    embedder obtainable (no key and no ``MOCK_LLM=1``) this raises
    :class:`RetrievalError`: an index the caller can read but cannot rank densely is a
    configuration failure, not an empty answer.
    """
    index = _resolve_index()
    if index is None:
        return []
    return HybridRetriever(index).retrieve(query, k=k, filters=filters)

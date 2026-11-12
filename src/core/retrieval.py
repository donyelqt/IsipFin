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

RRF_K = 60
DEFAULT_K = 8
_FILTER_KEYS = ("section", "doc_id")

_INDEX_DIR = Path(os.environ.get("INDEX_DIR", "").strip() or "data/index")

_INDEX: ChunkIndex | None = None
_INDEX_LOADED = False


@dataclass(frozen=True)
class ScoredChunk:

    chunk_id: str
    child_text: str
    parent_text: str
    page: int
    section_path: list[str]
    doc_id: str
    dense_score: float
    sparse_score: float
    fused_rank: int


def _ranks(scores: np.ndarray | None, size: int | None = None) -> np.ndarray:
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
    total = np.zeros(size, dtype=np.float64)
    for ranks in (_ranks(dense, size), _ranks(sparse, size)):
        contributed = ranks > 0
        if not contributed.any():
            continue
        total[contributed] += 1.0 / (RRF_K + ranks[contributed].astype(np.float64))
    return total


def _order(fused: np.ndarray, records: Iterable[ChunkRecord]) -> list[int]:
    keyed = [(-float(fused[index]), record.chunk_id, index) for index, record in enumerate(records)]
    keyed.sort()
    return [index for _, _, index in keyed]


def _filter_positions(index: ChunkIndex, filters: dict[str, Any] | None) -> list[int]:
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
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(f"k must be an integer, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return k


class HybridRetriever:

    def __init__(self, index: ChunkIndex) -> None:
        self._index = index

    @property
    def index(self) -> ChunkIndex:
        return self._index

    def retrieve(
        self, query: str, k: int = DEFAULT_K, filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        limit = _resolve_k(k)
        if not query.strip():
            return []

        positions = _filter_positions(self._index, filters)
        if not positions:
            return []
        candidates = self._index.subset(positions)
        records = candidates.records

        raw_dense = candidates.dense_scores(query)
        raw_sparse = candidates.sparse_scores(query)
        out_of_band = np.full(len(records), -np.inf, dtype=np.float64)

        per_leg = min(limit, len(records))
        fused = _fuse(
            _top_k(raw_dense, per_leg, len(records)),
            _top_k(raw_sparse, per_leg, len(records)),
            len(records),
        )
        ordered = _order(fused, records)[:limit]

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
    if scores is None or scores.size == 0:
        return scores
    scores = np.asarray(scores, dtype=np.float64)
    if k >= size:
        return scores
    order = np.argsort(-scores, kind="stable")[:k]
    masked = np.full(size, -np.inf, dtype=np.float64)
    masked[order] = scores[order]
    return masked


class RetrievalError(Exception):
    pass


_EMBEDDING_MODE_ENV = "MOCK_LLM"
_API_KEY_ENV = "OPENAI_API_KEY"
_BASE_URL_ENV = "OPENAI_BASE_URL"
_EMBEDDING_BASE_URL_ENV = "EMBEDDING_BASE_URL"
_EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class _Unset:
    pass


_UNSET = _Unset()


def _mock_embedder() -> tuple[Embedder, str]:
    import hashlib

    dimensions = 64

    def embedder(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
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
    try:
        candidate = Path(__file__).resolve().parents[2] / "scripts" / "build_index.py"
        if not candidate.is_file():
            return None
        spec = importlib.util.spec_from_file_location("isipfin_build_index", candidate)
        if spec is None or spec.loader is None:
            return None
        script = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = script
        try:
            spec.loader.exec_module(script)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        builder = getattr(script, "build_embedder", None)
        if not callable(builder):
            return None
        embedder, description = builder()
        return embedder, str(description)
    except Exception as exc:
        raise RetrievalError(str(exc)) from exc


def _owned_embedder(embedder: Embedder, description: str) -> tuple[Embedder, str]:

    def embed(texts: list[str]) -> list[list[float]]:
        try:
            return embedder(texts)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"the query-time embedder failed: {exc}") from exc

    return embed, description


def _resolve_embedder() -> tuple[Embedder, str]:
    if os.environ.get(_EMBEDDING_MODE_ENV, "").strip() == "1":
        return _owned_embedder(*(_script_build_embedder() or _mock_embedder()))
    from scripts.build_index import build_embedder

    try:
        return _owned_embedder(*build_embedder())
    except Exception as exc:
        raise RetrievalError(str(exc)) from exc


_EMBEDDER: tuple[Embedder, str] | None = None
_EMBEDDER_RESOLVED = False


def _embedder_for_query() -> Embedder:
    global _EMBEDDER, _EMBEDDER_RESOLVED
    resolved = _EMBEDDER
    if not _EMBEDDER_RESOLVED or resolved is None:
        resolved = _resolve_embedder()
        _EMBEDDER = resolved
        _EMBEDDER_RESOLVED = True
        logger.info("retrieval embedder: %s", resolved[1])
    return resolved[0]


def _embedder_width(embedder: Embedder) -> int | None:
    try:
        vectors = embedder(["dimensionality probe"])
    except Exception:
        return None
    if not vectors or not isinstance(vectors[0], (list, tuple)):
        return None
    return len(vectors[0])


def _check_dense_dimensions(embedder: Embedder, expected: int, where: Path | str) -> None:
    width = _embedder_width(embedder)
    if width is not None and width != expected:
        raise RetrievalError(
            f"the embedder resolved from the environment is {width}-dimensional but the "
            f"index under {where} holds {expected}-dimensional vectors — they cannot be "
            "compared. Rebuild the index with scripts/build_index.py using the same "
            "embedder (MOCK_LLM=1 selects the deterministic 64-dimensional one)."
        )


def _embedder_argument(embedder: Embedder | _Unset) -> Embedder | None:
    if isinstance(embedder, _Unset):
        return _embedder_for_query()
    return embedder


def build_index(doc: ParsedDocument, embedder: Embedder | None = None) -> HybridIndexHandle:
    handle = HybridIndexHandle(ChunkIndex(()).replace_document(doc, embedder=embedder))
    _set_index(handle.index)
    return handle


class HybridIndexHandle:

    def __init__(self, index: ChunkIndex, directory: Path | None = None) -> None:
        self._index = index
        self._retriever = HybridRetriever(index)
        self._directory = Path(directory) if directory is not None else None

    @property
    def index(self) -> ChunkIndex:
        return self._index

    def retrieve(
        self, query: str, k: int = DEFAULT_K, filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        return self._retriever.retrieve(query, k=k, filters=filters)

    def save(self, dir: str | Path | None = None) -> None:
        target = Path(dir) if dir is not None else (self._directory or _INDEX_DIR)
        self._index.save(target)
        self._directory = Path(target)

    def directory(self) -> Path | None:
        return self._directory


def load_index(
    dir: str | Path | None = None, embedder: Embedder | _Unset = _UNSET
) -> HybridIndexHandle | None:
    target = Path(dir) if dir is not None else _INDEX_DIR
    active = _embedder_argument(embedder)
    loaded = ChunkIndex.load(target, embedder=active)
    if loaded is None:
        return None
    if active is not None and loaded.dim is not None:
        _check_dense_dimensions(active, int(loaded.dim), target)
    _set_index(loaded)
    return HybridIndexHandle(loaded, directory=target)


def _set_index(index: ChunkIndex) -> None:
    global _INDEX, _INDEX_LOADED
    _INDEX = index
    _INDEX_LOADED = True


def _resolve_index() -> ChunkIndex | None:
    global _INDEX, _INDEX_LOADED
    if not _INDEX_LOADED:
        try:
            loaded = ChunkIndex.load(_INDEX_DIR)
        except Exception:
            loaded = None
        if loaded is not None and loaded.dim is not None:
            embedder = _embedder_for_query()
            _check_dense_dimensions(embedder, int(loaded.dim), Path(_INDEX_DIR))
            with_embedder = ChunkIndex.load(_INDEX_DIR, embedder=embedder)
            loaded = loaded if with_embedder is None else with_embedder
        _INDEX = loaded
        _INDEX_LOADED = True
    return _INDEX


def retrieve(query: str, k: int = 8, filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
    index = _resolve_index()
    if index is None:
        return []
    return HybridRetriever(index).retrieve(query, k=k, filters=filters)

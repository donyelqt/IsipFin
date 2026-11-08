r"""Chunk-index artifacts: chunk records, FAISS dense vectors, BM25 sparse token lists.

Layout under ``data/index/`` (docs/ARCHITECTURE.md data model, ADR-002 decision 6):

| Artifact | Carries | Does NOT carry |
|---|---|---|
| ``chunks.jsonl`` | **source of truth**: one record per child chunk with both
texts and all provenance (``chunk_id``, ``parent_id``, ``child_text``,
``parent_text``, ``page``, ``section_path``, ``doc_id``, ``ordinal``) | vectors, scores |
| ``faiss/index.faiss`` | vectors only — one L2-normalized ``float32`` row per
child chunk, ids equal to the record position in ``chunks.jsonl`` | texts, metadata |
| ``faiss/meta.json`` | dimension + vector count, so a stale artifact is
detected instead of trusted | anything load-bearing |
| ``bm25.pkl`` | token lists only (``\w+``, lowercased) — a pure tokenization
cache, never a fitted model object | texts, statistics |

Any artifact can be rebuilt from ``chunks.jsonl``: FAISS by re-embedding ``child_text``
(the embedder is injected by the caller — this module never builds an OpenAI client) and BM25
by re-tokenizing. ``ChunkIndex.load`` performs exactly that self-heal when an artifact is
missing, corrupt, or stale.

BM25 is constructed lazily and only for a non-empty corpus: ``BM25Okapi([])`` raises
``ZeroDivisionError`` in ``_initialize`` (``avgdl = num_doc / corpus_size``), so an empty
document must never reach the constructor. Token lists are persisted rather than a fitted
``BM25Okapi`` so the pickle carries data, not library state.

Vectors are L2-normalized before insertion, which makes inner-product search on
``faiss.IndexFlatIP`` exact cosine similarity (ADR-002: exact search is the right choice at
this corpus size; IVF/HNSW buys nothing below ~50k chunks and costs a training step).
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.core.chunking import chunk_document
from src.core.pdf_parser import ParsedDocument

# A callable mapping a batch of texts to a batch of dense vectors. Injected by the
# caller (tests pass a deterministic fake; production wiring is T6's llm-gateway).
# The embedder is never persisted: it is a query-time dependency, not index data.
Embedder = Callable[[list[str]], list[list[float]]]

# --- persistence layout ---------------------------------------------------------
DEFAULT_INDEX_DIR = Path("data/index")
FAISS_DIRNAME = "faiss"
FAISS_FILENAME = "index.faiss"
BM25_FILENAME = "bm25.pkl"
CHUNKS_FILENAME = "chunks.jsonl"
META_FILENAME = "meta.json"

# Sparse tokenization: \w+ over the lowercased text (financial tokens such as
# "211,915" arrive as "211" and "915", BM25's usual trade for exact keyword hits).
_TOKEN = re.compile(r"\w+")

# Fields every chunks.jsonl line must carry (assignment rule 6).
_REQUIRED_FIELDS = (
    "chunk_id",
    "parent_id",
    "child_text",
    "parent_text",
    "page",
    "section_path",
    "doc_id",
    "ordinal",
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class ChunkRecord:
    """One persisted child chunk plus the parent payload it expands to.

    ``child_text`` is what is embedded and BM25-scored; ``parent_text`` is what reaches
    synthesis. Both live in the source-of-truth JSONL, so no retriever ever needs the
    original document again.
    """

    chunk_id: str
    parent_id: str
    child_text: str
    parent_text: str
    page: int
    section_path: tuple[str, ...]
    doc_id: str
    ordinal: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping; ``section_path`` serializes as a list per the contract."""
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "child_text": self.child_text,
            "parent_text": self.parent_text,
            "page": self.page,
            "section_path": list(self.section_path),
            "doc_id": self.doc_id,
            "ordinal": self.ordinal,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str) -> ChunkRecord:
        """Rebuild a record from one JSONL line, failing loudly on a malformed one."""
        missing = [field for field in _REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"{source}: missing field(s) {', '.join(missing)}")
        section = payload["section_path"]
        if not isinstance(section, list) or not all(isinstance(part, str) for part in section):
            raise ValueError(f"{source}: section_path must be a list of strings")
        return cls(
            chunk_id=str(payload["chunk_id"]),
            parent_id=str(payload["parent_id"]),
            child_text=str(payload["child_text"]),
            parent_text=str(payload["parent_text"]),
            page=int(payload["page"]),
            section_path=tuple(section),
            doc_id=str(payload["doc_id"]),
            ordinal=int(payload["ordinal"]),
        )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length; a zero row stays zero (never divide by zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return np.ascontiguousarray(matrix / safe, dtype=np.float32)


def _embed_texts(embedder: Embedder | None, texts: Sequence[str]) -> np.ndarray | None:
    """Embed ``texts`` into an ``(n, dim)`` L2-normalized float32 matrix.

    Returns ``None`` when there is nothing to embed (no texts, or no embedder injected),
    which callers read as "the dense leg is unavailable", never as an error.
    """
    if embedder is None or not texts:
        return None
    vectors = embedder(list(texts))
    if vectors is None:
        return None
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {len(texts)} texts; "
            "one vector per text is required"
        )
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedder must return a rectangular list of numeric vectors") from exc
    if matrix.ndim != 2:
        raise ValueError(
            f"embedder returned a {matrix.ndim}-D array; expected one vector per text (2-D)"
        )
    return _l2_normalize(matrix)


def _as_matrix(vectors: np.ndarray | None, expected: int) -> np.ndarray | None:
    """Validate a pre-built vector matrix against the record count.

    A ``(0, dim)`` matrix is normalized back to ``None`` — an empty index has no dense
    engine at all rather than one that can never hold a row.
    """
    if vectors is None:
        return None
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2-D (n_texts, dim) array")
    if matrix.shape[0] == 0:
        return None
    if matrix.shape[0] != expected:
        raise ValueError(
            f"vectors hold {matrix.shape[0]} rows for {expected} records; "
            "vector rows must align with record order"
        )
    return matrix


def _read_jsonl(path: Path) -> list[ChunkRecord]:
    """Read the source-of-truth JSONL; blank lines are skipped, bad lines raise."""
    records: list[ChunkRecord] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            records.append(ChunkRecord.from_dict(payload, source=f"{path}:{number}"))
    return records


def _write_jsonl(path: Path, records: Iterable[ChunkRecord]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False))
            handle.write("\n")


def _read_tokens(path: Path, expected: int) -> list[list[str]] | None:
    """Read the persisted token lists, or ``None`` when absent or out of sync.

    The pickle is a cache written by :meth:`ChunkIndex.save`; an unreadable or
    mismatching one is ignored and the caller re-tokenizes from ``child_text``.
    """
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(payload, list) or len(payload) != expected:
        return None
    if not all(
        isinstance(tokens, list) and all(isinstance(token, str) for token in tokens)
        for tokens in payload
    ):
        return None
    return [list(tokens) for tokens in payload]


def _read_vectors(path: Path, expected: int) -> np.ndarray | None:
    if not path.is_file() or expected == 0:
        return None
    try:
        index = faiss.read_index(str(path))
    except Exception:
        return None
    if index.ntotal != expected or index.d <= 0:
        return None
    # IndexFlat keeps exact float32 rows; reconstruct_n is a direct copy, so the
    # save -> load round-trip is bit-identical and scores cannot drift.
    return np.ascontiguousarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)


class ChunkIndex:
    """Child-chunk records plus the two engines that rank them.

    Record order is the identity: FAISS row *i*, BM25 document *i*, and
    ``records[i]`` always describe the same chunk, and every score array returned by
    this class is indexed by record position.
    """

    def __init__(
        self,
        records: Sequence[ChunkRecord],
        embedder: Embedder | None = None,
        vectors: np.ndarray | None = None,
        token_lists: Sequence[Sequence[str]] | None = None,
    ) -> None:
        self._records: tuple[ChunkRecord, ...] = tuple(records)
        self._embedder = embedder
        if token_lists is None:
            tokens = [tokenize(record.child_text) for record in self._records]
        else:
            tokens = [list(item) for item in token_lists]
            if len(tokens) != len(self._records):
                raise ValueError(
                    f"token_lists hold {len(tokens)} entries for {len(self._records)} records"
                )
        self._token_lists: list[list[str]] = tokens
        self._vectors = _as_matrix(vectors, len(self._records))
        self._dim: int | None = None if self._vectors is None else int(self._vectors.shape[1])
        self._dense: faiss.IndexFlatIP | None = None
        self._sparse: BM25Okapi | None = None
        self._build_engines()

    # --- engines -----------------------------------------------------------------

    def _build_engines(self) -> None:
        if self._vectors is not None and self._vectors.shape[0]:
            dense = faiss.IndexFlatIP(int(self._vectors.shape[1]))
            # Vectors are always stored in record order and contiguous, so a plain
            # ``add`` gives row i == record i. (``add_with_ids`` is not implemented
            # for IndexFlat, and would only let us skip rows we never skip.)
            dense.add(self._vectors)
            self._dense = dense
        # BM25Okapi([]) divides by corpus_size and raises ZeroDivisionError; an empty
        # corpus therefore simply has no sparse engine.
        if self._token_lists:
            self._sparse = BM25Okapi(self._token_lists)

    # --- accessors ---------------------------------------------------------------

    @property
    def records(self) -> tuple[ChunkRecord, ...]:
        """Immutable record view, ordered; score arrays are indexed by this order."""
        return self._records

    @property
    def token_lists(self) -> list[list[str]]:
        """Persisted sparse tokenization (token lists only, per ADR-002)."""
        return self._token_lists

    @property
    def dim(self) -> int | None:
        """Embedding dimensionality, or ``None`` for an index with no vectors."""
        return self._dim

    def subset(self, positions: Sequence[int]) -> ChunkIndex:
        """Return an index over the records at ``positions`` (same rules as filtering).

        Filtering happens before ranking (assignment rule 7), so a filtered retriever is
        a genuine sub-index: the kept rows take their vectors and token lists with them,
        and score arrays stay indexed by this sub-index's record order.
        """
        kept = [int(position) for position in positions]
        if any(position < 0 or position >= len(self._records) for position in kept):
            raise IndexError("subset position out of range")
        records = [self._records[position] for position in kept]
        token_lists = [self._token_lists[position] for position in kept]
        vectors: np.ndarray | None = None
        if self._vectors is not None and kept:
            if any(position >= self._vectors.shape[0] for position in kept):
                raise IndexError("subset position has no vector row")
            vectors = np.ascontiguousarray(self._vectors[kept], dtype=np.float32)
        return ChunkIndex(
            records, embedder=self._embedder, vectors=vectors, token_lists=token_lists
        )

    def __len__(self) -> int:
        return len(self._records)

    # --- scoring -----------------------------------------------------------------

    def dense_scores(self, query: str) -> np.ndarray | None:
        """Cosine similarity of ``query`` against every record, by record position.

        ``None`` when the dense leg is unavailable: no embedder injected (vectors
        alone cannot embed a new query) or an index with no vectors.
        """
        if self._dense is None or self._embedder is None or not self._records:
            return None
        try:
            matrix = _embed_texts(self._embedder, [query])
        except ValueError:
            # A malformed embedder degrades the dense leg only — never retrieval itself.
            return None
        if matrix is None:
            return None
        total = int(self._dense.ntotal)
        scores, ids = self._dense.search(matrix, total)
        by_position = np.zeros(total, dtype=np.float64)
        for score, position in zip(scores[0], ids[0], strict=True):
            if int(position) < 0:
                continue
            by_position[int(position)] = float(score)
        return by_position

    def sparse_scores(self, query: str) -> np.ndarray | None:
        """BM25 score of ``query`` against every record, by record position.

        ``None`` when the sparse leg is unavailable (no chunks, or the query carries no
        ``\\w+`` tokens at all — BM25 can rank nothing there). A query whose tokens match
        no document still scores: BM25 returns all zeros, which rank like any other
        scores (see the RRF note in ``src/core/retrieval.py``).
        """
        if self._sparse is None or not self._records:
            return None
        if not tokenize(query):
            return None
        scores = self._sparse.get_scores(tokenize(query))
        return np.asarray(scores, dtype=np.float64)

    # --- ingestion ---------------------------------------------------------------

    def replace_document(self, doc: ParsedDocument, embedder: Embedder | None = None) -> ChunkIndex:
        """Return an index with ``doc``'s chunks replacing that document's prior ones.

        Idempotent by ``doc_id``: existing rows for the same document are dropped, other
        documents' rows (and their vectors) are kept, and only new children are
        embedded — add is an append, delete is a rebuild, per ADR-002. A duplicate
        ``chunk_id`` is never built twice. Returns a new index; the receiver is unchanged.
        """
        active = embedder if embedder is not None else self._embedder
        parents, children = chunk_document(doc)
        parent_text = {parent.parent_id: parent.text for parent in parents}

        new_records: list[ChunkRecord] = []
        incoming: set[str] = set()
        for child in children:
            if child.chunk_id in incoming:
                continue  # duplicate chunk_id: never build one twice
            incoming.add(child.chunk_id)
            new_records.append(
                ChunkRecord(
                    chunk_id=child.chunk_id,
                    parent_id=child.parent_id,
                    child_text=child.text,
                    parent_text=parent_text[child.parent_id],
                    page=child.page,
                    section_path=tuple(child.section_path),
                    doc_id=child.doc_id,
                    ordinal=child.ordinal,
                )
            )

        kept_records: list[ChunkRecord] = []
        kept_rows: list[np.ndarray] = []
        for position, record in enumerate(self._records):
            if record.doc_id == doc.doc_id:
                continue  # same document: its old rows are replaced, not duplicated
            if record.chunk_id in incoming:
                continue  # chunk_id collision across documents: first writer wins
            kept_records.append(record)
            if self._vectors is not None and position < self._vectors.shape[0]:
                kept_rows.append(self._vectors[position])

        records = [*kept_records, *new_records]
        kept_matrix: np.ndarray | None = (
            np.vstack(kept_rows) if kept_rows else np.empty((0, 0), dtype=np.float32)
        )
        if kept_matrix is not None and kept_matrix.size == 0:
            kept_matrix = None
        new_matrix = _embed_texts(active, [record.child_text for record in new_records])

        if kept_matrix is None:
            vectors: np.ndarray | None = new_matrix
        elif new_matrix is None:
            vectors = kept_matrix
        elif kept_matrix.shape[1] == new_matrix.shape[1]:
            vectors = np.vstack([kept_matrix, new_matrix])
        else:
            # Dimension drift between a previously persisted index and the embedder
            # used now: rebuild every vector so the index stays self-consistent.
            vectors = _embed_texts(active, [record.child_text for record in records])
        return ChunkIndex(records, embedder=active, vectors=vectors)

    # --- persistence -------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write the three artifacts; ``chunks.jsonl`` remains the source of truth."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        _write_jsonl(target / CHUNKS_FILENAME, self._records)
        with open(target / BM25_FILENAME, "wb") as handle:
            pickle.dump(
                [list(tokens) for tokens in self._token_lists],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        # An empty index has no vectors: drop any FAISS directory a previous save left
        # behind, so "no faiss/ dir" always means "no vectors", never "stale vectors".
        faiss_dir = target / FAISS_DIRNAME
        if self._dense is None:
            shutil.rmtree(faiss_dir, ignore_errors=True)
            return target
        faiss_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._dense, str(faiss_dir / FAISS_FILENAME))
        (faiss_dir / META_FILENAME).write_text(
            json.dumps({"dim": self._dim, "count": int(self._dense.ntotal)}),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> ChunkIndex | None:
        """Load a persisted index, or ``None`` when nothing is persisted there.

        A directory with no ``chunks.jsonl`` has no index. An empty one loads as a valid
        empty index (zero records, no FAISS engine, no BM25 engine) so retrieval
        returns ``[]`` instead of raising. Missing or stale FAISS/BM25 artifacts are
        rebuilt from the JSONL — re-embedded when an embedder is available, otherwise
        re-tokenized.
        """
        target = Path(directory)
        chunks_path = target / CHUNKS_FILENAME
        if not chunks_path.is_file():
            return None
        records = _read_jsonl(chunks_path)

        vectors = _read_vectors(target / FAISS_DIRNAME / FAISS_FILENAME, len(records))
        if vectors is None and records and embedder is not None:
            vectors = _embed_texts(embedder, [record.child_text for record in records])

        token_lists = _read_tokens(target / BM25_FILENAME, len(records))
        if token_lists is None:
            token_lists = [tokenize(record.child_text) for record in records]
        return cls(records, embedder=embedder, vectors=vectors, token_lists=token_lists)

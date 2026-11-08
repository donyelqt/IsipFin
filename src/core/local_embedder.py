"""In-process Hugging-Face embeddings via ``fastembed`` (ONNX Runtime, no torch).

``EMBEDDING_PROVIDER=local`` (the default) resolves the embedder this module builds: a
real sentence-embedding model — ``BAAI/bge-small-en-v1.5`` by default, 384 dimensions —
run locally through ONNX Runtime. It needs no API key and no network *after* the first
use, which is what makes the app's keyless demo path answer questions about meaning
rather than about sha256 digests.

Why ``fastembed`` and not ``sentence-transformers``: the latter pulls PyTorch (~2.5 GB)
into the image, and the Docker cold-start the README promises is a 150 MB ONNX download
here against a multi-gigabyte wheel there. The weights are the same Hugging Face files
either way; only the runtime differs.

Two properties this module is built around:

* **Lazy.** Importing it imports no ML stack and downloads nothing. ``fastembed`` is
  imported, and the model fetched, on the first :meth:`LocalEmbedder.__call__` — so a
  process that never embeds (a UI that only reads an index) never pays for a model.
* **Cached once per process.** :func:`get_local_embedder` hands out one instance per
  model name, so building the FAISS index and querying it re-use a single loaded model;
  a per-call load would make ingest unusable.

The type contract is the one ``src.core.index.Embedder`` declares — a callable mapping a
batch of texts to a batch of vectors — so this class implements ``__call__`` and returns
plain ``list[list[float]]`` (never a numpy array: the dimension guard in
``src.core.retrieval`` probes the *shape* of the returned rows, and an array row would
hide it). A missing install or a failed download raises :class:`LocalEmbeddingError`,
never a bare ``ImportError`` from inside a node.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any

#: Default model: the small English flag-embedding variant fastembed ships, 384-dim.
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#: Environment variable naming the local model (falls back to the default above).
LOCAL_EMBEDDING_MODEL_ENV = "LOCAL_EMBEDDING_MODEL"

#: Dimensions of the models this repo ships configured, so ``dim`` never has to load a
#: model (and never has to download one) just to describe itself.
_KNOWN_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class LocalEmbeddingError(RuntimeError):
    """The local embedder is unusable: ``fastembed`` is missing or the model failed.

    Carries the actionable sentence a caller can surface verbatim — which install or
    which download went wrong, and which alternative paths (``EMBEDDING_PROVIDER=api``,
    ``MOCK_LLM=1``) need no local model at all.
    """


def local_model_name() -> str:
    return os.environ.get(LOCAL_EMBEDDING_MODEL_ENV, "").strip() or DEFAULT_LOCAL_EMBEDDING_MODEL


def _as_list(vector: Any) -> list[float]:
    """One vector as a plain ``list[float]``, whatever fastembed hands back.

    fastembed yields numpy arrays; ``tolist`` keeps the row a Python list so
    ``src.core.retrieval``'s width probe (``isinstance(vectors[0], (list, tuple))``)
    can actually read it, and so the persisted FAISS rows are plain floats.
    """
    tolist = getattr(vector, "tolist", None)
    values = tolist() if callable(tolist) else list(vector)
    return [float(value) for value in values]


def _metadata_dimension(model_name: str) -> int | None:
    """Width declared by fastembed's model registry, without loading the model.

    ``None`` when fastembed is absent or does not know the name — the caller then falls
    back to a real probe rather than guessing a width.
    """
    try:
        from fastembed import TextEmbedding
    except Exception:
        return None
    try:
        supported = TextEmbedding.list_supported_models()
    except Exception:
        return None
    for entry in supported:
        if not isinstance(entry, dict) or entry.get("model") != model_name:
            continue
        width = entry.get("dim")
        if isinstance(width, int) and width > 0:
            return width
    return None


class LocalEmbedder:
    """A ``fastembed`` text-embedding model behind the ``Embedder`` call shape.

    Construction is cheap and touches nothing: the ONNX model is loaded on first use and
    kept for the life of the instance.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or local_model_name()
        self._dimensions: int | None = _KNOWN_DIMENSIONS.get(self._model_name)
        self._model: Any = None
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        """Vector width this embedder produces.

        Known models answer from the table; an unknown one asks fastembed's registry and,
        failing that, probes the loaded model once. Raises :class:`LocalEmbeddingError`
        only when no width can be established at all (fastembed absent and the model
        unknown), which is the same failure the first embed would report.
        """
        if self._dimensions is not None:
            return self._dimensions
        width = _metadata_dimension(self._model_name)
        if width is None:
            vectors = self._embed(["dimensionality probe"])
            if not vectors:
                raise LocalEmbeddingError(
                    f"the local embedding model {self._model_name!r} returned no vector to "
                    "measure, so its width is unknown"
                )
            width = len(vectors[0])
        self._dimensions = width
        return width

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in one ONNX batch, preserving input order."""
        return self._embed(list(texts))

    # --- internals ---------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Shared batch path (``__call__`` and the ``dim`` probe)."""
        if not texts:
            return []
        model = self._model_instance()
        try:
            vectors = [_as_list(vector) for vector in model.embed(texts)]
        except LocalEmbeddingError:
            raise
        except Exception as exc:  # ONNX/ORT failures raise many types
            raise LocalEmbeddingError(
                f"the local embedding model {self._model_name!r} failed to embed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if len(vectors) != len(texts):
            raise LocalEmbeddingError(
                f"the local embedding model {self._model_name!r} returned {len(vectors)} "
                f"vectors for {len(texts)} texts"
            )
        return vectors

    def _model_instance(self) -> Any:
        """The loaded ``TextEmbedding``, loading (and downloading) it on first use."""
        model = self._model
        if model is not None:
            return model
        with self._load_lock:
            # Re-check under the lock: two threads racing the first embed must load once.
            if self._model is None:
                self._model = self._load_model()
            return self._model

    def _load_model(self) -> Any:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise LocalEmbeddingError(
                "the local embedder needs the 'fastembed' package, which is not installed "
                "(pip install -r requirements.txt pins it). Set EMBEDDING_PROVIDER=api to "
                "use an OpenAI-compatible endpoint instead, or MOCK_LLM=1 for the "
                "deterministic offline path that needs no model at all."
            ) from exc
        try:
            return TextEmbedding(model_name=self._model_name)
        except Exception as exc:  # a failed download, an unknown model, an ONNX error
            raise LocalEmbeddingError(
                f"could not load the local embedding model {self._model_name!r} "
                f"({type(exc).__name__}: {exc}). Its weights are downloaded from Hugging "
                "Face on first use, so this needs network access (and ~150 MB of disk) "
                "once; afterwards they are reused from the local cache. Set "
                "LOCAL_EMBEDDING_MODEL to a smaller model, EMBEDDING_PROVIDER=api to use "
                "a hosted endpoint, or MOCK_LLM=1 for the offline deterministic path."
            ) from exc


#: One instance per model name, created on demand and shared for the whole process.
_CACHE: dict[str, LocalEmbedder] = {}
_CACHE_LOCK = threading.Lock()


def get_local_embedder(model_name: str | None = None) -> LocalEmbedder:
    """The process-wide :class:`LocalEmbedder` for ``model_name`` (default: configured).

    The cache is what lets index-build and query share one loaded model: constructing a
    second one would re-read (or re-download) the same weights and halve throughput.
    """
    name = model_name or local_model_name()
    with _CACHE_LOCK:
        embedder = _CACHE.get(name)
        if embedder is None:
            embedder = LocalEmbedder(name)
            _CACHE[name] = embedder
        return embedder


def build_local_embedder(model_name: str | None = None) -> tuple[LocalEmbedder, str]:
    """The cached local embedder plus the description surfaced in the UI/logs.

    The description names the model and its width (read from the dimension table, so it
    costs no download) — the same three facts the mock/api branches report, so an
    operator can tell from one line which index they are about to build or query.
    """
    embedder = get_local_embedder(model_name)
    return embedder, (
        f"local embedder, {embedder.model_name}, {embedder.dim} dims, "
        "fastembed/ONNX, offline after first use"
    )

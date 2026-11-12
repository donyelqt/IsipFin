from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any

DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

LOCAL_EMBEDDING_MODEL_ENV = "LOCAL_EMBEDDING_MODEL"

_KNOWN_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class LocalEmbeddingError(RuntimeError):
    pass


def local_model_name() -> str:
    return os.environ.get(LOCAL_EMBEDDING_MODEL_ENV, "").strip() or DEFAULT_LOCAL_EMBEDDING_MODEL


def _as_list(vector: Any) -> list[float]:
    tolist = getattr(vector, "tolist", None)
    values = tolist() if callable(tolist) else list(vector)
    return [float(value) for value in values]


def _metadata_dimension(model_name: str) -> int | None:
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
        return self._embed(list(texts))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._model_instance()
        try:
            vectors = [_as_list(vector) for vector in model.embed(texts)]
        except LocalEmbeddingError:
            raise
        except Exception as exc:
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
        model = self._model
        if model is not None:
            return model
        with self._load_lock:
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
        except Exception as exc:
            raise LocalEmbeddingError(
                f"could not load the local embedding model {self._model_name!r} "
                f"({type(exc).__name__}: {exc}). Its weights are downloaded from Hugging "
                "Face on first use, so this needs network access (and ~150 MB of disk) "
                "once; afterwards they are reused from the local cache. Set "
                "LOCAL_EMBEDDING_MODEL to a smaller model, EMBEDDING_PROVIDER=api to use "
                "a hosted endpoint, or MOCK_LLM=1 for the offline deterministic path."
            ) from exc


_CACHE: dict[str, LocalEmbedder] = {}
_CACHE_LOCK = threading.Lock()


def get_local_embedder(model_name: str | None = None) -> LocalEmbedder:
    name = model_name or local_model_name()
    with _CACHE_LOCK:
        embedder = _CACHE.get(name)
        if embedder is None:
            embedder = LocalEmbedder(name)
            _CACHE[name] = embedder
        return embedder


def build_local_embedder(model_name: str | None = None) -> tuple[LocalEmbedder, str]:
    embedder = get_local_embedder(model_name)
    return embedder, (
        f"local embedder, {embedder.model_name}, {embedder.dim} dims, "
        "fastembed/ONNX, offline after first use"
    )

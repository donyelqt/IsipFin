#!/usr/bin/env python3
"""Build the hybrid index over the corpus PDFs in ``data/pdfs/`` (task T5).

Run with no arguments to index every PDF in ``data/pdfs/``, or ``--pdf PATH`` to index
exactly one file. The corpus is the PDFs themselves — this script never fetches anything
and never needs a URL.

Idempotency contract
--------------------
The manifest of indexed documents is ``data/index/docs.jsonl``: one JSON line per
document with the frozen fields ``doc_id``, ``filename``, ``sha256``, ``page_count``,
``sections``. ``doc_id`` *is* the sha256 of the file's bytes (``parse_pdf`` sets it), so
a document already in the catalog is byte-identical to the one indexed: it is reported
as ``skipped`` ("no changes") and neither parsed, embedded, nor re-saved. A re-run that
had nothing to do performs no write to ``docs.jsonl`` at all.

Only the stdlib, the already-installed analysis libraries, and the deterministic
offline embedder are imported — tests must never import this script (docs/CONSTRAINTS.md:
the suite is hermetic, and ``testpaths = ["tests"]`` keeps it that way).

Exit codes
----------
0  success (every document either built or provably already indexed)
1  failure: an unreadable PDF, an embedding error, a missing ``OPENAI_API_KEY`` while
   ``EMBEDDING_PROVIDER=api``, or an ``EMBEDDING_PROVIDER`` value that names no backend
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict

from src.core.index import ChunkIndex, Embedder
from src.core.pdf_parser import ParsedDocument, ParseError, parse_pdf
from src.utils.settings import load_env

if TYPE_CHECKING:
    # Annotation only. The runtime value lives in src.core.retrieval; importing it
    # here would make this script pay for the retrieval stack on every invocation.
    from src.core.retrieval import HybridIndexHandle

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = REPO_ROOT / "data" / "pdfs"
INDEX_DIR = REPO_ROOT / "data" / "index"
CATALOG_FILENAME = "docs.jsonl"

# The five frozen catalog fields (docs/ARCHITECTURE.md data model). Order is the wire
# format; anything else added to a line would be a different contract.
CATALOG_FIELDS = ("doc_id", "filename", "sha256", "page_count", "sections")


# One line of ``docs.jsonl``: the five frozen fields above. ``size_bytes`` is optional
# and read-only — it is never written, and its absence falls back to the file's size.
class CatalogEntry(TypedDict):
    """One document-catalog line: the frozen fields, plus the optional size hint."""

    doc_id: str
    filename: str
    sha256: str
    page_count: int
    sections: list[str]
    size_bytes: NotRequired[int]


DOC_ID_CHARS = 12
MOCK_ENV = "MOCK_LLM"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"
DEFAULT_EMBEDDING_PROVIDER = "local"
API_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"
EMBEDDING_BASE_URL_ENV = "EMBEDDING_BASE_URL"
EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MOCK_DIMENSIONS = 64


class BuildError(Exception):
    """Raised for a corpus that cannot be indexed; the CLI renders it as a message."""


# --- embedders ------------------------------------------------------------------


class MockEmbedder:
    """A deterministic, offline embedder (``MOCK_LLM=1``).

    A text is hashed into a fixed dimensional unit vector. It carries no semantic
    content, so a mock-indexed corpus answers nothing about meaning — but it builds,
    saves, reloads and scores *exactly* like a real index, so the pipeline, the
    artifacts and this script's idempotency are all exercised with no key and no
    network. The hash is sha256-derived rather than :func:`hash` so the vectors are
    identical on every interpreter, platform and run.
    """

    def __init__(self, dimensions: int = MOCK_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Vector width this embedder produces."""
        return self._dimensions

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        """Map every text to one deterministic unit vector of ``dimensions`` floats."""
        return [_mock_vector(text, self._dimensions) for text in texts]


def _mock_vector(text: str, dimensions: int) -> list[float]:
    """One unit vector derived from ``text``: sha256 of the text, folded to ``dimensions``."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = [float(digest[offset] - 127.5) for offset in range(dimensions) if offset < len(digest)]
    if len(vector) < dimensions:  # digest shorter than the requested width
        vector = (vector * ((dimensions // max(1, len(vector))) + 1))[:dimensions]
    norm = float(sum(value * value for value in vector)) ** 0.5
    if norm == 0.0:
        return [1.0 if position == 0 else 0.0 for position in range(dimensions)]
    return [value / norm for value in vector]


class OpenAIEmbedder:
    """An embedder over any OpenAI-compatible endpoint, built from the environment.

    ``OPENAI_API_KEY`` is the only mandatory variable; a missing one is reported by
    :func:`build_embedder` before any work starts. ``OPENAI_BASE_URL`` may point at
    Ollama, vLLM or any gateway (ADR-005 uses one client for chat and embeddings).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - pinned in requirements.txt
            raise BuildError(
                "the OpenAI client is not installed; run `pip install -r requirements.txt`"
            ) from exc
        self._model = model
        self._dimensions = dimensions
        self._client = OpenAI(api_key=api_key, base_url=base_url or DEFAULT_BASE_URL)
        self._endpoint = base_url or DEFAULT_BASE_URL

    @property
    def model(self) -> str:
        """Embedding model id this client requests."""
        return self._model

    @property
    def endpoint(self) -> str:
        """Base URL the client was built against."""
        return self._endpoint

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text in one request and return the vectors in order."""
        if not texts:
            return []
        # The SDK models every keyword precisely (model as a literal union, dimensions
        # as ``Omit``, extra_headers as a Mapping, …), so a ``**``-unpacked dict cannot
        # satisfy its overloads. The three keys this class owns are passed explicitly
        # instead, each already typed the way the SDK expects: ``dimensions`` is
        # omitted entirely when unset (never sent as 0, which is not a valid width).
        try:
            response = (
                self._client.embeddings.create(
                    model=self._model,
                    input=list(texts),
                    dimensions=self._dimensions,
                )
                if self._dimensions is not None
                else self._client.embeddings.create(model=self._model, input=list(texts))
            )
        except Exception as exc:  # the client raises many transport-level types
            raise BuildError(
                f"embedding request to {self._endpoint} failed ({type(exc).__name__}: {exc})"
            ) from exc
        vectors = [list(item.embedding) for item in response.data]
        if len(vectors) != len(texts):
            raise BuildError(
                f"{self._endpoint} returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return vectors


def build_embedder() -> tuple[Embedder, str]:
    """Resolve the embedder from the environment and return it with a description.

    ``MOCK_LLM=1`` selects the deterministic offline embedder, and is checked *first* so
    the keyless CI/reviewer path never consults a provider or a model at all. Otherwise
    ``EMBEDDING_PROVIDER`` chooses the backend — ``local`` (the default) is an in-process
    fastembed ONNX model needing no key, ``api`` is the OpenAI-compatible client. A key
    missing for ``api``, or an unknown provider name, fails loudly here — before a corpus
    is parsed and half an index is written — rather than producing an index with no dense
    leg. Both ingest (``src/components/upload.py``) and query
    (``src/core/retrieval.py``) resolve through this one function, so they cannot
    disagree about which provider — or which width — an index was built with.
    """
    if os.environ.get(MOCK_ENV, "").strip() == "1":
        mock_embedder = MockEmbedder()
        return mock_embedder, (
            f"mock embedder, deterministic, {mock_embedder.dimensions} dims, offline"
        )

    provider = (
        os.environ.get(EMBEDDING_PROVIDER_ENV, "").strip().lower() or DEFAULT_EMBEDDING_PROVIDER
    )
    if provider == "local":
        return _build_local_embedder()
    if provider == "api":
        return _build_api_embedder()
    raise BuildError(
        f"{EMBEDDING_PROVIDER_ENV}={provider!r} is not a known embedding provider; use "
        f"'local' (fastembed, the default, no key needed), 'api' (an OpenAI-compatible "
        f"/embeddings endpoint), or {MOCK_ENV}=1 for the deterministic offline path."
    )


def _build_local_embedder() -> tuple[Embedder, str]:
    """The in-process fastembed embedder (``EMBEDDING_PROVIDER=local``, the default).

    The ML import and the model download are deferred to ``src.core.local_embedder``'s
    first use, so resolving the policy here stays cheap and this script never pays for
    ONNX when the mock path is taken. The local module's typed failure is re-raised as
    this module's :class:`BuildError`, so every resolution failure — from any provider —
    reaches the CLI as one message.
    """
    from src.core.local_embedder import LocalEmbeddingError, build_local_embedder

    try:
        return build_local_embedder()
    except LocalEmbeddingError as exc:
        raise BuildError(str(exc)) from exc


def _build_api_embedder() -> tuple[Embedder, str]:
    """The OpenAI-compatible embedder (``EMBEDDING_PROVIDER=api`` — today's behaviour).

    A missing ``OPENAI_API_KEY`` fails here, before any corpus work, rather than
    producing an index with no dense leg.
    """
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise BuildError(
            f"embedding needs a key: {API_KEY_ENV} is not set and {MOCK_ENV} is not '1'. "
            f"Either export {API_KEY_ENV} (and optionally {BASE_URL_ENV} / "
            f"{EMBEDDING_MODEL_ENV}), or run with {MOCK_ENV}=1 for the deterministic "
            "offline path that needs no key and no network."
        )
    # Embeddings may live on a different provider than chat: Groq serves chat only
    # and has no /embeddings endpoint, so EMBEDDING_BASE_URL (falling back to the
    # chat base URL) keeps the two independently configurable.
    base_url = (
        os.environ.get(EMBEDDING_BASE_URL_ENV, "").strip()
        or os.environ.get(BASE_URL_ENV, "").strip()
        or None
    )
    model = os.environ.get(EMBEDDING_MODEL_ENV, "").strip() or DEFAULT_EMBEDDING_MODEL
    try:
        embedder: OpenAIEmbedder = OpenAIEmbedder(api_key=api_key, base_url=base_url, model=model)
    except BuildError as exc:
        raise exc
    except Exception as exc:  # pragma: no cover - constructor failures are rare
        raise BuildError(f"could not build the OpenAI-compatible client: {exc}") from exc
    return embedder, f"OpenAI-compatible embedder, {model} via {embedder.endpoint}"


# --- catalog --------------------------------------------------------------------


def catalog_path(index_dir: Path = INDEX_DIR) -> Path:
    """The one-line-per-document catalog, written alongside T4's index artifacts."""
    return index_dir / CATALOG_FILENAME


def read_catalog(path: Path | None = None) -> dict[str, CatalogEntry]:
    """The catalog as ``{doc_id: entry}``; ``{}`` when nothing has been indexed yet."""
    target = path or catalog_path()
    if not target.is_file():
        return {}
    entries: dict[str, CatalogEntry] = {}
    with open(target, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f"{target}:{number} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict) or "doc_id" not in payload:
                raise BuildError(f"{target}:{number} is not a catalog line (no 'doc_id')")
            missing = [field for field in CATALOG_FIELDS if field not in payload]
            if missing:
                # A line without the frozen fields cannot be compared digests against,
                # so a document it describes could be re-indexed over identical bytes.
                raise BuildError(f"{target}:{number} is missing catalog field(s) {missing}")
            # The line is JSON, so its values are coerced into the entry's declared
            # types here; every one of the frozen fields is present (checked above).
            entry: CatalogEntry = {
                "doc_id": str(payload["doc_id"]),
                "filename": str(payload.get("filename", "")),
                "sha256": str(payload.get("sha256", "")),
                "page_count": int(payload.get("page_count", 0)),
                "sections": list(payload.get("sections", [])),
            }
            if "size_bytes" in payload:
                entry["size_bytes"] = int(payload["size_bytes"])
            entries[str(payload["doc_id"])] = entry
    return entries


def write_catalog(entries: Iterable[CatalogEntry], path: Path | None = None) -> None:
    """Rewrite the catalog from scratch, in ``doc_id`` order, with the frozen fields.

    A rewrite (rather than an append) keeps the file sorted and unique even when the
    callers change, and guarantees ``doc_id`` uniqueness: one document, one line.
    """
    target = path or catalog_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda entry: str(entry["doc_id"]))
    lines = [json.dumps(entry, sort_keys=True) for entry in ordered]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def sha256_of(path: Path) -> str:
    """Digest of a file's bytes, read in chunks so large documents stay cheap."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def sections_of(doc: ParsedDocument) -> list[str]:
    """The document's section paths, flattened to ``"a > b"`` strings, in first-seen order.

    A block with no section (a cover page before any heading) contributes nothing, so
    ``sections`` lists only real paths, deduplicated in document order.
    """
    sections: list[str] = []
    seen: set[str] = set()
    for block in doc.blocks:
        joined = " > ".join(block.section_path)
        if joined and joined not in seen:
            seen.add(joined)
            sections.append(joined)
    return sections


# --- indexing -------------------------------------------------------------------


def discover_pdfs(directory: Path, explicit: Sequence[Path] | None = None) -> list[Path]:
    """The PDFs to index: the explicitly named files, else every PDF in ``directory``.

    Non-PDF files in ``directory`` are ignored rather than fatal, so a stray manifest,
    note or partial download cannot stop a corpus build.
    """
    if explicit is not None:
        return [Path(item) for item in explicit]
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )


def validate_pdf(path: Path) -> None:
    """Fail loudly on a missing or non-PDF file, before it reaches the parser."""
    if not path.is_file():
        raise BuildError(f"no such PDF: {path}")
    with open(path, "rb") as handle:
        header = handle.read(5)
    if not header.startswith(b"%PDF"):
        rendering = (
            "".join(chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}" for byte in header)
            or "nothing"
        )
        raise BuildError(
            f"{path} is not a PDF: expected leading bytes %PDF, got {rendering}. "
            "Download it with scripts/download_corpus.py or drop a real PDF here."
        )


@dataclass(frozen=True)
class DocReport:
    """What happened to one PDF: the summary line and the catalog entry it produced."""

    doc_id: str  # full digest
    filename: str
    page_count: int
    parents: int
    children: int
    action: str  # "built" or "skipped"
    sections: int
    size_bytes: int

    @property
    def short_id(self) -> str:
        """The 12-character id used in the summary table (parent ids use it too)."""
        return self.doc_id[:DOC_ID_CHARS]


def _load_existing(index_dir: Path, embedder: Embedder) -> HybridIndexHandle | None:
    """The index already on disk, as a handle, or ``None`` when nothing is persisted.

    T4 owns that loader: the chunks, the FAISS rows and the BM25 pickle are re-read
    together, so a build that adds one document never disturbs the others' rows.
    ``load_index`` is imported here, and ``HybridIndexHandle`` under ``TYPE_CHECKING``
    at module scope, so a build that never touches an existing index does not pay
    for the retrieval stack at runtime while the annotation still resolves.

    A ``RetrievalError`` — the width mismatch between the configured embedder and the
    vectors already on disk is the one that matters — becomes this script's
    :class:`BuildError`, so the CLI prints the actionable message and exits 1 instead of
    letting a mismatch escape as a traceback.
    """
    from src.core.retrieval import RetrievalError, load_index

    try:
        return load_index(index_dir, embedder)
    except RetrievalError as exc:
        raise BuildError(str(exc)) from exc


def index_one(
    path: Path, embedder: Embedder, catalog: dict[str, CatalogEntry], index_dir: Path
) -> tuple[DocReport, ParsedDocument | None]:
    """Index one PDF, or skip it when the catalog already holds its digest.

    The digest is read from the file itself, which is the only cheap proof that the
    bytes on disk are the bytes that were indexed: ``doc_id`` is the sha256 of the
    document's bytes, so a catalog match means the content is unchanged and the indexed
    chunks cannot have changed either. The document is then reported as ``skipped`` and
    is never parsed, embedded or written.

    Otherwise T4's ``build_index`` chunks the document and embeds its children through
    the injected embedder, the handle is saved into ``index_dir`` with T4's own layout
    (``chunks.jsonl``, ``bm25.pkl``, ``faiss/index.faiss`` + ``faiss/meta.json``).

    Returns the report plus the parsed document it described, or ``None`` for a
    document that was skipped (its catalog line is reused verbatim, so its sections are
    never recomputed from a second read of the file).
    """
    validate_pdf(path)
    digest = sha256_of(path)
    if digest in catalog:
        entry = catalog[digest]
        # Count the rows this document actually has in the persisted index, so a
        # "skipped" line reports the index's real state rather than the catalog's
        # subset. The index is loaded lazily and only when something is skipped.
        parents, children = 0, 0
        persisted = _load_existing(index_dir, embedder)
        if persisted is not None:
            rows = [r for r in persisted.index.records if r.doc_id == digest]
            children = len(rows)
            parents = len({r.parent_id for r in rows})
        return (
            DocReport(
                doc_id=digest,
                filename=str(entry.get("filename", path.name)),
                page_count=int(entry.get("page_count", 0)),
                parents=parents,
                children=children,
                action="skipped",
                sections=len(entry.get("sections", [])),
                size_bytes=int(entry.get("size_bytes", path.stat().st_size)),
            ),
            None,
        )

    try:
        doc = parse_pdf(path)
    except ParseError as exc:
        raise BuildError(f"{path} could not be parsed: {exc}") from exc

    if digest != doc.doc_id:
        # parse_pdf hashes the same bytes this function does; a mismatch means one of
        # the two reads was not the file we think it is, and an index built on a wrong
        # digest is a silently wrong index.
        raise BuildError(
            f"{path}: computed sha256 {digest[:16]}… but the parser reported "
            f"{doc.doc_id[:16]}… — the file changed while it was being read"
        )

    # Build on the index already on disk, not on an empty one: T4's build_index() starts
    # from scratch, and saving that handle would erase every other document's rows. The
    # merge is ChunkIndex.replace_document() — same-document rows are dropped, every
    # other document's rows (and their vectors) are carried over.
    existing = _load_existing(index_dir, embedder)
    try:
        base = existing.index if existing is not None else ChunkIndex(())
        merged = base.replace_document(doc, embedder=embedder)
        merged.save(index_dir)
    except Exception as exc:  # FAISS/BM25/IO failures must not look like success
        raise BuildError(f"could not write the index under {index_dir}: {exc}") from exc

    this_document = [record for record in merged.records if record.doc_id == digest]
    parents = len({record.parent_id for record in this_document})
    return (
        DocReport(
            doc_id=digest,
            filename=path.name,
            page_count=doc.page_count,
            parents=parents,
            children=len(this_document),
            action="built",
            sections=len(sections_of(doc)),
            size_bytes=path.stat().st_size,
        ),
        doc,
    )


def _entry_for(report: DocReport, doc: ParsedDocument) -> CatalogEntry:
    """The frozen five-field catalog line for one document (docs/ARCHITECTURE.md)."""
    return {
        "doc_id": report.doc_id,
        "filename": report.filename,
        "sha256": report.doc_id,
        "page_count": report.page_count,
        "sections": sections_of(doc),
    }


def _format_table(reports: Sequence[DocReport]) -> str:
    """The per-document summary table, aligned on the id/filename columns."""
    header = ("doc_id", "filename", "pages", "parents", "children", "action")
    rows = [
        (
            r.short_id,
            r.filename,
            str(r.page_count),
            str(r.parents),
            str(r.children),
            r.action,
        )
        for r in reports
    ]
    widths = [
        (
            max(len(header[column]), *(len(row[column]) for row in rows))
            if rows
            else len(header[column])
        )
        for column in range(len(header))
    ]
    separator = "  ".join("-" * widths[column] for column in range(len(header)))
    lines = ["  ".join(header[c].ljust(widths[c]) for c in range(len(header))).rstrip(), separator]
    for row in rows:
        lines.append("  ".join(row[c].ljust(widths[c]) for c in range(len(header))).rstrip())
    return "\n".join(lines)


def _index_size(index_dir: Path) -> str:
    """Total on-disk size of the index artifacts (the catalog itself excluded)."""
    total = 0
    for child in sorted(index_dir.rglob("*")):
        if child.is_file() and child.name != CATALOG_FILENAME:
            total += child.stat().st_size
    if total == 0:
        return "0 B"
    value = float(total)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def build(
    pdf_paths: Sequence[Path],
    pdf_dir: Path,
    index_dir: Path,
    embedder_name: str,
    embedder: Embedder,
) -> int:
    """Index every requested PDF and summarize; returns the process exit code."""
    if not pdf_paths:
        print(f"no PDFs to index: {pdf_dir} contains no .pdf files")
        print(f"  put the corpus PDF in {pdf_dir} (or pass --pdf PATH), then run this again")
        return 1

    catalog = read_catalog(catalog_path(index_dir))
    entries: dict[str, CatalogEntry] = {str(entry["doc_id"]): entry for entry in catalog.values()}
    reports: list[DocReport] = []
    documents: dict[str, ParsedDocument] = {}
    failures: list[str] = []

    print(f"embedder: {embedder_name}")
    print(f"index:    {index_dir}")
    print(f"pdfs:     {len(pdf_paths)} to consider under {pdf_dir}")

    for path in pdf_paths:
        try:
            report, doc = index_one(path, embedder, entries, index_dir)
        except BuildError as exc:
            failures.append(str(exc))
            print(f"  ✗ {path.name}: {exc}", file=sys.stderr)
            continue
        reports.append(report)
        if doc is not None:
            documents[report.doc_id] = doc
        if report.action == "built":
            print(f"  ✓ built {report.short_id}  {path.name}")
        else:
            print(f"  ✓ skipped {report.short_id}  {path.name} (already indexed, no changes)")

    if reports:
        print()
        print(_format_table(reports))
        for report in reports:
            # A skipped document keeps the sections already in the catalog verbatim;
            # a freshly built one writes the sections of the bytes that were indexed.
            if report.doc_id not in entries:
                entries[report.doc_id] = _entry_for(report, documents[report.doc_id])
        # Same path the catalog was read from: --index-dir must own the catalog too, or
        # a build into a scratch directory rewrites the default index's catalog.
        write_catalog(list(entries.values()), catalog_path(index_dir))
        print()
        print(f"catalog: {catalog_path(index_dir)} ({len(entries)} documents)")
        print(f"index:   {_index_size(index_dir)} in {index_dir}")

    if failures:
        print()
        print(
            f"failed: {len(failures)} of {len(pdf_paths)} PDF(s) could not be indexed",
            file=sys.stderr,
        )
        for message in failures:
            print(f"  - {message}", file=sys.stderr)
        return 1

    if reports and all(report.action == "skipped" for report in reports):
        print()
        print("no changes: every PDF was already indexed (sha256 match)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_index.py",
        description="Build the hybrid index over the corpus PDFs in data/pdfs/.",
        epilog=(
            "Idempotent by sha256: a PDF whose digest is already in data/index/docs.jsonl "
            "is skipped and reported as 'no changes'."
        ),
    )
    parser.add_argument(
        "--pdf",
        action="append",
        default=None,
        metavar="PATH",
        help="index exactly this PDF (repeatable); default: every PDF in data/pdfs",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(PDF_DIR),
        help=f"corpus directory scanned when --pdf is not given (default: {PDF_DIR})",
    )
    parser.add_argument(
        "--index-dir",
        default=str(INDEX_DIR),
        help=f"index output directory, reused T4's layout (default: {INDEX_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    load_env()
    args = parse_args(argv)
    index_dir = Path(args.index_dir)
    pdf_dir = Path(args.pdf_dir)
    try:
        embedder, embedder_name = build_embedder()
        explicit = [Path(item) for item in args.pdf] if args.pdf else None
        for path in explicit or []:
            if not path.is_file():
                raise BuildError(f"no such PDF: {path}")
        pdf_paths = discover_pdfs(pdf_dir, explicit)
        return build(pdf_paths, pdf_dir, index_dir, embedder_name, embedder)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

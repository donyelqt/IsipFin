"""The ingest surface: multi-PDF uploader, index-build progress, document catalog (task T18).

ADR-009's boundary rule reads "no business logic in the UI: the UI calls the graph and the
store and renders what they decided". Ingest is the one place where the UI is allowed to
*drive* a pipeline step, because docs/PLAN.md T18 requires "upload → index build progress →
empty state → first Q&A" and the app builds its own index from this panel as well as from
``scripts/build_index.py``. So this module orchestrates exactly one run and nothing else:
it calls the ingest modules only (:func:`src.core.pdf_parser.parse_pdf`,
``ChunkIndex.replace_document``/``ChunkIndex.save`` — which is where
``src.core.chunking`` runs), reuses the frozen catalog format and the embedder policy that
live in ``scripts/build_index.py`` (imported, never reimplemented), and touches nothing
outside ``data/``. It builds no prompt, never calls ``src.chains.graph``, and never imports
``openai``: the embedder is resolved by the builder's own policy, so a real client is the
builder's object.

Idempotency is inherited, not re-invented. ``doc_id`` is the sha256 of the file's bytes
(:func:`parse_pdf` sets it), so a PDF already in ``data/index/docs.jsonl`` is reported as
``skipped`` with no parse, no embed and no write — which is precisely what
``scripts/build_index.py:index_one`` does, the function this panel reproduces for a
user-chosen set of files. Every other document's rows survive: the new document is merged
into the index already on disk rather than replacing it.

Public surface
--------------
``read_catalog``            the frozen catalog as ``{doc_id: entry}`` (``{}`` when unbuilt)
``index_status``            ready / documents / chunks / on-disk size / reason; never raises
``ingest_documents``        parse → embed → save a set of uploaded PDFs, per-file outcomes
``render_upload_panel``     the uploader, the index-build progress and its status cards
``render_document_catalog`` the catalog card: one row per indexed document
``render_index_status``     the small "index ready" card
``render_file_result(s)``   one status card (indexed / already indexed / failed + reason)
``document_names``          ``doc_id`` → filename, for the chat panel's citation labels

Every visible state exists here: the uploader's "nothing selected" caption, a status
container with a live progress bar while a build runs, one card per uploaded file carrying
the outcome and the reason, a scanned-page warning (the parser's own "no text layer" marker,
surfaced rather than dropped), an index-not-built line on the status card, and per-file error
cards that never stop the other uploads.
"""

from __future__ import annotations

import functools
import hashlib
import html
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import streamlit as st

from src.core.index import ChunkIndex
from src.core.pdf_parser import ParseError, parse_pdf

if TYPE_CHECKING:  # annotation only: the runtime value is whatever the builder resolved
    from src.core.index import Embedder

__all__ = [
    "CATALOG_FIELDS",
    "CATALOG_FILENAME",
    "INDEX_DIR",
    "PDF_DIR",
    "UPLOAD_STATE_KEY",
    "CatalogReadError",
    "EmbedderUnavailable",
    "FileResult",
    "IndexStatus",
    "IngestError",
    "IngestResult",
    "document_names",
    "index_status",
    "ingest_documents",
    "read_catalog",
    "render_document_catalog",
    "render_file_result",
    "render_file_results",
    "render_index_status",
    "render_upload_panel",
]

# --- locations ------------------------------------------------------------------
# ``scripts/build_index.py`` derives its own default catalog directory from the repository
# root, and the uploader must write the index the query path (and a later script run) reads.
# The same absolute derivation is what keeps a UI-built index indistinguishable from a
# script-built one no matter which directory Streamlit was started from.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Corpus directory for uploaded PDFs, mirroring the data model in docs/ARCHITECTURE.md.
PDF_DIR = REPO_ROOT / "data" / "pdfs"

#: Index directory the artifacts are read from and written to. ``INDEX_DIR`` in the environment
#: overrides it: the mock embedder is 64-dimensional and a real one 384, and a single directory
#: cannot hold both, so a reader running `MOCK_LLM=1` after a real build points this at the mock
#: index rather than rebuilding. Left unset it is the ordinary `data/index`.
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "").strip() or (REPO_ROOT / "data" / "index"))

#: The catalog file, one JSON line per document (the frozen wire format).
CATALOG_FILENAME = "docs.jsonl"

#: The five frozen catalog fields (docs/ARCHITECTURE.md data model); order is the wire format.
CATALOG_FIELDS = ("doc_id", "filename", "sha256", "page_count", "sections")

#: Leading bytes every PDF must carry. A browser-reported ``.pdf`` extension is never
#: trusted over the bytes themselves (the same check ``download_corpus.require_pdf`` makes).
PDF_MAGIC = b"%PDF"
_SNIFF_BYTES = 8

#: Longest stored filename — the cap ``download_corpus.py`` applies to installed files.
_MAX_FILENAME_CHARS = 80

#: The marker :func:`parse_pdf` writes for a page with no text layer (a scanned image).
#: The literal is duplicated here deliberately: it is the only signal the parser emits for
#: the pages an analyst must be warned about, and it carries no field of its own.
NO_TEXT_LAYER = "[no text layer]"

#: Section paths listed per document before the catalog card truncates the rest.
_MAX_SECTIONS_LISTED = 12

#: Session-state key holding the last ingest result, so its per-file cards survive the rerun
#: that any other widget on the page triggers. A twentieth of a megabyte of status text is
#: UI wiring, exactly what ``st.session_state`` is documented to hold.
UPLOAD_STATE_KEY = "isip-upload-result"
_ACTION_KEY = "isip-upload-index"

#: Uploader label, action label, and the "nothing selected" caption.
UPLOAD_LABEL = "Upload PDF documents"
ACTION_LABEL = "Index document"
ACTION_LABEL_PLURAL = "Index documents"
UPLOAD_EMPTY = "No files selected — upload one or more PDFs to index them."

#: Status card copy for the index that has never been built.
INDEX_EMPTY = "No index yet — upload a PDF to build one."
CATALOG_EMPTY = "No document indexed yet — upload a PDF to add one."

#: Section path rendering: ``scripts/build_index.sections_of`` flattens to ``"a > b"``.
_SECTION_SEPARATOR = " > "
_SHORT_DOC_ID = 12

# --- outcomes -------------------------------------------------------------------


@dataclass(frozen=True)
class FileResult:
    """What happened to one uploaded PDF: the outcome and the numbers behind it.

    ``state`` is ``"indexed"`` (parsed, embedded, written), ``"skipped"`` (its digest was
    already in the catalog — no work done), or ``"failed"`` (nothing written; ``detail``
    carries the reason). ``parents``/``children`` are counts out of the merged index for
    this document only, which is the same accounting ``scripts/build_index.py`` reports.
    """

    filename: str
    state: Literal["indexed", "skipped", "failed"]
    doc_id: str | None = None
    page_count: int | None = None
    parents: int = 0
    children: int = 0
    sections: int = 0
    size_bytes: int = 0
    detail: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class IndexStatus:
    """What the on-disk index holds right now, read without resolving any embedder.

    ``ready`` means "there are chunks to retrieve over"; every other state carries the
    reason so the panel can say why the index is unusable instead of showing an empty one.
    """

    ready: bool
    doc_count: int
    chunk_count: int
    size_bytes: int
    size_human: str
    reason: str | None = None


@dataclass(frozen=True)
class IngestResult:
    """The outcome of one ingest run: one :class:`FileResult` per upload, plus the index."""

    files: tuple[FileResult, ...] = ()
    index: IndexStatus = field(
        default_factory=lambda: IndexStatus(False, 0, 0, 0, "0 B", INDEX_EMPTY)
    )

    @property
    def indexed(self) -> tuple[FileResult, ...]:
        """The uploads that were parsed, embedded and written."""
        return tuple(item for item in self.files if item.state == "indexed")

    @property
    def skipped(self) -> tuple[FileResult, ...]:
        """The uploads already indexed under the same digest."""
        return tuple(item for item in self.files if item.state == "skipped")

    @property
    def failed(self) -> tuple[FileResult, ...]:
        """The uploads that failed, each with its reason on the card."""
        return tuple(item for item in self.files if item.state == "failed")

    def summary(self) -> str:
        """The one-line verdict rendered as the status container's final label."""

        total = len(self.files)
        parts = [
            f"{len(self.indexed)} indexed",
            f"{len(self.skipped)} already indexed",
            f"{len(self.failed)} failed",
        ]
        return f"Indexed {total} upload(s): " + ", ".join(parts)


class IngestError(ValueError):
    """Base class for ingest-surface failures: the catalog is unreadable or unwritable.

    A ``ValueError`` base is deliberate — a malformed catalog line is, like a bad JSON body,
    an invalid value, and ``json.JSONDecodeError`` is a ``ValueError`` subclass too, so the
    builder's own error maps in without a second exception hierarchy.
    """


class CatalogReadError(IngestError):
    """The document catalog could not be read or parsed (one bad line is enough)."""


class EmbedderUnavailable(IngestError):
    """No embedder could be resolved, so nothing new can be embedded.

    The message is the builder's own: it names the missing variable and the ``MOCK_LLM=1``
    alternative, which is the actionable answer an analyst needs on this card.
    """


# --- the builder as a policy source ---------------------------------------------


def _builder() -> Any:
    """``scripts/build_index.py`` as a module, for the policy this panel must not duplicate.

    The frozen catalog format, the section flattening and the embedder resolution all live
    there, and reimplementing any of them would let the UI and the script disagree about
    which wire format "indexed" means. Retrieval resolves the same script the same way
    (``src/core/retrieval._script_build_embedder``).
    """
    try:
        import scripts.build_index as module
    except Exception as exc:  # a missing file or a broken dependency is a typed failure here
        raise IngestError(f"scripts/build_index.py is unavailable: {exc}") from exc
    return module


def _resolve_embedder() -> tuple[Embedder, str]:
    """The ingest-time embedder, resolved by the builder's environment policy.

    The policy — including ``EMBEDDING_PROVIDER`` — lives entirely in
    ``scripts/build_index.build_embedder``, which this calls; this function only rebinds
    the failure type, so ingest and query cannot disagree about the provider or the
    width. In short: ``MOCK_LLM=1`` selects the deterministic offline embedder; otherwise
    ``EMBEDDING_PROVIDER`` picks ``local`` (fastembed in-process, the default, no key) or
    ``api`` (an OpenAI-compatible client from ``OPENAI_API_KEY``, whose absence fails
    here, before any PDF is parsed or half an index written). The type is made this
    module's own (the builder raises its ``BuildError``); the message is kept verbatim.
    """
    try:
        embedder, description = _builder().build_embedder()
    except IngestError as exc:  # the builder module itself is unusable
        raise EmbedderUnavailable(str(exc)) from exc
    except Exception as exc:  # the builder's typed policy failure (missing key, bad client)
        raise EmbedderUnavailable(str(exc)) from exc
    return embedder, str(description)


# --- catalog and index reads -----------------------------------------------------


def _catalog_path(index_dir: Path | None) -> Path:
    """The catalog file inside ``index_dir`` (or this module's default index directory)."""
    target = Path(index_dir) if index_dir is not None else INDEX_DIR
    return target / CATALOG_FILENAME


def read_catalog(index_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Read the document catalog as ``{doc_id: entry}``; ``{}`` when nothing is indexed.

    Each entry carries the five frozen fields (``doc_id``, ``filename``, ``sha256``,
    ``page_count``, ``sections``) plus ``size_bytes`` when the writer recorded one. The file
    format is the builder's, so one malformed line raises :class:`CatalogReadError` rather
    than silently dropping the document it describes — a document the catalog forgets is a
    document that would be re-indexed over identical bytes.
    """
    try:
        entries = _builder().read_catalog(_catalog_path(index_dir))
    except IngestError:
        raise
    except Exception as exc:  # the builder's BuildError on a bad line, or an unreadable file
        raise CatalogReadError(str(exc)) from exc
    # Sorted so a caller rendering this is deterministic; the builder writes it sorted.
    return {str(key): dict(value) for key, value in sorted(entries.items())}


def index_status(index_dir: Path | None = None) -> IndexStatus:
    """Describe the index on disk without resolving an embedder or loading any chunk.

    Reads the artifact sizes and counts ``chunks.jsonl``'s lines — the source of truth holds
    one record per child chunk, so a line count is the chunk count with no JSON parse. Never
    raises: an unreadable or malformed index is reported as not-ready with its reason, which
    is what the panel renders.
    """
    target = Path(index_dir) if index_dir is not None else INDEX_DIR
    chunks = target / "chunks.jsonl"
    try:
        total = _artifact_bytes(target)
        documents = len(read_catalog(target))
        if not chunks.is_file():
            return IndexStatus(False, documents, 0, total, _human_size(total), INDEX_EMPTY)
        rows = _count_records(chunks)
        if rows == 0:
            return IndexStatus(
                False,
                documents,
                0,
                total,
                _human_size(total),
                f"the index at {target} is empty (no chunks have been written yet)",
            )
        return IndexStatus(True, documents, rows, total, _human_size(total))
    except (OSError, IngestError) as exc:
        return IndexStatus(False, 0, 0, 0, "0 B", f"could not read the index at {target}: {exc}")


def document_names(
    entries: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    index_dir: Path | None = None,
) -> dict[str, str]:
    """``doc_id`` → display name, for the chat panel's citation labels.

    A citation's raw sha256 is not something a reader can follow, so the transcript maps
    each ``doc_id`` to the filename that was indexed. Pass ``entries`` (already read for
    the caller's own reasons) or let this function read the catalog itself.
    """
    catalog = read_catalog(index_dir) if entries is None else dict(entries)
    return {
        doc_id: str(entry.get("filename") or doc_id[:_SHORT_DOC_ID])
        for doc_id, entry in catalog.items()
    }


def _artifact_bytes(target: Path) -> int:
    """On-disk size of the index artifacts, the catalog itself excluded (as the builder counts)."""
    total = 0
    for child in sorted(target.rglob("*")):
        if child.is_file() and child.name != CATALOG_FILENAME:
            total += child.stat().st_size
    return total


def _count_records(path: Path) -> int:
    """Non-blank lines of ``chunks.jsonl``: one line is one child chunk."""
    with open(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())


def _human_size(value: int) -> str:
    """Human-readable byte count, the unit ladder ``download_corpus.fmt_size`` uses."""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


# --- ingest -----------------------------------------------------------------------


def ingest_documents(
    files: Sequence[Any],
    *,
    index_dir: Path | None = None,
    pdf_dir: Path | None = None,
    embedder: Embedder | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> IngestResult:
    """Parse, embed and save a set of uploaded PDFs, one file at a time.

    Mirrors ``scripts/build_index.py index_one`` for a user-chosen set: sniff the bytes,
    copy the PDF into ``data/pdfs/``, skip a document whose digest the catalog already
    holds, parse it, merge it into the index already on disk (never replacing it), and
    append its catalog line. Files are handled independently, so one unreadable PDF costs
    one failed card and never the other uploads.

    Args:
        files: Streamlit ``UploadedFile`` objects, or anything with ``.name`` and ``.read()``.
        index_dir: Index directory; defaults to this module's ``INDEX_DIR``.
        pdf_dir: Corpus directory the PDFs are copied into; defaults to ``PDF_DIR``.
        embedder: Injected embedder; ``None`` resolves one through the builder's policy.
        on_progress: Optional ``(fraction, message)`` progress callback.

    Returns:
        The :class:`IngestResult` for every file, with the index's state after the run.

    Raises:
        IngestError: The embedder policy cannot be resolved (the builder's message).
    """
    target = Path(index_dir) if index_dir is not None else INDEX_DIR
    corpus = Path(pdf_dir) if pdf_dir is not None else PDF_DIR
    resolved: dict[str, Any] = {}
    total = len(files)
    results: list[FileResult] = []
    for position, file in enumerate(files, start=1):
        progress = functools.partial(_scaled_report, on_progress, position - 1, total)
        results.append(
            _ingest_one(
                file,
                corpus_dir=corpus,
                index_dir=target,
                embedder=embedder,
                resolved=resolved,
                progress=progress,
            )
        )
    if on_progress is not None:
        on_progress(1.0, "Index build finished")
    return IngestResult(tuple(results), index_status(target))


def _scaled_report(
    on_progress: Callable[[float, str], None] | None,
    done: int,
    total: int,
    share: float,
    message: str,
) -> None:
    """Turn one file's 0..1 progress into the run's 0..1 progress across ``total`` files."""
    if on_progress is None:
        return
    overall = (done + min(max(share, 0.0), 1.0)) / total
    on_progress(overall, message)


def _ingest_one(
    file: Any,
    *,
    corpus_dir: Path,
    index_dir: Path,
    embedder: Embedder | None,
    resolved: dict[str, Any],
    progress: Callable[[float, str], None],
) -> FileResult:
    """Ingest one upload; a failure returns its card and never stops the remaining files."""
    filename = _upload_name(file)
    progress(0.1, f"Reading {filename}")
    try:
        payload = file.read()
    except Exception as exc:  # an upload stream that cannot be read is a per-file failure
        return _failed(filename, f"the upload could not be read: {exc}")
    if not payload:
        return _failed(filename, "the file is empty")

    prefix = payload[:_SNIFF_BYTES]
    if not prefix.startswith(PDF_MAGIC):
        return _failed(
            filename,
            f"not a PDF: expected leading bytes {PDF_MAGIC.decode()}, got {_readable(prefix)}",
        )
    digest = hashlib.sha256(payload).hexdigest()

    progress(0.25, f"Saving {filename}")
    try:
        path = _pdf_target(corpus_dir, filename, digest)
        corpus_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    except OSError as exc:
        return _failed(filename, f"could not save the PDF into {corpus_dir}: {exc}")

    catalog = read_catalog(index_dir)
    if digest in catalog:
        entry = catalog[digest]
        progress(1.0, f"{filename} already indexed")
        return FileResult(
            filename=filename,
            state="skipped",
            doc_id=digest,
            page_count=entry.get("page_count"),
            size_bytes=len(payload),
            detail=(
                "already indexed as "
                f"{entry.get('filename') or filename} — identical bytes, nothing rebuilt"
            ),
        )

    if "error" in resolved:
        # The embedder policy already failed this run; retrying it per file would only
        # repeat the same message N times on N cards.
        return _failed(filename, str(resolved["error"]))
    try:
        active, _description = _load_embedder(embedder, resolved)
    except EmbedderUnavailable as exc:
        resolved["error"] = str(exc)
        return _failed(filename, str(exc))

    progress(0.45, f"Parsing {filename}")
    try:
        document = parse_pdf(path)
    except ParseError as exc:
        return _failed(filename, f"could not be parsed: {exc}")
    if document.doc_id != digest:
        # parse_pdf hashes the same bytes this function did, so a mismatch means the file
        # changed under us — an index built on the wrong digest is a silently wrong index.
        return _failed(
            filename,
            "the document changed while it was being read "
            f"(parsed digest {document.doc_id[:_SHORT_DOC_ID]}…, read {digest[:_SHORT_DOC_ID]}…)",
        )

    progress(0.7, f"Chunking and embedding {filename}")
    try:
        # Build on the index already on disk, not on an empty one: build_index() starts
        # from scratch and saving that handle would erase every other document's rows.
        base = ChunkIndex.load(index_dir, active) or ChunkIndex(())
        merged = base.replace_document(document, embedder=active)
        merged.save(index_dir)
    except Exception as exc:  # FAISS/BM25/IO failures must never look like success
        return _failed(filename, f"could not write the index under {index_dir}: {exc}")

    rows = [record for record in merged.records if record.doc_id == digest]
    progress(0.95, f"Indexed {filename}")
    try:
        _append_catalog_entry(index_dir, digest, document)
    except IngestError as exc:
        # The index is written; a catalog that did not follow is a real failure and must be
        # visible, because the next run would re-index this exact document.
        return FileResult(
            filename=filename,
            state="failed",
            doc_id=digest,
            detail=f"indexed, but the catalog could not be updated: {exc}",
        )
    return FileResult(
        filename=filename,
        state="indexed",
        doc_id=digest,
        page_count=document.page_count,
        parents=len({record.parent_id for record in rows}),
        children=len(rows),
        sections=len(_sections_of(document)),
        size_bytes=len(payload),
        warning=_scanned_warning(document),
    )


def _load_embedder(embedder: Embedder | None, resolved: dict[str, Any]) -> tuple[Embedder, str]:
    """The embedder for this run: the injected one, else the builder's policy, cached once."""
    if embedder is not None:
        return embedder, "caller-supplied embedder"
    if "embedder" not in resolved:
        resolved["embedder"] = _resolve_embedder()
    embedder_pair: tuple[Embedder, str] = resolved["embedder"]
    return embedder_pair


def _sections_of(document: Any) -> list[str]:
    """The document's section paths, flattened by the builder's own helper."""
    sections: Any = _builder().sections_of(document)
    return [str(item) for item in sections]


def _append_catalog_entry(index_dir: Path, digest: str, document: Any) -> None:
    """Add this document's frozen five-field line to the catalog, leaving the others alone.

    The entries are re-read at write time so two uploads in one run cannot overwrite each
    other's lines: a write based on a stale read would drop the first document.
    """
    entries = read_catalog(index_dir)
    entries[str(digest)] = {
        "doc_id": digest,
        "filename": str(document.filename),
        "sha256": digest,
        "page_count": int(document.page_count),
        "sections": _sections_of(document),
    }
    _builder().write_catalog(list(entries.values()), _catalog_path(index_dir))


def _failed(filename: str, detail: str) -> FileResult:
    """A failed upload's card: the reason, and nothing written for it."""
    return FileResult(filename=filename, state="failed", detail=detail)


def _scanned_warning(document: Any) -> str | None:
    """Name the pages the parser flagged as having no text layer, or ``None`` when there are none.

    A scanned page is recorded, not dropped (ADR-004), so those pages contribute no
    searchable text — the analyst who uploaded them has to hear that from the UI.
    """
    pages = sorted(
        block.page for block in document.blocks if str(block.text).strip() == NO_TEXT_LAYER
    )
    if not pages:
        return None
    listed = ", ".join(str(page) for page in pages[:_MAX_SECTIONS_LISTED])
    rest = len(pages) - min(len(pages), _MAX_SECTIONS_LISTED)
    tail = f", and {rest} more" if rest > 0 else ""
    return (
        f"{len(pages)} page(s) have no text layer (scanned images): {listed}{tail} "
        "— they contribute no searchable text"
    )


def _upload_name(file: Any) -> str:
    """The display name of an upload, falling back to a placeholder when it carries none."""
    name = str(getattr(file, "name", "") or "").strip()
    return name or "uploaded document"


def _pdf_target(corpus_dir: Path, filename: str, digest: str) -> Path:
    """Where an upload is copied, never clobbering a different file under the same name.

    Two analysts can upload two documents both called ``annual-report.pdf``; writing over
    the first would leave an index whose ``doc_id`` no longer matches the file on disk. A
    digest-suffixed name keeps both.
    """
    stored = _sanitized_name(filename)
    candidate = corpus_dir / stored
    if candidate.is_file() and _digest_of(candidate) != digest:
        stem, suffix = candidate.stem, candidate.suffix
        candidate = corpus_dir / f"{stem}-{digest[:_SHORT_DOC_ID]}{suffix}"
    return candidate


def _sanitized_name(filename: str) -> str:
    """A filename safe to store: no directories, no control characters, capped length."""
    stem = Path(filename).name or "document.pdf"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "document.pdf"
    if len(cleaned) > _MAX_FILENAME_CHARS:
        stem_part, suffix = Path(cleaned).stem, Path(cleaned).suffix
        keep = max(_MAX_FILENAME_CHARS - len(suffix), 1)
        cleaned = f"{stem_part[:keep]}{suffix}"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


def _digest_of(path: Path) -> str:
    """sha256 of a file's bytes, read in chunks so a large document stays cheap."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _readable(prefix: bytes) -> str:
    """Printable rendering of raw leading bytes: ``PK\\x03\\x04``, never a bare control code."""
    return "".join(chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}" for byte in prefix)


# --- rendering ---------------------------------------------------------------------

_STATE_LABELS = {"indexed": "indexed", "skipped": "already indexed", "failed": "failed"}

#: Token colours per outcome — the design system has exactly one colour per signal class.
_STATE_COLORS = {"indexed": "var(--success)", "skipped": "var(--warning)", "failed": "var(--error)"}


def render_upload_panel(
    *,
    index_dir: Path | None = None,
    pdf_dir: Path | None = None,
    embedder: Embedder | None = None,
    store: Any | None = None,
    state_key: str = UPLOAD_STATE_KEY,
) -> IngestResult | None:
    """The whole ingest surface: uploader, action, status cards, index status, catalog card.

    Args:
        index_dir: Index directory; defaults to this module's ``INDEX_DIR``.
        pdf_dir: Corpus directory the PDFs are copied into; defaults to ``PDF_DIR``.
        embedder: Injected embedder; ``None`` resolves one through the builder's policy.
        store: Optional chat store, used to append the ``upload.parsed`` observability row
            (``docs/OBSERVABILITY.md``'s vocabulary) for each indexed document.
        state_key: Session-state key holding the last result, so the per-file cards survive
            the rerun any other widget triggers. Pass a distinct key for a second panel.

    Returns:
        The last :class:`IngestResult` — the one just produced when the action was clicked,
        else the stored one, else ``None`` before anything has been ingested.
    """
    target = Path(index_dir) if index_dir is not None else INDEX_DIR
    stored = st.session_state.get(state_key)
    if isinstance(stored, IngestResult):
        render_file_results(stored.files)

    uploaded = st.file_uploader(UPLOAD_LABEL, type=["pdf"], accept_multiple_files=True)
    if not uploaded:
        st.caption(UPLOAD_EMPTY)
    else:
        files = tuple(uploaded)
        action = ACTION_LABEL_PLURAL if len(files) > 1 else ACTION_LABEL
        if st.button(f"{action} ({len(files)})", key=_ACTION_KEY):
            result = _run_ingest(files, pdf_dir=pdf_dir, embedder=embedder, store=store)
            if result is not None:
                st.session_state[state_key] = result
                stored = result

    render_index_status(index_status(target))
    render_document_catalog(index_dir=target)
    return stored if isinstance(stored, IngestResult) else None


def _run_ingest(
    files: Sequence[Any],
    *,
    pdf_dir: Path | None,
    embedder: Embedder | None,
    store: Any | None,
) -> IngestResult | None:
    """Run one ingest inside a live status container, then card every outcome."""
    with st.status(f"Indexing {len(files)} document(s)", expanded=True) as status:
        bar = st.progress(0.0)

        def advance(share: float, message: str) -> None:
            bounded = min(max(share, 0.0), 1.0)
            bar.progress(round(bounded, 4))
            status.update(label=f"{message} · {int(bounded * 100)}%")

        try:
            result = ingest_documents(
                files, pdf_dir=pdf_dir, embedder=embedder, on_progress=advance
            )
        except IngestError as exc:
            status.update(label="Index build failed", state="error")
            st.error(f"Could not build the index: {exc}")
            return None
        status.update(label=result.summary(), state="complete" if not result.failed else "error")

    render_file_results(result.files)
    _record_upload_events(store, result)
    return result


def _record_upload_events(store: Any | None, result: IngestResult) -> None:
    """Append one ``upload.parsed`` row per indexed document to the events stream.

    Counts, ids and page totals only — never document text (``docs/OBSERVABILITY.md``'s
    logging rule). A store failure degrades visibly with a warning instead of being
    swallowed: losing an observability row in silence is what the table exists to prevent.
    """
    if store is None:
        return
    failures: list[str] = []
    for item in result.indexed:
        try:
            store.record_event(
                "upload.parsed",
                payload={
                    "doc_id": item.doc_id,
                    "filename": item.filename,
                    "pages": item.page_count,
                    "parents": item.parents,
                    "children": item.children,
                },
            )
        except Exception as exc:  # a broken sink must not undo a completed ingest
            failures.append(str(exc))
    if failures:
        st.warning(f"{len(failures)} upload event(s) could not be recorded: {failures[0]}")


def render_file_results(results: Sequence[FileResult]) -> None:
    """Render the outcome cards of one ingest run."""
    for result in results:
        render_file_result(result)


def render_file_result(result: FileResult) -> None:
    """Render one upload's card: the outcome, the reason, and the numbers when it succeeded.

    Args:
        result: One :class:`FileResult` as produced by :func:`ingest_documents`.
    """
    with st.container(border=True):
        st.markdown(
            _header_row(
                html.escape(result.filename),
                _STATE_LABELS[result.state],
                _STATE_COLORS[result.state],
            ),
            unsafe_allow_html=True,
        )
        detail = result.detail
        if result.state == "indexed" and not detail:
            detail = _indexed_detail(result)
        if detail:
            st.markdown(_detail_line(html.escape(detail)), unsafe_allow_html=True)
        if result.warning:
            st.markdown(
                _detail_line(html.escape(result.warning), "var(--warning)"),
                unsafe_allow_html=True,
            )


def _indexed_detail(result: FileResult) -> str:
    """The counts line of a built card: pages, parents, children, sections."""
    parts: list[str] = []
    if result.page_count is not None:
        parts.append(f"{result.page_count} pages")
    parts.extend([f"{result.parents} parents", f"{result.children} chunks"])
    if result.sections:
        parts.append(f"{result.sections} sections")
    return " · ".join(parts)


def render_index_status(status: IndexStatus) -> None:
    """Render the small index-status card: what is indexed, or why nothing can be asked yet.

    Args:
        status: The :class:`IndexStatus` as read by :func:`index_status`.
    """
    with st.container(border=True):
        st.markdown("**Index status**")
        if status.ready:
            st.markdown(_header_row("index ready", "ok", "var(--success)"), unsafe_allow_html=True)
            st.caption(
                f"{status.doc_count} document(s) · {status.chunk_count} chunks · "
                f"{status.size_human} on disk"
            )
        else:
            st.markdown(
                _header_row("index not built", "empty", "var(--warning)"),
                unsafe_allow_html=True,
            )
            st.caption(status.reason or INDEX_EMPTY)


def render_document_catalog(
    entries: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    index_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Render the catalog card — one expandable row per indexed document.

    Args:
        entries: A catalog already read by the caller (``{doc_id: entry}``). Omit to read
            the catalog from ``index_dir`` here.
        index_dir: Index directory to read when ``entries`` is not given.

    Returns:
        The catalog that was rendered, so the caller does not read it a second time.
    """
    target = Path(index_dir) if index_dir is not None else INDEX_DIR
    catalog = (
        read_catalog(target)
        if entries is None
        else {str(key): dict(value) for key, value in entries.items()}
    )
    with st.container(border=True):
        st.markdown(f"**Documents indexed · {len(catalog)}**")
        if not catalog:
            st.caption(CATALOG_EMPTY)
        for entry in catalog.values():
            _render_catalog_row(entry)
        st.caption(f"Catalog: {target / CATALOG_FILENAME}")
    return catalog


def _render_catalog_row(entry: Mapping[str, Any]) -> None:
    """One catalog row: filename, short id, page and section counts, sections in an expander."""
    filename = html.escape(str(entry.get("filename") or "unnamed document"))
    doc_id = str(entry.get("doc_id") or "")
    pages = int(entry.get("page_count") or 0)
    sections = [str(item) for item in entry.get("sections") or ()]
    label = f"{filename} · {doc_id[:_SHORT_DOC_ID]} · {pages} pages · {len(sections)} sections"
    with st.expander(label):
        st.caption(f"Document id: {html.escape(doc_id)}")
        if not sections:
            st.caption("No section paths recorded — the parser found no headings in this file.")
            return
        for section in sections[:_MAX_SECTIONS_LISTED]:
            st.markdown(f"- {html.escape(section)}")
        if len(sections) > _MAX_SECTIONS_LISTED:
            st.caption(f"… and {len(sections) - _MAX_SECTIONS_LISTED} more section paths")


# --- html fragments ---------------------------------------------------------------


def _header_row(name: str, state: str, color: str) -> str:
    """A card's filename on the left, its coloured outcome on the right.

    Every interpolated value arrives escaped by the caller: a filename is attacker-chosen
    text and must never reach raw HTML.
    """
    return (
        '<div style="display:flex;align-items:baseline;justify-content:space-between;'
        f'gap:8px"><span style="font-weight:600;color:var(--text)">{name}</span>'
        f'<span style="color:{color};font-weight:600;white-space:nowrap">{state}</span></div>'
    )


def _detail_line(text: str, color: str = "var(--text-muted)") -> str:
    """A card's detail line in the given token colour (muted by default)."""
    return f'<div style="color:{color};font-size:0.8rem">{text}</div>'

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

if TYPE_CHECKING:
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

REPO_ROOT = Path(__file__).resolve().parents[2]

PDF_DIR = REPO_ROOT / "data" / "pdfs"

INDEX_DIR = Path(os.environ.get("INDEX_DIR", "").strip() or (REPO_ROOT / "data" / "index"))

CATALOG_FILENAME = "docs.jsonl"

CATALOG_FIELDS = ("doc_id", "filename", "sha256", "page_count", "sections")

PDF_MAGIC = b"%PDF"
_SNIFF_BYTES = 8

_MAX_FILENAME_CHARS = 80

NO_TEXT_LAYER = "[no text layer]"

_MAX_SECTIONS_LISTED = 12

UPLOAD_STATE_KEY = "isip-upload-result"
_ACTION_KEY = "isip-upload-index"

UPLOAD_LABEL = "Upload PDF documents"
ACTION_LABEL = "Index document"
ACTION_LABEL_PLURAL = "Index documents"
UPLOAD_EMPTY = "No files selected — upload one or more PDFs to index them."

INDEX_EMPTY = "No index yet — upload a PDF to build one."
CATALOG_EMPTY = "No document indexed yet — upload a PDF to add one."

_SECTION_SEPARATOR = " > "
_SHORT_DOC_ID = 12


@dataclass(frozen=True)
class FileResult:

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

    ready: bool
    doc_count: int
    chunk_count: int
    size_bytes: int
    size_human: str
    reason: str | None = None


@dataclass(frozen=True)
class IngestResult:
    files: tuple[FileResult, ...] = ()
    index: IndexStatus = field(
        default_factory=lambda: IndexStatus(False, 0, 0, 0, "0 B", INDEX_EMPTY)
    )

    @property
    def indexed(self) -> tuple[FileResult, ...]:
        return tuple(item for item in self.files if item.state == "indexed")

    @property
    def skipped(self) -> tuple[FileResult, ...]:
        return tuple(item for item in self.files if item.state == "skipped")

    @property
    def failed(self) -> tuple[FileResult, ...]:
        return tuple(item for item in self.files if item.state == "failed")

    def summary(self) -> str:

        total = len(self.files)
        parts = [
            f"{len(self.indexed)} indexed",
            f"{len(self.skipped)} already indexed",
            f"{len(self.failed)} failed",
        ]
        return f"Indexed {total} upload(s): " + ", ".join(parts)


class IngestError(ValueError):
    pass


class CatalogReadError(IngestError):
    pass


class EmbedderUnavailable(IngestError):
    pass


def _builder() -> Any:
    try:
        import scripts.build_index as module
    except Exception as exc:
        raise IngestError(f"scripts/build_index.py is unavailable: {exc}") from exc
    return module


def _resolve_embedder() -> tuple[Embedder, str]:
    try:
        embedder, description = _builder().build_embedder()
    except IngestError as exc:
        raise EmbedderUnavailable(str(exc)) from exc
    except Exception as exc:
        raise EmbedderUnavailable(str(exc)) from exc
    return embedder, str(description)


def _catalog_path(index_dir: Path | None) -> Path:
    target = Path(index_dir) if index_dir is not None else INDEX_DIR
    return target / CATALOG_FILENAME


def read_catalog(index_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    try:
        entries = _builder().read_catalog(_catalog_path(index_dir))
    except IngestError:
        raise
    except Exception as exc:
        raise CatalogReadError(str(exc)) from exc
    return {str(key): dict(value) for key, value in sorted(entries.items())}


def index_status(index_dir: Path | None = None) -> IndexStatus:
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
    catalog = read_catalog(index_dir) if entries is None else dict(entries)
    return {
        doc_id: str(entry.get("filename") or doc_id[:_SHORT_DOC_ID])
        for doc_id, entry in catalog.items()
    }


def _artifact_bytes(target: Path) -> int:
    total = 0
    for child in sorted(target.rglob("*")):
        if child.is_file() and child.name != CATALOG_FILENAME:
            total += child.stat().st_size
    return total


def _count_records(path: Path) -> int:
    with open(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def ingest_documents(
    files: Sequence[Any],
    *,
    index_dir: Path | None = None,
    pdf_dir: Path | None = None,
    embedder: Embedder | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> IngestResult:
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
    filename = _upload_name(file)
    progress(0.1, f"Reading {filename}")
    try:
        payload = file.read()
    except Exception as exc:
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
        return _failed(
            filename,
            "the document changed while it was being read "
            f"(parsed digest {document.doc_id[:_SHORT_DOC_ID]}…, read {digest[:_SHORT_DOC_ID]}…)",
        )

    progress(0.7, f"Chunking and embedding {filename}")
    try:
        base = ChunkIndex.load(index_dir, active) or ChunkIndex(())
        merged = base.replace_document(document, embedder=active)
        merged.save(index_dir)
    except Exception as exc:
        return _failed(filename, f"could not write the index under {index_dir}: {exc}")

    rows = [record for record in merged.records if record.doc_id == digest]
    progress(0.95, f"Indexed {filename}")
    try:
        _append_catalog_entry(index_dir, digest, document)
    except IngestError as exc:
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
    if embedder is not None:
        return embedder, "caller-supplied embedder"
    if "embedder" not in resolved:
        resolved["embedder"] = _resolve_embedder()
    embedder_pair: tuple[Embedder, str] = resolved["embedder"]
    return embedder_pair


def _sections_of(document: Any) -> list[str]:
    sections: Any = _builder().sections_of(document)
    return [str(item) for item in sections]


def _append_catalog_entry(index_dir: Path, digest: str, document: Any) -> None:
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
    return FileResult(filename=filename, state="failed", detail=detail)


def _scanned_warning(document: Any) -> str | None:
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
    name = str(getattr(file, "name", "") or "").strip()
    return name or "uploaded document"


def _pdf_target(corpus_dir: Path, filename: str, digest: str) -> Path:
    stored = _sanitized_name(filename)
    candidate = corpus_dir / stored
    if candidate.is_file() and _digest_of(candidate) != digest:
        stem, suffix = candidate.stem, candidate.suffix
        candidate = corpus_dir / f"{stem}-{digest[:_SHORT_DOC_ID]}{suffix}"
    return candidate


def _sanitized_name(filename: str) -> str:
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
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _readable(prefix: bytes) -> str:
    return "".join(chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}" for byte in prefix)


_STATE_LABELS = {"indexed": "indexed", "skipped": "already indexed", "failed": "failed"}

_STATE_COLORS = {"indexed": "var(--success)", "skipped": "var(--warning)", "failed": "var(--error)"}


def render_upload_panel(
    *,
    index_dir: Path | None = None,
    pdf_dir: Path | None = None,
    embedder: Embedder | None = None,
    store: Any | None = None,
    state_key: str = UPLOAD_STATE_KEY,
) -> IngestResult | None:
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
        except Exception as exc:
            failures.append(str(exc))
    if failures:
        st.warning(f"{len(failures)} upload event(s) could not be recorded: {failures[0]}")


def render_file_results(results: Sequence[FileResult]) -> None:
    for result in results:
        render_file_result(result)


def render_file_result(result: FileResult) -> None:
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
    parts: list[str] = []
    if result.page_count is not None:
        parts.append(f"{result.page_count} pages")
    parts.extend([f"{result.parents} parents", f"{result.children} chunks"])
    if result.sections:
        parts.append(f"{result.sections} sections")
    return " · ".join(parts)


def render_index_status(status: IndexStatus) -> None:
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


def _header_row(name: str, state: str, color: str) -> str:
    return (
        '<div style="display:flex;align-items:baseline;justify-content:space-between;'
        f'gap:8px"><span style="font-weight:600;color:var(--text)">{name}</span>'
        f'<span style="color:{color};font-weight:600;white-space:nowrap">{state}</span></div>'
    )


def _detail_line(text: str, color: str = "var(--text-muted)") -> str:
    return f'<div style="color:{color};font-size:0.8rem">{text}</div>'

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
    from src.core.retrieval import HybridIndexHandle

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = REPO_ROOT / "data" / "pdfs"
INDEX_DIR = REPO_ROOT / "data" / "index"
CATALOG_FILENAME = "docs.jsonl"

CATALOG_FIELDS = ("doc_id", "filename", "sha256", "page_count", "sections")


class CatalogEntry(TypedDict):
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
    pass


class MockEmbedder:

    def __init__(self, dimensions: int = MOCK_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return [_mock_vector(text, self._dimensions) for text in texts]


def _mock_vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = [float(digest[offset] - 127.5) for offset in range(dimensions) if offset < len(digest)]
    if len(vector) < dimensions:
        vector = (vector * ((dimensions // max(1, len(vector))) + 1))[:dimensions]
    norm = float(sum(value * value for value in vector)) ** 0.5
    if norm == 0.0:
        return [1.0 if position == 0 else 0.0 for position in range(dimensions)]
    return [value / norm for value in vector]


class OpenAIEmbedder:

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise BuildError(
                "the OpenAI client is not installed; run `pip install -r requirements.txt`"
            ) from exc
        self._model = model
        self._dimensions = dimensions
        self._client = OpenAI(api_key=api_key, base_url=base_url or DEFAULT_BASE_URL)
        self._endpoint = base_url or DEFAULT_BASE_URL

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
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
        except Exception as exc:
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
    from src.core.local_embedder import LocalEmbeddingError, build_local_embedder

    try:
        return build_local_embedder()
    except LocalEmbeddingError as exc:
        raise BuildError(str(exc)) from exc


def _build_api_embedder() -> tuple[Embedder, str]:
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise BuildError(
            f"embedding needs a key: {API_KEY_ENV} is not set and {MOCK_ENV} is not '1'. "
            f"Either export {API_KEY_ENV} (and optionally {BASE_URL_ENV} / "
            f"{EMBEDDING_MODEL_ENV}), or run with {MOCK_ENV}=1 for the deterministic "
            "offline path that needs no key and no network."
        )
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
    except Exception as exc:
        raise BuildError(f"could not build the OpenAI-compatible client: {exc}") from exc
    return embedder, f"OpenAI-compatible embedder, {model} via {embedder.endpoint}"


def catalog_path(index_dir: Path = INDEX_DIR) -> Path:
    return index_dir / CATALOG_FILENAME


def read_catalog(path: Path | None = None) -> dict[str, CatalogEntry]:
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
                raise BuildError(f"{target}:{number} is missing catalog field(s) {missing}")
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
    target = path or catalog_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda entry: str(entry["doc_id"]))
    lines = [json.dumps(entry, sort_keys=True) for entry in ordered]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def sections_of(doc: ParsedDocument) -> list[str]:
    sections: list[str] = []
    seen: set[str] = set()
    for block in doc.blocks:
        joined = " > ".join(block.section_path)
        if joined and joined not in seen:
            seen.add(joined)
            sections.append(joined)
    return sections


def discover_pdfs(directory: Path, explicit: Sequence[Path] | None = None) -> list[Path]:
    if explicit is not None:
        return [Path(item) for item in explicit]
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )


def validate_pdf(path: Path) -> None:
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
    doc_id: str
    filename: str
    page_count: int
    parents: int
    children: int
    action: str
    sections: int
    size_bytes: int

    @property
    def short_id(self) -> str:
        return self.doc_id[:DOC_ID_CHARS]


def _load_existing(index_dir: Path, embedder: Embedder) -> HybridIndexHandle | None:
    from src.core.retrieval import RetrievalError, load_index

    try:
        return load_index(index_dir, embedder)
    except RetrievalError as exc:
        raise BuildError(str(exc)) from exc


def index_one(
    path: Path, embedder: Embedder, catalog: dict[str, CatalogEntry], index_dir: Path
) -> tuple[DocReport, ParsedDocument | None]:
    validate_pdf(path)
    digest = sha256_of(path)
    if digest in catalog:
        entry = catalog[digest]
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
        raise BuildError(
            f"{path}: computed sha256 {digest[:16]}… but the parser reported "
            f"{doc.doc_id[:16]}… — the file changed while it was being read"
        )

    existing = _load_existing(index_dir, embedder)
    try:
        base = existing.index if existing is not None else ChunkIndex(())
        merged = base.replace_document(doc, embedder=embedder)
        merged.save(index_dir)
    except Exception as exc:
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
    return {
        "doc_id": report.doc_id,
        "filename": report.filename,
        "sha256": report.doc_id,
        "page_count": report.page_count,
        "sections": sections_of(doc),
    }


def _format_table(reports: Sequence[DocReport]) -> str:
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
            if report.doc_id not in entries:
                entries[report.doc_id] = _entry_for(report, documents[report.doc_id])
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

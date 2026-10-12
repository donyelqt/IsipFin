#!/usr/bin/env python3
"""Fetch the FY2023 corpus document into ``data/pdfs/`` (task T5).

There is no default source. Microsoft's own "Microsoft 2023 Annual Report" artifact
(linked from https://www.microsoft.com/investor/reports/ar23/download-center/) is the
canonical document, but that endpoint currently serves a **DOCX** — a ZIP whose first
bytes are ``PK\\x03\\x04`` — not a PDF, so no URL is assumed. Pass ``--url`` to install
from a PDF source, or ``--check`` to probe a candidate first.

Idempotency contract
--------------------
A second run is a no-op. The digest of every installed file is recorded in
``data/pdfs/manifest.json``; when the file on disk matches its recorded digest the
script performs no HTTP request, writes nothing, and reports "already present". A file
placed by hand is adopted rather than re-downloaded, and its digest is recorded so the
next run can be a no-op. ``--force`` re-downloads regardless.

Installs are atomic: the response is streamed into a temp file *in the destination
directory* and only renamed into place after the magic-byte check passes, so an
interrupted run can never leave a truncated ``.pdf`` behind.

Hard guarantee: a non-PDF response is never written as a ``.pdf``. The first bytes of
the response are checked against ``%PDF`` before the rename, and a rejection names the
bytes actually received ("expected ``%PDF``, got ``PK``") together with the declared
``Content-Type``.

Exit codes
----------
0  the document is installed, or was already present
1  usage error, or the download / validation failed
2  ``--check``: the source answered but is not a PDF
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Official artifact linked from Microsoft's ar23 download center. NOTE: this endpoint
# currently serves a DOCX (ZIP, b"PK\x03\x04"), not a PDF, so it is offered as an
# opt-in --url argument only and is never the default. Override with --url.
CORPUS_URL = "https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/2023_Annual_Report"

# Leading bytes every PDF must carry. A server that labels its response
# "application/pdf" is still never trusted over the bytes themselves.
PDF_MAGIC = b"%PDF"

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1
SNIFF_BYTES = 16
CHUNK_BYTES = 64 * 1024
MAX_FILENAME_CHARS = 80
DEFAULT_TIMEOUT = 180.0
USER_AGENT = "isipfin-corpus-downloader/1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_DIR = REPO_ROOT / "data" / "pdfs"

# Leading bytes -> human-readable type. Order matters: the first match wins.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "PDF document"),
    (b"PK\x03\x04", "ZIP container (DOCX / XLSX / PPTX / EPUB)"),
    (b"PK\x05\x06", "empty ZIP archive"),
    (b"\x1f\x8b", "gzip stream"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"GIF87a", "GIF image"),
    (b"GIF89a", "GIF image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"%!PS", "PostScript document"),
    (b"{\\rtf", "RTF document"),
    (b"\xd0\xcf\x11\xe0", "OLE2 compound file (legacy .doc / .xls)"),
    (b"SQLite format 3\x00", "SQLite database"),
)


class DownloadError(Exception):
    """Raised when a source cannot be fetched, or the response is not a PDF."""


@dataclass(frozen=True)
class Fetched:
    """A completed HTTP fetch: the local file, its size, digest, and declared type."""

    path: Path
    size_bytes: int
    sha256: str
    content_type: str


@dataclass(frozen=True)
class Source:
    """One installed document, as recorded in the manifest."""

    filename: str
    url: str
    sha256: str
    size_bytes: int
    downloaded_at: str

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "url": self.url,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "downloaded_at": self.downloaded_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Source:
        return cls(
            filename=str(payload["filename"]),
            url=str(payload["url"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            downloaded_at=str(payload.get("downloaded_at", "")),
        )


# --- helpers ---------------------------------------------------------------------


def fmt_size(size_bytes: int) -> str:
    """Human-readable byte count, so sizes stay obvious in the output."""
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def escape_bytes(prefix: bytes) -> str:
    """Printable rendering of raw bytes: ``PK\\x03\\x04``, never a bare control code."""
    return "".join(chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}" for byte in prefix)


def sniff_kind(prefix: bytes) -> str:
    """Best-effort identification of a response body from its leading bytes."""
    for signature, kind in _SIGNATURES:
        if prefix.startswith(signature):
            return kind
    head = prefix.decode("utf-8", errors="replace").lstrip("\ufeff \t\r\n").lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        return "HTML page (an error or interstitial, not the document)"
    if head.startswith("<?xml"):
        return "XML document"
    if not prefix:
        return "empty response"
    return "unknown binary"


def sha256_of(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks so large documents stay cheap."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def read_prefix(path: Path, count: int = SNIFF_BYTES) -> bytes:
    """The first ``count`` bytes of a file."""
    with open(path, "rb") as handle:
        return handle.read(count)


def filename_from_url(url: str) -> str:
    """Stable destination filename derived from a URL: its last path segment.

    The result is identical for identical URLs, contains no query or fragment, carries a
    ``.pdf`` suffix, and has a bounded, filesystem-safe length.
    """
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    stem = unquote(Path(parsed.path).name)
    stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    stem = stem.strip("._-")[:MAX_FILENAME_CHARS]
    if not stem:
        stem = "corpus"
    if stem.lower().endswith(".pdf"):
        return stem
    return f"{stem}.pdf"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- manifest --------------------------------------------------------------------


def manifest_path(directory: Path) -> Path:
    return directory / MANIFEST_FILENAME


def read_manifest(directory: Path) -> dict[str, Source]:
    """The recorded sources keyed by filename; ``{}`` when nothing was installed yet."""
    path = manifest_path(directory)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DownloadError(f"{path} is unreadable ({exc}); remove it or pass --force") from exc
    sources = payload.get("sources", {}) if isinstance(payload, dict) else {}
    if not isinstance(sources, dict):
        raise DownloadError(f"{path} has no 'sources' object; remove it or pass --force")
    return {name: Source.from_dict(entry) for name, entry in sources.items()}


def write_manifest(directory: Path, sources: dict[str, Source]) -> None:
    """Write the manifest atomically, so an interrupted run cannot truncate it."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MANIFEST_VERSION,
        "sources": {name: sources[name].to_dict() for name in sorted(sources)},
    }
    target = manifest_path(directory)
    temporary = target.with_name(f".{MANIFEST_FILENAME}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


# --- transport -------------------------------------------------------------------


def fetch(url: str, target: Path, timeout: float) -> Fetched:
    """Stream ``url`` into ``target``; the caller decides whether to keep the file."""
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf, */*"})
    hasher = hashlib.sha256()
    size = 0
    try:
        with urlopen(request, timeout=timeout) as response, open(target, "wb") as sink:
            declared = response.headers.get_content_type() if response.headers else ""
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                sink.write(chunk)
                hasher.update(chunk)
                size += len(chunk)
    except HTTPError as exc:
        target.unlink(missing_ok=True)
        raise DownloadError(f"{url} answered HTTP {exc.code} ({exc.reason})") from exc
    except (URLError, TimeoutError, OSError) as exc:
        target.unlink(missing_ok=True)
        raise DownloadError(f"{url} could not be fetched: {exc}") from exc
    return Fetched(path=target, size_bytes=size, sha256=hasher.hexdigest(), content_type=declared)


def require_pdf(fetched: Fetched, url: str) -> bytes:
    """Reject a non-PDF response, naming the bytes actually received."""
    prefix = read_prefix(fetched.path)
    if prefix.startswith(PDF_MAGIC):
        return prefix
    declared = fetched.content_type or "none"
    raise DownloadError(
        f"{url} is not a PDF: expected leading bytes %PDF, got {escape_bytes(prefix) or 'nothing'} "
        f"({sniff_kind(prefix)}). The server declared Content-Type: {declared} and sent "
        f"{fmt_size(fetched.size_bytes)}. Nothing was installed — pick a source that serves "
        f"a PDF, or convert the document first."
    )


# --- install / check -------------------------------------------------------------


def check_source(url: str, timeout: float) -> int:
    """Probe a candidate URL without installing anything into ``data/pdfs/``."""
    print(f"checking {url}")
    with tempfile.TemporaryDirectory(prefix="isipfin-corpus-check-") as scratch:
        target = Path(scratch) / filename_from_url(url)
        try:
            fetched = fetch(url, target, timeout=timeout)
            prefix = require_pdf(fetched, url)
        except DownloadError as exc:
            print(f"check failed: {exc}", file=sys.stderr)
            return 1
        if not prefix.startswith(PDF_MAGIC):  # unreachable: require_pdf already raised
            return 2
        print(f"  type          {sniff_kind(prefix)}")
        print(f"  size          {fmt_size(fetched.size_bytes)} ({fetched.size_bytes} bytes)")
        print(f"  sha256        {fetched.sha256}")
        print(f"  content-type  {fetched.content_type or 'none'}")
        print("  result        usable as corpus PDF; nothing was installed")
        print(f"  install with  python {Path(__file__).name} --url {url!r}")
        return 0


def install_source(url: str, directory: Path, filename: str, timeout: float, force: bool) -> int:
    """Install ``url`` into ``directory/filename``; a no-op when the digest is unchanged."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    sources = read_manifest(directory)

    if destination.is_file() and destination.stat().st_size > 0:
        digest = sha256_of(destination)
        recorded = sources.get(destination.name)
        if not force and recorded is not None and recorded.sha256 == digest:
            print(f"already present: {destination}")
            print(
                f"  {fmt_size(recorded.size_bytes)} ({recorded.size_bytes} bytes), "
                f"sha256 {recorded.sha256[:16]}… — no changes"
            )
            return 0
        if force and recorded is not None and recorded.sha256 == digest:
            print(f"forcing a fresh download of {url}")
        elif recorded is None:
            # A file placed by hand: adopt it instead of re-downloading, and record its
            # digest so every later run is a no-op without guessing at the source.
            size = destination.stat().st_size
            sources[destination.name] = Source(
                filename=destination.name,
                url=url,
                sha256=digest,
                size_bytes=size,
                downloaded_at=utc_now(),
            )
            write_manifest(directory, sources)
            print(
                "already present (digest recorded, not verified against the source): "
                f"{destination}"
            )
            print(f"  {fmt_size(size)} ({size} bytes), sha256 {digest[:16]}… — no changes")
            return 0
        elif recorded.sha256 != digest:
            print(
                f"digest mismatch for {destination} (recorded {recorded.sha256[:16]}…, "
                f"on disk {digest[:16]}…): re-downloading"
            )
    elif destination.is_file():
        print(f"{destination} exists but is empty: re-downloading")

    print(f"downloading {url}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=directory, prefix=f".{destination.name}.", suffix=".part"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fetched = fetch(url, temporary, timeout=timeout)
        require_pdf(fetched, url)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    os.replace(temporary, destination)
    sources[destination.name] = Source(
        filename=destination.name,
        url=url,
        sha256=fetched.sha256,
        size_bytes=fetched.size_bytes,
        downloaded_at=utc_now(),
    )
    write_manifest(directory, sources)
    print(f"installed {destination}")
    print(f"  {fmt_size(fetched.size_bytes)} ({fetched.size_bytes} bytes), sha256 {fetched.sha256}")
    print(f"  content-type {fetched.content_type or 'none'}")
    return 0


# --- entrypoint ------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_corpus.py",
        description="Download the FY2023 corpus document into data/pdfs/.",
        epilog=(
            "Idempotent: a file whose sha256 matches data/pdfs/manifest.json is left "
            "alone ('already present'). --check probes a URL without installing it."
        ),
    )
    parser.add_argument(
        "--url",
        required=True,
        help=(
            "source URL to download. There is no default: Microsoft's ar23 download "
            f"center currently serves a DOCX, so pick a PDF source and pass it here "
            f"(e.g. {CORPUS_URL}). Use --check to probe a candidate first."
        ),
    )
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_PDF_DIR),
        help=f"destination directory (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="destination filename (default: derived from the URL's last path segment)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even when the file is already present"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="download to a temp file, validate it, report type/size/sha256, install nothing",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"socket timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:]) if argv is None else list(argv)
    try:
        args = parse_args(arguments)
    except SystemExit as exc:
        if exc.code != 2 or "--url" in arguments:
            raise
        # A bare invocation: argparse rejected the missing required --url. Add the one
        # fact that decides what the user does next — the official source is a DOCX.
        print(
            "error: no source URL given (--url is required; there is deliberately no default)",
            file=sys.stderr,
        )
        print(
            "note: Microsoft's ar23 download center currently serves a DOCX (ZIP, first "
            f'bytes "PK"), not a PDF: {CORPUS_URL}',
            file=sys.stderr,
        )
        print(
            "      probe a candidate source with --check, then install it with --url.",
            file=sys.stderr,
        )
        print(
            f"      or drop a PDF into {DEFAULT_PDF_DIR} and run scripts/build_index.py.",
            file=sys.stderr,
        )
        usage = "usage: download_corpus.py --url URL [--dir DIR] [--out NAME] [--force] [--check]"
        print(usage, file=sys.stderr)
        return 2
    filename = args.out or filename_from_url(args.url)
    directory = Path(args.dir)
    try:
        if args.check:
            return check_source(args.url, args.timeout)
        return install_source(args.url, directory, filename, args.timeout, args.force)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

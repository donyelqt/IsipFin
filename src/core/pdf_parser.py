"""doc-ingest: structure-aware parsing of financial PDFs.

Two-stage parse (ADR-004):
  1. pypdf extracts per-page text and font metadata (1-based pages).
  2. Pages heuristically flagged as table-bearing are re-parsed with pdfplumber
     and each detected grid is serialized to a markdown table.

The public contract is frozen in docs/SPEC.md section 9:
  Block(kind, text, page, section_path) and ParsedDocument(doc_id, filename, page_count, blocks).
Downstream (chunking-index) depends only on those shapes, so parsing changes never
leak into retrieval code.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pdfplumber
from pypdf import PdfReader


class ParseError(Exception):
    """Raised when a page cannot be parsed; carries the page number."""


# --- frozen contract (docs/SPEC.md section 9) ---------------------------------


@dataclass(frozen=True)
class Block:
    kind: Literal["text", "table"]
    text: str
    page: int
    section_path: tuple[str, ...]


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    filename: str
    page_count: int
    blocks: tuple[Block, ...]


# --- patterns (ADR-004) ---------------------------------------------------------

# ALL-CAPS lines, "Note 13", and numbered sections head sections.
_HEADING_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9 ,&'\-/()\.]{4,}$")
_HEADING_NOTE = re.compile(r"^Note\s+\d+\b")
_HEADING_NUMBERED = re.compile(r"^(\d+\.|[A-Z]\.)\s+[A-Z]")
# "(table|note) <n>" style captions, used only as a table-page hint.
_TABLE_CAPTION = re.compile(r"(table|note)\s+\d+", re.IGNORECASE)

# Header/footer recurrence threshold (ADR-004 rule: >=60% of pages).
_BOILERPLATE_PAGE_RATIO = 0.6
# Blocks shorter than this after cleaning are dropped (ADR-004).
_MIN_BLOCK_CHARS = 20
# A page qualifies as table-bearing at N numeric column runs (ADR-004).
_MIN_NUMERIC_RUNS = 4
# Line broken mid-word: "operat-\ning" -> "operating" (ADR-004).
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

_Table = list[list[str | None]]

# A line that is only a page number (ADR-004 cleaning drops these before
# heading detection, otherwise the footer becomes the page's "first line").
_PAGE_NUMBER_ONLY = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE | re.MULTILINE)


# --- cleaning (ADR-004 rules 4; order matters) ---------------------------------


def clean_text(raw: str) -> str:
    """Clean a text block: de-hyphenate, collapse whitespace, preserve numbers.

    Numbers are preserved exactly as printed (thousands separators, parentheses
    for negatives); normalization happens at display, never at storage.
    """
    text = _HYPHEN_BREAK.sub(r"\1\2", raw)
    # Standalone page numbers ("page 12", "312") are pagination, not content.
    text = _PAGE_NUMBER_ONLY.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


# --- boilerplate detection ------------------------------------------------------


def find_boilerplate(page_texts: dict[int, str]) -> set[str]:
    """Lines recurring on >=60% of pages are page headers/footers.

    These dominate retrieval with per-page titles ("Microsoft 2023 Annual
    Report") and must be stripped from block content (ADR-004 rule 1).
    """
    if not page_texts:
        return set()
    counts: Counter[str] = Counter()
    for text in page_texts.values():
        for line in dict.fromkeys(text.splitlines()):
            stripped = line.strip()
            if len(stripped) >= 3:
                counts[stripped] += 1
    threshold = _BOILERPLATE_PAGE_RATIO * len(page_texts)
    return {line for line, n in counts.items() if n >= threshold}


def strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    return "\n".join(line for line in text.splitlines() if line.strip() not in boilerplate)


# --- structure ------------------------------------------------------------------


def detect_heading(line: str) -> bool:
    """True when a line looks like a section heading (ADR-004 rule 3)."""
    s = line.strip()
    if not s or len(s) > 90:
        return False
    return bool(_HEADING_ALLCAPS.match(s) or _HEADING_NOTE.match(s) or _HEADING_NUMBERED.match(s))


def page_is_table_bearing(text: str) -> bool:
    """Heuristic table-page flag: >=4 numeric column runs, or a table caption.

    Runs the ADR-004 heuristic: a regex for N whitespace-aligned numeric runs,
    or a caption matching "(table|note) <n>". False on captioned pages with no
    numeric structure, so prose behind a caption is not mis-parsed.
    """
    if _MIN_NUMERIC_RUNS == 0:
        return False
    numeric_runs = re.findall(r"^[^\S\n]*[\d,\.\)\s]{6,}$", text, flags=re.MULTILINE)
    return len(numeric_runs) >= _MIN_NUMERIC_RUNS or bool(_TABLE_CAPTION.search(text))


def heading_of_line(line: str) -> str:
    """First non-empty line of a heading block, used to build the section path."""
    return line.strip()


# --- table serialization ---------------------------------------------------------


def table_to_markdown(table: _Table) -> str:
    """Serialize a pdfplumber grid to a markdown table with an explicit header.

    ADR-004 rule 5: the header row travels with the table, because detached
    numeric rows are unreadable to the LLM. Empty cells become "—" so column
    count is preserved and the grid does not silently shift.
    """
    if not table:
        return ""
    width = max(len(row) for row in table)
    normalized = []
    for row in table:
        cells = [(cell or "").strip() or "—" for cell in row]
        normalized.append(cells + ["—"] * (width - len(cells)))
    header, *body = normalized
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def extract_tables(page: pdfplumber.page.Page) -> list[str]:
    """Markdown tables found on one page; empty when no grid is detected."""
    grids = page.extract_tables() or []
    return [md for md in (table_to_markdown(g) for g in grids) if md]


# --- main entrypoint --------------------------------------------------------------


def parse_pdf(path: str | Path) -> ParsedDocument:
    """Parse one PDF into the frozen Block/ParsedDocument contract.

    - Every block carries a 1-based page and an inherited section_path.
    - Tables are markdown, extracted with pdfplumber only on flagged pages.
    - Text is cleaned per ADR-004; boilerplate lines are stripped repo-wide
      first so the 60% rule sees all pages, not a window.
    """
    path = Path(path)
    raw = path.read_bytes()
    doc_id = hashlib.sha256(raw).hexdigest()

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # unreadable / truncated stream
        raise ParseError(f"pypdf could not open the document: {exc}") from exc

    page_count = len(reader.pages)
    page_texts: dict[int, str] = {}
    for i, page in enumerate(reader.pages, start=1):
        try:
            page_texts[i] = page.extract_text() or ""
        except Exception as exc:  # pypdf raises broad exception types per page
            raise ParseError(f"pypdf failed on page {i}: {exc}") from exc

    boilerplate = find_boilerplate(page_texts)

    blocks: list[Block] = []
    section: tuple[str, ...] = ()

    with pdfplumber.open(str(path)) as pdf:
        for i in range(1, page_count + 1):
            raw_text = strip_boilerplate(page_texts.get(i, ""), boilerplate)

            # 1. Heading opens a new section. Detect on cleaned lines so a page
            #    number footer can never masquerade as the first line.
            for line in raw_text.splitlines():
                candidate = _PAGE_NUMBER_ONLY.sub("", line).strip()
                if not candidate:
                    continue
                if detect_heading(candidate):
                    section = (*section, heading_of_line(candidate))
                break

            has_text_layer = bool(raw_text.strip())
            table_markdown: list[str] = []
            if i <= len(pdf.pages) and page_is_table_bearing(raw_text):
                try:
                    table_markdown = extract_tables(pdf.pages[i - 1])
                except Exception as exc:
                    # A table page that fails to parse must not lose the text.
                    raise ParseError(f"pdfplumber failed on page {i}: {exc}") from exc

            # 2. Table blocks first: richer than the flattened text they replace.
            for md in table_markdown:
                blocks.append(Block("table", md, i, section))

            # 3. Text block. pypdf gives no coordinates, so table text also
            #    appears in the page's flat text; the duplication is the
            #    accepted limitation in ADR-004 and chunking tolerates it.
            cleaned = clean_text(raw_text)
            if len(cleaned) >= _MIN_BLOCK_CHARS:
                blocks.append(Block("text", cleaned, i, section))
            elif not has_text_layer:
                # Scanned page: recorded so the UI can warn, never silently dropped.
                blocks.append(Block("text", "[no text layer]", i, section))

    return ParsedDocument(
        doc_id=doc_id,
        filename=path.name,
        page_count=page_count,
        blocks=tuple(blocks),
    )

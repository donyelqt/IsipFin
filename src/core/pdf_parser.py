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
    pass


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


_HEADING_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9 ,&'\-/()\.]{4,}$")
_HEADING_NOTE = re.compile(r"^Note\s+\d+\b")
_HEADING_NUMBERED = re.compile(r"^(\d+\.|[A-Z]\.)\s+[A-Z]")
_MIN_HEADING_CHARS = 10
_TABLE_CAPTION = re.compile(r"(table|note)\s+\d+", re.IGNORECASE)

_BOILERPLATE_PAGE_RATIO = 0.6
_MIN_BLOCK_CHARS = 20
_MIN_NUMERIC_RUNS = 4
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

_Table = list[list[str | None]]

_PAGE_NUMBER_ONLY = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE | re.MULTILINE)


def clean_text(raw: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", raw)
    text = _PAGE_NUMBER_ONLY.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def find_boilerplate(page_texts: dict[int, str]) -> set[str]:
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


def detect_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90:
        return False
    if _HEADING_NOTE.match(s) or _HEADING_NUMBERED.match(s):
        return True
    if not _HEADING_ALLCAPS.match(s):
        return False
    if len(s) < _MIN_HEADING_CHARS:
        return False
    return not any(word.islower() for word in s.split())


def page_is_table_bearing(text: str) -> bool:
    if _MIN_NUMERIC_RUNS == 0:
        return False
    numeric_runs = re.findall(r"^[^\S\n]*[\d,\.\)\s]{6,}$", text, flags=re.MULTILINE)
    return len(numeric_runs) >= _MIN_NUMERIC_RUNS or bool(_TABLE_CAPTION.search(text))


def heading_of_line(line: str) -> str:
    return line.strip()


def table_to_markdown(table: _Table) -> str:
    if not table:
        return ""
    rows = [[(cell or "").strip() for cell in row] for row in table]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [[cell or "—" for cell in row] + ["—"] * (width - len(row)) for row in rows]
    header, *body = normalized
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def extract_tables(page: pdfplumber.page.Page) -> list[str]:
    grids = page.extract_tables() or []
    return [md for md in (table_to_markdown(g) for g in grids) if md]


def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    raw = path.read_bytes()
    doc_id = hashlib.sha256(raw).hexdigest()

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ParseError(f"pypdf could not open the document: {exc}") from exc

    page_count = len(reader.pages)
    page_texts: dict[int, str] = {}
    for i, page in enumerate(reader.pages, start=1):
        try:
            page_texts[i] = page.extract_text() or ""
        except Exception as exc:
            raise ParseError(f"pypdf failed on page {i}: {exc}") from exc

    boilerplate = find_boilerplate(page_texts)

    blocks: list[Block] = []
    section: tuple[str, ...] = ()

    with pdfplumber.open(str(path)) as pdf:
        for i in range(1, page_count + 1):
            raw_text = strip_boilerplate(page_texts.get(i, ""), boilerplate)

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
                    raise ParseError(f"pdfplumber failed on page {i}: {exc}") from exc

            for md in table_markdown:
                blocks.append(Block("table", md, i, section))

            cleaned = clean_text(raw_text)
            if len(cleaned) >= _MIN_BLOCK_CHARS:
                blocks.append(Block("text", cleaned, i, section))
            elif not has_text_layer:
                blocks.append(Block("text", "[no text layer]", i, section))

    return ParsedDocument(
        doc_id=doc_id,
        filename=path.name,
        page_count=page_count,
        blocks=tuple(blocks),
    )

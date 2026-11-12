from __future__ import annotations

import re
from dataclasses import dataclass, replace

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.pdf_parser import Block, ParsedDocument

PARENT_TOKENS = 1000
TABLE_SPLIT_TOKENS = 1200
CHILD_TOKENS = 250
CHILD_OVERLAP_TOKENS = 50

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_MD_ROW_SEPARATOR = re.compile(r"^[\s|:\-]+$")

_SENTENCE_END = re.compile(r"([.!?])(?=\s)|([.!?])$")

_BLOCK_JOIN = "\n\n"
_ROW_JOIN = "\n"


@dataclass(frozen=True)
class ParentChunk:
    parent_id: str
    text: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    has_table: bool
    doc_id: str


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_id: str
    text: str
    page: int
    section_path: tuple[str, ...]
    doc_id: str
    ordinal: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _table_rows(table_text: str) -> list[str]:
    return [line for line in table_text.splitlines() if line.strip()]


def _is_markdown_table(rows: list[str]) -> bool:
    if len(rows) < 3:
        return False
    if not rows[0].lstrip().startswith("|"):
        return False
    if not _MD_ROW_SEPARATOR.match(rows[1]):
        return False
    return all(line.lstrip().startswith("|") for line in rows[2:])


def split_table_rows(table_text: str) -> list[str]:
    rows = _table_rows(table_text)
    if len(rows) < 3 or not _is_markdown_table(rows):
        return [table_text]
    if estimate_tokens(table_text) <= TABLE_SPLIT_TOKENS:
        return [table_text]

    header = rows[:2]
    fragments: list[list[str]] = []
    group: list[str] = []

    for row in rows[2:]:
        candidate = [*header, *group, row]
        if group and estimate_tokens(_ROW_JOIN.join(candidate)) > TABLE_SPLIT_TOKENS:
            fragments.append([*header, *group])
            group = []
        group.append(row)
    if group:
        fragments.append([*header, *group])

    return [_ROW_JOIN.join(fragment) for fragment in fragments]


def _oversized_text_units(block: Block) -> list[Block]:
    fragments = _paragraph_fragments(block.text)
    if len(fragments) <= 1:
        fragments = _sentence_fragments(block.text)
    if len(fragments) <= 1:
        return [block]
    return [replace(block, text=piece) for piece in fragments]


def _parent_units(blocks: tuple[Block, ...]) -> list[Block]:
    units: list[Block] = []
    for block in blocks:
        if not block.text.strip():
            continue
        if block.kind == "table":
            pieces = split_table_rows(block.text)
            units.extend(replace(block, text=piece) for piece in pieces)
        elif estimate_tokens(block.text) > PARENT_TOKENS:
            units.extend(_oversized_text_units(block))
        else:
            units.append(block)
    return units


def _text_of(units: list[Block]) -> str:
    return _BLOCK_JOIN.join(unit.text for unit in units)


def _paragraph_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            fragments.append("\n".join(current))
            current = []
    if current:
        fragments.append("\n".join(current))
    return [fragment.strip() for fragment in fragments if fragment.strip()]


def _looks_like_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _sentence_fragments(paragraph: str) -> list[str]:
    units: list[str] = []
    sentence: list[str] = []
    table_run: list[str] = []

    def flush_sentence() -> None:
        if sentence:
            units.append(" ".join(sentence))
            sentence.clear()

    def flush_table() -> None:
        if table_run:
            units.append(_ROW_JOIN.join(table_run))
            table_run.clear()

    for line in paragraph.splitlines():
        if _looks_like_table_line(line):
            flush_sentence()
            table_run.append(line.strip())
            continue
        flush_table()
        pieces = _SENTENCE_END.split(line)
        for index in range(0, len(pieces), 3):
            text_part = pieces[index].strip()
            if index + 2 < len(pieces):
                terminator = pieces[index + 1] or pieces[index + 2] or ""
            else:
                terminator = ""
            if text_part:
                sentence.append(text_part)
            if terminator:
                sentence[-1] = f"{sentence[-1]}{terminator}"
                flush_sentence()
    flush_sentence()
    flush_table()
    return [unit for unit in units if unit.strip()]


def _over_budget(current: list[Block], unit: Block) -> bool:
    return estimate_tokens(_text_of([*current, unit])) > PARENT_TOKENS


def _group_units(units: list[Block]) -> list[list[Block]]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    section: tuple[str, ...] | None = None

    def close() -> None:
        nonlocal current, section
        if current:
            groups.append(current)
        current = []
        section = None

    for unit in units:
        if current and (unit.section_path != section or _over_budget(current, unit)):
            close()
        current.append(unit)
        section = unit.section_path
    close()
    return groups


def _build_parents(doc: ParsedDocument) -> list[ParentChunk]:
    parents: list[ParentChunk] = []
    for group in _group_units(_parent_units(doc.blocks)):
        parents.append(_parent_from(group, doc_id=doc.doc_id, index=len(parents)))
    return parents


def _parent_from(units: list[Block], doc_id: str, index: int) -> ParentChunk:
    pages = [unit.page for unit in units]
    return ParentChunk(
        parent_id=f"{doc_id[:12]}-p{index}",
        text=_text_of(units),
        page_start=min(pages),
        page_end=max(pages),
        section_path=units[0].section_path,
        has_table=any(unit.kind == "table" for unit in units),
        doc_id=doc_id,
    )


def _child_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=list(SEPARATORS),
        chunk_size=CHILD_TOKENS,
        chunk_overlap=CHILD_OVERLAP_TOKENS,
        length_function=estimate_tokens,
        keep_separator=True,
        strip_whitespace=False,
    )


def split_children(text: str) -> list[str]:
    if not text.strip():
        return []
    return [piece for piece in _child_splitter().split_text(text) if piece.strip()]


def _children_of(parent: ParentChunk) -> list[ChildChunk]:
    children: list[ChildChunk] = []
    for ordinal, piece in enumerate(split_children(parent.text)):
        children.append(
            ChildChunk(
                chunk_id=f"{parent.parent_id}-c{ordinal}",
                parent_id=parent.parent_id,
                text=piece,
                page=parent.page_start,
                section_path=parent.section_path,
                doc_id=parent.doc_id,
                ordinal=ordinal,
            )
        )
    return children


def chunk_document(
    doc: ParsedDocument,
) -> tuple[tuple[ParentChunk, ...], tuple[ChildChunk, ...]]:
    parents = _build_parents(doc)
    children = [child for parent in parents for child in _children_of(parent)]
    return tuple(parents), tuple(children)

"""chunking-index: parent/child chunking of a parsed document.

Contract (docs/SPEC.md section 9, ADR-002, docs/STRESS-TEST-RAG rows 4-5):

- **Parents** (~1000 estimated tokens) are heading-aligned: consecutive blocks
  sharing a ``section_path`` merge into one parent, and a changed
  ``section_path`` opens a new one. A parent carries ``page_start`` /
  ``page_end`` — the min/max page of its blocks — so a section or table that
  spans two pages keeps its full range (``page_end >= page_start`` invariant,
  STRESS-TEST-RAG row 5).
- **Children** (~250 estimated tokens, 50-token overlap) are cut from each
  parent's text with ``RecursiveCharacterTextSplitter`` over the
  financial-friendly separator ladder ``["\\n\\n", "\\n", ". ", " ", ""]``:
  paragraph, then table row, then sentence, then word.
- **A parent holds one semantic stretch, not one block.** A block that is
  itself over the parent budget is expanded into sub-block units before
  grouping — blank-line paragraphs first, then sentences for a block with no
  blank-line borders — so a page of flat text with several sections side by
  side produces several coherent parents instead of one ~1300-token parent
  that dilutes every child cut from it. A unit that alone exceeds the budget
  still becomes its own parent; content is never cut to fit.

Two rules are load-bearing and implemented here rather than delegated:

1. **Table atomicity.** Parent construction never cuts a table block — a
   table lives whole inside exactly one parent, because a table split mid-row
   is unreadable to the LLM and a citation would point at a fragment without
   its header. The single escape hatch: a table exceeding
   ``TABLE_SPLIT_TOKENS`` (~1200 tokens) is split *only* at row boundaries
   into fragments that each repeat the table's header row, so every fragment
   is still self-describing (STRESS-TEST-RAG row 4). The same rule holds for
   a table flattened into page text: consecutive pipe-delimited lines inside
   a text block are treated as one atomic run at the sentence level.
2. **Offline token estimation.** Token counts use ``estimate_tokens``
   (``max(1, len(text) // 4)``), never a real tokenizer: tiktoken downloads
   its BPE vocabulary over HTTP on first use, which would break the
   zero-network test rule in docs/CONSTRAINTS.md. The approximation needs no
   network, no model, and no state, so it is identical on every machine.

Determinism: ids are derived from position only — ``parent_id`` is
``f"{doc_id[:12]}-p{n}"`` with ``n`` a 0-based document-order index, and
``chunk_id`` is ``f"{parent_id}-c{ordinal}"`` with ``ordinal`` 0-based within
the parent. No uuid, no randomness, no timestamps: the same
``ParsedDocument`` always produces the same chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.pdf_parser import Block, ParsedDocument

# --- tuned parameters (ADR-002 parameter table; SPEC section 11) -----------------

# Parent target size, in estimated tokens.
PARENT_TOKENS = 1000
# Above this a table is no longer atomic and is split at row boundaries.
TABLE_SPLIT_TOKENS = 1200
# Child size / overlap, in estimated tokens.
CHILD_TOKENS = 250
CHILD_OVERLAP_TOKENS = 50

# Separator ladder for child splitting: paragraph, single newline (table rows),
# sentence, word, character. Financial prose prefers the earliest separator
# that fits, so tables break between rows and prose breaks between sentences.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_MD_ROW_SEPARATOR = re.compile(r"^[\s|:\-]+$")

# Sentence terminator: a period (or ! or ?) followed by a space or end of
# line. The trailing space is kept so a split never detaches the terminator
# from the sentence it closes, and a decimal point or a period inside a
# number is not a terminator because no space follows it.
_SENTENCE_END = re.compile(r"([.!?])(?=\s)|([.!?])$")

_BLOCK_JOIN = "\n\n"
_ROW_JOIN = "\n"


# --- frozen contract (docs/SPEC.md section 9) ------------------------------------


@dataclass(frozen=True)
class ParentChunk:
    parent_id: str
    text: str
    page_start: int  # 1-based, inclusive
    page_end: int  # >= page_start
    section_path: tuple[str, ...]
    has_table: bool
    doc_id: str


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_id: str
    text: str
    page: int  # 1-based
    section_path: tuple[str, ...]
    doc_id: str
    ordinal: int  # 0-based order within the parent


# --- token estimation (offline by contract) --------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` with no network and no model.

    The standard 4-characters-per-token approximation, floored at 1 so any
    non-empty string costs at least one token. A real BPE tokenizer cannot be
    used here: loading one downloads its vocabulary, and the build/test
    contract requires offline determinism.
    """
    return max(1, len(text) // 4)


# --- table atomicity -------------------------------------------------------------


def _table_rows(table_text: str) -> list[str]:
    """Non-blank physical lines of a markdown table; the header row is first."""
    return [line for line in table_text.splitlines() if line.strip()]


def _is_markdown_table(rows: list[str]) -> bool:
    """True when ``rows`` look like a whole markdown table: header row,
    alignment row, then data rows — all pipe-delimited."""
    if len(rows) < 3:
        return False
    if not rows[0].lstrip().startswith("|"):
        return False
    if not _MD_ROW_SEPARATOR.match(rows[1]):
        return False
    return all(line.lstrip().startswith("|") for line in rows[2:])


def split_table_rows(table_text: str) -> list[str]:
    """Split an oversized markdown table at row boundaries, repeating the header.

    Every fragment after the first repeats the table's first two lines (header
    row and its alignment row) so a continuation fragment still names its own
    columns. Rows are grouped in document order and a group is closed as soon
    as adding another row would exceed ``TABLE_SPLIT_TOKENS``.

    The table is returned as a single fragment when it is already within
    budget, when it is too small to split (fewer than three rows), or when it
    is not recognizable markdown — an unrecognizable table is left whole
    rather than guessed at, so no content is silently lost.
    """
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


# --- parent construction ---------------------------------------------------------


def _oversized_text_units(block: Block) -> list[Block]:
    """Parent-construction units for a text block over ``PARENT_TOKENS``.

    Paragraph borders are tried first (blank lines), then sentences, then —
    as a last resort for a single sentence longer than the budget — the
    block is returned whole. Splitting stops at the first level that yields
    more than one unit, so a block that is one long paragraph is not shredded
    further than needed to open a parent border.

    The block is returned whole when no finer split exists: content is never
    cut to fit the budget, and a parent may exceed the target with one unit.
    """
    fragments = _paragraph_fragments(block.text)
    if len(fragments) <= 1:
        fragments = _sentence_fragments(block.text)
    if len(fragments) <= 1:
        return [block]
    return [replace(block, text=piece) for piece in fragments]


def _parent_units(blocks: tuple[Block, ...]) -> list[Block]:
    """Expand blocks into parent-construction units.

    Blank blocks carry no content and are dropped. Every other block passes
    through as-is, except a block over ``PARENT_TOKENS``, which becomes a
    sequence of units:

    - A **table** over ``TABLE_SPLIT_TOKENS`` becomes row-split fragments,
      each still a table block so parent construction still treats it as
      atomic and each repeating the header row (STRESS-TEST-RAG row 4).
    - A **text** block over the budget becomes one unit per blank-line
      paragraph, and one unit per sentence when the block has no blank-line
      borders at all. Flat page text from the parser holds several sections
      side by side with no paragraph structure; splitting it keeps one parent
      from absorbing an entire page of unrelated prose into a single
      ~1300-token child-diluting blob.

    The caller merges those paragraph units back up to the ~1000-token budget,
    so paragraph borders are a split *opportunity*, never a mandate.
    """
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
    """Parent text: blocks in document order, separated by a paragraph break."""
    return _BLOCK_JOIN.join(unit.text for unit in units)


def _paragraph_fragments(text: str) -> list[str]:
    """Split text into its blank-line-separated paragraphs, in document order.

    Blank lines are the paragraph signal the parser preserves inside a text
    block; consecutive non-blank lines form one fragment. Fragments are
    returned stripped, and blank ones dropped, so a whitespace-only gap never
    becomes a parent-construction unit.
    """
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
    """True for a markdown table row — never a split point inside a table.

    A flattened table survives in page text as a run of pipe-delimited lines;
    cutting such a run leaves rows without their header.
    """
    return line.lstrip().startswith("|")


def _sentence_fragments(paragraph: str) -> list[str]:
    """Sentence-level units of one paragraph, table rows kept atomic.

    Used only for a paragraph that is itself over the parent budget, where no
    blank line offers a split. Sentences are the next-coarser unit after a
    paragraph, so a cut lands between statements rather than mid-sentence,
    and a run of consecutive table lines is emitted whole rather than cut
    between rows (STRESS-TEST-RAG row 4).
    """
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
        # A sentence is flushed only when it actually closes on this line:
        # a physical line break is not a sentence break, so continuation
        # lines stay inside the sentence they continue. The terminator is
        # glued directly onto its text with no space — "…$18.4 billion."
        # reads as the original sentence, not as a detached dot — and a
        # decimal point inside a number is never a break because no space
        # follows it.
        pieces = _SENTENCE_END.split(line)
        for index in range(0, len(pieces), 3):
            text_part = pieces[index].strip()
            # ``re.split`` with this pattern only populates the group that
            # matched, so the other one comes back as ``None``.
            if index + 2 < len(pieces):
                terminator = pieces[index + 1] or pieces[index + 2] or ""
            else:
                terminator = ""
            if text_part:
                sentence.append(text_part)
            if terminator:
                # The terminator closes the sentence: glue it on with no
                # space so the sentence reads as printed ("…$18.4 billion.").
                sentence[-1] = f"{sentence[-1]}{terminator}"
                flush_sentence()
    flush_sentence()
    flush_table()
    return [unit for unit in units if unit.strip()]


def _over_budget(current: list[Block], unit: Block) -> bool:
    """True when adding ``unit`` to ``current`` would exceed the parent budget."""
    return estimate_tokens(_text_of([*current, unit])) > PARENT_TOKENS


def _group_units(units: list[Block]) -> list[list[Block]]:
    """Group units into heading-aligned parents.

    Two boundaries close a parent: a section change (``section_path`` differs
    from the units already grouped) and the ~1000-token budget. A unit that
    alone exceeds the budget — an oversized table fragment, or a single
    runaway paragraph — still becomes its own parent, because content is
    never cut to fit.

    Because the units themselves are now paragraph-bounded, a budget close
    lands on a paragraph border rather than mid-sentence, so a parent holds
    one coherent stretch of a section instead of an arbitrary slice of a page.
    """
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
    """Group a document's blocks into heading-aligned parent chunks."""
    parents: list[ParentChunk] = []
    for group in _group_units(_parent_units(doc.blocks)):
        parents.append(_parent_from(group, doc_id=doc.doc_id, index=len(parents)))
    return parents


def _parent_from(units: list[Block], doc_id: str, index: int) -> ParentChunk:
    pages = [unit.page for unit in units]
    return ParentChunk(
        parent_id=f"{doc_id[:12]}-p{index}",
        text=_text_of(units),
        page_start=min(pages),  # 1-based
        page_end=max(pages),  # >= page_start: a merged cross-page range keeps both pages
        section_path=units[0].section_path,
        has_table=any(unit.kind == "table" for unit in units),
        doc_id=doc_id,
    )


# --- child construction ----------------------------------------------------------


def _child_splitter() -> RecursiveCharacterTextSplitter:
    """A fresh splitter for every call.

    Splitter instances are stateful across calls in some versions; building
    one per call keeps every document's children independent and repeatable.
    """
    return RecursiveCharacterTextSplitter(
        separators=list(SEPARATORS),
        chunk_size=CHILD_TOKENS,
        chunk_overlap=CHILD_OVERLAP_TOKENS,
        length_function=estimate_tokens,
        keep_separator=True,
        strip_whitespace=False,
    )


def split_children(text: str) -> list[str]:
    """Cut one parent's text into child chunks (~250 tokens, 50-token overlap)."""
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


# --- public entrypoint -----------------------------------------------------------


def chunk_document(
    doc: ParsedDocument,
) -> tuple[tuple[ParentChunk, ...], tuple[ChildChunk, ...]]:
    """Chunk a parsed document into parent chunks and child chunks.

    Parents are heading-aligned (~1000 estimated tokens) and never cut a table
    block: a table larger than ~1200 tokens is split at row boundaries with
    its header row repeated on every fragment. A block that is itself over
    the budget is expanded into paragraph-level (then sentence-level) units
    first, so a parent holds one coherent stretch of a section rather than a
    whole page of unrelated prose; a unit that alone exceeds the budget still
    becomes its own parent. Children are ~250-token
    windows with ~50-token overlap cut from each parent's text, each recording
    its parent, the parent's ``page_start`` and section path, and a 0-based
    ordinal within that parent.

    Deterministic and offline: identical input documents produce identical
    output, with ids derived from position only.
    """
    parents = _build_parents(doc)
    children = [child for parent in parents for child in _children_of(parent)]
    return tuple(parents), tuple(children)

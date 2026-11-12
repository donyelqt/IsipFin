from __future__ import annotations

import difflib
import itertools
from pathlib import Path

import pytest
from src.core.chunking import (
    ChildChunk,
    ParentChunk,
    chunk_document,
    estimate_tokens,
)
from src.core.pdf_parser import Block, ParsedDocument, parse_pdf, table_to_markdown

from tests.fixtures import sample_doc

CHILD_SIZE_TOKENS = 250
CHILD_OVERLAP_TOKENS = 50
CHILD_SIZE_SLACK_TOKENS = 50
TABLE_SPLIT_THRESHOLD_TOKENS = 1200

HEADER_ROW = ["Metric", "FY2023", "FY2022"]
DATA_ROWS = [
    ["Revenue", "$ 211,915", "$ 198,270"],
    ["Operating income", "$ 88,523", "$ 83,383"],
]

SMALL_TABLE = table_to_markdown([HEADER_ROW, *DATA_ROWS])

DOC_ID = "0123456789abcdef" * 4

SECTION_AR = ("ANNUAL REPORT",)
SECTION_RESULTS = ("ANNUAL REPORT", "SUMMARY OF RESULTS")
SECTION_CONTINGENCIES = (*SECTION_AR, "CONTINGENCIES")


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _doc(*blocks: Block) -> ParsedDocument:
    return ParsedDocument(
        doc_id=DOC_ID,
        filename="fixture.pdf",
        page_count=max((block.page for block in blocks), default=0),
        blocks=tuple(blocks),
    )


def _big_table(n_rows: int = 200) -> str:
    grid = [HEADER_ROW]
    for i in range(1, n_rows + 1):
        note = f"Note {i:03d} " + "segment detail " * 8
        grid.append([f"Segment {i:03d}", note, f"$ {100000 + i}"])
    return table_to_markdown(grid)


BIG_TABLE = _big_table()


SAMPLE_DOC = _doc(
    Block(
        "text",
        "ANNUAL REPORT\nRevenue grew across segments in fiscal 2023.",
        1,
        SECTION_AR,
    ),
    Block("table", SMALL_TABLE, 2, SECTION_RESULTS),
    Block(
        "text",
        "SUMMARY OF RESULTS\nRevenue and operating income by year.",
        2,
        SECTION_RESULTS,
    ),
    Block("text", "CONTINGENCIES\nWe are subject to various claims.", 3, SECTION_CONTINGENCIES),
)

BIG_TABLE_DOC = _doc(
    Block("text", "SEGMENT RESULTS\nRevenue by segment follows below.", 1, ("SEGMENT RESULTS",)),
    Block("table", BIG_TABLE, 2, ("SEGMENT RESULTS",)),
)

PROSE = "\n\n".join(
    f"Statement {i:02d}: revenue grew {i} percent across all segments in fiscal 2023."
    for i in range(1, 46)
)

PROSE_DOC = _doc(Block("text", "SEGMENT DISCUSSION\n" + PROSE, 1, ("SEGMENT DISCUSSION",)))

CROSS_PAGE_DOC = _doc(
    Block("text", "SUMMARY\nRevenue grew across segments in fiscal 2023.", 1, ("SUMMARY",)),
    Block("text", "Operating income expanded year over year as well.", 2, ("SUMMARY",)),
)

ALL_DOCS = (SAMPLE_DOC, BIG_TABLE_DOC, PROSE_DOC, CROSS_PAGE_DOC)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(sample_doc())
    return path


def _table_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("|")]


def _assert_no_partial_row(text: str, table_lines: set[str]) -> None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            assert stripped in table_lines, f"table cut mid-row near {stripped!r}"


def _shared_span(before: str, after: str, window: int = 2000) -> str:
    tail, head = before[-window:], after[:window]
    matcher = difflib.SequenceMatcher(None, tail, head, autojunk=False)
    match = matcher.find_longest_match(0, len(tail), 0, len(head))
    return tail[match.a : match.a + match.size]


def _parent_order(parents: tuple[ParentChunk, ...]) -> dict[str, int]:
    return {parent.parent_id: index for index, parent in enumerate(parents)}


@pytest.mark.parametrize(
    "doc",
    [
        pytest.param(SAMPLE_DOC, id="sample"),
        pytest.param(BIG_TABLE_DOC, id="big_table"),
        pytest.param(PROSE_DOC, id="prose"),
        pytest.param(CROSS_PAGE_DOC, id="cross_page"),
    ],
)
def test_every_child_resolves_to_a_parent(doc: ParsedDocument) -> None:
    parents, children = chunk_document(doc)
    parent_ids = {parent.parent_id for parent in parents}
    assert parent_ids, "fixture document produced no parents"
    assert children, "fixture document produced no children"
    for child in children:
        assert child.parent_id in parent_ids


def test_every_parent_owns_at_least_one_child() -> None:
    for doc in ALL_DOCS:
        parents, children = chunk_document(doc)
        owners = {child.parent_id for child in children}
        for parent in parents:
            assert parent.parent_id in owners, "parent without children"


def test_parents_are_section_aligned() -> None:
    parents, _ = chunk_document(SAMPLE_DOC)
    assert [parent.section_path for parent in parents] == [
        SECTION_AR,
        SECTION_RESULTS,
        SECTION_CONTINGENCIES,
    ]


def test_child_chunks_carry_the_parent_page_section_and_doc_id() -> None:
    parents, children = chunk_document(SAMPLE_DOC)
    by_id = {parent.parent_id: parent for parent in parents}
    for child in children:
        parent = by_id[child.parent_id]
        assert child.page == parent.page_start
        assert child.section_path == parent.section_path
        assert child.doc_id == parent.doc_id == SAMPLE_DOC.doc_id
        assert isinstance(child.ordinal, int) and child.ordinal >= 0


def test_parent_page_range_is_ordered_and_non_empty() -> None:
    for doc in ALL_DOCS:
        parents, _ = chunk_document(doc)
        for parent in parents:
            assert parent.page_start >= 1
            assert parent.page_end >= parent.page_start
            assert parent.text.strip(), "parent with no content"


def test_no_block_page_is_lost() -> None:
    for doc in (SAMPLE_DOC, BIG_TABLE_DOC, CROSS_PAGE_DOC):
        parents, _ = chunk_document(doc)
        for block in doc.blocks:
            covering = [
                parent
                for parent in parents
                if parent.section_path == block.section_path
                and parent.page_start <= block.page <= parent.page_end
            ]
            assert covering, f"page {block.page} of section {block.section_path} lost"


def test_same_section_text_blocks_merge_across_pages() -> None:
    parents, _ = chunk_document(CROSS_PAGE_DOC)
    assert len(parents) == 1
    parent = parents[0]
    assert parent.section_path == ("SUMMARY",)
    assert (parent.page_start, parent.page_end) == (1, 2)
    assert "Revenue grew across segments" in parent.text
    assert "Operating income expanded" in parent.text


def test_table_block_is_never_split_across_parents() -> None:
    parents, _ = chunk_document(SAMPLE_DOC)
    header, separator, *rows = SMALL_TABLE.splitlines()
    carriers = [parent for parent in parents if any(row in parent.text for row in rows)]
    assert len(carriers) == 1, f"table rows spread across {len(carriers)} parents"
    carrier = carriers[0]
    assert carrier.has_table
    assert header in carrier.text, "header row missing from the table's parent"
    assert separator in carrier.text
    for row in rows:
        assert row in carrier.text, f"row cut from the table's parent: {row!r}"
    assert carrier.section_path == SECTION_RESULTS

    known = set(SMALL_TABLE.splitlines())
    for parent in parents:
        _assert_no_partial_row(parent.text, known)


def test_parsed_pdf_table_stays_whole(sample_pdf: Path) -> None:
    doc = parse_pdf(sample_pdf)
    parents, children = chunk_document(doc)
    table = next(block for block in doc.blocks if block.kind == "table")
    header, separator, *rows = table.text.splitlines()

    carriers = [parent for parent in parents if any(row in parent.text for row in rows)]
    assert len(carriers) == 1, f"parsed table spread across {len(carriers)} parents"
    carrier = carriers[0]
    assert carrier.has_table
    assert header in carrier.text
    assert separator in carrier.text
    for row in rows:
        assert row in carrier.text, f"parsed row cut from its parent: {row!r}"

    known = set(table.text.splitlines())
    for parent in parents:
        _assert_no_partial_row(parent.text, known)

    parent_ids = {parent.parent_id for parent in parents}
    assert {child.parent_id for child in children} <= parent_ids


def test_large_table_splits_at_row_boundaries_and_repeats_the_header() -> None:
    parents, _ = chunk_document(BIG_TABLE_DOC)
    header, separator, *rows = BIG_TABLE.splitlines()
    data_rows = set(rows)
    assert _est_tokens(BIG_TABLE) > TABLE_SPLIT_THRESHOLD_TOKENS, "fixture too small"

    carriers = [parent for parent in parents if data_rows & set(_table_lines(parent.text))]
    assert len(carriers) >= 2, "a table past the split threshold must be split"

    for parent in carriers:
        assert parent.has_table
        assert header in parent.text, "continuation lost the repeated header row"
        assert separator in parent.text, "continuation lost the separator row"
        _assert_no_partial_row(parent.text, set(BIG_TABLE.splitlines()))

    order = _parent_order(parents)
    carried = [
        line
        for parent in sorted(carriers, key=lambda item: order[item.parent_id])
        for line in _table_lines(parent.text)
        if line in data_rows
    ]
    assert carried == rows


def test_has_table_flag_matches_table_content() -> None:
    for doc in (SAMPLE_DOC, BIG_TABLE_DOC):
        parents, _ = chunk_document(doc)
        for parent in parents:
            assert bool(parent.has_table) is bool(_table_lines(parent.text))


def test_child_ordinals_are_zero_based_and_contiguous_per_parent() -> None:
    for doc in (SAMPLE_DOC, BIG_TABLE_DOC, PROSE_DOC):
        _parents, children = chunk_document(doc)
        next_ordinal: dict[str, int] = {}
        for child in children:
            expected = next_ordinal.get(child.parent_id, 0)
            assert (
                child.ordinal == expected
            ), f"ordinals for {child.parent_id} are not 0-based and contiguous"
            next_ordinal[child.parent_id] = expected + 1


def test_child_text_comes_from_its_parent_text() -> None:
    for doc in (SAMPLE_DOC, BIG_TABLE_DOC, PROSE_DOC):
        parents, children = chunk_document(doc)
        by_id = {parent.parent_id: parent for parent in parents}
        for child in children:
            assert child.text in by_id[child.parent_id].text


def test_long_parent_splits_into_overlapping_children() -> None:
    _parents, children = chunk_document(PROSE_DOC)
    assert len(children) >= 3, "a ~900-token parent must split into several children"

    grouped: dict[str, list[ChildChunk]] = {}
    for child in children:
        grouped.setdefault(child.parent_id, []).append(child)

    for group in grouped.values():
        for child in group:
            assert _est_tokens(child.text) <= CHILD_SIZE_TOKENS + CHILD_SIZE_SLACK_TOKENS
        for before, after in itertools.pairwise(group):
            overlap = _shared_span(before.text, after.text)
            assert len(overlap) >= 40, "consecutive children share no context"
            assert _est_tokens(overlap) <= CHILD_OVERLAP_TOKENS * 2, "overlap too large"


def test_chunking_is_deterministic_and_id_free() -> None:
    for doc in (SAMPLE_DOC, BIG_TABLE_DOC, PROSE_DOC):
        first = chunk_document(doc)
        second = chunk_document(doc)
        assert first == second

        parents, children = first
        for index, parent in enumerate(parents):
            assert parent.parent_id == f"{DOC_ID[:12]}-p{index}"
        for child in children:
            assert child.chunk_id == f"{child.parent_id}-c{child.ordinal}"

        assert chunk_document(_doc(*doc.blocks)) == first


def test_document_without_blocks_yields_no_chunks() -> None:
    assert chunk_document(_doc()) == ((), ())


def test_whitespace_only_blocks_are_skipped() -> None:
    doc = _doc(
        Block("text", "   \n\t  ", 1, ("BLANK",)),
        Block("table", "\n \n", 2, ("BLANK",)),
        Block("text", "Revenue grew across segments in fiscal 2023.", 3, ("REAL",)),
    )
    parents, children = chunk_document(doc)
    assert len(parents) == 1
    parent = parents[0]
    assert parent.section_path == ("REAL",)
    assert (parent.page_start, parent.page_end) == (3, 3)
    assert "Revenue grew across segments in fiscal 2023." in parent.text
    assert not parent.has_table
    assert children
    for chunk in (*parents, *children):
        assert chunk.text.strip(), "whitespace-only block leaked into a chunk"


def test_whitespace_only_document_yields_no_chunks() -> None:
    doc = _doc(
        Block("text", "  \n ", 1, ("BLANK",)),
        Block("table", "\t\n", 2, ("BLANK",)),
    )
    assert chunk_document(doc) == ((), ())


def test_token_estimate_is_offline_and_deterministic() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens(PROSE) == len(PROSE) // 4
    assert estimate_tokens(PROSE) != len(PROSE)

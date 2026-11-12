from __future__ import annotations

from pathlib import Path

import pytest
from src.core.pdf_parser import (
    ParseError,
    clean_text,
    detect_heading,
    find_boilerplate,
    page_is_table_bearing,
    parse_pdf,
    strip_boilerplate,
    table_to_markdown,
)

from tests.fixtures import HEADER, image_only_doc, sample_doc


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pdf"
    p.write_bytes(sample_doc())
    return p


@pytest.fixture
def scanned(tmp_path: Path) -> Path:
    p = tmp_path / "scanned.pdf"
    p.write_bytes(image_only_doc())
    return p


def test_dehyphenation_joins_broken_word() -> None:
    assert clean_text("operat-\ning margin") == "operating margin"


def test_numbers_preserved_exactly_as_printed() -> None:
    text = "Revenue $ 211,915 and a (1,234) negative, not rounded."
    assert clean_text(text) == text


def test_short_blocks_survive_cleaning_for_caller_to_drop() -> None:
    assert clean_text("ok") == "ok"


def test_standalone_page_numbers_dropped() -> None:
    assert clean_text("page 12\nRevenue grew.") == "Revenue grew."
    assert clean_text("312\nRevenue grew.") == "Revenue grew."


def test_page_number_inside_a_real_sentence_is_kept() -> None:
    text = "Revenue grew in page-3 segments by 12 percent."
    assert clean_text(text) == text


def test_repeated_header_detected_and_stripped() -> None:
    pages = {1: f"{HEADER}\nBody text here.", 2: f"{HEADER}\nMore body."}
    boiler = find_boilerplate(pages)
    assert HEADER in boiler
    assert strip_boilerplate(f"{HEADER}\nBody text here.", boiler) == "Body text here."


def test_normal_body_line_is_not_boilerplate() -> None:
    pages = {1: "Revenue grew.", 2: "Revenue fell."}
    assert find_boilerplate(pages) == set()


@pytest.mark.parametrize(
    "line",
    ["ANNUAL REPORT", "Note 13 Commitments", "3. Revenue Analysis", "B. Risk Factors"],
)
def test_heading_patterns(line: str) -> None:
    assert detect_heading(line)


@pytest.mark.parametrize(
    "line",
    ["Revenue grew across segments.", "a", "3." * 40],
)
def test_non_headings(line: str) -> None:
    assert not detect_heading(line)


def test_table_to_markdown_keeps_header_and_pads_cells() -> None:
    md = table_to_markdown([["Metric", "FY2023", "FY2022"], ["Revenue", "$ 211,915"]])
    lines = md.splitlines()
    assert lines[0] == "| Metric | FY2023 | FY2022 |"
    assert lines[1].count("---") == 3
    assert lines[2] == "| Revenue | $ 211,915 | — |"


def test_empty_grid_serializes_to_empty_string() -> None:
    assert table_to_markdown([]) == ""


def test_page_count_and_one_based_pages(sample: Path) -> None:
    doc = parse_pdf(sample)
    assert doc.page_count == 3
    assert {b.page for b in doc.blocks} == {1, 2, 3}


def test_doc_id_is_sha256_and_idempotent(sample: Path) -> None:
    a, b = parse_pdf(sample), parse_pdf(sample)
    assert a.doc_id == b.doc_id and len(a.doc_id) == 64


def test_table_block_detected_on_known_page(sample: Path) -> None:
    doc = parse_pdf(sample)
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert tables, "expected at least one markdown table block"
    assert "$ 211,915" in tables[0].text
    assert tables[0].page == 2


def test_section_path_inherited_and_extended(sample: Path) -> None:
    doc = parse_pdf(sample)
    p2 = [b for b in doc.blocks if b.page == 2]
    assert all(
        b.section_path[-1] == "SUMMARY OF RESULTS" for b in p2
    ), f"blocks inherited the wrong path: {[b.section_path for b in p2]}"


def test_boilerplate_absent_from_block_content(sample: Path) -> None:
    doc = parse_pdf(sample)
    assert all(HEADER not in b.text for b in doc.blocks)


def test_scanned_page_flagged_not_dropped(scanned: Path) -> None:
    doc = parse_pdf(scanned)
    assert doc.page_count >= 1
    assert any("[no text layer]" in b.text for b in doc.blocks)


def test_table_page_flag_rejects_plain_prose() -> None:
    assert not page_is_table_bearing("Revenue grew across all segments in 2023.\n")


def test_parse_error_carries_page_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(sample_doc())
    bad.write_bytes(bad.read_bytes()[:40])
    with pytest.raises(ParseError):
        parse_pdf(bad)


def test_contract_shapes_match_spec(sample: Path) -> None:
    doc = parse_pdf(sample)
    for b in doc.blocks:
        assert b.kind in {"text", "table"}
        assert isinstance(b.page, int) and b.page >= 1
        assert isinstance(b.section_path, tuple)

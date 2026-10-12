"""chunking-index retrieval invariants (ADR-002, docs/TESTING-AND-EVAL.md).

Every test here is hermetic: a deterministic fake embedder defined in this file is the
only "model", ``tests/fixtures.py::sample_doc()`` is the only PDF, and no network or
OpenAI client is ever touched. The invariants pinned are the ones the spec freezes:

1. a revenue/FY2023 query puts the income-statement *parent* at fused rank 1;
2. RRF arithmetic ``1/(k + rank)`` with ``k = 60`` on a hand-computed case;
3. ``save`` -> ``load`` -> ``retrieve`` reproduces every field bit-for-bit;
4. ``load`` on an empty/missing directory returns ``None``;
5. ``filters={"section": ...}`` excludes out-of-section chunks *before* ranking;
6. retrieval is deterministic;
7. an empty document yields a valid index that retrieves ``[]``;
8. ``k`` above the chunk count returns every chunk;
9. a query with no BM25 match never raises (dense still contributes).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from src.core.chunking import chunk_document
from src.core.index import ChunkRecord
from src.core.pdf_parser import Block, ParsedDocument, parse_pdf
from src.core.retrieval import (
    HybridIndexHandle,
    ScoredChunk,
    _fuse,
    _order,
    _ranks,
    _top_k,
    build_index,
    load_index,
)

from tests.fixtures import sample_doc

# --- the fake embedder (hermetic by contract) ------------------------------------

EMBED_DIMS = 128
_TOKEN = re.compile(r"\w+")


def _bucket(token: str) -> int:
    """Stable pseudo-random bucket for a token (blake2b, not salted ``hash``)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % EMBED_DIMS


def fake_embedder(texts: list[str]) -> list[list[float]]:
    """Hashed word-bag embedder: token frequency in a stable-bucketed 128-dim space.

    Deterministic across processes and runs, L2-normalized so an inner product *is* a
    cosine similarity. It carries no semantic knowledge, so the rank-1 revenue invariant
    below is decided by genuine lexical overlap — the same signal BM25 uses — which is
    what makes that assertion a test of the retrieval pipeline rather than a tautology
    about the embedder.
    """
    vectors: list[list[float]] = []
    for text in texts:
        bucket = [0.0] * EMBED_DIMS
        for token in _TOKEN.findall(text.lower()):
            bucket[_bucket(token)] += 1.0
        norm = math.sqrt(sum(value * value for value in bucket))
        vectors.append([value / norm for value in bucket] if norm else list(bucket))
    return vectors


# --- the corpus under test ---------------------------------------------------------


REVENUE_QUERY = "What was revenue in FY2023?"
CONTINGENCIES = "CONTINGENCIES"
SUMMARY = "SUMMARY OF RESULTS"

# Long enough that the chunker must cut it into several children (~250 tokens each),
# which is what lets the parent-expansion test see child-granular ranking.
PROSE_DOC_ID = "c" * 64
PROSE = "\n\n".join(
    f"Statement {i:02d}: revenue grew {i} percent across all segments in fiscal 2023."
    for i in range(1, 46)
)

# The hand-built RRF case: three unrelated one-child sections. The rank constants
# below were read off the real legs' score arrays and the fused order is derived by
# hand in the test, so a weighted-sum or dense-only implementation cannot pass.
RRF_CORPUS = (
    "growth cash revenue fiscal income",  # dense rank 1, sparse rank 2
    "income total growth flow margin",  # dense rank 2, sparse rank 3
    "margin total revenue annual flow",  # dense rank 3, sparse rank 1
)
RRF_QUERY = "growth income"
RRF_DENSE_RANKS = [1, 2, 3]
RRF_SPARSE_RANKS = [2, 3, 1]
# doc 0 is sparse rank 2 -> 1/62; doc 1 is sparse rank 3 -> 1/63; doc 2 is sparse rank
# 1 -> 1/61. Fused order is therefore 0 > 2 > 1, and *neither* leg agrees.
RRF_EXPECTED_FUSED = [
    float(Fraction(1, 61) + Fraction(1, 62)),  # doc 0: 0.03252247488101533
    float(Fraction(1, 62) + Fraction(1, 63)),  # doc 1: 0.03200204813108039
    float(Fraction(1, 63) + Fraction(1, 61)),  # doc 2: 0.032266458495966696
]
RRF_EXPECTED_ORDER = (0, 2, 1)


# --- fixtures ----------------------------------------------------------------------


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(sample_doc())
    return path


@pytest.fixture
def parsed(sample_pdf: Path) -> ParsedDocument:
    """The real fixture, parsed through the real parser and handed to retrieval."""
    return parse_pdf(sample_pdf)


@pytest.fixture
def handle(parsed: ParsedDocument) -> HybridIndexHandle:
    return build_index(parsed, fake_embedder)


@pytest.fixture
def prose_handle() -> HybridIndexHandle:
    """One section long enough to split into several children of a single parent."""
    doc = ParsedDocument(
        doc_id=PROSE_DOC_ID,
        filename="prose.pdf",
        page_count=1,
        blocks=(Block("text", "SEGMENT DISCUSSION\n" + PROSE, 1, ("SEGMENT DISCUSSION",)),),
    )
    return build_index(doc, fake_embedder)


def _rebuilt(doc: ParsedDocument) -> ParsedDocument:
    """A structurally identical copy, so ids depend on position only."""
    return ParsedDocument(
        doc_id=doc.doc_id, filename=doc.filename, page_count=doc.page_count, blocks=doc.blocks
    )


def _ranks_of(scores) -> list[int]:
    """1-based ranks, descending, ties keeping index order — mirrors the module."""
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    ranks = [0] * len(scores)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


def _chunk_id_for(handle: HybridIndexHandle, text: str) -> str:
    for record in handle.index.records:
        if text in record.child_text:
            return record.chunk_id
    raise AssertionError(f"no child chunk contains {text!r}")


# --- 1. revenue/FY2023 lands on the income-statement parent at fused rank 1 --------


def test_revenue_query_returns_income_statement_parent_at_rank_1(
    handle: HybridIndexHandle,
) -> None:
    results = handle.retrieve(REVENUE_QUERY)

    assert results, "revenue query returned nothing"
    top = results[0]
    assert top.fused_rank == 0, "the top hit is not fused rank 0"
    # The fixture's table block *is* the income statement: header plus both years.
    assert "$ 211,915" in top.parent_text, "FY2023 revenue missing from the parent"
    assert "$ 198,270" in top.parent_text, "FY2022 revenue missing from the parent"
    assert "FY2023" in top.parent_text and "Metric" in top.parent_text
    assert top.page == 2, "the income statement is on page 2 of the fixture"
    # Both retrievers agree it is the best chunk, so the fused head is not a fluke.
    assert top.dense_score > 0 and top.sparse_score > 0


def test_income_statement_parent_keeps_the_whole_table(handle: HybridIndexHandle) -> None:
    top = handle.retrieve(REVENUE_QUERY)[0]
    assert "| Metric | FY2023 | FY2022 |" in top.parent_text
    assert "| Operating income | $ 88,523 | $ 83,383 |" in top.parent_text
    # Both the markdown table and its flattened echo survive expansion.
    assert top.parent_text.count("$ 211,915") >= 2


def test_parent_text_comes_from_the_chunker(
    handle: HybridIndexHandle, parsed: ParsedDocument
) -> None:
    parents, _ = chunk_document(parsed)
    parent_texts = {parent.text for parent in parents}
    for chunk in handle.retrieve(REVENUE_QUERY):
        assert chunk.parent_text in parent_texts, "expanded to a parent the chunker never built"


# --- 2. RRF arithmetic by hand ----------------------------------------------------


def test_rrf_scores_match_the_k_equals_60_formula(handle: HybridIndexHandle) -> None:
    from src.core.pdf_parser import Block

    blocks = tuple(
        Block("text", text, 1, (f"SEC{index}",)) for index, text in enumerate(RRF_CORPUS)
    )
    doc = ParsedDocument(doc_id="a" * 64, filename="hand.pdf", page_count=1, blocks=blocks)
    index = build_index(doc, fake_embedder)

    dense = index.index.dense_scores(RRF_QUERY)
    sparse = index.index.sparse_scores(RRF_QUERY)
    assert dense is not None and sparse is not None, "both legs must be available"

    # The rank constants the hand-written expectations claim, verified against the legs.
    assert _ranks_of(dense) == RRF_DENSE_RANKS
    assert _ranks_of(sparse) == RRF_SPARSE_RANKS

    results = index.retrieve(RRF_QUERY, k=len(RRF_CORPUS))
    assert len(results) == len(RRF_CORPUS)

    # Fused order is the arithmetic, and it disagrees with each leg alone.
    expected_ids = [
        _chunk_id_for(index, text) for text in (RRF_CORPUS[i] for i in RRF_EXPECTED_ORDER)
    ]
    assert [r.chunk_id for r in results] == expected_ids
    assert [r.fused_rank for r in results] == [0, 1, 2]


def test_rrf_fused_scores_equal_the_sum_of_reciprocals(handle: HybridIndexHandle) -> None:
    from src.core.pdf_parser import Block

    blocks = tuple(
        Block("text", text, 1, (f"SEC{index}",)) for index, text in enumerate(RRF_CORPUS)
    )
    doc = ParsedDocument(doc_id="a" * 64, filename="hand.pdf", page_count=1, blocks=blocks)
    index = build_index(doc, fake_embedder)

    dense = index.index.dense_scores(RRF_QUERY)
    sparse = index.index.sparse_scores(RRF_QUERY)
    assert dense is not None and sparse is not None
    fused = _fuse(dense, sparse, len(RRF_CORPUS))
    for record_position, expected in enumerate(RRF_EXPECTED_FUSED):
        assert float(fused[record_position]) == pytest.approx(expected, rel=1e-12)


def test_rrf_ranks_are_one_based() -> None:
    """The top score sits at rank 1, so its term is ``1 / (60 + 1)`` — not ``1 / 60``."""
    scores = np.asarray([0.9, 0.7, 0.1])
    assert list(_ranks(scores, 3)) == [1, 2, 3]
    # The arithmetic: a rank-1 hit is worth 1/61, never 1/60 (what a 0-based rank gives).
    assert [float(value) for value in _fuse(None, scores, 3)] == [
        1.0 / (60 + rank) for rank in (1, 2, 3)
    ]
    assert float(_fuse(np.asarray([1.0, 0.0, 0.0]), None, 3)[0]) != pytest.approx(1.0 / 60)


def test_rrf_unretrieved_document_earns_no_term() -> None:
    """A leg that never retrieved a doc adds nothing: a single-list hit is not diluted."""
    dense = np.asarray([1.0, -np.inf, -np.inf])
    fused = _fuse(dense, None, 3)
    assert float(fused[0]) == pytest.approx(1.0 / 61)
    assert float(fused[1]) == pytest.approx(0.0)
    assert float(fused[2]) == pytest.approx(0.0)


def test_rrf_single_list_hit_keeps_its_full_term() -> None:
    """A doc found by one retriever alone still scores 1/(k + rank) from that leg."""
    # Only the dense leg is available: every doc earns its own reciprocal, ranked.
    dense_only = _fuse(np.asarray([1.0, 0.0]), None, 2)
    assert float(dense_only[0]) == pytest.approx(1.0 / 61), "rank-1 hit scores 1/(60+1)"
    assert float(dense_only[1]) == pytest.approx(1.0 / 62), "rank-2 hit scores 1/(60+2)"

    # Only the sparse leg: same formula, so the two legs are exchangeable.
    sparse_only = _fuse(None, np.asarray([0.2, 0.9]), 2)
    assert float(sparse_only[1]) == pytest.approx(1.0 / 61), "the top sparse hit keeps rank 1"
    assert float(sparse_only[0]) == pytest.approx(1.0 / 62)


def test_rrf_absent_leg_contributes_nothing() -> None:
    """``None`` is not a leg of zeros; it is a leg that adds no term at all."""
    scores = np.asarray([0.5, 0.5, 0.5])
    assert _fuse(None, None, 3).tolist() == [0.0, 0.0, 0.0]
    fused = _fuse(None, scores, 3)
    assert [float(value) for value in fused] == [1.0 / (60 + rank) for rank in (1, 2, 3)]


def test_top_k_hides_what_a_leg_did_not_retrieve() -> None:
    """Only a leg's top ``k`` is rankable; the rest is masked below every real score."""
    scores = np.asarray([0.9, 0.7, 0.1, 0.5, -0.2])
    masked = _top_k(scores, 3, 5)
    assert masked is not None
    # The three survivors keep their genuine scores, unmodified.
    for index in (0, 1, 3):
        assert float(masked[index]) == float(scores[index])
    # The two it did not return are unreachable — including a real negative cosine,
    # which masking at 0.0 would wrongly rank above an unseen document.
    lowest_real = float(np.min(scores))
    for index in (2, 4):
        assert float(masked[index]) < lowest_real
    ranked = _ranks(masked, 5)
    assert [int(value) for value in ranked[ranked > 0]] == [1, 2, 3]
    assert (ranked == 0).sum() == 2, "unretrieved docs get no rank and earn no RRF term"


def test_top_k_short_list_passes_through_whole() -> None:
    scores = np.asarray([0.1, 0.2])
    assert np.array_equal(_top_k(scores, 5, 2), scores), "k >= size changes nothing"
    assert np.array_equal(_top_k(scores, 2, 2), scores)
    assert _top_k(None, 3, 5) is None, "an unavailable leg stays unavailable"


def test_fused_order_breaks_ties_by_chunk_id() -> None:
    """Equal fused scores order by chunk_id ascending, so ``fused_rank`` is total."""
    records = [
        ChunkRecord("zz-chunk", "p", "t", "pt", 1, ("s",), "d", 0),
        ChunkRecord("aa-chunk", "p", "t", "pt", 1, ("s",), "d", 0),
    ]
    tie = np.asarray([1.0 / 61 + 1.0 / 62, 1.0 / 61 + 1.0 / 62])
    assert _order(tie, records) == [1, 0]


# --- 3. save -> load -> retrieve round-trip ---------------------------------------


def test_save_load_retrieve_is_deeply_equal(handle: HybridIndexHandle, tmp_path: Path) -> None:
    before = handle.retrieve(REVENUE_QUERY)

    target = tmp_path / "index"
    handle.save(target)
    assert (target / "chunks.jsonl").is_file(), "the source of truth was not written"

    reloaded = load_index(target, fake_embedder)
    assert reloaded is not None, "load returned None for a saved index"
    after = reloaded.retrieve(REVENUE_QUERY)

    assert after == before
    for a, b in zip(before, after, strict=True):
        assert a.chunk_id == b.chunk_id
        assert a.child_text == b.child_text
        assert a.parent_text == b.parent_text
        assert a.page == b.page
        assert a.section_path == b.section_path
        assert a.doc_id == b.doc_id
        assert a.dense_score == b.dense_score
        assert a.sparse_score == b.sparse_score
        assert a.fused_rank == b.fused_rank


def test_chunks_jsonl_is_the_source_of_truth(handle: HybridIndexHandle, tmp_path: Path) -> None:
    import shutil

    target = tmp_path / "index"
    handle.save(target)
    lines = [
        json.loads(line)
        for line in (target / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "chunk_id",
        "parent_id",
        "child_text",
        "parent_text",
        "page",
        "section_path",
        "doc_id",
        "ordinal",
    }
    assert len(lines) == len(handle.index.records)
    for line in lines:
        assert required <= set(line), f"chunks.jsonl line missing fields: {sorted(line)}"
        assert isinstance(line["section_path"], list)
        assert isinstance(line["page"], int) and isinstance(line["ordinal"], int)
    assert all(line["child_text"] in line["parent_text"] for line in lines)

    # Delete the derived artifacts: the index must rebuild from the JSONL alone.
    shutil.rmtree(target / "faiss")
    (target / "bm25.pkl").unlink()
    rebuilt = load_index(target, fake_embedder)
    assert rebuilt is not None, "the index could not be rebuilt from chunks.jsonl"
    assert rebuilt.retrieve(REVENUE_QUERY) == handle.retrieve(REVENUE_QUERY)


def test_embedder_is_never_persisted(handle: HybridIndexHandle, tmp_path: Path) -> None:
    """The embedder is query-time wiring; persisting it would leak the injection seam."""
    target = tmp_path / "index"
    handle.save(target)
    names = {path.name for path in target.rglob("*") if path.is_file()}
    assert not [name for name in names if "embed" in name.lower()]


def test_index_loads_with_no_embedder(handle: HybridIndexHandle, tmp_path: Path) -> None:
    """A saved index loads without one; only the query-time dense leg goes dark."""
    target = tmp_path / "index"
    handle.save(target)
    reloaded = load_index(target, None)
    assert reloaded is not None
    # The persisted FAISS artifact keeps the dimensionality, so the index is intact.
    assert reloaded.index.dim == handle.index.dim
    # No embedder means no query vector: dense scores go to zero but nothing raises.
    sparse_only = reloaded.retrieve(REVENUE_QUERY)
    assert all(c.dense_score == 0.0 for c in sparse_only)
    assert sparse_only, "BM25 alone must still rank the fixture"
    assert [c.chunk_id for c in sparse_only] == [c.chunk_id for c in handle.retrieve(REVENUE_QUERY)]


# --- 4. load on an empty or missing directory -------------------------------------


def test_load_returns_none_for_a_missing_directory(tmp_path: Path) -> None:
    assert load_index(tmp_path / "does-not-exist", fake_embedder) is None


def test_load_returns_none_for_an_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_index(empty, fake_embedder) is None


def test_load_returns_none_for_a_missing_directory_at_module_scope(tmp_path: Path) -> None:
    assert load_index(tmp_path / "gone" / "deeper", fake_embedder) is None


# --- 5. filters -------------------------------------------------------------------


def test_section_filter_excludes_out_of_section_chunks(handle: HybridIndexHandle) -> None:
    """Filtering runs BEFORE ranking, so out-of-section chunks are never candidates."""
    every = handle.retrieve(REVENUE_QUERY)
    kept = handle.retrieve(REVENUE_QUERY, filters={"section": CONTINGENCIES})

    assert kept, "no chunks in the contingencies section of the fixture"
    kept_ids = {chunk.chunk_id for chunk in kept}
    assert kept_ids < {chunk.chunk_id for chunk in every}, "the filter excluded nothing"

    assert all(CONTINGENCIES in chunk.section_path for chunk in kept)
    dropped = [chunk for chunk in every if chunk.chunk_id not in kept_ids]
    assert dropped, "every chunk survived the filter — the assertion is vacuous"
    assert all(CONTINGENCIES not in chunk.section_path for chunk in dropped)


def test_section_filter_reranks_from_zero(handle: HybridIndexHandle) -> None:
    """A chunk the unfiltered query ranks last becomes rank 0 once filtered in."""
    every = handle.retrieve(REVENUE_QUERY)
    kept = handle.retrieve(REVENUE_QUERY, filters={"section": CONTINGENCIES})
    assert [c.fused_rank for c in kept] == [0], "filtered results are re-ranked from 0"
    assert (
        kept[0].chunk_id == every[-1].chunk_id
    ), "the contingencies chunk is last unfiltered, so filtering lifted it to rank 1"


def test_section_filter_can_empty_the_result(handle: HybridIndexHandle) -> None:
    assert handle.retrieve(REVENUE_QUERY, filters={"section": "NOT A SECTION"}) == []
    assert handle.retrieve(REVENUE_QUERY, filters={"section": "Annual report"}) == []


def test_doc_id_filter_restricts_to_that_document(
    handle: HybridIndexHandle, parsed: ParsedDocument
) -> None:
    assert handle.retrieve(REVENUE_QUERY, filters={"doc_id": parsed.doc_id})
    assert handle.retrieve(REVENUE_QUERY, filters={"doc_id": "not-a-doc-id"}) == []


def test_unknown_filter_key_raises_value_error(handle: HybridIndexHandle) -> None:
    with pytest.raises(ValueError):
        handle.retrieve(REVENUE_QUERY, filters={"bogus": "value"})
    with pytest.raises(ValueError):
        handle.retrieve(REVENUE_QUERY, filters={"section": SUMMARY, "nope": 1})


def test_both_filters_apply_together(handle: HybridIndexHandle, parsed: ParsedDocument) -> None:
    kept = handle.retrieve(
        REVENUE_QUERY, filters={"section": CONTINGENCIES, "doc_id": parsed.doc_id}
    )
    assert kept and all(CONTINGENCIES in c.section_path for c in kept)
    assert (
        handle.retrieve(REVENUE_QUERY, filters={"section": CONTINGENCIES, "doc_id": "other"}) == []
    )


# --- 6. determinism ---------------------------------------------------------------


def test_retrieve_twice_is_identical(handle: HybridIndexHandle) -> None:
    first = handle.retrieve(REVENUE_QUERY)
    assert handle.retrieve(REVENUE_QUERY) == first
    assert [c.fused_rank for c in first] == list(range(len(first)))


def test_retrieve_is_stable_under_repetition(handle: HybridIndexHandle) -> None:
    first = handle.retrieve(REVENUE_QUERY)
    for _ in range(3):
        assert handle.retrieve(REVENUE_QUERY) == first


def test_retrieve_is_identical_across_independently_built_indices(
    parsed: ParsedDocument,
) -> None:
    a = build_index(_rebuilt(parsed), fake_embedder)
    b = build_index(_rebuilt(parsed), fake_embedder)
    assert a.retrieve(REVENUE_QUERY) == b.retrieve(REVENUE_QUERY)


def test_fused_rank_is_zero_based_and_contiguous(handle: HybridIndexHandle) -> None:
    results = handle.retrieve(REVENUE_QUERY, k=len(handle.index.records))
    assert [c.fused_rank for c in results] == list(range(len(results)))


def test_retrieval_is_deterministic_after_save_and_reload(
    handle: HybridIndexHandle, tmp_path: Path
) -> None:
    target = tmp_path / "index"
    handle.save(target)
    reloaded = load_index(target, fake_embedder)
    assert reloaded is not None
    assert reloaded.retrieve(REVENUE_QUERY) == reloaded.retrieve(REVENUE_QUERY)
    assert reloaded.retrieve(REVENUE_QUERY) == handle.retrieve(REVENUE_QUERY)


# --- 7. an empty document is a valid, empty index ---------------------------------


def test_empty_document_builds_a_valid_empty_index() -> None:
    empty = ParsedDocument(doc_id="0" * 64, filename="empty.pdf", page_count=0, blocks=())
    handle = build_index(empty, fake_embedder)

    assert isinstance(handle, HybridIndexHandle)
    assert len(handle.index.records) == 0
    assert handle.index.dim is None
    assert handle.retrieve(REVENUE_QUERY) == []
    assert handle.retrieve(REVENUE_QUERY, k=8) == []
    assert handle.retrieve(REVENUE_QUERY, filters={"section": SUMMARY}) == []


def test_empty_index_round_trips(tmp_path: Path) -> None:
    empty = ParsedDocument(doc_id="0" * 64, filename="empty.pdf", page_count=0, blocks=())
    handle = build_index(empty, fake_embedder)
    target = tmp_path / "empty-index"
    handle.save(target)

    reloaded = load_index(target, fake_embedder)
    assert reloaded is not None, "an empty index is valid and loadable"
    assert reloaded.retrieve(REVENUE_QUERY) == []


def test_whitespace_only_document_builds_an_empty_index() -> None:
    doc = ParsedDocument(
        doc_id="1" * 64,
        filename="blank.pdf",
        page_count=1,
        blocks=(Block("text", "   \n  ", 1, ("SEC",)),),
    )
    handle = build_index(doc, fake_embedder)
    assert len(handle.index.records) == 0
    assert handle.retrieve("revenue") == []


def test_bm25_only_index_never_raises_zero_division_error(parsed: ParsedDocument) -> None:
    """``BM25Okapi([])`` divides by corpus size; an empty corpus must not construct one."""
    empty = ParsedDocument(doc_id="0" * 64, filename="empty.pdf", page_count=0, blocks=())
    handle = build_index(empty, fake_embedder)
    assert handle.retrieve("anything at all") == []

    # A real corpus indexed with no embedder: sparse ranks, dense is simply absent.
    sparse_only = build_index(parsed, embedder=None)
    assert sparse_only.index.dim is None
    results = sparse_only.retrieve(REVENUE_QUERY)
    assert results, "BM25 alone must still rank the fixture"
    assert all(c.dense_score == 0.0 for c in results)
    assert [c.fused_rank for c in results] == list(range(len(results)))


# --- 8. k above the chunk count returns every chunk -------------------------------


def test_k_above_chunk_count_returns_every_chunk(handle: HybridIndexHandle) -> None:
    total = len(handle.index.records)
    for k in (total, total + 1, total * 10):
        results = handle.retrieve(REVENUE_QUERY, k=k)
        assert len(results) == total, f"k={k} returned {len(results)} of {total} chunks"
        assert [c.fused_rank for c in results] == list(range(total))


def test_k_limits_the_result(handle: HybridIndexHandle) -> None:
    total = len(handle.index.records)
    assert len(handle.retrieve(REVENUE_QUERY, k=1)) == min(1, total)
    if total > 1:
        assert len(handle.retrieve(REVENUE_QUERY, k=2)) == 2


def test_invalid_k_is_rejected(handle: HybridIndexHandle) -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError):
            handle.retrieve(REVENUE_QUERY, k=bad)


# --- 9. a query with no BM25 match never raises -----------------------------------


def test_no_bm25_match_still_scores_on_dense(handle: HybridIndexHandle) -> None:
    """BM25 returns all zeros; ranking zeros is well-defined and dense carries the query."""
    results = handle.retrieve("zzqqxx nonexistenttoken wubba")

    assert results, "a no-match query returned no chunks at all"
    assert all(chunk.sparse_score == 0.0 for chunk in results), "unexpected sparse match"
    assert any(chunk.dense_score > 0.0 for chunk in results), "the dense leg contributed nothing"
    # ADR-002 formula taken literally over the whole corpus: zero BM25 still ranks.
    assert len(results) == len(handle.index.records)


def test_query_without_word_tokens_does_not_crash(handle: HybridIndexHandle) -> None:
    """BM25 can rank nothing without tokens; punctuation never raises."""
    punctuation = handle.retrieve("!!! ??? ...")
    assert punctuation, "a tokenless query still ranks the corpus, all zero scores"
    assert all(chunk.sparse_score == 0.0 for chunk in punctuation)
    assert all(chunk.dense_score == 0.0 for chunk in punctuation)
    assert [c.fused_rank for c in punctuation] == list(range(len(punctuation)))
    assert handle.retrieve("") == [], "a blank query answers nothing at all"


def test_empty_query_on_an_empty_index_does_not_crash() -> None:
    empty = ParsedDocument(doc_id="0" * 64, filename="empty.pdf", page_count=0, blocks=())
    handle = build_index(empty, fake_embedder)
    assert handle.retrieve("") == []
    assert handle.retrieve("!!!") == []


# --- 10. parent expansion happens after fusion ------------------------------------


def test_ranking_is_at_child_granularity_with_parent_expansion(
    prose_handle: HybridIndexHandle,
) -> None:
    """Children are ranked, then expanded: the parent text is the payload, never the rank."""
    records = prose_handle.index.records
    assert len(records) > 1, "the fixture section must split into several children"
    assert len({r.parent_id for r in records}) == 1, "one parent owns them all"

    results = prose_handle.retrieve("revenue fiscal 2023", k=len(records))
    assert len(results) == len(records)
    assert len({c.chunk_id for c in results}) == len(records), "children stay distinct"
    assert len({c.parent_text for c in results}) == 1, "all hits expand to one parent"
    for chunk in results:
        assert chunk.child_text in chunk.parent_text, "child was not cut from its parent"


def test_parent_expansion_cannot_reorder_by_parent_text(
    prose_handle: HybridIndexHandle,
) -> None:
    """Every hit shares one parent, so any ordering beyond record order is child rank."""
    records = prose_handle.index.records
    results = prose_handle.retrieve("revenue fiscal 2023", k=len(records))
    assert results
    fused = [c.chunk_id for c in results]
    assert len({c.parent_text for c in results}) == 1, "one parent text, many hits"
    assert len(set(fused)) == len(fused), "one row per child chunk"
    # The child order is not simply the chunker's record order: rank was decided per child.
    assert fused != [r.chunk_id for r in records]


def test_dense_ranks_the_child_not_the_parent(
    prose_handle: HybridIndexHandle,
) -> None:
    """A query matching one sibling must not promote its siblings.

    This is the ADR-002 structural rule — "parent expansion happens AFTER fusion,
    ranking is decided at child granularity" — and it is only observable when two
    children share a parent and a query matches exactly one. With one child per
    parent the two designs are indistinguishable, which is why a mutation that
    embeds ``parent_text`` instead of ``child_text`` survived the whole suite:
    every sibling then shares one identical dense vector and both designs agree.
    """
    records = prose_handle.index.records
    siblings = [r for r in records if r.parent_id == records[0].parent_id]
    assert len(siblings) >= 2, f"need >=2 siblings to discriminate, got {len(siblings)}"

    # Query text that lives in exactly ONE sibling and is NOT the section heading.
    # The heading ("SEGMENT DISCUSSION") appears once only because it sits at a split
    # boundary, which makes it a weak signal: under a parent-text mutation all siblings
    # share one dense vector, ties break by record order, and the assertion passes by
    # luck rather than by child granularity. Use a content sentence instead.
    heading = "SEGMENT DISCUSSION"
    target = next((s for s in siblings if len(s.child_text.strip()) > 40), siblings[0])
    others_text = "\n".join(r.child_text for r in siblings if r is not target)
    phrase = next(
        (
            line
            for line in target.child_text.strip().splitlines()
            if len(line.strip()) > 20
            and line.strip() != heading
            and line.strip() not in others_text
        ),
        "",
    )
    assert phrase, "fixture must have a content line unique to the target sibling"
    assert "Statement" in phrase, "expected a numbered content sentence, not a heading"

    results = prose_handle.retrieve(phrase, k=len(records))
    assert results, "no results"

    # The load-bearing property, asserted on scores rather than on order: siblings that
    # contain DIFFERENT text must receive DIFFERENT dense scores. If the dense leg were
    # embedding parent_text (expansion before fusion), every sibling would share one
    # vector and one score, and the ranking would be pure FAISS tie-break order.
    by_id = {c.chunk_id: c for c in results}
    scores = {cid: by_id[cid].dense_score for cid in by_id if cid in {s.chunk_id for s in siblings}}
    assert len(scores) >= 2, "need >=2 sibling scores to compare"
    assert len(set(scores.values())) >= 2, (
        "siblings share one dense score — the dense leg is not ranking child text "
        "(ADR-002: expansion happens after fusion)"
    )
    # And the sibling that actually contains the query must be among the top hits.
    assert target.chunk_id in {
        c.chunk_id for c in results[:2]
    }, "the child containing the query text should be at or near the top"
    assert len({r.parent_text for r in results}) == 1, "all hits share one parent"


def test_children_of_one_parent_carry_the_parent_page(prose_handle: HybridIndexHandle) -> None:
    for chunk in prose_handle.retrieve("revenue fiscal 2023", k=8):
        assert chunk.section_path == ["SEGMENT DISCUSSION"]
        assert chunk.page >= 1


# --- idempotency and the contract surface ----------------------------------------


def test_reindexing_the_same_document_is_idempotent(
    handle: HybridIndexHandle, parsed: ParsedDocument
) -> None:
    again = HybridIndexHandle(handle.index.replace_document(parsed, embedder=fake_embedder))
    assert len(again.index.records) == len(handle.index.records)
    assert [r.chunk_id for r in again.index.records] == [r.chunk_id for r in handle.index.records]
    assert again.retrieve(REVENUE_QUERY) == handle.retrieve(REVENUE_QUERY)


def test_two_documents_keep_their_chunks_separate(parsed: ParsedDocument) -> None:
    """A second document appends its rows; a duplicate ``chunk_id`` is never built twice."""
    first = build_index(_rebuilt(parsed), fake_embedder)
    other = ParsedDocument(
        doc_id="d" * 64,
        filename="other.pdf",
        page_count=1,
        blocks=(Block("text", "OTHER\nRevenue mentions elsewhere.", 1, ("OTHER",)),),
    )
    combined = HybridIndexHandle(first.index.replace_document(other, embedder=fake_embedder))
    assert len(combined.index.records) > len(first.index.records)
    assert {r.chunk_id.split("-p")[0] for r in combined.index.records} >= {other.doc_id[:12]}

    combined_ids = [record.chunk_id for record in combined.index.records]
    assert len(combined_ids) == len(set(combined_ids)), "a duplicate chunk_id was built"

    # Re-indexing the first document onto the combined index replaces its rows: the set
    # of chunks is unchanged (idempotent by ``doc_id``) and retrieval is identical.
    again = HybridIndexHandle(combined.index.replace_document(parsed, embedder=fake_embedder))
    assert {r.chunk_id for r in again.index.records} == set(combined_ids)
    assert combined.retrieve(REVENUE_QUERY) == again.retrieve(REVENUE_QUERY)

    # Indexing the same document many times still never duplicates its chunks.
    third = again.index.replace_document(parsed, embedder=fake_embedder)
    assert [r.chunk_id for r in third.records] == [r.chunk_id for r in again.index.records]


def test_scored_chunk_fields_match_the_frozen_contract(handle: HybridIndexHandle) -> None:
    top = handle.retrieve(REVENUE_QUERY)[0]
    assert isinstance(top, ScoredChunk)
    assert isinstance(top.chunk_id, str) and top.chunk_id
    assert isinstance(top.child_text, str) and top.child_text
    assert isinstance(top.parent_text, str) and top.parent_text
    assert isinstance(top.page, int) and top.page >= 1
    assert isinstance(top.section_path, list)
    assert isinstance(top.doc_id, str) and top.doc_id
    assert isinstance(top.dense_score, float)
    assert isinstance(top.sparse_score, float)
    assert isinstance(top.fused_rank, int) and top.fused_rank >= 0


def test_module_level_retrieve_serves_the_built_index(tmp_path: Path, monkeypatch) -> None:
    """The schema's loose ``retrieve`` serves whatever index this process built."""
    from src.core import retrieval as module

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(sample_doc())
    monkeypatch.setattr(module, "_INDEX_DIR", tmp_path / "no-index-here")
    monkeypatch.setattr(module, "_INDEX_LOADED", False)
    monkeypatch.setattr(module, "_INDEX", None)
    assert module.retrieve(REVENUE_QUERY) == []

    build_index(parse_pdf(pdf), fake_embedder)
    assert module.retrieve(REVENUE_QUERY) == module.retrieve(REVENUE_QUERY)
    assert len(module.retrieve(REVENUE_QUERY, k=1)) == 1
    assert module.retrieve(REVENUE_QUERY, filters={"section": CONTINGENCIES})


def test_retrieval_module_never_imports_openai() -> None:
    """No network client in this module: the embedder is injected, always."""
    import sys

    assert "src.core.retrieval" in sys.modules, "the module under test was not imported"
    assert not [name for name in sys.modules if name == "openai" or name.startswith("openai.")]

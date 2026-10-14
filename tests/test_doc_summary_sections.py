"""``DocSummary.sections`` is a tuple of heading *paths*, and every reader reads one.

The audit that produced this file measured the symptom on the real corpus: a ``topic``
refusal rendered *"these documents cover i"* — the first *character* of the section heading
``ISSUER PURCHASES OF EQUITY SECURITIES, DIVIDENDS, AND STOCK PERFORMANCE`` — because the
catalog was built as a flat ``list[str]`` and ``refuse._subject_headings`` read ``path[0]``
expecting the first *level* of the path. ``_catalog_text`` was corrupted the same way, and
only ``analyze_question``'s ``_section_text`` defended against the wrong shape.

The invariant, now enforced at the type by a parse helper
(:func:`~src.chains.state.section_paths`): a catalog reader hands back the strings
``docs.jsonl`` actually stores (``"A > B > C"``) and the summary's ``sections`` is
``tuple[tuple[str, ...], ...]`` — one tuple of heading levels per section path. A flat list
of strings cannot be built again by a caller that goes through the helper, which is why the
helper is the fix and not a runtime guard each caller would have to remember.

Everything here is hermetic: the gateway is never reached (``refuse`` is deterministic and
``analyze_question``'s planner is patched out), the catalog rows are hand-written JSON in
the corpus's own shape, and nothing touches ``data/`` or the network.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from src.chains.nodes.analyze import (
    _catalog_json,
    _covered_years,
    _section_text,
    analyze_question,
)
from src.chains.nodes.refuse import _subject_headings, _topic_copy, refuse
from src.chains.state import DocSummary, QueryPlan, section_paths
from src.utils.llm import ChatResult, LLMParams

#: The corpus's own catalog row (``data/index/docs.jsonl`` line 1), trimmed to two real
#: paths so the assertions stay readable. ``sections`` on disk is a flat list of
#: ``"A > B > C"`` strings — the wire format the app must convert on the way in.
_CATALOG_ROW: dict[str, Any] = {
    "doc_id": "311fa88413b7dae376ba0f19994f906ec0cc8a44b2e861e5fdfde542e858738b",
    "filename": "2023_Annual_Report.pdf",
    "page_count": 85,
    "sections": [
        "ISSUER PURCHASES OF EQUITY SECURITIES, DIVIDENDS, AND STOCK PERFORMANCE",
        "ISSUER PURCHASES OF EQUITY SECURITIES, DIVIDENDS, AND STOCK PERFORMANCE > "
        "STOCK PERFORMANCE > RESEARCH AND DEVELOPMENT",
    ],
}

#: The root heading of every path in that row: what a ``topic`` refusal must name.
_ROOT = "ISSUER PURCHASES OF EQUITY SECURITIES, DIVIDENDS, AND STOCK PERFORMANCE"

#: What the flat shape made the same catalog render: ``path[0]`` on a string indexes a
#: character, so the whole first letter of the root heading was the "subject".
_BUGGY_SUBJECT = "I"


def _summary() -> DocSummary:
    """One :class:`DocSummary` built from the corpus row through the shared helper."""

    return DocSummary(
        doc_id=str(_CATALOG_ROW["doc_id"]),
        filename=str(_CATALOG_ROW["filename"]),
        sections=section_paths(_CATALOG_ROW.get("sections")),
        page_count=int(_CATALOG_ROW["page_count"]),
    )


def _flat_summary() -> DocSummary:
    """The same catalog the way ``app.py::_doc_catalog`` builds it: flat strings.

    The parameter is annotated ``Any`` on purpose: this is the one shape that must never
    exist, and the test holds it to show what it renders. mypy is not run over ``tests/``,
    so the untyped alias keeps a runtime-only fingerprint of the broken builder honest
    rather than asserting the broken shape through a suppression.
    """

    flat: Any = [str(item) for item in _CATALOG_ROW["sections"]]
    return DocSummary(
        doc_id=str(_CATALOG_ROW["doc_id"]),
        filename=str(_CATALOG_ROW["filename"]),
        sections=flat,
        page_count=int(_CATALOG_ROW["page_count"]),
    )


# --- 1. the shape, at the type --------------------------------------------------------


def test_a_catalog_row_becomes_a_path_of_headings_not_a_flat_list() -> None:
    """``sections`` holds one tuple of levels per path, so ``path[0]`` is the first level."""

    assert _summary().sections == (
        (_ROOT,),
        (
            _ROOT,
            "STOCK PERFORMANCE",
            "RESEARCH AND DEVELOPMENT",
        ),
    )


def test_the_helper_handles_every_shape_the_catalog_writer_can_produce() -> None:
    """One string, a list of strings, and paths already split — all land as tuples of levels."""

    assert section_paths("A > B") == (("A", "B"),)
    assert section_paths(["A > B", "A > B > C"]) == (("A", "B"), ("A", "B", "C"))
    assert section_paths([("A", "B")]) == (("A", "B"),)
    # ``docs.jsonl`` stores no separators in a single-level heading, so a bare
    # whitespace-separated list of paths must not be split into one path per word.
    assert section_paths("A") == (("A",),)
    assert section_paths(None) == ()
    assert section_paths(42) == ()


# --- 2. the user-visible symptom: the topic refusal names the heading ------------------


def test_a_topic_refusal_names_the_section_heading_not_its_first_character() -> None:
    """The regression this file exists for: "cover i" was the first glyph of the heading."""

    assert _topic_copy([_summary()]) == f"these documents cover {_ROOT.lower()}"


def test_the_first_level_of_a_path_is_the_subject_the_refusal_renders() -> None:
    """A nested path still reads as one subject: its root, in catalog order, de-duplicated."""

    assert _subject_headings([_summary()]) == [_ROOT]


def test_the_flat_shape_still_renders_a_single_character_and_is_the_bug() -> None:
    """Pinned so the symptom is visible: the flat catalog yields ``"i"``, not the heading."""

    assert _subject_headings([_flat_summary()]) == [_BUGGY_SUBJECT]


def test_a_flat_shaped_summary_is_repaired_at_the_readers_boundary() -> None:
    """A caller that still hands back the flat shape is converted, not read as glyphs.

    The next caller will get the shape wrong — ``docs.jsonl`` stores it flat and nothing on
    disk says otherwise — so both readers resolve ``DocSummary.sections`` through
    ``section_paths`` before they read it. Neither node may raise on it and neither may
    render a character, so repair is the only honest outcome.
    """

    answer = refuse(
        {
            "query_plan": QueryPlan(kind="out_of_scope", missing_aspect="topic"),
            "standalone_question": "What does the report say about convertible bonds?",
            "doc_catalog": [_flat_summary()],
        }
    )
    assert isinstance(answer["answer_md"], str)
    assert _ROOT.lower() in answer["answer_md"].lower()
    assert "cover i." not in answer["answer_md"].lower()


# --- 3. the refusal end to end --------------------------------------------------------


def test_the_refusal_node_renders_the_heading_for_a_topic_question() -> None:
    """The whole node, not the helper: the answer text a user reads carries the heading."""

    plan = QueryPlan(kind="out_of_scope", missing_aspect="topic")
    answer = refuse(
        {
            "query_plan": plan,
            "standalone_question": "What does the report say about convertible bonds?",
            "doc_catalog": [_summary()],
        }
    )
    assert isinstance(answer["answer_md"], str)
    assert _ROOT.lower() in answer["answer_md"].lower()


# --- 4. analyze's defensive reader, now the narrowed contract --------------------------


def test_a_section_path_is_rendered_as_the_path_it_is() -> None:
    """``analyze_question`` renders one path as ``"A > B"``, the form the catalog stores."""

    assert _section_text(("A", "B")) == "A > B"
    assert _section_text(("A",)) == "A"


def test_analyze_still_reads_a_path_a_caller_handed_back_as_one_string() -> None:
    """``_section_text`` stays shape-tolerant: a whole string is one path, not a glyph run.

    The tolerance is deliberate and retained — ``analyze`` reads the catalog through a
    ``list[Any]`` state boundary, so it must not assume the shape a caller built.
    """

    assert _section_text("A > B") == "A > B"
    assert _section_text(["A", "B"]) == "A > B"
    assert _section_text(None) == ""
    assert _section_text(7) == ""


def test_the_catalog_the_planner_sees_carries_the_whole_heading_path() -> None:
    """The payload ``analyze`` sends keeps the levels of the path, not the first letter."""

    text = _catalog_json([_summary()])
    entries = json.loads(text)
    assert [entry["sections"] for entry in entries] == [
        [
            _ROOT,
            f"{_ROOT} > STOCK PERFORMANCE > RESEARCH AND DEVELOPMENT",
        ]
    ]


def test_a_flat_shaped_catalog_is_repaired_before_the_planner_sees_it() -> None:
    """The same conversion ``refuse`` applies, on the planner's payload.

    ``analyze`` reads the catalog through the same ``list[Any]`` boundary, so a flat summary
    is resolved before it is rendered rather than shipped as ``"I"``.
    """

    text = _catalog_json([_flat_summary()])
    entries = json.loads(text)
    assert [entry["sections"] for entry in entries] == [
        [
            _ROOT,
            f"{_ROOT} > STOCK PERFORMANCE > RESEARCH AND DEVELOPMENT",
        ]
    ]


def test_the_fiscal_year_gate_reads_the_years_the_whole_catalog_shows(planner: Any) -> None:
    """``_covered_years`` joins the whole catalog text, so the report's year stands for its span."""

    assert _covered_years([_summary()]) == {2021, 2022, 2023}


def test_the_uncovered_year_gate_still_refuses_without_a_model_call(planner: Any) -> None:
    """The gate this shape must not disturb: FY2019 never reaches the planner."""

    stub = planner()
    plan = analyze_question(
        {
            "question": "What was revenue in FY2019?",
            "doc_catalog": [_summary()],
        }
    )["query_plan"]
    assert isinstance(plan, QueryPlan)
    assert plan.kind == "out_of_scope"
    assert plan.missing_aspect == "time_period"
    assert stub.calls == []


# --- the seam the node tests stub, mirroring tests/test_nodes_analyze.py ----------------


class _PlannerStub:
    """A gateway client recording whether it was consulted at all."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(
            text="{}",
            parsed={
                "kind": "simple",
                "sub_queries": [],
                "required_sections": [],
                "fiscal_period": "2023",
                "missing_aspect": None,
            },
            prompt_tokens=8,
            completion_tokens=2,
            latency_ms=1.0,
            model="stub-model",
        )


@pytest.fixture
def planner(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch ``build_llm`` and hand back a factory installing that one stub."""

    holder: dict[str, _PlannerStub] = {}

    def factory(params: LLMParams | None = None) -> _PlannerStub:
        return holder["stub"]

    stub = _PlannerStub()
    holder["stub"] = stub
    monkeypatch.setattr("src.chains.nodes.analyze.build_llm", factory)
    return lambda: stub

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

_ROOT = "ISSUER PURCHASES OF EQUITY SECURITIES, DIVIDENDS, AND STOCK PERFORMANCE"

_BUGGY_SUBJECT = "I"


def _summary() -> DocSummary:

    return DocSummary(
        doc_id=str(_CATALOG_ROW["doc_id"]),
        filename=str(_CATALOG_ROW["filename"]),
        sections=section_paths(_CATALOG_ROW.get("sections")),
        page_count=int(_CATALOG_ROW["page_count"]),
    )


def _flat_summary() -> DocSummary:

    flat: Any = [str(item) for item in _CATALOG_ROW["sections"]]
    return DocSummary(
        doc_id=str(_CATALOG_ROW["doc_id"]),
        filename=str(_CATALOG_ROW["filename"]),
        sections=flat,
        page_count=int(_CATALOG_ROW["page_count"]),
    )


def test_a_catalog_row_becomes_a_path_of_headings_not_a_flat_list() -> None:

    assert _summary().sections == (
        (_ROOT,),
        (
            _ROOT,
            "STOCK PERFORMANCE",
            "RESEARCH AND DEVELOPMENT",
        ),
    )


def test_the_helper_handles_every_shape_the_catalog_writer_can_produce() -> None:

    assert section_paths("A > B") == (("A", "B"),)
    assert section_paths(["A > B", "A > B > C"]) == (("A", "B"), ("A", "B", "C"))
    assert section_paths([("A", "B")]) == (("A", "B"),)
    assert section_paths("A") == (("A",),)
    assert section_paths(None) == ()
    assert section_paths(42) == ()


def test_a_topic_refusal_names_the_section_heading_not_its_first_character() -> None:

    assert _topic_copy([_summary()]) == f"these documents cover {_ROOT.lower()}"


def test_the_first_level_of_a_path_is_the_subject_the_refusal_renders() -> None:

    assert _subject_headings([_summary()]) == [_ROOT]


def test_the_flat_shape_still_renders_a_single_character_and_is_the_bug() -> None:

    assert _subject_headings([_flat_summary()]) == [_BUGGY_SUBJECT]


def test_a_flat_shaped_summary_is_repaired_at_the_readers_boundary() -> None:

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


def test_the_refusal_node_renders_the_heading_for_a_topic_question() -> None:

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


def test_a_section_path_is_rendered_as_the_path_it_is() -> None:

    assert _section_text(("A", "B")) == "A > B"
    assert _section_text(("A",)) == "A"


def test_analyze_still_reads_a_path_a_caller_handed_back_as_one_string() -> None:

    assert _section_text("A > B") == "A > B"
    assert _section_text(["A", "B"]) == "A > B"
    assert _section_text(None) == ""
    assert _section_text(7) == ""


def test_the_catalog_the_planner_sees_carries_the_whole_heading_path() -> None:

    text = _catalog_json([_summary()])
    entries = json.loads(text)
    assert [entry["sections"] for entry in entries] == [
        [
            _ROOT,
            f"{_ROOT} > STOCK PERFORMANCE > RESEARCH AND DEVELOPMENT",
        ]
    ]


def test_a_flat_shaped_catalog_is_repaired_before_the_planner_sees_it() -> None:

    text = _catalog_json([_flat_summary()])
    entries = json.loads(text)
    assert [entry["sections"] for entry in entries] == [
        [
            _ROOT,
            f"{_ROOT} > STOCK PERFORMANCE > RESEARCH AND DEVELOPMENT",
        ]
    ]


def test_the_fiscal_year_gate_reads_the_years_the_whole_catalog_shows(planner: Any) -> None:

    assert _covered_years([_summary()]) == {2021, 2022, 2023}


def test_the_uncovered_year_gate_still_refuses_without_a_model_call(planner: Any) -> None:

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


class _PlannerStub:

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

    holder: dict[str, _PlannerStub] = {}

    def factory(params: LLMParams | None = None) -> _PlannerStub:
        return holder["stub"]

    stub = _PlannerStub()
    holder["stub"] = stub
    monkeypatch.setattr("src.chains.nodes.analyze.build_llm", factory)
    return lambda: stub

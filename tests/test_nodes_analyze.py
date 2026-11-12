from __future__ import annotations

from typing import Any

import pytest
from src.chains.checks import asked_years, catalog_years, year_coverage_gap
from src.chains.nodes.analyze import (
    MISSING_ENTITY,
    MISSING_TIME_PERIOD,
    analyze_question,
)
from src.chains.state import DocSummary, GraphState, QueryPlan
from src.utils.llm import ChatResult, LLMParams

CATALOG: list[DocSummary] = [
    DocSummary(
        doc_id="311fa88413b7dae376ba0f19994f906ec0cc8a44b2e861e5fdfde542e858738b",
        filename="2023_Annual_Report.pdf",
        page_count=85,
        sections=(
            ("ISSUER PURCHASES OF EQUITY SECURITIES, DIVIDENDS, AND STOCK PERFORMANCE",),
            (
                "ISSUER PURCHASES OF EQUITY SECURITIES, DIVIDENDS, AND STOCK PERFORMANCE",
                "STOCK PERFORMANCE",
            ),
            ("NOTES TO FINANCIAL STATEMENTS",),
        ),
    )
]

COVERED = {2021, 2022, 2023}

DARK_CATALOG: list[DocSummary] = [
    DocSummary(doc_id="d2", filename="annual-report.pdf", sections=(("Revenue",),), page_count=4)
]


class PlannerStub:

    def __init__(self, parsed: dict[str, Any]) -> None:
        self.parsed = parsed
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
            parsed=self.parsed,
            prompt_tokens=8,
            completion_tokens=2,
            latency_ms=1.0,
            model="stub-model",
        )


@pytest.fixture
def planner(monkeypatch: pytest.MonkeyPatch) -> Any:

    holder: dict[str, PlannerStub] = {}

    def factory(params: LLMParams | None = None) -> PlannerStub:
        return holder["stub"]

    def install(parsed: dict[str, Any] | None = None) -> PlannerStub:
        holder["stub"] = PlannerStub(parsed or _simple_plan())
        return holder["stub"]

    monkeypatch.setattr("src.chains.nodes.analyze.build_llm", factory)
    return install


def _simple_plan() -> dict[str, Any]:

    return {
        "kind": "simple",
        "sub_queries": [],
        "required_sections": [],
        "fiscal_period": "2023",
        "missing_aspect": None,
    }


def _state(question: str, catalog: list[DocSummary] = CATALOG) -> GraphState:
    return GraphState(question=question, doc_catalog=catalog)


def _plan_of(out: dict[str, Any]) -> QueryPlan:
    plan = out.get("query_plan")
    assert isinstance(plan, QueryPlan)
    return plan


def test_the_catalog_shows_the_years_an_annual_report_spans() -> None:

    text = "2023_Annual_Report.pdf ISSUER PURCHASES OF EQUITY SECURITIES > STOCK PERFORMANCE"
    assert catalog_years(text) == COVERED


@pytest.mark.parametrize(
    ("question", "years"),
    [
        ("What was revenue in FY2019?", [2019]),
        ("What was revenue in fiscal year 2019?", [2019]),
        ("What was revenue in 2019?", [2019]),
        ("What was revenue in FY2023?", [2023]),
        ("Compare revenue between FY2023 and FY2022.", [2022, 2023]),
        ("And the margin trend?", []),
        ("Revenue of $211,915 million and $ 168,088 million", []),
    ],
)
def test_a_question_names_only_the_years_it_prints(question: str, years: list[int]) -> None:

    assert sorted(asked_years(question)) == years


def test_the_gate_fires_on_a_year_the_documents_do_not_cover() -> None:
    assert year_coverage_gap(COVERED, "What was revenue in FY2019?") is True


def test_the_gate_leaves_a_covered_year_alone() -> None:

    for year in ("FY2021", "FY2022", "FY2023", "fiscal year 2023", "2023"):
        assert year_coverage_gap(COVERED, f"What was revenue in {year}?") is False


def test_the_gate_stays_out_of_the_way_when_the_question_names_no_period() -> None:

    assert year_coverage_gap(COVERED, "What was revenue?") is False
    assert year_coverage_gap(COVERED, "And the margin trend?") is False


def test_the_gate_stays_out_of_the_way_of_a_comparison() -> None:

    assert year_coverage_gap(COVERED, "Compare revenue between FY2019 and FY2023.") is False
    assert year_coverage_gap(COVERED, "Compare revenue between FY2019 and FY2020.") is False


def test_the_gate_stays_out_of_the_way_of_an_entity_gap() -> None:

    assert year_coverage_gap(COVERED, "What was Amazon's revenue in 2023?") is False


def test_the_gate_cannot_fire_against_a_catalog_it_cannot_read() -> None:

    assert year_coverage_gap(set(), "What was revenue in FY2019?") is False


def test_a_year_the_documents_do_not_cover_is_refused_without_a_model_call(
    planner: Any,
) -> None:

    stub = planner()
    out = analyze_question(_state("What was revenue in FY2019?"))
    plan = _plan_of(out)

    assert plan.kind == "out_of_scope"
    assert plan.missing_aspect == MISSING_TIME_PERIOD
    assert plan.fiscal_period == "2019"
    assert plan.sub_queries == []
    assert plan.required_sections == []
    assert stub.calls == [], "a gated question must never reach the planner"


def test_the_gated_plan_is_the_plan_refuse_already_knows_how_to_render(planner: Any) -> None:

    planner()
    plan = _plan_of(analyze_question(_state("What was revenue in FY2019?")))

    assert plan.missing_aspect == "time_period"
    assert plan.is_out_of_scope


def test_a_gated_question_returns_only_the_plan(planner: Any) -> None:

    planner()
    out = analyze_question(_state("What was revenue in FY2019?"))

    assert set(out) == {"query_plan"}


def test_a_gated_question_is_not_marked_as_a_degraded_planner(planner: Any) -> None:

    planner()
    out = analyze_question(_state("What was revenue in FY2019?"))

    assert "planner_degraded" not in out


@pytest.mark.parametrize(
    "question",
    [
        "What was revenue in FY2019?",
        "What was revenue in fiscal year 2019?",
        "What was revenue in 2019?",
    ],
)
def test_every_printed_form_of_the_asked_year_is_refused(planner: Any, question: str) -> None:

    stub = planner()
    plan = _plan_of(analyze_question(_state(question)))

    assert plan.kind == "out_of_scope"
    assert stub.calls == []


def test_a_question_in_scope_still_reaches_the_planner(planner: Any) -> None:

    stub = planner(_simple_plan())
    plan = _plan_of(analyze_question(_state("What was Microsoft's revenue in FY2023?")))

    assert plan.kind == "simple"
    assert len(stub.calls) == 1


def test_a_comparison_still_reaches_the_planner(planner: Any) -> None:

    stub = planner(
        {
            "kind": "multi_part",
            "sub_queries": ["What was revenue in FY2023?", "What was revenue in FY2022?"],
            "required_sections": [],
            "fiscal_period": None,
            "missing_aspect": None,
        }
    )
    plan = _plan_of(analyze_question(_state("Compare revenue between FY2023 and FY2022.")))

    assert plan.kind == "multi_part"
    assert plan.sub_queries == ["What was revenue in FY2023?", "What was revenue in FY2022?"]
    assert len(stub.calls) == 1


def test_an_entity_gap_still_reaches_the_planner(planner: Any) -> None:

    stub = planner(
        {
            "kind": "out_of_scope",
            "sub_queries": [],
            "required_sections": [],
            "fiscal_period": "2023",
            "missing_aspect": MISSING_ENTITY,
        }
    )
    plan = _plan_of(analyze_question(_state("What was Amazon's revenue in 2023?")))

    assert plan.kind == "out_of_scope"
    assert plan.missing_aspect == MISSING_ENTITY
    assert len(stub.calls) == 1


def test_a_follow_up_with_no_period_still_reaches_the_planner(planner: Any) -> None:

    stub = planner()
    plan = _plan_of(analyze_question(_state("And the margin trend?")))

    assert plan.kind == "simple"
    assert len(stub.calls) == 1


def test_a_catalog_with_no_readable_year_defers_to_the_planner(planner: Any) -> None:

    stub = planner()
    plan = _plan_of(analyze_question(_state("What was revenue in FY2019?", DARK_CATALOG)))

    assert plan.kind == "simple"
    assert len(stub.calls) == 1


def test_an_empty_catalog_still_reaches_the_planner(planner: Any) -> None:

    stub = planner()
    plan = _plan_of(analyze_question(_state("What was revenue in FY2019?", [])))

    assert plan.kind == "simple"
    assert len(stub.calls) == 1


def test_coverage_follows_the_catalog_rather_than_a_hardcoded_corpus(planner: Any) -> None:

    planner()
    older = [DocSummary(doc_id="d3", filename="2021_Annual_Report.pdf", sections=(), page_count=90)]

    covered = _plan_of(analyze_question(_state("What was revenue in FY2020?", older)))
    assert covered.kind == "simple"

    gap = _plan_of(analyze_question(_state("What was revenue in FY2018?", older)))
    assert gap.kind == "out_of_scope"
    assert gap.fiscal_period == "2018"

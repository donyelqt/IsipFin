"""``analyze_question``'s deterministic fiscal-year gate (ADR-010, applied to routing).

The live keyed golden run that motivated this: ``What was revenue in FY2019?`` was answered
with three years of real revenue instead of refused. The corpus covers FY2021-FY2023 — the
corpus's own mock states it, and ``refuse`` has always been able to print it — yet the gate
was reachable **only** by paying for a planner POST and hoping the planner noticed. Three
identical temperature-0 calls to the same provider returned three different answers (a
``simple`` plan, a ``simple`` plan with an empty aspect, and a raised ``LLMError``), so the
refusals correctness of the whole run rested on provider-side luck.

``refuse`` already knew the coverage. The gate moves that same knowledge in front of the
planner, so the one class of question the catalog settles outright no longer depends on a
model call at all. This is the routing-level sibling of the numeric pre-pass
``groundedness_check`` runs on figures, and it is asserted here the same way — as pure
functions over real text, with the gateway stubbed and no network.

Everything is hermetic: the gateway is patched out (so a *non-gated* path still has to work
through the planner), the catalog is a hand-built ``DocSummary``, and the symbolic half runs
for real.
"""

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

# --- the catalog the golden set runs against ----------------------------------------
#
# One annual report whose own year is 2023. The section paths are real
# ``data/index/docs.jsonl`` entries, so the coverage this test asserts is the coverage the
# real corpus shows, not a fixture's guess.
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

#: What the catalog shows and what an annual report's statements therefore span.
COVERED = {2021, 2022, 2023}

#: A catalog the gate cannot read anything out of: no filename year, no section year.
DARK_CATALOG: list[DocSummary] = [
    DocSummary(doc_id="d2", filename="annual-report.pdf", sections=(("Revenue",),), page_count=4)
]


# --- the stub gateway ---------------------------------------------------------------


class PlannerStub:
    """A :class:`~src.utils.llm.ChatClient` recording whether it was consulted at all.

    The invariant the gate exists to protect is *"this question never reaches the model"*, so
    a stub that counts calls is the assertion. A planner that answers ``simple`` to
    everything is deliberate: if the gate fails to fire, the plan that comes back is a
    retrieval plan and the failure is visible rather than hidden by a cooperating stub.
    """

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
    """Patch the one seam ``build_llm`` is (ADR-005) and hand back the installed stub.

    The factory records every call, so a test that expects no model call asserts
    ``stub.calls == []`` rather than trusting the gate silently.
    """

    holder: dict[str, PlannerStub] = {}

    def factory(params: LLMParams | None = None) -> PlannerStub:
        return holder["stub"]

    def install(parsed: dict[str, Any] | None = None) -> PlannerStub:
        holder["stub"] = PlannerStub(parsed or _simple_plan())
        return holder["stub"]

    monkeypatch.setattr("src.chains.nodes.analyze.build_llm", factory)
    return install


def _simple_plan() -> dict[str, Any]:
    """A planner reply that routes to retrieval, so a missing gate is loud."""

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


# --- 1. what the catalog and a question each show ------------------------------------


def test_the_catalog_shows_the_years_an_annual_report_spans() -> None:
    """``"2023_Annual_Report.pdf"`` stands for FY2021-FY2023, the span its statements carry."""

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
    """Figures and pronouns are not periods, so the gate is never handed a phantom year."""

    assert sorted(asked_years(question)) == years


# --- 2. the gate decides the year the catalog cannot cover ----------------------------


def test_the_gate_fires_on_a_year_the_documents_do_not_cover() -> None:
    assert year_coverage_gap(COVERED, "What was revenue in FY2019?") is True


def test_the_gate_leaves_a_covered_year_alone() -> None:
    """Every year the corpus answers still routes to the planner — the gate only ever adds."""

    for year in ("FY2021", "FY2022", "FY2023", "fiscal year 2023", "2023"):
        assert year_coverage_gap(COVERED, f"What was revenue in {year}?") is False


def test_the_gate_stays_out_of_the_way_when_the_question_names_no_period() -> None:
    """A period-less question has nothing to compare, so the planner's judgement stands."""

    assert year_coverage_gap(COVERED, "What was revenue?") is False
    assert year_coverage_gap(COVERED, "And the margin trend?") is False


def test_the_gate_stays_out_of_the_way_of_a_comparison() -> None:
    """Several periods asked is a ``multi_part`` call, and one uncovered year must not
    refuse a comparison the documents can honestly answer."""

    assert year_coverage_gap(COVERED, "Compare revenue between FY2019 and FY2023.") is False
    assert year_coverage_gap(COVERED, "Compare revenue between FY2019 and FY2020.") is False


def test_the_gate_stays_out_of_the_way_of_an_entity_gap() -> None:
    """The catalog holds no entity list, so ``Amazon`` stays the planner's refusal to make."""

    assert year_coverage_gap(COVERED, "What was Amazon's revenue in 2023?") is False


def test_the_gate_cannot_fire_against_a_catalog_it_cannot_read() -> None:
    """No coverage shown means nothing to compare: the planner's judgement stands, and
    ``refuse`` renders its own "cannot be determined" copy rather than a fabricated range."""

    assert year_coverage_gap(set(), "What was revenue in FY2019?") is False


# --- 3. the node routes a gated question without consulting the planner ----------------


def test_a_year_the_documents_do_not_cover_is_refused_without_a_model_call(
    planner: Any,
) -> None:
    """The regression this file exists for: FY2019 must refuse, and must cost no POST."""

    stub = planner()
    out = analyze_question(_state("What was revenue in FY2019?"))
    plan = _plan_of(out)

    assert plan.kind == "out_of_scope"
    assert plan.missing_aspect == MISSING_TIME_PERIOD
    # The asked period is recorded so the refusal copy can name what was asked for.
    assert plan.fiscal_period == "2019"
    assert plan.sub_queries == []
    assert plan.required_sections == []
    assert stub.calls == [], "a gated question must never reach the planner"


def test_the_gated_plan_is_the_plan_refuse_already_knows_how_to_render(planner: Any) -> None:
    """``missing_aspect=time_period`` is what P1's schema and ``refuse``'s copy expect, so a
    symbolic refusal and a planner's refusal reach the reader as one indistinguishable outcome."""

    planner()
    plan = _plan_of(analyze_question(_state("What was revenue in FY2019?")))

    assert plan.missing_aspect == "time_period"
    assert plan.is_out_of_scope


def test_a_gated_question_returns_only_the_plan(planner: Any) -> None:
    """No gateway call was made, so no gateway accounting is reported."""

    planner()
    out = analyze_question(_state("What was revenue in FY2019?"))

    assert set(out) == {"query_plan"}


def test_a_gated_question_is_not_marked_as_a_degraded_planner(planner: Any) -> None:
    """The gate is a reasoned outcome, not P1's failure handling: the debug panel must be
    able to tell a symbolic refusal from a planner that could not answer."""

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
    """The corpus writes periods three ways; the gate reads all of them."""

    stub = planner()
    plan = _plan_of(analyze_question(_state(question)))

    assert plan.kind == "out_of_scope"
    assert stub.calls == []


# --- 4. what still goes to the planner ------------------------------------------------


def test_a_question_in_scope_still_reaches_the_planner(planner: Any) -> None:
    """The gate is narrow precisely so it does not take the planner's job away."""

    stub = planner(_simple_plan())
    plan = _plan_of(analyze_question(_state("What was Microsoft's revenue in FY2023?")))

    assert plan.kind == "simple"
    assert len(stub.calls) == 1


def test_a_comparison_still_reaches_the_planner(planner: Any) -> None:
    """``multi_part`` is a judgement about the question's shape, which the catalog cannot make."""

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
    """``Amazon`` is the one refusal the catalog cannot settle: there is no entity list to
    read, so the planner's comparison of the question against the catalog stays the answer."""

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
    """``"And the margin trend?"`` inherits its period from history, not from its own words —
    the gate has nothing to read here and must not invent a gap."""

    stub = planner()
    plan = _plan_of(analyze_question(_state("And the margin trend?")))

    assert plan.kind == "simple"
    assert len(stub.calls) == 1


def test_a_catalog_with_no_readable_year_defers_to_the_planner(planner: Any) -> None:
    """Coverage that cannot be determined is not coverage that can be violated."""

    stub = planner()
    plan = _plan_of(analyze_question(_state("What was revenue in FY2019?", DARK_CATALOG)))

    assert plan.kind == "simple"
    assert len(stub.calls) == 1


def test_an_empty_catalog_still_reaches_the_planner(planner: Any) -> None:
    """The empty-catalog refusal is ``refuse``'s own ``no_documents`` reason, not the gate's."""

    stub = planner()
    plan = _plan_of(analyze_question(_state("What was revenue in FY2019?", [])))

    assert plan.kind == "simple"
    assert len(stub.calls) == 1


def test_coverage_follows_the_catalog_rather_than_a_hardcoded_corpus(planner: Any) -> None:
    """A 2021 report's statements span FY2019-FY2021, so FY2020 is answerable there even
    though the FY2023 corpus would refuse it, and FY2018 is a gap under both."""

    planner()
    older = [DocSummary(doc_id="d3", filename="2021_Annual_Report.pdf", sections=(), page_count=90)]

    covered = _plan_of(analyze_question(_state("What was revenue in FY2020?", older)))
    assert covered.kind == "simple"

    gap = _plan_of(analyze_question(_state("What was revenue in FY2018?", older)))
    assert gap.kind == "out_of_scope"
    assert gap.fiscal_period == "2018"

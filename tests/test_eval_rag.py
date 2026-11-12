from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> Any:

    path = _REPO_ROOT / "scripts" / "eval_rag.py"
    spec = importlib.util.spec_from_file_location("eval_rag_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_module()


def _case(
    case_id: str,
    expect: dict[str, Any] | None = None,
    follow_up_of: str | None = None,
) -> Any:

    return harness.Case(
        case_id=case_id,
        question=f"question for {case_id}",
        expect=expect or {"kind": "figure"},
        tags=(),
        follow_up_of=follow_up_of,
        line_no=1,
    )


def _canary_case(case_id: str) -> Any:

    return _case(
        case_id,
        {
            "kind": "figure",
            "asked_year": "FY2023",
            "adjacent_year_figure": "$198,270",
            "adjacent_year_label": "FY2022",
        },
    )


def _result(case_id: str, case: Any = None, **scores: Any) -> Any:

    return harness.CaseResult(case=case or _case(case_id), **scores)


@pytest.mark.parametrize(
    "metric",
    ["retrieval_hit_rate@5", "citation_presence", "figure_match", "refusal_correctness"],
)
def test_a_rate_meeting_its_floor_passes(metric: str) -> None:

    field = harness._FIELD[metric]
    results = [_result(f"c{index}", **{field: True}) for index in range(7)]
    verdicts = harness._verdicts(results)

    assert verdicts[metric] == "PASS"
    assert harness._exit_code(results, verdicts) == (harness.EXIT_PASS, "")


def test_a_one_point_zero_rate_against_a_zero_point_nine_floor_prints_pass() -> None:

    results = [_result(f"c{index}", retrieval_hit=True) for index in range(7)]
    verdicts = harness._verdicts(results)

    assert harness._metric_value("retrieval_hit_rate@5", results) == "7/7 (1.00)"
    assert harness.METRICS["retrieval_hit_rate@5"] == pytest.approx(0.9)
    assert verdicts["retrieval_hit_rate@5"] == "PASS"


def test_a_rate_meeting_the_floor_exactly_passes() -> None:

    results = [
        *(_result(f"h{index}", retrieval_hit=True) for index in range(18)),
        *(_result(f"m{index}", retrieval_hit=False) for index in range(2)),
    ]
    verdicts = harness._verdicts(results)

    assert harness._metric_value("retrieval_hit_rate@5", results) == "18/20 (0.90)"
    assert verdicts["retrieval_hit_rate@5"] == "PASS"
    below = [
        *(_result(f"h{index}", retrieval_hit=True) for index in range(17)),
        *(_result(f"m{index}", retrieval_hit=False) for index in range(3)),
    ]

    assert harness._metric_value("retrieval_hit_rate@5", below) == "17/20 (0.85)"
    assert harness._verdicts(below)["retrieval_hit_rate@5"] == "FAIL"


def test_a_rate_below_its_floor_fails_and_reaches_the_exit_code() -> None:

    results = [
        *(_result(f"h{index}", figure_ok=True) for index in range(5)),
        *(_result(f"m{index}", figure_ok=False) for index in range(2)),
    ]
    verdicts = harness._verdicts(results)

    assert harness._metric_value("figure_match", results) == "5/7 (0.71)"
    assert harness.METRICS["figure_match"] == pytest.approx(0.95)
    assert verdicts["figure_match"] == "FAIL"

    code, reason = harness._exit_code(results, verdicts)
    assert code == harness.EXIT_REGRESSION
    assert "figure_match" in reason


def test_one_failed_case_out_of_many_fails_the_metric() -> None:

    results = [
        *(_result(f"c{index}", citation_ok=True) for index in range(9)),
        _result("c9", citation_ok=False),
    ]
    verdicts = harness._verdicts(results)

    assert harness._metric_value("citation_presence", results) == "9/10 (0.90)"
    assert harness.METRICS["citation_presence"] == pytest.approx(1.0)
    assert verdicts["citation_presence"] == "FAIL"


def test_the_canary_passes_with_no_contamination() -> None:

    clean = dict(wrong_year=False, wrong_year_state=harness.WRONG_YEAR_CLEAN)
    results = [
        _result("canary0", case=_canary_case("canary0"), **clean),
        _result("canary1", case=_canary_case("canary1"), **clean),
    ]
    verdicts = harness._verdicts(results)

    assert verdicts["wrong_year_contamination"] == "PASS"
    assert harness._exit_code(results, verdicts) == (harness.EXIT_PASS, "")


def test_the_canary_fails_on_one_contaminated_case() -> None:

    clean = dict(wrong_year=False, wrong_year_state=harness.WRONG_YEAR_CLEAN)
    dirty = dict(wrong_year=True, wrong_year_state=harness.WRONG_YEAR_CONTAMINATED)
    results = [
        _result("canary0", case=_canary_case("canary0"), **clean),
        _result("canary1", case=_canary_case("canary1"), **dirty),
    ]
    verdicts = harness._verdicts(results)

    assert harness.METRICS["wrong_year_contamination"] == pytest.approx(0.0)
    assert verdicts["wrong_year_contamination"] == "FAIL"

    code, reason = harness._exit_code(results, verdicts)
    assert code == harness.EXIT_REGRESSION
    assert "wrong_year_contamination" in reason


def test_wrong_year_false_on_a_clean_answer_means_it_is_not_contaminated() -> None:

    case = _canary_case("clean-fy2023")
    result = harness.CaseResult(
        case=case,
        answer_md="Microsoft's FY2023 revenue was $211,915 million [1], up from "
        "$198,270 million in FY2022 [2].",
    )
    harness._score_case(case, result, harness._full_scopes())

    assert result.wrong_year is False
    assert result.wrong_year_state == harness.WRONG_YEAR_CLEAN
    assert result.wrong_year_detail == ""

    dirty = harness.CaseResult(
        case=case, answer_md="Microsoft's FY2023 revenue was $198,270 million [1]."
    )
    harness._score_case(case, dirty, harness._full_scopes())

    assert dirty.wrong_year is True
    assert dirty.wrong_year_state == harness.WRONG_YEAR_CONTAMINATED
    assert "$198,270" in dirty.wrong_year_detail


def test_a_clean_case_costs_the_canary_nothing() -> None:

    results = [
        _result(
            f"canary{index}",
            case=_canary_case(f"canary{index}"),
            wrong_year=False,
            wrong_year_state=harness.WRONG_YEAR_CLEAN,
        )
        for index in range(3)
    ]

    assert harness._metric_value("wrong_year_contamination", results) == "0/3+0"
    assert harness._verdicts(results)["wrong_year_contamination"] == "PASS"


_THREE_YEAR_TABLE = (
    "| Metric | FY2023 | FY2022 | FY2021 |\n"
    "|---|---|---|---|\n"
    "| Revenue | $211,915 | $198,270 | $168,088 |"
)


def _canary_verdict(answer_md: str, figure: str = "$198,270", year: str = "FY2022") -> bool:

    case = _canary_case("canary-read")
    result = harness.CaseResult(case=case, answer_md=answer_md)
    harness._score_case(case, result, harness._full_scopes())
    return result.wrong_year


def test_a_correct_three_year_table_is_clean() -> None:

    assert not _canary_verdict(_THREE_YEAR_TABLE)

    for figure in ("$211,915", "$198,270", "$168,088"):
        assert not _canary_verdict(_THREE_YEAR_TABLE, figure=figure)

    two_metrics = (
        "| Metric | FY2023 | FY2022 |\n"
        "|---|---|---|\n"
        "| Revenue | $211,915 | $198,270 |\n"
        "| Operating income | $88,523 | $83,383 |"
    )
    assert not _canary_verdict(two_metrics)
    assert not _canary_verdict(two_metrics, figure="$83,383")


def test_prose_naming_the_asked_year_but_quoting_an_adjacent_figure_is_contaminated() -> None:

    assert _canary_verdict("Revenue was $198,270 million in FY2023 [1].")
    assert _canary_verdict(
        "Revenue was $198,270 million in FY2023 [1]; operating income" " was $88,523 million [2]."
    )
    assert not _canary_verdict(
        "Revenue was $211,915 million in FY2023, up from $198,270 million in FY2022 [1]."
    )


def test_a_table_header_naming_fy2023_with_the_fy2022_figure_under_it_is_contaminated() -> None:

    transposed = "| Metric | FY2023 | FY2022 |\n|---|---|---|\n| Revenue | $198,270 | $211,915 |"
    assert _canary_verdict(transposed)

    assert not _canary_verdict(
        "| Metric | FY2023 | FY2022 |\n|---|---|---|\n| FY2023 revenue | $211,915 |" " $198,270 |"
    )


def test_a_table_cell_is_judged_against_its_own_column_not_the_merged_row() -> None:

    assertions = harness._assertions(_THREE_YEAR_TABLE)

    assert ("$198,270", "Revenue FY2022") in assertions
    assert ("$211,915", "Revenue FY2023") in assertions
    assert ("$168,088", "Revenue FY2021") in assertions
    for text, _label in assertions:
        assert "|" not in text
    assert len(assertions) == 3


def test_a_lone_table_row_with_no_header_is_read_as_the_row_text_it_is() -> None:

    assert _canary_verdict("| Revenue | $211,915 | $198,270 |")
    assert _canary_verdict("| Revenue | FY2023 | $211,915 | $198,270 |")
    assert _canary_verdict("Revenue | $211,915 | $198,270")


def test_an_alignment_row_and_a_headerless_row_carry_no_column_attribution() -> None:

    assert not harness._is_alignment_row(["Metric", "FY2023"])
    assert harness._is_alignment_row(["---", ":-:", " -- "])
    assert not harness._is_alignment_row(["", "FY2023"])

    assert harness._assertions("| $211,915 | $198,270 |") == [
        ("$211,915", ""),
        ("$198,270", ""),
    ]


def test_an_unmeasured_metric_is_na_and_never_fails_the_run() -> None:

    results = [_result("c0", retrieval_hit=None), _result("c1", retrieval_hit=None)]
    verdicts = harness._verdicts(results)

    assert verdicts["retrieval_hit_rate@5"] == "NA"
    assert harness._metric_value("retrieval_hit_rate@5", results) == "NA"
    assert harness._exit_code(results, verdicts) == (harness.EXIT_PASS, "")


def test_na_is_not_a_fail_even_when_the_comparison_would_have_failed() -> None:

    for metric in harness._FIELD:
        results = [_result("c0", **{harness._FIELD[metric]: None})]
        verdicts = harness._verdicts(results)

        assert verdicts[metric] == "NA", metric
        code, _ = harness._exit_code(results, verdicts)
        assert code == harness.EXIT_PASS, metric


def test_the_canary_is_na_when_no_case_carries_a_checkable_answer() -> None:

    results = [
        _result(
            "canary0",
            case=_canary_case("canary0"),
            wrong_year_state=harness.NOT_CHECKABLE,
        ),
        _result("plain", wrong_year_state=""),
    ]
    verdicts = harness._verdicts(results)

    assert verdicts["wrong_year_contamination"] == "NA"
    assert harness._exit_code(results, verdicts) == (harness.EXIT_PASS, "")


def test_the_summary_prints_the_verdicts_the_exit_code_reads(
    capsys: pytest.CaptureFixture[str],
) -> None:

    results = [
        _result("c0", retrieval_hit=True),
        _result("c1", retrieval_hit=True),
        _result("c2", retrieval_hit=True),
    ]
    verdicts = harness._verdicts(results)
    ctx = harness._RunContext(
        docs=[],
        mock_mode=True,
        index_dir=Path("unused"),
        scopes=harness._mock_scopes(),
    )
    report = harness.DatasetReport(cases=[], malformed=[], duplicate_ids=[])
    code = harness._print_summary(results, verdicts, ctx, report, [])

    printed = capsys.readouterr().out
    assert code == harness.EXIT_PASS
    assert "retrieval_hit_rate@5" in printed
    assert ">= 0.9" in printed
    assert "<= 0" in printed
    assert "exit code 0" in printed


def test_the_observed_value_is_printed_for_an_na_metric() -> None:

    results = [_result("c0", retrieval_observed=True), _result("c1", retrieval_observed=False)]

    assert harness._observed_value("retrieval_hit_rate@5", results) == "1/2 (0.50)"


def test_the_documented_metric_table_is_what_the_harness_uses() -> None:

    assert dict(harness.METRICS) == {
        "retrieval_hit_rate@5": 0.9,
        "citation_presence": 1.0,
        "figure_match": 0.95,
        "wrong_year_contamination": 0.0,
        "refusal_correctness": 1.0,
    }
    assert harness.METRICS["wrong_year_contamination"] == pytest.approx(0.0)
    assert "wrong_year_contamination" in harness.HARD_METRICS


def test_a_missing_dataset_is_a_harness_error_not_a_crash(tmp_path: Path) -> None:

    with pytest.raises(harness.DatasetError):
        harness.parse_dataset(tmp_path / "absent.jsonl")


def test_loading_the_module_left_no_index_or_env_state_behind() -> None:

    graph_modules = [name for name in sys.modules if name.startswith("src.chains.graph")]
    assert not graph_modules, "the harness module body must not import the graph at import time"


def test_no_metric_field_reads_a_missing_attribute() -> None:

    declared = {field.name for field in harness.CaseResult.__dataclass_fields__.values()}
    for metric, field in harness._FIELD.items():
        assert field in declared, f"{metric} reads undeclared field {field!r}"
    for metric, field in harness._OBSERVED_FIELD.items():
        assert field in declared, f"{metric} observes undeclared field {field!r}"


RATE_LIMITED_REASON = (
    "RateLimited: model rate limited the request (Error code: 429 - "
    "{'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` on tokens "
    "per minute (TPM): Limit 8000, Used 6034, Requested 2511.', 'type': 'tokens'}})"
)


@pytest.mark.parametrize(
    "reason",
    [
        RATE_LIMITED_REASON,
        "Auth: chat needs a key: none of LLM_API_KEY, OPENAI_API_KEY or GROQ_API_KEY is set",
        "ContextLength: the prompt is too long for the model's context window",
        "Timeout: the provider did not answer within the timeout",
    ],
)
def test_a_planner_that_could_not_run_is_not_a_semantic_verdict(reason: str) -> None:

    assert harness._infrastructure_degradation(reason) is True
    assert (
        harness._infrastructure_degradation(
            "SchemaViolationError: model response did not match the requested schema"
        )
        is False
    )
    assert harness._infrastructure_degradation("") is False
    assert harness._infrastructure_degradation("   ") is False


def test_a_rate_limited_case_is_not_scored_as_a_refusal() -> None:

    case = _case("degraded-figure", {"kind": "figure", "must_appear": ["$88,523"]})
    result = harness.CaseResult(
        case=case,
        answer_md="**Refused** — I can't answer that: the question is outside the scope of "
        "the documents in this session.",
        refusal_reason="unspecified",
        plan_kind="out_of_scope",
        route="refuse",
        degraded_reason=RATE_LIMITED_REASON,
    )

    harness._score_case(case, result, harness._full_scopes())

    assert result.infrastructure_degraded is True
    assert result.refusal_ok is None, "the answer to 'was this refusal right?' is unknown"
    assert result.citation_ok is None, "no retrieval ran, so there was no citation to score"
    assert result.figure_ok is None, "no answer was produced, so there was no figure to match"
    assert result.missing_figures == ["$88,523"], "still observed: the figure is absent"
    assert result.refusal_observed is False, "what it saw is recorded, not hidden"
    assert any("could not run" in note for note in result.notes)


def test_the_same_case_without_the_degradation_signal_still_scores() -> None:

    case = _case("real-refusal", {"kind": "figure", "must_appear": ["$88,523"]})
    result = harness.CaseResult(
        case=case,
        answer_md="**Refused** — outside the scope of the documents in this session.",
        refusal_reason="unspecified",
        plan_kind="out_of_scope",
        degraded_reason="",
    )

    harness._score_case(case, result, harness._full_scopes())

    assert result.refusal_ok is False
    assert result.figure_ok is False
    assert result.citation_ok is False
    assert result.infrastructure_degraded is False


def test_a_degraded_refusal_case_keeps_its_real_missing_aspect_check() -> None:

    case = _case("degraded-refusal", {"kind": "refusal", "missing_aspect": "entity"})
    result = harness.CaseResult(
        case=case,
        answer_md="**Refused** — outside the scope of the documents in this session.",
        refusal_reason="unspecified",
        plan_kind="out_of_scope",
        degraded_reason=RATE_LIMITED_REASON,
    )

    harness._score_case(case, result, harness._full_scopes())

    assert result.refusal_ok is None
    assert result.refusal_observed is False, "'unspecified' where 'entity' was expected"
    assert result.refusal_detail.startswith("refused, but missing_aspect")
    assert result.citation_ok is None, "a refusal is never expected to cite"


def test_the_golden_set_shape_does_not_read_a_degradation_as_a_failure() -> None:

    figures = {
        "q01": "$211,915",
        "q02": "$88,523",
        "q03": "$211,915",
        "q04": "$72,361",
        "q05": "Intelligent Cloud",
        "q06": "",
        "q07": "$88,523",
        "q08": "Intelligent Cloud",
    }
    results = []
    for case_id, must in figures.items():
        case = _case(case_id, {"kind": "figure", "must_appear": [must] if must else []})
        results.append(
            harness.CaseResult(
                case=case,
                answer_md="**Refused** — outside the scope of the documents.",
                refusal_reason="unspecified",
                plan_kind="out_of_scope",
                degraded_reason=RATE_LIMITED_REASON,
            )
        )
    gated = _case("q09", {"kind": "refusal", "missing_aspect": "time_period"})
    results.append(
        harness.CaseResult(
            case=gated,
            answer_md="**Refused** — the documents cover fiscal years FY2021-FY2023.",
            refusal_reason="time_period",
            plan_kind="out_of_scope",
            degraded_reason="",
        )
    )
    degraded_entity = _case("q10", {"kind": "refusal", "missing_aspect": "entity"})
    results.append(
        harness.CaseResult(
            case=degraded_entity,
            answer_md="**Refused** — outside the scope of the documents.",
            refusal_reason="unspecified",
            plan_kind="out_of_scope",
            degraded_reason=RATE_LIMITED_REASON,
        )
    )

    for result in results:
        harness._score_case(result.case, result, harness._full_scopes())
    verdicts = harness._verdicts(results)

    assert not [r for r in results if r.failed], "no case failed a check"
    assert verdicts["refusal_correctness"] == "PASS", "q09 is the only measurable refusal"
    assert harness._metric_value("refusal_correctness", results) == "1/1 (1.00)"
    for metric in ("citation_presence", "figure_match", "retrieval_hit_rate@5"):
        assert verdicts[metric] == "NA", metric
    code, reason = harness._exit_code(results, verdicts)
    assert code == harness.EXIT_REGRESSION
    assert "planner could not run" in reason
    assert "q01" in reason


def test_the_summary_names_the_unmeasured_cases_apart_from_the_failing_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:

    failing = _result("bad", refusal_ok=False)
    unmeasured = _result("gated-out", degraded_reason=RATE_LIMITED_REASON)
    results = [failing, unmeasured]
    ctx = harness._RunContext(
        docs=[],
        mock_mode=False,
        index_dir=Path("unused"),
        scopes=harness._full_scopes(),
    )
    report = harness.DatasetReport(cases=[], malformed=[], duplicate_ids=[])

    code = harness._print_summary(results, harness._verdicts(results), ctx, report, [])

    printed = capsys.readouterr().out
    assert "cases failing        : bad" in printed
    assert "cases not measured   : gated-out (planner could not run)" in printed
    assert code == harness.EXIT_REGRESSION


LOG_ANSWER_Q01 = (
    "Microsoft\u2019s revenue for fiscal year\u202f2023 was "
    "**$\u202f211,915\u202fmillion**\u202f[2]"
)
SPACED_ANSWER_Q01 = "Microsoft's revenue for fiscal year 2023 was **$ 211,915 million** [2]"
FY2022_FIGURE = "$198,270"


def test_the_logged_answer_matches_the_figure_it_prints() -> None:

    assert harness._contains_figure(LOG_ANSWER_Q01, "$211,915")
    assert harness._contains_figure(SPACED_ANSWER_Q01, "$211,915")


def test_the_figure_metric_scores_the_logged_answer_as_a_match() -> None:

    case = _case("q01", {"kind": "figure", "must_appear": ["$211,915"]})
    result = harness.CaseResult(case=case, answer_md=LOG_ANSWER_Q01)
    harness._score_case(case, result, harness._full_scopes())

    assert result.figure_ok is True
    assert result.missing_figures == []


def test_the_exact_answer_the_run_printed_reaches_the_pass_bar() -> None:

    results = [
        harness.CaseResult(
            case=_case("q01", {"kind": "figure", "must_appear": ["$211,915"]}),
            answer_md=LOG_ANSWER_Q01,
        ),
        harness.CaseResult(
            case=_case("q02", {"kind": "figure", "must_appear": ["$88,523"]}),
            answer_md="Operating income in fiscal year 2023 was $55,884 million [6].",
        ),
        harness.CaseResult(
            case=_case("q04", {"kind": "figure", "must_appear": ["$72,361"]}),
            answer_md="**Refused** — outside the scope of the documents.",
        ),
    ]
    for result in results:
        harness._score_case(result.case, result, harness._full_scopes())

    assert harness._metric_value("figure_match", results) == "1/3 (0.33)"
    assert harness._verdicts(results)["figure_match"] == "FAIL"
    assert harness._exit_code(results, harness._verdicts(results))[0] == harness.EXIT_REGRESSION


def test_an_absent_figure_is_still_missing() -> None:

    answer = "**Refused** — outside the scope of the documents in this session."
    case = _case("q01", {"kind": "figure", "must_appear": ["$211,915"]})
    result = harness.CaseResult(case=case, answer_md=answer)
    harness._score_case(case, result, harness._full_scopes())

    assert result.figure_ok is False
    assert result.missing_figures == ["$211,915"]
    assert not harness._contains_figure(answer, "$211,915")


def test_an_answer_printing_a_different_number_does_not_match() -> None:

    assert not harness._contains_figure("Revenue was $111,915 million in FY2023 [1].", "$211,915")
    assert not harness._contains_figure("Revenue was $211.915 thousand.", "$211,915")
    assert not harness._contains_figure("Revenue was $21,19,15 million.", "$211,915")
    assert not harness._contains_figure("Revenue was about $212 billion.", "$211,915")


def test_the_adjacent_year_figures_do_not_match() -> None:

    for answer in (
        "Revenue was $198,270 million in FY2023 [1].",
        "Operating income was $83,383 million in FY2023 [2].",
        "| Operating income | $ 83,383 |",
    ):
        assert not harness._contains_figure(answer, "$211,915"), answer
        assert not harness._contains_figure(answer, "$88,523"), answer


def test_the_adjacent_year_figure_still_matches_when_it_is_the_expected_one() -> None:

    assert harness._contains_figure(
        "Revenue fell from $211,915 to $ 198,270 million [1].", "$198,270"
    )


def test_a_longer_number_containing_the_expected_one_does_not_match() -> None:

    assert not harness._contains_figure(
        "The company earned $1,198,270 million in FY2023.", "$198,270"
    )
    assert not harness._contains_figure("The subtotal was $1198270.", "$211,915")


def test_the_spacing_emphasis_and_grouping_variants_all_match() -> None:

    expected = "$211,915"
    variants = (
        "$211,915",
        "$ 211,915",
        "$\u00a0211,915",
        "$\u202f211,915",
        "211,915",
        "211915",
        "**$ 211,915**",
        "*$211,915*",
        "\u20ac211,915",
        "Revenue was **$ 211,915 million** [1].",
        "| Revenue | $ 211,915 | $ 198,270 |",
        "Revenue grew to $ 211,915 million, up 7% [1].",
    )
    for variant in variants:
        assert harness._contains_figure(variant, expected), variant


def test_a_thin_space_between_the_digit_groups_is_the_same_figure() -> None:

    assert harness._contains_figure("Revenue was $ 211\u2009915 million.", "$211,915")
    assert harness._contains_figure("Revenue was 211\u2007915 million.", "211,915")


def test_a_figure_the_answer_never_prints_does_not_match() -> None:

    for absent in ("$212,915", "$211,915,000", "$1,119,150", "$198,270"):
        assert not harness._contains_figure(LOG_ANSWER_Q01, absent), absent


def test_a_non_numeric_expectation_keeps_its_literal_reading() -> None:

    assert harness._contains_figure(
        "Intelligent Cloud drove revenue with $87,907 million [3].", "Intelligent Cloud"
    )
    assert not harness._contains_figure("The answer concerns Amazon.", "Intelligent Cloud")
    assert not harness._contains_figure("Intelligent Cloud", "$87,907")


def test_an_empty_expectation_is_never_present() -> None:

    assert not harness._contains_figure(LOG_ANSWER_Q01, "")
    assert not harness._contains_figure(LOG_ANSWER_Q01, "   ")


def test_the_canary_verdict_is_unchanged_by_the_normalisation() -> None:

    assert _canary_verdict("Revenue was $198,270 million in FY2023 [1].")
    assert _canary_verdict("Revenue was $ 198,270\u202fmillion in FY2023 [1].")
    for figure in ("$211,915", "$198,270", "$168,088"):
        assert not _canary_verdict(_THREE_YEAR_TABLE, figure=figure)
    spaced = (
        "| Metric | FY2023 | FY2022 | FY2021 |\n"
        "|---|---|---|---|\n"
        "| Revenue | $ 211,915 | $ 198,270 | $ 168,088 |"
    )
    assert not _canary_verdict(spaced)
    assert not _canary_verdict(spaced, figure="$198,270")
    assert _canary_verdict("Revenue | $211,915 | $198,270")


def test_the_canary_still_names_the_adjacent_year_it_found() -> None:

    case = _canary_case("canary-report")
    result = harness.CaseResult(case=case, answer_md="Revenue was $198,270 million in FY2023 [1].")
    harness._score_case(case, result, harness._full_scopes())

    assert result.wrong_year is True
    assert FY2022_FIGURE in result.wrong_year_detail
    assert result.wrong_year_state == harness.WRONG_YEAR_CONTAMINATED


class _Clock:

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _pacer(
    clock: _Clock,
    slept: list[float],
    *,
    tpm_limit: int = 8000,
    tpd_limit: int = 200_000,
    allowed_today: int = 200_000,
    mock_mode: bool = False,
) -> Any:

    return harness._TokenPacer(
        tpm_limit=tpm_limit,
        tpd_limit=tpd_limit,
        allowed_today=allowed_today,
        mock_mode=mock_mode,
        now=clock,
        sleep=slept.append,
    )


def _cost(costs: Sequence[int]) -> Any:

    return {
        f"prior{index}": harness.CaseResult(
            case=_case(f"prior{index}"),
            tokens=(cost, 0),
        )
        for index, cost in enumerate(costs)
    }


def test_a_case_that_fits_the_window_dispatches_with_no_wait() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept)

    assert pacer.projected_cost({}) == 0
    assert pacer.gate(pacer.projected_cost({}), label="(before q01)") == 0.0
    assert slept == []

    prior = _cost([7_000])
    assert pacer.projected_cost(prior) == 7_000
    clock.advance(30.0)
    assert pacer.gate(pacer.projected_cost(prior), label="(before q02)") == 0.0
    assert slept == []


def test_a_case_whose_projection_fits_is_not_made_to_wait_on_a_budget_it_is_under() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept)
    prior = _cost([10_038])

    pacer.record(10_038)
    clock.advance(61.0)
    assert pacer.gate(pacer.projected_cost(prior), label="(before q02)") == 0.0
    assert slept == []


def test_a_case_that_would_overshoot_the_window_waits_before_dispatching() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept)

    pacer.record(6_000)
    clock.advance(10.0)

    waited = pacer.gate(6_500, label="(before q03)")
    assert waited > 0.0
    assert slept, "an overshooting projection must sleep before it dispatches"
    assert slept[0] == waited

    assert slept[0] == 50.0


def test_the_wait_it_takes_is_the_one_that_makes_room_not_one_more_second() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept)

    pacer.record(5_000)
    clock.advance(59.5)
    assert pacer.gate(4_000, label="(before q04)") == 1.0
    assert slept == [1.0]


def test_the_window_forgets_old_spend_so_a_long_run_cannot_deadlock() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept)

    pacer.record(7_500)
    clock.advance(5.0)
    assert pacer._minutes_window() == [(0.0, 7_500)], "the spend is inside the window"
    assert pacer.gate(5_000, label="(before q05)") > 0.0, "it overshoots while still in window"

    clock.advance(56.0)
    assert pacer._minutes_window() == [], "and the spend has rolled out of the window"
    assert pacer.gate(5_000, label="(before q06)") == 0.0, "so the next case fits"
    assert slept and slept[-1] > 0.0, "and the wait it did take was on the earlier gate"


def test_the_same_projection_overshoots_twice_before_the_window_clears() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept)

    pacer.record(7_000)
    clock.advance(1.0)
    assert pacer.gate(6_000, label="(before q07)") > 0.0
    first = len(slept)

    clock.advance(float(slept[-1]))
    pacer.record(7_500)
    assert pacer.gate(6_000, label="(before q08)") > 0.0, "the window is full again"
    assert len(slept) == first + 1


class _StubInvoke:

    def __init__(self, cost: int) -> None:
        self.cost = cost
        self.calls: list[str] = []

    def __call__(
        self,
        question: str,
        *,
        history: object = None,
        params: object = None,
        doc_catalog: object = None,
    ) -> Any:
        self.calls.append(question)

        class _Envelope:
            answer_md = "**Refused** — the documents cover FY2021-FY2023."
            plan = None
            refusal_reason = "time_period"
            grounded = True
            planner_degraded = False
            citations: ClassVar[list[Any]] = []
            retrieval_debug: ClassVar[list[Any]] = []
            node_trace: ClassVar[list[Any]] = []

        _Envelope.node_trace = [
            {"node": "analyze_question", "tokens_in": self.cost, "tokens_out": 31},
            {"node": "synthesize_answer", "tokens_in": self.cost, "tokens_out": 19},
        ]
        return _Envelope()


def test_the_trace_becomes_the_next_cases_projection_and_the_windows_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    import src.chains.graph as graph_module

    clock = _Clock()
    slept: list[float] = []
    stub = _StubInvoke(3_000)
    monkeypatch.setattr(graph_module, "invoke", stub, raising=True)

    ctx = harness._RunContext(
        docs=[],
        mock_mode=False,
        index_dir=Path("unused"),
        scopes=harness._full_scopes(),
    )
    pacer = _pacer(clock, slept)

    results: dict[str, Any] = {}
    first = harness._run_case(_case("q02"), ctx, [], results, pacer=pacer)
    assert first.tokens == (6_000, 50), "the envelope's own counts land in the result"
    assert first.token_cost == 6_050
    results[first.case.case_id] = first
    assert pacer._minutes_window() == [(0.0, 6_050)]
    assert pacer.projected_cost(results) == 6_050, "the projection is the measured cost"

    clock.advance(1.0)
    second = harness._run_case(_case("q03"), ctx, [], results, pacer=pacer)

    assert second.tokens == (6_000, 50)
    assert len(slept) == 1, "the second case waited before it dispatched"
    assert slept[0] == 59.0, "and the wait cleared the window the first case filled"
    assert stub.calls == [first.case.question, second.case.question]
    assert pacer.waits == [59.0]


def test_an_unpaced_run_is_unchanged_by_the_pacer_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    import src.chains.graph as graph_module

    stub = _StubInvoke(1_000)
    monkeypatch.setattr(graph_module, "invoke", stub, raising=True)

    ctx = harness._RunContext(
        docs=[],
        mock_mode=False,
        index_dir=Path("unused"),
        scopes=harness._full_scopes(),
    )
    case = _case("q09", {"kind": "refusal", "missing_aspect": "time_period"})

    clock = _Clock()
    paced = harness._run_case(case, ctx, [], {}, pacer=_pacer(clock, []))
    unpaced = harness._run_case(case, ctx, [], {})

    assert paced.tokens == unpaced.tokens
    assert paced.answer_md == unpaced.answer_md
    assert paced.refusal_reason == unpaced.refusal_reason
    assert paced.plan_ok == unpaced.plan_ok


def test_the_pacer_is_inert_under_mock_llm() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept, mock_mode=True)

    prior = _cost([6_000])
    assert pacer.projected_cost(prior) == 0
    assert pacer.gate(6_000, label="(before q01)") == 0.0

    pacer.record(6_000)
    assert pacer.gate(6_000, label="(before q02)") == 0.0
    assert pacer.waits == []
    assert slept == [], "no sleep of any length, on any path"
    assert pacer._minutes_window() == []
    assert pacer.check_daily(999_999) == ""

    pacer.tpm_limit = 4000
    pacer._spend.append((clock(), 7_500))
    assert pacer.gate(3_000, label="(before q03)") == 0.0
    assert slept == []


def test_eval_tpm_limit_overrides_the_per_minute_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    assert harness._tpm_limit_from_env() == harness.DEFAULT_TPM_LIMIT
    monkeypatch.setenv("EVAL_TPM_LIMIT", "30000")
    assert harness._tpm_limit_from_env() == 30000

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept, tpm_limit=harness._tpm_limit_from_env())
    pacer.record(10_038)
    assert pacer.gate(10_038, label="(before q02)") == 0.0, "fits in the raised window"
    assert slept == []


def test_an_unusable_ceiling_falls_back_to_the_default_rather_than_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    for bad in ("", "  ", "eight-thousand", "0", "-1"):
        monkeypatch.delenv("EVAL_TPM_LIMIT", raising=False)
        if bad:
            monkeypatch.setenv("EVAL_TPM_LIMIT", bad)
        assert harness._tpm_limit_from_env() == harness.DEFAULT_TPM_LIMIT, bad


def test_a_run_that_would_spend_past_the_days_share_is_refused_before_it_starts() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept, allowed_today=50_000)

    projected = harness._projected_run_cost(_full_set())
    assert projected > 50_000, "the fixture must be a genuinely short-quota key"

    refusal = pacer.check_daily(projected)
    assert refusal, "a run the day cannot pay for must be refused, not merely slowed down"
    assert "50,000" in refusal and "tokens short" in refusal
    assert harness.TPD_OVERRIDE_HINT in refusal


def test_a_run_inside_the_days_share_starts() -> None:

    clock = _Clock()
    pacer = _pacer(clock, [], allowed_today=200_000)
    assert pacer.check_daily(harness._projected_run_cost(_full_set())) == ""

    assert pacer.check_daily(harness._projected_run_cost([_case("q01")])) == ""


def test_eval_tpd_override_disables_the_daily_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    clock = _Clock()
    pacer = _pacer(clock, [])

    assert pacer.check_daily(99_000_000) != ""
    pacer.daily_gate = harness._daily_gate_enabled()
    monkeypatch.setenv("EVAL_TPD_OVERRIDE", "1")
    assert harness._daily_gate_enabled() is False
    pacer.daily_gate = harness._daily_gate_enabled()
    assert pacer.check_daily(99_000_000) == ""


def test_the_preflight_prints_the_projection_and_the_fraction_of_the_day(
    capsys: pytest.CaptureFixture[str],
) -> None:

    clock = _Clock()
    pacer = _pacer(clock, [])
    projected = harness._projected_run_cost(_full_set())
    assert projected == 8 * harness.CANONICAL_CASE_TOKENS + harness.FOLLOW_UP_CASE_TOKENS

    harness._print_preflight(projected, pacer, list(_full_set()))
    printed = capsys.readouterr().out
    assert "tpm ceiling" in printed and "8,000" in printed
    assert f"{projected:,}" in printed
    assert "% of the day" in printed


def test_a_429_that_lands_after_pacing_still_degrades_into_the_unmeasured_path() -> None:

    case = _case("q02", {"kind": "figure", "must_appear": ["$211,915"]})
    result = harness.CaseResult(
        case=case,
        answer_md="**Refused** — outside the scope of the documents.",
        refusal_reason="unspecified",
        plan_kind="out_of_scope",
        missing_aspect=None,
        degraded_reason=RATE_LIMITED_REASON,
        tokens=(3_477, 0),
    )
    harness._score_case(case, result, harness._full_scopes())

    assert result.infrastructure_degraded is True
    assert result.failed is False
    assert result.refusal_ok is None and result.figure_ok is None

    verdicts = harness._verdicts([result])
    code, reason = harness._exit_code([result], verdicts)
    assert code == harness.EXIT_REGRESSION
    assert "planner could not run" in reason
    assert verdicts["figure_match"] == "NA", "a 429 is never scored as an answer"


def test_the_pacer_does_not_quieten_the_canary() -> None:

    case = _canary_case("q03")
    result = harness.CaseResult(case=case, answer_md="Revenue was $198,270 million in FY2023 [1].")
    harness._score_case(case, result, harness._full_scopes())

    assert result.wrong_year is True
    assert harness._exit_code([result], harness._verdicts([result]))[0] == harness.EXIT_REGRESSION


def test_the_traces_token_counts_are_read_from_the_run_itself() -> None:

    events = [
        {"node": "analyze_question", "tokens_in": 2038, "tokens_out": 411},
        {"node": "hybrid_retrieve"},
        {"node": "synthesize_answer", "tokens_in": 5900, "tokens_out": 308},
        {"node": "groundedness_check", "tokens_in": 2100, "tokens_out": 1},
    ]
    assert harness._trace_tokens(events) == (10_038, 720)
    assert sum(harness._trace_tokens(events)) == 10_758

    class _Event:
        def __init__(self, node: str, tokens_in: object = None, tokens_out: object = None) -> None:
            self.node = node
            self.tokens_in = tokens_in
            self.tokens_out = tokens_out

    shaped = [_Event("analyze_question", 2038, 411), _Event("synthesize_answer", 5900, 308)]
    assert harness._trace_tokens(shaped) == (7938, 719)

    assert harness._trace_tokens([]) == (0, 0)
    assert harness._trace_tokens(None) == (0, 0)


def test_the_projection_is_the_most_expensive_prior_not_the_average() -> None:

    clock = _Clock()
    pacer = _pacer(clock, [])
    assert pacer.projected_cost(_cost([5_000, 8_100])) == 8_100
    assert pacer.projected_cost(_cost([100, 5_000])) == 5_000


def test_a_projection_larger_than_the_window_runs_and_lets_the_provider_decide() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept, tpm_limit=8_000)

    projection = pacer._project(20_000)
    assert projection.wait_seconds == 0.0
    assert "cannot fit" in projection.reason or "no wait" in projection.reason
    assert slept == []


def test_a_case_that_costs_nothing_is_never_paced() -> None:

    clock = _Clock()
    slept: list[float] = []
    pacer = _pacer(clock, slept)

    assert pacer.projected_cost(_cost([0, 0])) == 0
    assert pacer.gate(0, label="(before q09)") == 0.0
    assert slept == []


def test_the_projected_run_cost_counts_refusal_cases_as_free() -> None:

    in_scope = _case("q01")
    refusal = _case("q09", {"kind": "refusal", "missing_aspect": "time_period"})
    assert harness._projected_run_cost([in_scope, refusal]) == harness.CANONICAL_CASE_TOKENS


def _full_set() -> list[Any]:

    cases = [
        _case("q01"),
        _case("q02"),
        _case("q03"),
        _case("q04"),
        _case("q05"),
        _case("q06"),
        _case("q07"),
        _case(
            "q08",
            {"kind": "entity_figure"},
            follow_up_of="q07",
        ),
        _case("q09", {"kind": "refusal", "missing_aspect": "time_period"}),
    ]
    return cases

"""Aggregate-verdict invariants of the golden QA harness (``scripts/eval_rag.py``).

Every test here is hermetic: no network, no ``.env``, no writes into the repo's tree, and
no read of ``data/`` or ``tests/data/``. The results lists are built in-process from
:class:`scripts.eval_rag.CaseResult` objects, so the whole aggregate path
(:func:`_verdicts` → :func:`_exit_code` → :func:`_print_summary`) is exercised without
running the graph.

The invariants pinned are the ones a measured keyed run inverted (NVIDIA Nemotron-3 Super
120B, a real 384-dim index):

1. a rate metric's bar in ``docs/TESTING-AND-EVAL.md`` is a **floor**, so a rate that meets
   or exceeds it PASSes — the old ``value > bar`` printed FAIL for a 1.00 hit rate against a
   0.9 floor and PASS for a 0.71 figure-match rate against a 0.95 floor, and the run ended
   "metrics missed their bar" with exit 1 on a clean pipeline;
2. a rate strictly below its floor FAILs, and that FAIL reaches the exit code;
3. the wrong-year canary keeps its own comparison: its bar is 0 and its value is a
   hard-fail **count**, so ``value > bar`` (any contamination fails, none passes) stays
   exactly as it was;
4. ``NA`` — a metric with no measurable case — is not a failure: it never contributes a
   FAIL to the exit code, and a run whose only unmeasured metrics are ``NA`` exits 0.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> Any:
    """Import ``scripts/eval_rag.py`` as a module without executing its entry point.

    The file is deliberately outside ``testpaths`` (it is the runner, not a test), so it is
    loaded by path rather than by a ``scripts`` package that does not exist. The module body
    only defines things — every side effect (``load_env``, graph imports, the sweep) lives
    inside ``main()`` — so importing it touches neither the network nor ``.env``.
    """

    path = _REPO_ROOT / "scripts" / "eval_rag.py"
    spec = importlib.util.spec_from_file_location("eval_rag_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_module()


def _case(case_id: str, expect: dict[str, Any] | None = None) -> Any:
    """One golden case, built in-process so no dataset file is read."""

    return harness.Case(
        case_id=case_id,
        question=f"question for {case_id}",
        expect=expect or {"kind": "figure"},
        tags=(),
        follow_up_of=None,
        line_no=1,
    )


def _canary_case(case_id: str) -> Any:
    """One year-confusion canary case: ``adjacent_year_figure`` is what makes it one."""

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
    """One finished case carrying the metric values ``scores`` names."""

    return harness.CaseResult(case=case or _case(case_id), **scores)


# --- 1. a rate at or above its bar passes --------------------------------------------


@pytest.mark.parametrize(
    "metric",
    ["retrieval_hit_rate@5", "citation_presence", "figure_match", "refusal_correctness"],
)
def test_a_rate_meeting_its_floor_passes(metric: str) -> None:
    """Every case scoring True is a 1.00 rate, which meets every bar in the table."""

    field = harness._FIELD[metric]
    results = [_result(f"c{index}", **{field: True}) for index in range(7)]
    verdicts = harness._verdicts(results)

    assert verdicts[metric] == "PASS"
    assert harness._exit_code(results, verdicts) == (harness.EXIT_PASS, "")


def test_a_one_point_zero_rate_against_a_zero_point_nine_floor_prints_pass() -> None:
    """The exact measured shape that printed FAIL: 7/7 hit rate against the >= 0.9 bar."""

    results = [_result(f"c{index}", retrieval_hit=True) for index in range(7)]
    verdicts = harness._verdicts(results)

    assert harness._metric_value("retrieval_hit_rate@5", results) == "7/7 (1.00)"
    assert harness.METRICS["retrieval_hit_rate@5"] == pytest.approx(0.9)
    assert verdicts["retrieval_hit_rate@5"] == "PASS"


def test_a_rate_meeting_the_floor_exactly_passes() -> None:
    """``>=`` is inclusive: 0.90 against a 0.90 bar is a pass, not a boundary failure."""

    results = [
        *(_result(f"h{index}", retrieval_hit=True) for index in range(18)),
        *(_result(f"m{index}", retrieval_hit=False) for index in range(2)),
    ]
    verdicts = harness._verdicts(results)

    assert harness._metric_value("retrieval_hit_rate@5", results) == "18/20 (0.90)"
    assert verdicts["retrieval_hit_rate@5"] == "PASS"
    # One drop below the floor is the other side of the same boundary: 0.89 fails.
    below = [
        *(_result(f"h{index}", retrieval_hit=True) for index in range(17)),
        *(_result(f"m{index}", retrieval_hit=False) for index in range(3)),
    ]

    assert harness._metric_value("retrieval_hit_rate@5", below) == "17/20 (0.85)"
    assert harness._verdicts(below)["retrieval_hit_rate@5"] == "FAIL"


# --- 2. a rate below its floor fails -------------------------------------------------


def test_a_rate_below_its_floor_fails_and_reaches_the_exit_code() -> None:
    """The exact measured shape that printed PASS: 5/7 figure match against >= 0.95."""

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
    """A single miss under a 1.0 bar is a fail: 9/10 against citation_presence."""

    results = [
        *(_result(f"c{index}", citation_ok=True) for index in range(9)),
        _result("c9", citation_ok=False),
    ]
    verdicts = harness._verdicts(results)

    assert harness._metric_value("citation_presence", results) == "9/10 (0.90)"
    assert harness.METRICS["citation_presence"] == pytest.approx(1.0)
    assert verdicts["citation_presence"] == "FAIL"


# --- 3. the zero-ceiling canary keeps its own comparison ------------------------------


def test_the_canary_passes_with_no_contamination() -> None:
    """A hard-fail count of zero against a bar of zero passes."""

    clean = dict(wrong_year=False, wrong_year_state=harness.WRONG_YEAR_CLEAN)
    results = [
        _result("canary0", case=_canary_case("canary0"), **clean),
        _result("canary1", case=_canary_case("canary1"), **clean),
    ]
    verdicts = harness._verdicts(results)

    assert verdicts["wrong_year_contamination"] == "PASS"
    assert harness._exit_code(results, verdicts) == (harness.EXIT_PASS, "")


def test_the_canary_fails_on_one_contaminated_case() -> None:
    """Its bar is a zero ceiling: one contaminated case is a regression on its own."""

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


# --- 3a. ``wrong_year`` is True-means-contaminated, and a clean case reads False ----


def test_wrong_year_false_on_a_clean_answer_means_it_is_not_contaminated() -> None:
    """The field's polarity, pinned against the reading that made it look broken.

    ``wrong_year`` is a *contamination flag*: ``True`` means the case asserted the adjacent
    year's figure as the asked year's, and ``False`` means it did not. A run whose five
    clean FY2023 answers print ``wrong_year: False`` is therefore correct — the answers
    quoted FY2023's figures, nothing was contaminated, and the aggregate printed ``0/3+0
    PASS``. It is a confusingly-named field (it reads like "the year is wrong"), never an
    inverted one.

    What would be a harness defect is the opposite polarity: a *clean* case scoring ``True``,
    or a contaminated case scoring ``False``. Both are asserted against the real scoring
    path, so a future inversion cannot pass this suite quietly.
    """

    # The shape the golden set actually produced: a correct FY2023 answer over a canary case.
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

    # And the transpose: the same shape with the adjacent year's figure asserted for the
    # asked year must score True, through the same scoring path.
    dirty = harness.CaseResult(
        case=case, answer_md="Microsoft's FY2023 revenue was $198,270 million [1]."
    )
    harness._score_case(case, dirty, harness._full_scopes())

    assert dirty.wrong_year is True
    assert dirty.wrong_year_state == harness.WRONG_YEAR_CONTAMINATED
    assert "$198,270" in dirty.wrong_year_detail


def test_a_clean_case_costs_the_canary_nothing() -> None:
    """``0/3+0`` is the honest reading of three clean canaries and no uncheckable case."""

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


# --- 3b. the canary reads a table row as a structured record --------------------------


#: The margin-trend shape P4 tells synthesis to produce: a header naming each period
#: column, an alignment row, then one body row per metric.
_THREE_YEAR_TABLE = (
    "| Metric | FY2023 | FY2022 | FY2021 |\n"
    "|---|---|---|---|\n"
    "| Revenue | $211,915 | $198,270 | $168,088 |"
)


def _canary_verdict(answer_md: str, figure: str = "$198,270", year: str = "FY2022") -> bool:
    """Whether the canary fires on ``answer_md``, through the per-case scoring path.

    The contract a canary has to keep is the one :func:`_presented_as_answer` implements,
    and the only caller that matters is the wrong-year block of :func:`_score_case`, so the
    helper scores one real :class:`CaseResult` with a real :class:`Case` rather than
    reaching into the helper directly.
    """

    case = _canary_case("canary-read")
    result = harness.CaseResult(case=case, answer_md=answer_md)
    harness._score_case(case, result, harness._full_scopes())
    return result.wrong_year


def test_a_correct_three_year_table_is_clean() -> None:
    """A row is a structured record: each figure belongs to its own column's header.

    The original bug. The canary split the answer on ``.!?`` and newlines, so a table's
    *row* was one of its "sentences" — and that sentence carried all three years' figures
    as a single string with neither the year label nor a comparative marker in it. A margin
    trend genuinely needs every year, so the canary fired on a correct answer. It now reads
    a row cell-wise against its column header: FY2023's figure under the FY2023 header is
    FY2023's, not a bare assertion.
    """

    assert not _canary_verdict(_THREE_YEAR_TABLE)

    # The first year is not the only one that has to survive: every column is clean.
    for figure in ("$211,915", "$198,270", "$168,088"):
        assert not _canary_verdict(_THREE_YEAR_TABLE, figure=figure)

    # Every body row of a multi-row table reads against the same headers.
    two_metrics = (
        "| Metric | FY2023 | FY2022 |\n"
        "|---|---|---|\n"
        "| Revenue | $211,915 | $198,270 |\n"
        "| Operating income | $88,523 | $83,383 |"
    )
    assert not _canary_verdict(two_metrics)
    assert not _canary_verdict(two_metrics, figure="$83,383")


def test_prose_naming_the_asked_year_but_quoting_an_adjacent_figure_is_contaminated() -> None:
    """The prose path is unchanged: one year named, another year's figure quoted.

    The transpose that has to keep firing. "Revenue was $198,270 million in FY2023" is the
    year trap in its sentence form, and the sentence's own text is all the attribution a
    prose assertion can carry — so it stays contaminated.
    """

    assert _canary_verdict("Revenue was $198,270 million in FY2023 [1].")
    # Not every sentence that names FY2023 is clearable: one that quotes FY2022's figure
    # for it is the trap however it is worded.
    assert _canary_verdict(
        "Revenue was $198,270 million in FY2023 [1]; operating income" " was $88,523 million [2]."
    )
    # The same reading over the correct answer: comparative language clears it.
    assert not _canary_verdict(
        "Revenue was $211,915 million in FY2023, up from $198,270 million in FY2022 [1]."
    )


def test_a_table_header_naming_fy2023_with_the_fy2022_figure_under_it_is_contaminated() -> None:
    """The table path reads strictly: the column header decides, not the row width.

    The fix must not make the canary quieter by making it blinder. A figure that sits under
    the FY2023 column header is *asserted* for FY2023 whatever the rest of the row says —
    including when the asked year's own figure is one cell to the left of it, which is
    exactly the income statement's layout. The trap is now caught structurally rather than
    missed because a neighbour vouchsafed it.
    """

    transposed = "| Metric | FY2023 | FY2022 |\n|---|---|---|\n| Revenue | $198,270 | $211,915 |"
    assert _canary_verdict(transposed)

    # And the row's own label naming a year is attribution too, so it clears.
    assert not _canary_verdict(
        "| Metric | FY2023 | FY2022 |\n|---|---|---|\n| FY2023 revenue | $211,915 |" " $198,270 |"
    )


def test_a_table_cell_is_judged_against_its_own_column_not_the_merged_row() -> None:
    """The unit of reading a table produces is per cell, with its column's header."""

    assertions = harness._assertions(_THREE_YEAR_TABLE)

    assert ("$198,270", "Revenue FY2022") in assertions
    assert ("$211,915", "Revenue FY2023") in assertions
    assert ("$168,088", "Revenue FY2021") in assertions
    # No assertion carries more than one cell's text: the merged row is never produced.
    for text, _label in assertions:
        assert "|" not in text
    assert len(assertions) == 3


def test_a_lone_table_row_with_no_header_is_read_as_the_row_text_it_is() -> None:
    """A row that travelled alone keeps the reading it can carry.

    Two shapes reach the canary with no header above them: a full ``| a | b |`` row on its
    own, and a claim whose outer pipes ``checks.py`` stripped when it logged it
    (``Revenue | $211,915 | $198,270``), which is prose for every purpose here. Neither has
    a column to attribute to, so both are read as the row text they are — still able to
    fire, never silently clean because a header was invented.
    """

    assert _canary_verdict("| Revenue | $211,915 | $198,270 |")
    assert _canary_verdict("| Revenue | FY2023 | $211,915 | $198,270 |")
    assert _canary_verdict("Revenue | $211,915 | $198,270")


def test_an_alignment_row_and_a_headerless_row_carry_no_column_attribution() -> None:
    """``|---|`` and a row of bare figures never name a column."""

    assert not harness._is_alignment_row(["Metric", "FY2023"])
    assert harness._is_alignment_row(["---", ":-:", " -- "])
    assert not harness._is_alignment_row(["", "FY2023"])

    # Body rows and no header row: the first row labels nothing, so nothing is a header and
    # no figure inherits a column it was not printed under.
    assert harness._assertions("| $211,915 | $198,270 |") == [
        ("$211,915", ""),
        ("$198,270", ""),
    ]


# --- 4. NA is not a failure -----------------------------------------------------------


def test_an_unmeasured_metric_is_na_and_never_fails_the_run() -> None:
    """A metric with no measurable case prints NA and exits 0, not 1."""

    results = [_result("c0", retrieval_hit=None), _result("c1", retrieval_hit=None)]
    verdicts = harness._verdicts(results)

    assert verdicts["retrieval_hit_rate@5"] == "NA"
    assert harness._metric_value("retrieval_hit_rate@5", results) == "NA"
    # Nothing scored and nothing crashed: the run is clean, whatever it could not measure.
    assert harness._exit_code(results, verdicts) == (harness.EXIT_PASS, "")


def test_na_is_not_a_fail_even_when_the_comparison_would_have_failed() -> None:
    """``NA`` skips the bar comparison entirely: 0 measurable cases is not a 0.00 rate.

    Under the old comparison a rate of ``0.0`` against a 0.95 floor was a *pass* and one of
    ``1.0`` a fail. Both readings are wrong for an empty scored subset — the metric has no
    value at all — so the branch emits ``NA`` before any comparison happens.
    """

    for metric in harness._FIELD:
        results = [_result("c0", **{harness._FIELD[metric]: None})]
        verdicts = harness._verdicts(results)

        assert verdicts[metric] == "NA", metric
        code, _ = harness._exit_code(results, verdicts)
        assert code == harness.EXIT_PASS, metric


def test_the_canary_is_na_when_no_case_carries_a_checkable_answer() -> None:
    """``not_checkable`` excludes a case from the canary rather than scoring it clean."""

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


# --- the aggregate block, printed over a synthetic results list -----------------------


def test_the_summary_prints_the_verdicts_the_exit_code_reads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The printed table and the exit code come from one verdict map."""

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
    # The floor-shaped line for the metric at its bar, and the ceiling line for the canary.
    assert ">= 0.9" in printed
    assert "<= 0" in printed
    assert "exit code 0" in printed


def test_the_observed_value_is_printed_for_an_na_metric() -> None:
    """An unmeasured metric still prints what it saw, labelled as an observation."""

    results = [_result("c0", retrieval_observed=True), _result("c1", retrieval_observed=False)]

    assert harness._observed_value("retrieval_hit_rate@5", results) == "1/2 (0.50)"


def test_the_documented_metric_table_is_what_the_harness_uses() -> None:
    """The five metrics and bars ``docs/TESTING-AND-EVAL.md`` names, in the harness."""

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
    """Reading the dataset is the loader's job; a bad path raises the typed error."""

    with pytest.raises(harness.DatasetError):
        harness.parse_dataset(tmp_path / "absent.jsonl")


def test_loading_the_module_left_no_index_or_env_state_behind() -> None:
    """Importing the harness touches no file and reads no key.

    ``main()`` is where ``load_env`` and the graph live, so a module-level import is
    side-effect free by construction — asserted here so a future module-body import (of
    the gateway, or of ``src.chains.graph``) cannot quietly break the hermetic contract.
    """

    graph_modules = [name for name in sys.modules if name.startswith("src.chains.graph")]
    assert not graph_modules, "the harness module body must not import the graph at import time"


def test_no_metric_field_reads_a_missing_attribute() -> None:
    """Every field the verdicts read is one :class:`CaseResult` really declares."""

    declared = {field.name for field in harness.CaseResult.__dataclass_fields__.values()}
    for metric, field in harness._FIELD.items():
        assert field in declared, f"{metric} reads undeclared field {field!r}"
    for metric, field in harness._OBSERVED_FIELD.items():
        assert field in declared, f"{metric} observes undeclared field {field!r}"


# --- 5. an infrastructure-degraded case is not measured, not scored --------------------
#
# The shape a rate-limited run actually produces: ``analyze_question`` catches the
# ``RateLimited`` the gateway raises and returns P1's no-plan output — ``kind=out_of_scope``,
# ``missing_aspect=None``, ``planner_degraded=True`` — so the graph routes to ``refuse`` and
# every case comes back refused. Scored as a semantic verdict, that refusal reads as a wrong
# one on every in-scope case, which is what took ``refusal_correctness`` to 2/9 and
# ``citation_presence`` to 1/7 on a run where the answer quality was never exercised at all.
#
# The harness has one job here: say the provider said no, and refuse to average over it.

#: The reason ``analyze.py:_degraded_plan`` logs for a 429 — a class name, then the message.
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
    """Each infrastructure class says "no plan was produced", never what the question was.

    A ``RateLimited``, ``Auth``, ``ContextLength`` or ``Timeout`` stops the gateway outside
    the question's content. ``SchemaViolationError`` is deliberately *not* in that set: the
    model answered and its answer was unusable, which is P1's failure handling, and a plan
    the model produced is a plan the harness can score.
    """

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
    """The regression this section exists for: q02's degraded refusal scored ``False`.

    Under the old scoring an in-scope question that was refused scored ``False`` because it
    was refused — with nothing about the question having been decided. The planner-degraded
    signal is what separates that from a real wrong refusal.
    """

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
    """The exclusion is narrow: only the degradation signal takes a case out of the rate.

    A genuinely refused in-scope question — a planner that ran and said ``out_of_scope`` —
    still scores ``False``, which is the metric doing its job. The fix labels an
    infrastructure case; it does not become a blanket exemption.
    """

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
    """A degradation is no licence for a wrong ``missing_aspect`` either.

    q10 is a refusal case whose plan degradation still lands it on ``refuse``: the harness
    records what the pipeline said about which aspect was missing, and says it could not
    verify it, rather than quietly passing or failing.
    """

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
    """The whole q01-q08 + q10 shape: nine unmeasured cases, one honest refusal.

    This is the aggregate a rate-limited run produces after the fix. Nothing scores a
    ``False``, because nothing was measured — and the run still does not pass, because
    nine cases carrying no verdict is a regression about the run, not a quality result.
    """

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
    # q09 is the one real refusal: FY2019, never reaching the planner at all (ADR-010's
    # fiscal-year gate), so it scores normally.
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
    # Excluded from every rate, and still a regression: the exclusion cannot become a pass.
    for metric in ("citation_presence", "figure_match", "retrieval_hit_rate@5"):
        assert verdicts[metric] == "NA", metric
    code, reason = harness._exit_code(results, verdicts)
    assert code == harness.EXIT_REGRESSION
    assert "planner could not run" in reason
    assert "q01" in reason


def test_the_summary_names_the_unmeasured_cases_apart_from_the_failing_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two different labels: "failing" is a bad answer, "not measured" is no verdict."""

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


# --- 6. figure match compares figures, not their orthography --------------------------
#
# The shape a keyed run against ``openai/gpt-oss-20b`` actually produced for q01 ("What was
# Microsoft's revenue in FY2023?"), verbatim from that run's journal:
#
#     expected       : $211,915
#     expected       : $211,915       MISSING
#     retrieval hit  : True (expected figure present in the fused top-5 texts)
#     answer         : Microsoft's revenue for fiscal year 2023 was **$ 211,915 million** [2]
#
# Four cases scored that way, so ``figure_match`` printed 0/3 (0.00) FAIL while
# ``citation_presence`` printed 4/4 (1.00) PASS on the *same* runs — answers that were
# cited and correct reported as missing the figure they printed. The comparison was a
# literal substring test of ``$211,915`` against the answer text, and P4 rule 2 tells the
# model to reproduce the document's own spacing, so the corpus's ``$ 211,915`` (a space
# after the symbol) is the answer the prompt contract asks for.
#
# The fix normalises spacing, emphasis, grouping and the currency sign — the orthography of
# a number — and never the digits. That distinction is what the rest of this section pins:
# the wrong-year figure must still be a different figure.

#: The answer q01's run printed, with the log's own narrow no-break spaces (``U+202F``)
#: after the symbol, before the unit and before the citation marker.
LOG_ANSWER_Q01 = (
    "Microsoft\u2019s revenue for fiscal year\u202f2023 was "
    "**$\u202f211,915\u202fmillion**\u202f[2]"
)
#: The same answer as the ticket writes it: plain ASCII with a normal space.
SPACED_ANSWER_Q01 = "Microsoft's revenue for fiscal year 2023 was **$ 211,915 million** [2]"
#: The figure the corpus prints for FY2022 — the adjacent year the canary watches.
FY2022_FIGURE = "$198,270"


def test_the_logged_answer_matches_the_figure_it_prints() -> None:
    """The exact regression: q01's own answer, scored against q01's own expectation."""

    assert harness._contains_figure(LOG_ANSWER_Q01, "$211,915")
    assert harness._contains_figure(SPACED_ANSWER_Q01, "$211,915")


def test_the_figure_metric_scores_the_logged_answer_as_a_match() -> None:
    """End to end through the scoring path, not just the helper: ``figure_ok`` is True."""

    case = _case("q01", {"kind": "figure", "must_appear": ["$211,915"]})
    result = harness.CaseResult(case=case, answer_md=LOG_ANSWER_Q01)
    harness._score_case(case, result, harness._full_scopes())

    assert result.figure_ok is True
    assert result.missing_figures == []


def test_the_exact_answer_the_run_printed_reaches_the_pass_bar() -> None:
    """The aggregate shape the run printed, with one of its three answers repaired.

    q01 now scores a match while q02 and q04 still miss, so the metric reports 1/3 (0.33)
    against a 0.95 floor and still reaches exit 1. The normalisation repairs the
    comparison, not the run's other outcomes.
    """

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
    """The check did not become vacuous: an answer without the figure still misses it."""

    answer = "**Refused** — outside the scope of the documents in this session."
    case = _case("q01", {"kind": "figure", "must_appear": ["$211,915"]})
    result = harness.CaseResult(case=case, answer_md=answer)
    harness._score_case(case, result, harness._full_scopes())

    assert result.figure_ok is False
    assert result.missing_figures == ["$211,915"]
    assert not harness._contains_figure(answer, "$211,915")


def test_an_answer_printing_a_different_number_does_not_match() -> None:
    """The same digits at another scale, regrouped, or in another arrangement."""

    assert not harness._contains_figure("Revenue was $111,915 million in FY2023 [1].", "$211,915")
    assert not harness._contains_figure("Revenue was $211.915 thousand.", "$211,915")
    assert not harness._contains_figure("Revenue was $21,19,15 million.", "$211,915")
    assert not harness._contains_figure("Revenue was about $212 billion.", "$211,915")


def test_the_adjacent_year_figures_do_not_match() -> None:
    """The distractors, which is what stops the normalisation from matching any number.

    ``$198,270`` and ``$83,383`` sit next to the asked figures in the corpus's income
    statement, so they are exactly what a loosened check would let through.
    """

    for answer in (
        "Revenue was $198,270 million in FY2023 [1].",
        "Operating income was $83,383 million in FY2023 [2].",
        "| Operating income | $ 83,383 |",
    ):
        assert not harness._contains_figure(answer, "$211,915"), answer
        assert not harness._contains_figure(answer, "$88,523"), answer


def test_the_adjacent_year_figure_still_matches_when_it_is_the_expected_one() -> None:
    """The transpose: a normalisation that swallowed the distractors would fail q03, whose
    expected figures include the prior year's."""

    assert harness._contains_figure(
        "Revenue fell from $211,915 to $ 198,270 million [1].", "$198,270"
    )


def test_a_longer_number_containing_the_expected_one_does_not_match() -> None:
    """One printed number is one token, so ``$1,198,270`` never reports ``$198,270``."""

    assert not harness._contains_figure(
        "The company earned $1,198,270 million in FY2023.", "$198,270"
    )
    assert not harness._contains_figure("The subtotal was $1198270.", "$211,915")


def test_the_spacing_emphasis_and_grouping_variants_all_match() -> None:
    """Every spelling of the same figure the corpus and a model may print between them."""

    expected = "$211,915"
    variants = (
        "$211,915",  # the expectation's own form
        "$ 211,915",  # the corpus's form: a space after the symbol
        "$\u00a0211,915",  # no-break space
        "$\u202f211,915",  # narrow no-break space — what the keyed run printed
        "211,915",  # no currency sign
        "211915",  # no currency sign, no separator
        "**$ 211,915**",  # bold around the whole figure
        "*$211,915*",  # italic
        "\u20ac211,915",  # another currency's sign — the digits are the figure
        "Revenue was **$ 211,915 million** [1].",  # a space before the unit
        "| Revenue | $ 211,915 | $ 198,270 |",  # a table cell, distractor beside it
        "Revenue grew to $ 211,915 million, up 7% [1].",  # another figure in the sentence
    )
    for variant in variants:
        assert harness._contains_figure(variant, expected), variant


def test_a_thin_space_between_the_digit_groups_is_the_same_figure() -> None:
    """The corpus groups its values with a thin space, so ``211 915`` is ``211,915``."""

    assert harness._contains_figure("Revenue was $ 211\u2009915 million.", "$211,915")
    assert harness._contains_figure("Revenue was 211\u2007915 million.", "211,915")


def test_a_figure_the_answer_never_prints_does_not_match() -> None:
    """The literal-vs-normalised boundary: a number the answer does not print is absent,
    whether it differs by one digit, by scale, or by being another year's figure."""

    for absent in ("$212,915", "$211,915,000", "$1,119,150", "$198,270"):
        assert not harness._contains_figure(LOG_ANSWER_Q01, absent), absent


def test_a_non_numeric_expectation_keeps_its_literal_reading() -> None:
    """q05 and q08 expect a segment name. Dropping furniture from letters would invent
    matches no answer makes, so it keeps the substring reading it always had."""

    assert harness._contains_figure(
        "Intelligent Cloud drove revenue with $87,907 million [3].", "Intelligent Cloud"
    )
    assert not harness._contains_figure("The answer concerns Amazon.", "Intelligent Cloud")
    assert not harness._contains_figure("Intelligent Cloud", "$87,907")


def test_an_empty_expectation_is_never_present() -> None:
    """The guard the caller relies on for ``must_appear`` entries that strip to nothing."""

    assert not harness._contains_figure(LOG_ANSWER_Q01, "")
    assert not harness._contains_figure(LOG_ANSWER_Q01, "   ")


# --- 6a. the wrong-year canary reads the same inputs unchanged ------------------------


def test_the_canary_verdict_is_unchanged_by_the_normalisation() -> None:
    """The normalisation may not quieten the year-confusion canary.

    Every shape here is the canary's own corpus: the preceding tests establish that
    ``_contains_figure`` now matches across spacing and emphasis, and the same recognition
    is what *finds* the adjacent-year figure for the canary to judge. None of these verdicts
    may move — those that fired before must still fire, and those that were clean must
    still be clean.
    """

    # The trap in prose: the asked year named beside the adjacent year's figure.
    assert _canary_verdict("Revenue was $198,270 million in FY2023 [1].")
    # The same trap in the log's own spacing: found as reliably now, if not more so.
    assert _canary_verdict("Revenue was $ 198,270\u202fmillion in FY2023 [1].")
    # A correct three-year table is still clean, and still clean for every column.
    for figure in ("$211,915", "$198,270", "$168,088"):
        assert not _canary_verdict(_THREE_YEAR_TABLE, figure=figure)
    # The same table in the corpus's spacing is still clean.
    spaced = (
        "| Metric | FY2023 | FY2022 | FY2021 |\n"
        "|---|---|---|---|\n"
        "| Revenue | $ 211,915 | $ 198,270 | $ 168,088 |"
    )
    assert not _canary_verdict(spaced)
    assert not _canary_verdict(spaced, figure="$198,270")
    # A headerless row of the asked and adjacent figures, the pre-trim claim shape.
    assert _canary_verdict("Revenue | $211,915 | $198,270")


def test_the_canary_still_names_the_adjacent_year_it_found() -> None:
    """The report still says *which* figure looked wrong, in the expectation's own form."""

    case = _canary_case("canary-report")
    result = harness.CaseResult(case=case, answer_md="Revenue was $198,270 million in FY2023 [1].")
    harness._score_case(case, result, harness._full_scopes())

    assert result.wrong_year is True
    assert FY2022_FIGURE in result.wrong_year_detail
    assert result.wrong_year_state == harness.WRONG_YEAR_CONTAMINATED

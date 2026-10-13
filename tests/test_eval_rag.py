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

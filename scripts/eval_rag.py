#!/usr/bin/env python3
"""The golden QA harness: ``tests/data/golden_qa.jsonl`` run end to end through ``invoke()``.

This is the second testing layer docs/TESTING-AND-EVAL.md describes — the honest quality
gate that also produces the journal's numbers. It is deliberately **not** a pytest module:
``pyproject.toml`` scopes ``testpaths = ["tests"]`` and this file lives in ``scripts/``,
so ``pytest -q`` never collects it. Nothing here stubs the graph: every case is one real
``src.chains.graph.invoke()`` call over the real persisted index in ``data/index/``.

What it measures (docs/TESTING-AND-EVAL.md §Layer 2, in the doc's own order, with the doc's
own bars — none of them lowered for any mode):

========================  ============================================  ==========
Metric                    Definition                                   Pass bar
========================  ============================================  ==========
Retrieval hit-rate@5      the chunk containing the printed answer      >= 0.9
                          is in fused top-5
Citation presence         every factual answer carries >= 1 valid      1.0
                          marker
Figure match               the expected figure appears verbatim        >= 0.95
Wrong-year contamination  an adjacent-year figure appears where the   0 (hard fail)
                          asked year's figure was expected
Refusal correctness       out-of-scope refused; in-scope answered     1.0 / >= 0.95
========================  ============================================  ==========

Two modes, one harness. Under ``MOCK_LLM=1`` the graph runs the deterministic offline LLM
and the hash embedder — the whole point of hermetic CI: no key, no quota, no network. That
mode also changes what the metrics above *can* mean, and this harness says so out loud
rather than pretending:

* **Retrieval hit-rate@5 — N/A.** The mock embedder is a sha256 hash of each text, so a
  natural-language question's dense vector is noise. The fused top-5 is a BM25 ordering
  with noise on top, and where the answer chunk lands carries no signal.
* **Figure match — N/A.** The mock LLM prints scripted phrases plus whatever the retrieved
  contexts contain, so a figure's presence measures the script, not the pipeline.
* **Citation presence — N/A.** The mock cites only the blocks that contain its figure, so
  "carries >= 1 valid marker" is a function of whether retrieval happened to find that
  block — the same missing signal, one layer downstream.
* **Refusal correctness — partially measurable.** The mock planner refuses only on fiscal
  periods the corpus does not cover, so the ``time_period`` case (Q9) is scoreable and the
  ``entity`` case (Q10) is not: its plan is never out-of-scope to begin with.

Each unmeasurable cell reads ``NA`` with the reason beside it. The observed value is still
printed, labelled as an observation rather than a verdict, so a reader can tell "this mode
cannot answer it" from "it failed".

The **wrong-year canary** is the exception that proves the rule: it reads the answer's own
figures and the layout they sit in — a sentence whole, a table cell against its column's
header — so it is lexical, mode-independent, and fully able to FAIL under ``MOCK_LLM=1``.
That is the point — the one regression that must never hide behind a mode.

Exit codes, printed with their meaning on every run:

=====  ==========================================================================
0      every meaningful metric met its bar
1      a golden regression: a pass bar was missed, the wrong-year canary fired,
       a case's graph run raised, or a case's planner could not run at all
2      a harness error: the dataset could not be run (no usable case, an unreadable
       file, an unresolvable ``follow_up_of``, a follow-up whose ancestor never ran,
       or a missing index)
=====  ==========================================================================

A case whose planner could not run — a 429, an auth failure, a context-length or a
timeout refusal from the provider — is reported as **not measured** and excluded from its
metrics, not scored as the ``out_of_scope`` refusal ADR-005's degradation produced. The
reason: that refusal is what P1 emits when *no plan could be produced*
(``missing_aspect=None`` plus ``planner_degraded=True``), so scoring it would read the
provider's quota as a semantic verdict about the question, and as a wrong one — the
degradation lands on in-scope and out-of-scope cases alike. The exclusion is recorded per
case (``cases not measured``) and drives the exit code, so hiding a case behind it can
never look like passing.

**Pacing (real-LLM mode only).** A keyed run against a per-minute token ceiling — Groq's
free tier is 8,000 TPM, and one in-scope case on this corpus measures about 10,038 tokens
(P1 2,038 + P3 5,900 + P5 2,100, plus ~600 for a P2 rewrite) — cannot run back-to-back:
every request would overshoot the window before it was sent. The gateway's own
``retry-after`` handling is *reactive*; this pacer is the proactive half. Before each case
it projects the cost the next case will add, waits until the rolling per-minute window has
room, and prints the arithmetic so a reader can tell a wait from a hang.

Two ceilings, because a provider publishes both, and the daily one is the harder wall. The
per-minute ceiling is smoothable by waiting. The daily one is not: the run spends it and the
day is over. The ten-case golden set measures about 100K tokens, roughly half of Groq free
tier's 200K/day allowance, so a full keyed run is *not* repeatable on a free key:
``MOCK_LLM=1`` is the reproducible path (zero tokens, deterministic), and the keyed provider
is there for a single-question demonstration. The harness refuses to start a run projected
to spend past the day's usable share unless ``EVAL_TPD_OVERRIDE=1`` says so outright.

Under ``MOCK_LLM=1`` the pacer is inert: no projection, no window, no sleep, and no token
line in the report. A mock case costs zero provider tokens.

Environment (all optional, all real-LLM mode only):

=============================  ==============  ===========================================
Variable                       Default        Meaning
=============================  ==============  ===========================================
``EVAL_TPM_LIMIT``             8000           the per-minute token ceiling to pace against
``EVAL_TPD_LIMIT``             200000         the daily token ceiling
``EVAL_TPD_BUDGET``            (fraction)     tokens this run may spend today
``EVAL_TPD_FRACTION``          1.0            fraction of the daily ceiling to spend
``EVAL_TPD_OVERRIDE``          (unset)        ``1`` disables the daily pre-flight gate
=============================  ==============  ===========================================

Usage::

    MOCK_LLM=1 python scripts/eval_rag.py       # hermetic, offline, deterministic, un-paced
    python scripts/eval_rag.py                  # real LLM (needs OPENAI_API_KEY), paced
    python scripts/eval_rag.py --only q01,q03   # score a subset of cases
    python scripts/eval_rag.py --data <path>    # another golden set
    python scripts/eval_rag.py --json out.json  # machine-readable report
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.settings import load_env

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "tests" / "data" / "golden_qa.jsonl"
#: ``INDEX_DIR`` in the environment overrides this, so the harness can be pointed at the mock
#: index without rebuilding: the mock embedder is 64-dimensional and a real one 384, and a
#: single directory cannot hold both.
DEFAULT_INDEX_DIR = Path(os.environ.get("INDEX_DIR", "").strip() or (REPO_ROOT / "data" / "index"))
CATALOG_NAME = "docs.jsonl"

EXIT_PASS = 0
#: A pass bar was missed, the wrong-year canary fired, a case's graph run raised, or a
#: case's planner could not run at all.
EXIT_REGRESSION = 1
#: The dataset could not be run at all: a broken file, a broken reference, or no index.
EXIT_HARNESS_ERROR = 2

# The five metrics docs/TESTING-AND-EVAL.md names, with the doc's own bars. ``hard`` marks
# wrong-year contamination: the year-confusion canary, where one occurrence is a regression
# on its own regardless of the aggregate rate.
METRICS: dict[str, float] = {
    "retrieval_hit_rate@5": 0.9,
    "citation_presence": 1.0,
    "figure_match": 0.95,
    "wrong_year_contamination": 0.0,
    "refusal_correctness": 1.0,
}
HARD_METRICS = frozenset({"wrong_year_contamination"})

#: How many fused children count as "top-5" for hit-rate (the metric's own name).
HIT_RANK = 5

# --- provider budget ----------------------------------------------------------------
#
# Groq's free tier enforces a **rolling 8,000 tokens-per-minute** ceiling, and the same
# number for a daily one. The measured cost of this corpus (``data/index``, 384-dim
# fastembed) is one in-scope case costing about 10,000 tokens:
#
# ============  ========  ==========================================================
# node          tokens    what it is
# ============  ========  ==========================================================
# P1 planner    ~2,038    305 (system) + 766 (catalog JSON) + 9 (question) + 128 (schema)
# P3 synthesis  ~5,900    the 23k chars of retrieved context — irreducible, it carries
#                            the passages
# P5 verify      ~2,100    the answer and its claims
# ============  ========  ==========================================================
#
# A P2 rewrite adds ~600 with history. So one case is 125% of *one minute's* budget, and
# the ten-case golden set is ~100,000 tokens — about half of a daily 200K allowance in a
# single run. Running the set back-to-back makes a 429 a certainty rather than
# embarrassingly bad luck: the request that cannot fit (``Requested 8087`` against a
# ``Used 6537 / 8000`` window) is not a race, it is arithmetic.
#
# The gateway already honours a 429's ``retry-after``, which is *reactive*. What was
# missing is a pacer that knows a case is about to overshoot and waits first.
#
# The limits below are the ceiling of each *env override*'s acceptance, not a different
# default: a ceiling of zero or below would make every projection overshoot, so the
# harness's own fallback applies rather than an infinite wait.
DEFAULT_TPM_LIMIT = 8000
#: Groq's published free-tier daily allowance, for the ``EVAL_TPD_LIMIT`` default. The
#: keyed models are ``openai/gpt-oss-120b``, ``openai/gpt-oss-20b`` and
#: ``qwen/qwen3.8-27b`` — all three carry 8K TPM / 200K TPD, so the daily number is the
#: one a run cannot work around. ``allam-2-7b`` raises it to 500K but *lowers* TPM to 6K
#: and is a 7B model that cannot hold the citation contract, so it is a bad trade and is
#: not what this default is for.
DEFAULT_TPD_LIMIT = 200_000
#: A run spends at most this fraction of the daily allowance before it is refused. A run
#: that spends everything a free key has leaves tomorrow with nothing, which is a footgun
#: a pacer should not aim at.
DEFAULT_TPD_FRACTION = 1.0
#: How long one projection may push the pacer into the future before it is a runaway
#: rather than a wait. Beyond this, the case runs and let the provider's own 429 decide:
#: a best estimate is a projection, not an instrument, and guessing badly must never
#: wedge the harness. Both bounds are generous against the measured numbers — a whole
#: ten-case sweep is ~12.5 minutes of TPM headroom.
TPM_WAIT_CAP_SECONDS = 30 * 60
TPD_WAIT_CAP_SECONDS = 60 * 60
#: The message a user must change to say "yes, spend it", spelled once so the harness,
#: its report and its tests all read the same escape hatch.
TPD_OVERRIDE_HINT = "EVAL_TPD_OVERRIDE=1"
#: Every space that carries no meaning around a printed figure. The corpus prints its
#: tables with typographic spaces — a thin space (``U+2009``) between a value and its
#: ``$``, three-per-em (``U+2004``) and figure space (``U+2007``) inside a row — and an
#: answer following P4 rule 2 ("reproduce exactly as printed") prints the same number with
#: a narrow no-break space (``U+202F``) after the currency symbol, which is what the keyed
#: run actually produced. A difference of spacing is never a difference of number, so none
#: of it may decide whether a figure is present. A line break is deliberately *not* here:
#: a printed figure never spans lines, and a match that crosses one is a false match.
_FIGURE_SPACES = " \u00a0\u2009\u200a\u2002\u2003\u2004\u2005\u2007\u202f\u205f\u3000"

#: The rendering furniture dropped from both sides of a comparison: the spaces above,
#: markdown emphasis (``**`` bold, ``*`` italic — an asterisk is never a digit of a
#: figure), a thousands separator, the currency symbol, and the percent marker. Every one
#: of them is a unit or a typographic convention marked on the number, never a digit of
#: it, so ``$ 211,915``, ``**$211,915**`` and ``211915`` reduce to the same string.
_FIGURE_FURNITURE = _FIGURE_SPACES + "*,$%"

_SPACE_TRANSLATION = str.maketrans("", "", _FIGURE_SPACES)
_FIGURE_TRANSLATION = str.maketrans("", "", _FIGURE_FURNITURE)

#: The separators that glue two digit *groups* of one printed number: the comma, and the
#: typographic spaces the corpus groups its values with (a thin space in ``211 915``). A
#: plain space and a line break are deliberately absent — in a table row they separate
#: *fields*, not digits (``17 $ 4,600 $ 5,270`` prints three figures), so gluing on them
#! would merge a row's neighbouring numbers into one token and hide the figure the
#: wrong-year canary is looking for.
_GROUP_SEPARATORS = ",\u2009\u200a\u2002\u2003\u2004\u2005\u2007\u202f\u205f"

#: One printed number inside a text: an optional currency sign, the spacing a sign may
#: carry on either side, then the digit groups in the corpus's own discipline —
#: ``\d{1,3}`` then any number of well-formed three-digit groups — and an optional decimal.
#: The strictness is the whole point: ``$21,19,15`` is not a grouping of any real figure, so
#: it tokenises to the three separate numbers ``21``, ``19`` and ``15`` rather than to
#! ``211915``. Reading a number token-wise — rather than as a substring of space-stripped
#: text — is what keeps a normalised ``198270`` from matching inside ``$1,198,270``: that
#: is one printed number, so it yields the single token ``1198270``.
_FIGURE_TOKEN_IN_TEXT = re.compile(
    r"\$?[" + _FIGURE_SPACES + r"]*"
    r"(?:\d{1,3}(?:[" + _GROUP_SEPARATORS + r"]\d{3})+|\d+)(?:\.\d+)?"
)

#: A normalised figure made of nothing but digits and decimal points — the shape that is
#: compared token-wise. A ``must_appear`` carrying letters (``Intelligent Cloud``) reduces
#: to a string this never matches, so it keeps the literal reading above.
_DIGIT_FIGURE = re.compile(r"[\d.]+")
#: A sentence carrying an adjacent-year figure is a *comparison* when it names that year or
#: uses comparative language. Both readings are the answer being right about both years,
#: not the year trap.
_COMPARATIVE_MARKERS = (
    "vs",
    "versus",
    "compared",
    "comparison",
    "against",
    "prior",
    "earlier",
    "previous",
    "last year",
    "from $",
    "rose from",
    "fell from",
    "grew from",
    "increased from",
    "decreased from",
    "declined from",
    "down from",
)

#: The visible replacement for a trimmed claim (``checks.TRIM_MARKER``, spelled here rather
#: than imported so the harness reads the pipeline's own marker without importing the
#: verification module at module scope). An answer that is *only* this marker had every
#: claim trimmed away, which is exactly the case the canary must not read as "no figure, so
#: nothing to check".
TRIM_MARKER = "(unsupported claim removed)"

#: The verifier's logger. It is where the claim text the node *produced* — before the trim
#: removed it from ``answer_md`` — is written, which is the thing the canary has to judge: a
#: wrong-year claim the verifier caught and trimmed never reaches the final answer, so a
#: canary that reads only ``answer_md`` scores it as clean.
VERIFY_LOGGER_NAME = "src.chains.nodes.verify"

#: The planner's logger. It is where a degraded plan is named — the ``planner_degraded``
#: warning P1's failure handling emits (``analyze.py:_degraded_plan``) — which is the signal
#: that this run's refusal describes the *gateway*, not the question.
ANALYZE_LOGGER_NAME = "src.chains.nodes.analyze"

#: The planner-degraded warning's own marker, matched verbatim so the signal is read only
#: from the record the node emits and never inferred from a message that happens to look
#: similar. The formatted record also carries the reason, which starts with the error class
#: (``RateLimited: …``, ``SchemaViolationError: …``, ``Auth: …``) — the part that says *why*
#: the planner degraded, and the part :func:`_infrastructure_degradation` reads.
PLANNER_DEGRADED_MARKER = "planner_degraded:"

#: Error-class prefixes that name an *infrastructure* condition rather than a semantic answer
#: to the question. A planner that degraded because a 429, an auth failure or a transport
#: error stopped it has said "I could not plan", which is not the same statement as P1's
#: reasoned ``out_of_scope`` + ``missing_aspect`` and must not be scored as one.
_INFRASTRUCTURE_REASONS = frozenset(
    {
        "RateLimited",
        "ContextLength",
        "Auth",
        "Timeout",
    }
)

#: How a dollar figure or a percentage is printed, used to tell "this text asserts a
#: figure" from "this text merely mentions a year" — a year is not a figure, and an answer
#: carrying only years has nothing for the canary to check.
_FIGURE_IN_TEXT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?%")

#: The one state one case's canary can be in that is not a measurement.
NOT_CHECKABLE = "not_checkable"
#: The two measured states.
WRONG_YEAR_CLEAN = "clean"
WRONG_YEAR_CONTAMINATED = "contaminated"


# --- what the current mode can measure ----------------------------------------------


@dataclass(frozen=True)
class _Scope:
    """Whether one metric carries signal in this mode, and why when it does not.

    ``partial_kinds`` is a scoped exemption, not a blanket one: a degraded mode can still
    score the cases whose answer it *can* produce. It names the refusal classes the mode
    classifies correctly, so an ``N/A`` lands on the cases that lost the signal — never on
    the cases the mode happens to decide fine.
    """

    name: str
    meaningful: bool
    reason: str = ""
    partial_kinds: tuple[str, ...] = ()

    def covers(self, case: Case) -> bool:
        """Whether this metric is measurable for ``case`` in the current mode."""

        if self.meaningful:
            return True
        if not self.partial_kinds:
            return False
        # A refusal the mode cannot classify is unmeasurable; every other case — an answer
        # given or a refusal the mode does classify — is still measured normally.
        if not case.is_refusal:
            return True
        return str(case.expect.get("missing_aspect", "")) in self.partial_kinds


def _mock_scopes() -> dict[str, _Scope]:
    """Mock-mode scope: what a hash embedder and a scripted LLM cannot decide.

    Each ``NA`` names the capability that is absent rather than the number that would have
    been produced, so a reader can tell "this mode cannot answer it" from "it failed".
    """

    return {
        "retrieval_hit_rate@5": _Scope(
            "retrieval_hit_rate@5",
            False,
            "the mock embedder is a sha256 hash of each text, so dense scores carry no "
            "semantic signal and the fused top-5 over a natural-language question is a "
            "BM25 ordering with noise on top",
        ),
        "citation_presence": _Scope(
            "citation_presence",
            False,
            "the mock LLM cites only the blocks that contain its figure, so 'carries >= 1 "
            "valid marker' is a property of where retrieval happened to land",
        ),
        "figure_match": _Scope(
            "figure_match",
            False,
            "the mock LLM answers from scripted phrases plus whatever the retrieved "
            "contexts happen to contain, so a figure's presence is a property of the "
            "script, not evidence about the pipeline",
        ),
        "wrong_year_contamination": _Scope("wrong_year_contamination", True),
        "refusal_correctness": _Scope(
            "refusal_correctness",
            False,
            "the mock planner refuses only on fiscal periods the corpus does not cover, so "
            "an entity-gap case has no classification to score",
            partial_kinds=("time_period",),
        ),
    }


def _full_scopes() -> dict[str, _Scope]:
    """Real-LLM scope: every metric is measurable for every case."""

    return {name: _Scope(name, True) for name in METRICS}


# --- loading the dataset ------------------------------------------------------------


class DatasetError(Exception):
    """The dataset could not be read at all — a harness error, never a case."""


class CaseError(Exception):
    """One line or one reference is unusable: reported, never a crash."""


@dataclass(frozen=True)
class Case:
    """One golden case, parsed from one non-blank ``.jsonl`` line."""

    case_id: str
    question: str
    expect: dict[str, Any]
    tags: tuple[str, ...]
    follow_up_of: str | None
    line_no: int

    @property
    def is_refusal(self) -> bool:
        return self.expect.get("kind") == "refusal"

    @property
    def wants_citation(self) -> bool:
        """Whether this case expects the answer to carry at least one citation."""

        return bool(self.expect.get("require_citations", True))

    @property
    def expected_appearances(self) -> list[str]:
        """The figures and names the answer must contain, in the doc's own wording."""

        return [str(f) for f in self.expect.get("must_appear", []) or [] if str(f).strip()]


@dataclass(frozen=True)
class DatasetReport:
    """What the loader found: the usable cases, plus every line it had to reject."""

    cases: list[Case]
    malformed: list[tuple[int, str]]
    duplicate_ids: list[str]


def parse_dataset(path: Path) -> DatasetReport:
    """Read the golden set. A bad line is reported and skipped; a bad file is fatal."""

    if not path.is_file():
        raise DatasetError(f"golden set not found: {path}")
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"golden set unreadable: {path}: {exc}") from exc

    cases: list[Case] = []
    malformed: list[tuple[int, str]] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []

    for line_no, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped:
            # A blank line in a JSONL file is noise, not a case: reported, not fatal.
            malformed.append((line_no, "blank line (skipped)"))
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            malformed.append((line_no, f"not valid JSON ({exc.msg} at column {exc.colno})"))
            continue
        try:
            case = _case_from(payload, line_no)
        except CaseError as exc:
            malformed.append((line_no, str(exc)))
            continue
        if case.case_id in seen:
            duplicate_ids.append(case.case_id)
        seen.add(case.case_id)
        cases.append(case)

    return DatasetReport(cases=cases, malformed=malformed, duplicate_ids=duplicate_ids)


_REQUIRED_STRINGS = ("id", "question")
_EXPECT_KINDS = ("figure", "entity_figure", "summary", "refusal")


def _case_from(payload: Any, line_no: int) -> Case:
    """One parsed line as a :class:`Case`, raising :class:`CaseError` on a contract break."""

    if not isinstance(payload, Mapping):
        raise CaseError(f"expected a JSON object, found {type(payload).__name__}")
    for key in _REQUIRED_STRINGS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CaseError(f"missing or empty required field {key!r}")
    expect = payload.get("expect")
    if not isinstance(expect, Mapping):
        raise CaseError("missing or non-object field 'expect'")
    if expect.get("kind") not in _EXPECT_KINDS:
        raise CaseError(f"expect.kind must be one of {sorted(_EXPECT_KINDS)}")
    if expect.get("kind") == "refusal" and not str(expect.get("missing_aspect", "")).strip():
        raise CaseError("a refusal case must name expect.missing_aspect")
    adjacent = expect.get("adjacent_year_figure")
    if adjacent and not str(expect.get("adjacent_year_label", "")).strip():
        raise CaseError(
            "an adjacent_year_figure must name expect.adjacent_year_label, so the canary "
            "can tell a year-trap assertion from a comparison"
        )

    tags_raw = payload.get("tags", [])
    if isinstance(tags_raw, str) or not isinstance(tags_raw, (list, tuple)):
        raise CaseError("field 'tags' must be a list of strings")
    follow_up = payload.get("follow_up_of")
    if follow_up is not None and (not isinstance(follow_up, str) or not follow_up.strip()):
        raise CaseError("field 'follow_up_of' must be a case id when present")

    return Case(
        case_id=str(payload["id"]).strip(),
        question=str(payload["question"]),
        expect=dict(expect),
        tags=tuple(str(tag) for tag in tags_raw),
        follow_up_of=follow_up.strip() if isinstance(follow_up, str) else None,
        line_no=line_no,
    )


def _follow_up_chain(cases: Sequence[Case], case: Case) -> list[Case]:
    """Every ancestor of ``case``, oldest first — the history a follow-up replays.

    A ``follow_up_of`` naming an absent id raises: an unresolvable chain would otherwise run
    the follow-up as a standalone question and quietly score the wrong behaviour.
    """

    by_id = {item.case_id: item for item in cases}
    chain: list[Case] = []
    cursor: str | None = case.follow_up_of
    guard = 0
    while cursor:
        guard += 1
        if guard > len(cases) + 1:
            raise CaseError(f"the follow-up chain for {case.case_id!r} is circular")
        ancestor = by_id.get(cursor)
        if ancestor is None:
            raise CaseError(
                f"case {case.case_id!r} follows up {cursor!r}, which is not in the dataset"
            )
        if ancestor.case_id not in {item.case_id for item in chain}:
            chain.append(ancestor)
        cursor = ancestor.follow_up_of
    chain.reverse()
    return chain


# --- running one case ---------------------------------------------------------------


@dataclass
class CaseResult:
    """Everything one case's run produced, and what was scored from it.

    ``error_kind`` separates two findings the exit codes treat differently: ``dataset``
    means the golden set is broken (exit 2), ``run`` means the graph failed on a golden
    question (exit 1, a regression).
    """

    case: Case
    answer_md: str = ""
    plan_kind: str = ""
    route: str = ""
    refusal_reason: str | None = None
    missing_aspect: str | None = None
    grounded: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
    ranks: list[tuple[str, int]] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    #: The tokens this case's run really cost, ``(in, out)`` read off the envelope's own
    #: ``node_trace`` — the same numbers ``@trace_node`` recorded and ``app.py``'s
    #: ``_trace_payload`` persists. It is measured truth, never an estimate of prompt
    #: length, so the pacer's projection is a cost this pipeline has actually paid.
    tokens: tuple[int, int] = (0, 0)
    latency_ms: float = 0.0
    history_turns: int = 0
    history_note: str = ""
    error: str | None = None
    error_kind: str | None = None

    retrieval_hit: bool | None = None
    retrieval_detail: str = ""
    retrieval_observed: bool | None = None
    citation_ok: bool | None = None
    citation_observed: bool | None = None
    figure_ok: bool | None = None
    figure_observed: bool | None = None
    missing_figures: list[str] = field(default_factory=list)
    wrong_year: bool = False
    wrong_year_detail: str = ""
    #: Which of the three canary states this case's run landed in. ``""`` when the case
    #: names no adjacent-year figure, so the field is only ever read for a canary case.
    wrong_year_state: str = ""
    #: The claim text the verifier produced *before* its trim, read off the verify node's
    #: log — the text the canary judges, because a wrong-year claim the pipeline caught and
    #: removed never appears in ``answer_md`` at all.
    pretrim_claims: list[str] = field(default_factory=list)
    #: How those claim texts were obtained, so the report can say what it measured.
    pretrim_note: str = ""
    #: The reason ``analyze_question``'s ``planner_degraded`` warning carried, or ``""`` when
    #: the planner did not degrade. Populated from the same capture-wrap as
    #: ``pretrim_claims``, so the refusal the run produced can be told apart from the one the
    #: question earned: a degraded plan is an infrastructure fact, not a semantic verdict.
    degraded_reason: str = ""
    refusal_ok: bool | None = None
    refusal_observed: bool | None = None
    refusal_detail: str = ""
    plan_ok: bool | None = None
    plan_detail: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def infrastructure_degraded(self) -> bool:
        """Whether this case never got a verdict because the planner could not run.

        Not a failure and not a pass: the case was *unmeasured*, so it is reported as its
        own category rather than folded into a rate in either direction.
        """

        return _infrastructure_degradation(self.degraded_reason)

    @property
    def token_cost(self) -> int:
        """This case's total tokens: prompt plus completion, as the providers count them.

        Both halves count against a TPM *and* a TPD ceiling — a completion token is a token
        in every published table — so the pacer sums them rather than picking the one a
        rate limit happens to name first.
        """

        return int(self.tokens[0]) + int(self.tokens[1])

    @property
    def failed(self) -> bool:
        """Whether this case failed a check measurable in the current mode."""

        return any(
            (
                self.retrieval_hit is False,
                self.citation_ok is False,
                self.figure_ok is False,
                self.wrong_year,
                self.refusal_ok is False,
                self.plan_ok is False,
            )
        ) or (self.error is not None and self.error_kind == "run")


def _compact(text: str) -> str:
    """Text without spaces, so the corpus's ``$ 211,915`` matches ``$211,915``."""

    return text.translate(_SPACE_TRANSLATION)


def _figure_key(text: str) -> str:
    """A printed figure reduced to the digits that carry its value.

    Four classes of difference between two spellings of the same figure are dropped, and
    only four:

    * **whitespace** — every space in :data:`_FIGURE_SPACES`. The corpus prints
      ``$ 211,915`` and ``$\\u2009211,915``; an answer following P4 rule 2 prints the same
      digits with a narrow no-break space, or bold around them, or a thin space before the
      unit. Spacing is typography, never value.
    * **markdown emphasis** — ``**`` bold and ``*`` italic. An asterisk is never a digit.
    * **grouping and currency furniture** — the thousands separator and the ``$``. The
      golden set names figures in the corpus's own form (``$211,915``) while a model is
      free to print the same number bare or with different grouping.
    * **the percent marker** — a rate's unit sign, never a digit of the rate.

    What is *kept* is what makes the comparison a figure check rather than "any number
    matches": the digit string itself, in its printed order, and every decimal point. So
    ``211,915`` stays ``211915`` and ``9.68`` stays ``9.68``, while ``$198,270`` and
    ``$211,915`` stay two different keys — which is what the wrong-year canary is built on,
    and what keeps a genuinely absent figure MISSING.
    """

    return text.translate(_FIGURE_TRANSLATION)


#: Comparison over a set, built once per answer rather than once per expectation: a case
#: with two expected figures scans the text's numbers once, not twice.
def _figure_tokens(text: str) -> frozenset[str]:
    """Every number printed in ``text`` as its normalised key, as a set.

    Built once per answer rather than once per expectation, so a case naming two expected
    figures scans the answer's numbers once.
    """

    return frozenset(_figure_key(match.group(0)) for match in _FIGURE_TOKEN_IN_TEXT.finditer(text))


def _contains_figure(text: str, figure: str) -> bool:
    """Whether ``figure`` appears in ``text``, in any spelling of the same number.

    Two readings, and the second is the one the measured run needed:

    * the **literal** form is tried first, so a verbatim match never depends on the
      normalisation being right and a non-numeric expectation (a segment name such as
      ``Intelligent Cloud``) keeps exactly the reading it always had;
    * then the **normalised** form: both sides are reduced by :func:`_figure_key` and the
      expected key is looked up among the numbers the text actually prints.

    The second reading is token-wise rather than a substring test over space-stripped
    text, which is what makes it safe. ``**$ 211,915 million**`` and ``$ 211,915``
    (narrow no-break space after the symbol, thin space before the unit, bold around the
    whole figure) both match ``$211,915`` — and ``$1,198,270`` does *not* match
    ``$198,270``, because that is one printed number, so it yields the token ``1198270``.

    An empty ``figure`` is never present in anything. What stays *absent* under both
    readings is what a genuinely missing answer prints: no number in it reduces to the
    expected one, so it is reported MISSING exactly as before.
    """

    if not figure:
        return False
    if figure in text:
        return True
    key = _figure_key(figure)
    # A non-numeric expectation — a segment name — has no key to look up. It either matched
    # literally above or it is absent; dropping furniture from letters would invent matches.
    if not key or not _DIGIT_FIGURE.fullmatch(key):
        return False
    return key in _figure_tokens(text)


#: One markdown table row: a line that opens and closes with a pipe (``| a | b |``). A row
#: is a structured record, not a sentence — its cells each belong to a column, and the
#: column's header is what makes a figure attributable to a period.
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
#: A cell of the alignment row that separates a table's header from its body (``|---|``):
#: only dashes, colons and spaces, never a label or a figure.
_TABLE_CELL_ALIGNMENT = re.compile(r"^[\s:-]*$")


def _sentences(text: str) -> list[str]:
    """The answer's sentences: markdown line breaks and sentence terminators both split."""

    chunks: list[str] = []
    for paragraph in text.replace("\r", "").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        buffer = ""
        for char in paragraph:
            buffer += char
            if char in ".!?":
                chunks.append(buffer)
                buffer = ""
        if buffer.strip():
            chunks.append(buffer)
    return [chunk for chunk in (item.strip() for item in chunks) if chunk]


def _row_cells(line: str) -> list[str]:
    """One markdown table row as its stripped cells (``| a | b |`` -> ``["a", "b"]``)."""

    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_label_cell(cell: str) -> bool:
    """Whether a table cell labels rather than states a figure (``Revenue``, ``FY2023``)."""

    return bool(cell) and not _FIGURE_IN_TEXT.search(cell)


def _is_alignment_row(cells: Sequence[str]) -> bool:
    """True for the ``|---|---|`` row that separates a table's header from its body."""

    return all(cell and _TABLE_CELL_ALIGNMENT.fullmatch(cell) for cell in cells)


def _assertions(answer_md: str) -> list[tuple[str, str]]:
    """Every place the answer prints a figure, with the label the layout ties it to.

    Prose is read exactly as it always was: a sentence is one place a figure can be
    asserted, and the sentence itself is all the attribution there is.

    A contiguous markdown table is read as a **structured record** instead. Its header row
    names each column (``| Metric | FY2023 | FY2022 |``), so a figure in a body cell is
    asserted for the period that cell's *own* column names — not for whatever the rest of
    the row happens to print. The label returned with a figure is therefore the column
    header above it plus the row's label cells, and never the row's other figures: three
    year columns under three correct headers are three attributed figures, which is a
    margin trend being right about every year rather than a year trap.

    A row with no header above it has no column to attribute to — a single row that
    travelled on its own, the way a pre-trim claim does — so it is read as the row text it
    is, which is the only reading it can carry. The result is in document order, so the
    first contaminated assertion is still the one a reader finds first.
    """

    units: list[tuple[str, str]] = []
    rows: list[list[str]] = []

    def flush_table() -> None:
        if rows:
            units.extend(_table_assertions(rows))
            rows.clear()

    for line in answer_md.replace("\r", "").split("\n"):
        if _TABLE_ROW.match(line):
            rows.append(_row_cells(line))
            continue
        flush_table()
        units.extend((sentence, "") for sentence in _sentences(line))
    flush_table()
    return units


def _table_assertions(rows: list[list[str]]) -> list[tuple[str, str]]:
    """One table's figures, each attributed to its own column's header.

    The header is the first row that labels something — the ``|---|---|`` alignment row and
    any row of bare figures carry no label, so neither can name a column. Body rows are the
    rest; the header row's own cells are read against the row's labels, as they are when
    there is no header at all.
    """

    body = [cells for cells in rows if not _is_alignment_row(cells)]
    header = body[0] if len(body) > 1 and any(_is_label_cell(cell) for cell in body[0]) else []

    units: list[tuple[str, str]] = []
    for row_index, cells in enumerate(body):
        labels = " ".join(cell for cell in cells if _is_label_cell(cell))
        for ordinal, cell in enumerate(cells):
            if not _FIGURE_IN_TEXT.search(cell):
                continue
            column = header[ordinal] if header and row_index > 0 and ordinal < len(header) else ""
            units.append((cell, " ".join(part for part in (labels, column) if part)))
    return units


def _presented_as_answer(answer_md: str, figure: str, year: str) -> bool:
    """Whether the answer asserts ``figure`` as the asked-for figure's value.

    The adjacent-year trap is a column problem: in the printed income statement the prior
    year's figure sits one cell to the left of the asked year's, so a year-blind read picks
    the wrong number from the right row. Each place the answer prints a figure is judged on
    its own, and a place clears only when it is explicitly comparative — either by
    comparative language or by naming the figure's own year. "Revenue was $198,270
    million" is the trap; "up from $198,270 million in FY2022" is the answer being right
    about both years.

    Where the figure is printed decides what it can be judged against, so the place is what
    carries the attribution:

    * **a sentence** is judged whole — the sentence text is the only context its figures
      have, exactly as before;
    * **a table cell** is judged against the header of its own column plus its row's label
      cells. A row's figures are separate assertions, so three year columns under three
      correct headers are clean, while a figure that sits under a header naming another
      year is asserted as that year's value — the trap, read structurally rather than
      missed because the row also carries the asked year somewhere else in its width.

    A table thus reads *more* strictly than prose rather than less: the asked year's figure
    printed in another cell of the same row no longer vouchsafes its neighbour.
    """

    if not figure:
        return False
    for text, label in _assertions(answer_md):
        if not _contains_figure(text, figure):
            continue
        if year and year in text:
            continue
        if year and year in label:
            continue
        if any(marker in text.lower() for marker in _COMPARATIVE_MARKERS):
            continue
        return True
    return False


def _states_figure(text: str) -> bool:
    """Whether ``text`` prints a figure at all — a dollar amount or a percentage.

    A year label is not a figure: an answer that says only "FY2023" has asserted no number
    for the canary to check, which is a different finding from a clean one.
    """

    return bool(_FIGURE_IN_TEXT.search(text))


@contextmanager
def _capturing_node_signals() -> Iterator[dict[str, Any]]:
    """Read what the nodes logged during one ``invoke()``: two signals, two loggers.

    The first is the claim text the verifier writes *before* it trims a claim away. The
    verify node logs every claim its deterministic pre-pass rejects, as the claim's own text,
    before ``trim_answer`` removes it from ``answer_md``. That logged text is the claim the
    pipeline produced — and it is therefore the text the canary has to judge: a wrong-year
    claim the verifier caught and trimmed never reaches the final answer, so a canary reading
    only ``answer_md`` scores the pipeline's best defence as a clean bill of health.

    The second is the planner's degradation reason. ``analyze_question`` logs one warning
    when a plan could not be produced, and that reason starts with the mapped error's class
    name — which is what separates "the gateway said no" from "the model answered". Both are
    read off the loggers' own records rather than inferred from the answer text, so a silent
    node leaves its bucket empty and the report says so rather than guessing. The handler is
    removed from every logger in ``finally``, so a run that raises leaves nothing installed.
    """

    claims: list[str] = []
    degradations: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.name == ANALYZE_LOGGER_NAME:
                # ``analyze.py:_degraded_plan`` logs ``"planner_degraded: %s; classifying
                # out_of_scope with no missing_aspect (question=%r)"`` — matched on the prefix
                # rather than the whole template, because the reason is the first argument.
                if PLANNER_DEGRADED_MARKER not in str(record.msg):
                    return
                if not isinstance(record.args, tuple) or not record.args:
                    return
                if not isinstance(record.args[0], str):
                    return
                degradations.append(record.args[0])
                return
            # Otherwise the verifier's logger. ``record.msg`` is the format template the
            # node logged with — ``"%s pre-pass trimmed %d claim(s): %s"`` — not the
            # interpolated text, so the record is identified by it and its arguments read
            # positionally: the node name, the count, then the claim list.
            if "pre-pass trimmed" not in str(record.msg) or not isinstance(record.args, tuple):
                return
            if len(record.args) != 3 or not isinstance(record.args[2], list):
                return
            for entry in record.args[2]:
                if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], str):
                    claims.append(entry[1])
                elif isinstance(entry, str):
                    claims.append(entry)

    installed = [logging.getLogger(name) for name in (VERIFY_LOGGER_NAME, ANALYZE_LOGGER_NAME)]
    handler = _Capture(level=logging.WARNING)
    for logger in installed:
        logger.addHandler(handler)
    try:
        yield {"claims": claims, "degraded": degradations}
    finally:
        for logger in installed:
            logger.removeHandler(handler)


def _infrastructure_degradation(reason: str) -> bool:
    """Whether the planner degraded for an infrastructure reason rather than a model verdict.

    The reason is the text ``analyze_question``'s ``planner_degraded`` warning carries, which
    begins with the mapped error's class name (ADR-007's gateway taxonomy). A ``RateLimited``,
    ``Auth``, ``ContextLength`` or ``Timeout`` names a provider condition: the planner was
    never reached, or was reached and never answered. That is the opposite of a schema
    violation, where the model *did* answer and its answer was unusable — P1's failure
    handling treats the latter as "no plan", and the harness treats the former as
    "no measurement", because nothing about the question was decided.
    """

    return reason.strip().split(":", 1)[0].strip() in _INFRASTRUCTURE_REASONS


def _texts_to_check(result: CaseResult) -> tuple[list[str], str]:
    """The claim texts the canary judges, and what they are.

    The claim text the synthesis node produced before the verifier trimmed it is the
    pipeline's actual assertion, so it is checked first: this is what makes the canary able
    to fail the run when the pipeline emits a wrong-year claim and the verifier catches it.
    The final answer is checked as well, because in a mode where nothing is trimmed it is
    the only text there is — and an answer still present in it is the failure a reader is
    looking for.
    """

    text = result.answer_md
    trimmed_away = TRIM_MARKER in text
    if result.pretrim_claims:
        source = "the pre-trim claim text the verify node logged"
        if trimmed_away:
            source += " (the final answer was trimmed)"
        return [*result.pretrim_claims, text], source
    if trimmed_away:
        # The answer is only the removed-claim marker and the verifier logged nothing: there
        # is no claim text to read, which is the state the canary cannot check.
        return [], "no claim text was logged before the claims were trimmed"
    return [text], "the final answer (nothing was trimmed)"


def _citation_entries(citations: Sequence[Any]) -> list[dict[str, Any]]:
    """The envelope's citations as plain dicts, whatever object shape they arrive in."""

    entries: list[dict[str, Any]] = []
    for citation in citations:
        if isinstance(citation, Mapping):
            entries.append(dict(citation))
            continue
        entries.append(
            {
                "ref_index": getattr(citation, "ref_index", None),
                "doc_id": getattr(citation, "doc_id", ""),
                "page": getattr(citation, "page", None),
                "section_path": list(getattr(citation, "section_path", ()) or ()),
                "snippet": getattr(citation, "snippet", ""),
            }
        )
    return entries


def _valid_citation(entry: Mapping[str, Any]) -> bool:
    """One citation is valid when it carries a 1-based ref_index it points with."""

    ref = entry.get("ref_index")
    return isinstance(ref, int) and not isinstance(ref, bool) and ref >= 1


def _ranks_and_texts(
    retrieval_debug: Sequence[Any],
) -> tuple[list[tuple[str, int]], list[str]]:
    """``[(chunk_id, fused_rank)]`` in rank order, plus the top hits' text."""

    pairs: list[tuple[int, str]] = []
    texts: list[str] = []
    for index, chunk in enumerate(retrieval_debug):
        if isinstance(chunk, Mapping):
            rank = chunk.get("fused_rank")
            chunk_id = str(chunk.get("chunk_id", "?"))
            text = f"{chunk.get('child_text', '')}\n{chunk.get('parent_text', '')}"
        else:
            rank = getattr(chunk, "fused_rank", None)
            chunk_id = str(getattr(chunk, "chunk_id", "?"))
            text = f"{getattr(chunk, 'child_text', '')}\n{getattr(chunk, 'parent_text', '')}"
        if not isinstance(rank, int) or isinstance(rank, bool):
            rank = index
        pairs.append((rank, chunk_id))
        if index < HIT_RANK:
            texts.append(str(text))
    pairs.sort()
    return [(chunk_id, rank) for rank, chunk_id in pairs], texts


def _route_of(plan_kind: str) -> str:
    """The routing token the plan kind takes (ADR-001's table)."""

    return {"out_of_scope": "refuse", "multi_part": "decompose_question"}.get(
        plan_kind, "hybrid_retrieve"
    )


# --- the token-budget pacer ---------------------------------------------------------


@dataclass
class _Projection:
    """What the pacer thinks starting one case will cost, before it starts.

    ``wait_seconds`` is how long the harness intends to sleep, not how long the run took:
    a reader can tell a pacer from a hang only if the intention is written down. ``reason``
    is the arithmetic behind it, printed verbatim.
    """

    tokens: int
    wait_seconds: float
    reason: str

    def __bool__(self) -> bool:
        """A projection with no cost and no wait is nothing to report."""

        return self.tokens > 0 or self.wait_seconds > 0.0


class _TokenPacer:
    """Paces one sweep against a provider's per-minute and per-day token ceilings.

    The whole design, in one sentence: a rolling window of what this run has actually
    spent, and a projection read from the cost a real prior case really paid. There is no
    estimate-from-prompt-length anywhere, because the measurement is already in the
    result: every case's LLM calls land in ``node_trace`` as ``tokens_in`` / ``tokens_out``,
    so the pacer reads the same numbers ``@trace_node`` recorded.

    Why a projection and not a budget check: the ceiling is rolling, so the honest question
    is not "is the window full" but "how long until a request of this size fits". The pacer
    answers that, waits exactly that long in whole seconds, and degrades to "run it and let
    the provider's own 429 decide" the moment its own numbers stop making sense.

    Two ceilings, and the daily one is the harder wall. The per-minute ceiling is smoothable
    — sleep until the span's worth of spend clears. The per-day ceiling cannot be smoothed
    at all: the run spends it and the day is over, so the harness refuses to start a run
    that would spend more than the day's usable share, or says so out loud first.
    """

    def __init__(
        self,
        *,
        tpm_limit: int,
        tpd_limit: int,
        mock_mode: bool,
        allowed_today: int,
        daily_gate: bool = True,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.tpm_limit = int(tpm_limit)
        self.tpd_limit = int(tpd_limit)
        self.mock_mode = mock_mode
        #: What the day's allowance leaves for this run, after whatever already spent it.
        #: A run that asks for more is refused in :meth:`check_daily`, not throttled: a
        #: sleep cannot put tokens back into a day.
        self.allowed_today = int(allowed_today)
        #: Whether the daily gate may refuse at all. ``EVAL_TPD_OVERRIDE=1`` turns it off —
        #: the escape hatch for a user who has decided the day's quota is worth spending.
        #: It changes nothing about the report: the projection still prints either way.
        self.daily_gate = bool(daily_gate)
        self._now = now
        self._sleep = sleep
        #: ``(monotonic timestamp, token cost)`` for every case finished in this run, in
        #: order. The per-minute ceiling is *rolling*, so the window is a list of stamps
        #: and not a bucket reset on the minute: a sweep running long straddles the
        #: boundary, and a bucket would forget nothing and let the projection overshoot.
        self._spend: list[tuple[float, int]] = []
        self.waits: list[float] = []

    # -- reading the provider's own numbers --------------------------------------------

    def _minutes_window(self) -> list[tuple[float, int]]:
        """The spend still inside the rolling per-minute window.

        An entry clears 60 seconds after it was stamped, which is the same span the
        provider counts against. Entries that have rolled out are dropped here, so the
        window forgets on its own and a long run cannot accumulate a debt that never
        clears.
        """

        now = self._now()
        return [item for item in self._spend if now - item[0] < 60.0]

    def _seconds_until_token_slot(self, tokens: int) -> float:
        """How long until a request of ``tokens`` fits inside the rolling window.

        The window drains from its oldest entry forward, so the moment a request fits is
        the moment the earliest entries have cleared: for each entry, if dropping it makes
        room, the wait is that entry's own expiry. A projection larger than the whole
        ceiling never fits at all, and says so rather than returning a wait that cannot
        happen.
        """

        if tokens <= 0 or self.tpm_limit <= 0:
            return 0.0
        if tokens > self.tpm_limit:
            # A request this size cannot fit in any window the rate limit allows. The
            # provider's own 429 is the honest handler here, and the pacer says so rather
            # than pretending a sleep can shrink a prompt.
            return math.inf
        window = self._minutes_window()
        used = sum(cost for _, cost in window)
        if used + tokens <= self.tpm_limit:
            return 0.0
        # Oldest first: each entry clears 60s after it was stamped. Dropping every entry
        # always makes room — what is left of the window is ``tokens``, which fits by the
        # check above — so this loop always returns while the window is well-formed.
        running = used
        for stamp, cost in window:
            running -= cost
            if running + tokens <= self.tpm_limit:
                return max(0.0, stamp + 60.0 - self._now())
        # Reached only if the window holds entries older than its own span, which is the
        # shape a bucket-style window would produce and this one does not. The safe
        # direction is to wait: dispatching a request that overshoots is the failure the
        # pacer exists to prevent, while waiting too long is only slow.
        return max(0.0, window[-1][0] + 60.0 - self._now())

    # -- the two questions the harness asks --------------------------------------------

    def _project(self, tokens: int) -> _Projection:
        """What starting a case costing ``tokens`` would take, right now.

        A case that costs nothing — an ancestor replayed for history, a degraded refusal, a
        mock run — has no projection: there is nothing to wait for, so silence is the
        correct answer and not an omission.
        """

        if tokens <= 0:
            return _Projection(0, 0.0, "")
        used = sum(cost for _, cost in self._minutes_window())
        seconds = self._seconds_until_token_slot(tokens)
        if seconds <= 0.0:
            return _Projection(tokens, 0.0, "")
        if not math.isfinite(seconds):
            return _Projection(
                tokens,
                0.0,
                f"~{tokens} tokens is larger than the whole {self.tpm_limit} TPM window, "
                "so no wait can make it fit: it runs, and the provider's own 429 decides",
            )
        waited = min(seconds, TPM_WAIT_CAP_SECONDS)
        return _Projection(
            tokens,
            waited,
            f"~{tokens} tokens against {used} in the rolling {self.tpm_limit} TPM window — "
            f"waiting {waited:.1f}s for the window to clear",
        )

    def _refuse_reason(self, tokens: int) -> str:
        """Why a projected run must not start, or ``""`` when it may."""

        shortfall = max(0, tokens - self.allowed_today)
        if shortfall <= 0:
            return ""
        return (
            f"~{tokens} tokens projected against the day's ~{self.allowed_today:,} of "
            f"{self.tpd_limit:,} — {shortfall:,} tokens short. A daily ceiling cannot be "
            f"slept off, and a run that spends past it cannot be re-run: refusing to "
            f"start, so today's quota survives. Set {TPD_OVERRIDE_HINT} to spend it anyway"
        )

    # -- the entry points the harness calls --------------------------------------------

    def check_daily(self, tokens: int) -> str:
        """Whether a run of ``tokens`` may start, as a refusal reason or ``""``.

        This is a *pre-flight* gate, not a throttle: a daily ceiling is a wall the day
        erases, so the honest options are to spend knowingly or not to spend at all. A run
        that is refused never starts, which keeps its tokens available for tomorrow. Under
        ``MOCK_LLM=1`` the projection is zero tokens, so nothing is ever refused.
        """

        if self.mock_mode or tokens <= 0 or not self.daily_gate:
            return ""
        return self._refuse_reason(tokens)

    def gate(
        self, tokens: int, *, label: str = "", quiet: bool = False, stream: Any = None
    ) -> float:
        """Wait for room before a case starts, and return the seconds actually waited.

        Zero means no wait, which is the common answer: a projection that fits is served
        immediately rather than made to wait on a budget it is not close to. The pacer's
        wait is reported even when it is zero, so a reader never has to guess whether a
        slow case was paced or hung.
        """

        if self.mock_mode or tokens <= 0:
            return 0.0
        projection = self._project(tokens)
        if projection.wait_seconds <= 0.0:
            self._note(quiet, stream, projection, label)
            return 0.0
        waited = self._sleep_for(projection.wait_seconds)
        self.waits.append(waited)
        self._note(quiet, stream, projection, label)
        return waited

    def _sleep_for(self, seconds: float) -> float:
        """Sleep in whole seconds, so the window's own 60s entries clear exactly.

        A fractional sleep under the expiry time would wake while the window is still full
        and dispatch a request the provider has already refused — the pacing would work and
        the 429 would arrive anyway.
        """

        whole = max(1.0, math.ceil(seconds))
        self._sleep(whole)
        return whole

    @staticmethod
    def _note(quiet: bool, stream: Any, projection: _Projection, label: str) -> None:
        """One line of pacing arithmetic, on the run's own stream, never in quiet mode."""

        if quiet or not projection:
            return
        print(f"  pacing         : {projection.reason} {label}".rstrip(), file=stream)

    def record(self, tokens: int) -> None:
        """Stamp a finished case's real cost into the rolling window."""

        if self.mock_mode or tokens <= 0:
            return
        self._spend.append((self._now(), int(tokens)))

    def projected_cost(self, results: Mapping[str, Any]) -> int:
        """What the next case is expected to cost: the highest measured prior in this run.

        The maximum rather than the mean, for a reason the arithmetic forces: the ceiling is
        a per-request ceiling as well as a rolling one, and the case that would overshoot
        first is the *most* expensive one. Averaging in a cheap case would under-predict it
        and dispatch a request the provider then has to refuse.

        With no prior case there is nothing measured to project from, so zero: the first
        case of a run dispatches un-paced, which costs at most one overshoot on the same
        reactive path the 429 already carries.
        """

        if self.mock_mode:
            return 0
        costs = [
            result.token_cost for result in results.values() if getattr(result, "token_cost", 0)
        ]
        return max(costs) if costs else 0


#: What a case *starts* costing, measured on this corpus (``data/index``, 384-dim
#: fastembed) against Groq's ``openai/gpt-oss-*`` keyed models. P1 2,038 + P3 5,900 + P5
#: 2,100 is one in-scope case; a case whose follow-up needs a P2 rewrite adds ~600. A
#: refusal case gates at the fiscal-year boundary and never reaches the provider at all.
#:
#: These are the *first* projection of a run — used to estimate a fresh keyed run's total
#: before anything has been measured. Once cases have run, their real ``node_trace`` cost
#: replaces these numbers, so a run is forecast on measured truth wherever one exists.
CANONICAL_CASE_TOKENS = 10_038
#: The extra a follow-up costs: P2's rewrite call, which replays the ancestor's history.
FOLLOW_UP_CASE_TOKENS = 600


def _projected_run_cost(cases: Sequence[Any], prior_cost: int = 0) -> int:
    """The projected total token cost of a requested set, for the pre-flight report.

    ``prior_cost`` is a measured cost this run has already established, used per case it has
    actually run — measured truth beats a table — with the table's canonical figure standing
    in for anything not yet measured. The daily gate needs a *total* because the daily
    ceiling is not smoothable: a per-case projection would refuse the run one case at a time
    and still spend the day.
    """

    if not cases:
        return 0
    total = 0
    for case in cases:
        if getattr(case, "is_refusal", False):
            # A refusal case gates on the fiscal years the corpus covers (ADR-010) and
            # never reaches a provider, so it costs nothing to project.
            continue
        per_case = prior_cost or CANONICAL_CASE_TOKENS
        if getattr(case, "follow_up_of", None):
            per_case += FOLLOW_UP_CASE_TOKENS
        total += per_case
    return total


def _env_int(name: str, default: int) -> int:
    """A positive integer from the environment, or ``default`` when it is not usable.

    A typo or a blank must not silently become zero — a zero ceiling would make every
    projection overshoot and every run wait forever — so anything unparseable or
    non-positive falls back to the documented default.
    """

    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_flag(name: str) -> bool:
    """Whether an environment flag is set to ``1``, ``true`` or ``yes``."""

    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _tpm_limit_from_env(default: int = DEFAULT_TPM_LIMIT) -> int:
    """The per-minute ceiling this run is held to, from ``EVAL_TPM_LIMIT`` or the default.

    The default is Groq's published free tier (8,000 TPM). A paid tier or another provider
    raises it by exporting a larger number; nothing else has to change.
    """

    return _env_int("EVAL_TPM_LIMIT", default)


def _allowed_today_from_env(limit: int) -> int:
    """How much of the daily allowance this run may spend before it is refused.

    ``EVAL_TPD_BUDGET`` is a token count and takes precedence — the case where a key has
    already spent part of its day, which is exactly the number to hand over.
    ``EVAL_TPD_FRACTION`` (default :data:`DEFAULT_TPD_FRACTION`) is the common case: a
    fraction of the ceiling.
    """

    raw = os.environ.get("EVAL_TPD_BUDGET", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return max(1, int(DEFAULT_TPD_LIMIT))
    try:
        fraction = float(os.environ.get("EVAL_TPD_FRACTION", "").strip() or DEFAULT_TPD_FRACTION)
    except ValueError:
        fraction = DEFAULT_TPD_FRACTION
    if fraction <= 0.0:
        return 0
    return max(0, int(limit * fraction))


def _daily_gate_enabled() -> bool:
    """Whether the pre-flight daily gate may refuse a run. ``EVAL_TPD_OVERRIDE=1`` says no.

    The gate never refuses under ``MOCK_LLM=1`` at all: a mock case costs zero tokens, so
    its projected spend is zero and there is nothing to refuse.
    """

    return not _env_flag("EVAL_TPD_OVERRIDE")


def _max_measured_cost(results: Mapping[str, Any]) -> int:
    """The highest token cost this run has actually paid so far, or zero.

    Read from the finished results, so a follow-up's projection is based on what the
    earlier cases really cost rather than on a table. Zero means nothing has run yet —
    the state a fresh run is in, and the state the canonical table stands in for.
    """

    costs = [result.token_cost for result in results.values() if getattr(result, "token_cost", 0)]
    return max(costs) if costs else 0


def _print_preflight(
    projected: int,
    pacer: _TokenPacer,
    cases: Sequence[Any],
    *,
    quiet: bool = False,
    stream: Any = None,
) -> None:
    """The run's projected spend, and what it means against the day's allowance.

    A silent pacer is indistinguishable from a hang, and a pacer that lets a user hand over
    half their daily quota without a word is a footgun: so the projection is printed before
    the first case dispatches, with the fraction of the day it would spend spelled out.
    """

    if quiet or pacer.mock_mode:
        return
    head = f"  tpm ceiling     : {pacer.tpm_limit:,} tokens/min"
    print(head, file=stream)
    if not projected:
        print("  projected spend : 0 tokens (nothing meets a provider)", file=stream)
        return
    share = projected / pacer.tpd_limit if pacer.tpd_limit else 0.0
    print(
        f"  projected spend : ~{projected:,} tokens for {len(cases)} case(s) — "
        f"{share:.0%} of the day's {pacer.tpd_limit:,} allowance",
        file=stream,
    )


# --- the five metrics, per case ------------------------------------------------------


def _score_case(case: Case, result: CaseResult, scopes: Mapping[str, _Scope]) -> None:
    """Apply docs/TESTING-AND-EVAL.md's five metrics to one finished run, in place."""

    expect = case.expect
    answer = result.answer_md

    # Did the planner run at all? When it could not, this run's ``out_of_scope`` is P1's
    # no-plan output rather than a classification of the question, and the metrics that read
    # the answer have nothing to read.
    infrastructure = _infrastructure_degradation(result.degraded_reason)

    # --- retrieval hit-rate@5: is the expected figure inside the fused top-5's text? ---
    markers = [str(m) for m in expect.get("chunk_markers", []) or [] if str(m).strip()]
    if not markers:
        result.retrieval_hit = None
        result.retrieval_detail = "the case names no chunk marker (nothing to measure)"
    elif not result.retrieved:
        result.retrieval_hit = None
        result.retrieval_detail = "no retrieval ran (the plan routed straight to refusal)"
    else:
        top_text = _compact("\n".join(result.retrieved))
        hit = any(_compact(marker) in top_text for marker in markers)
        result.retrieval_observed = hit
        result.retrieval_detail = (
            f"expected figure {'present' if hit else 'absent'} in the fused top-{HIT_RANK} texts"
        )
        if scopes["retrieval_hit_rate@5"].covers(case):
            result.retrieval_hit = hit
        else:
            result.retrieval_hit = None  # observed, printed, not scored

    # --- citation presence: every factual answer carries >= 1 valid marker ------------
    if infrastructure:
        # No plan, so no retrieval, so no answer to carry a marker. Observed, not scored.
        result.citation_observed = bool(result.citations) and all(
            _valid_citation(entry) for entry in result.citations
        )
        result.citation_ok = None
    elif not case.wants_citation:
        result.citation_ok = None
        result.citation_observed = None
        result.notes.append("citations are not expected for this case")
    elif case.is_refusal:
        # A refusal carries no citations by design (docs/SPEC.md §9): its answer is correct,
        # not ungrounded. Presence is not a bar a refusal case can miss.
        result.citation_ok = None
        result.citation_observed = None
        result.notes.append("a refusal carries no citations by design")
    elif not scopes["citation_presence"].covers(case):
        result.citation_observed = bool(result.citations) and all(
            _valid_citation(entry) for entry in result.citations
        )
        result.citation_ok = None
    else:
        result.citation_observed = bool(result.citations) and all(
            _valid_citation(entry) for entry in result.citations
        )
        result.citation_ok = result.citation_observed

    # --- figure match: the expected figure appears verbatim ---------------------------
    expected_figures = case.expected_appearances
    if not expected_figures:
        result.figure_ok = None
        result.figure_observed = None
        result.notes.append(
            "no expected figure for this case (summarization); figure match not scored"
        )
    elif infrastructure:
        # The graph routed to ``refuse`` before retrieval, so the figure was never reachable
        # from the answer: an absent figure here measures the quota, not the pipeline.
        result.missing_figures = list(expected_figures)
        result.figure_observed = False
        result.figure_ok = None
    elif not scopes["figure_match"].covers(case):
        result.missing_figures = [
            figure for figure in expected_figures if not _contains_figure(answer, figure)
        ]
        result.figure_observed = not result.missing_figures
        result.figure_ok = None
    else:
        result.missing_figures = [
            figure for figure in expected_figures if not _contains_figure(answer, figure)
        ]
        result.figure_observed = not result.missing_figures
        result.figure_ok = result.figure_observed

    # --- wrong-year contamination: the year-confusion canary --------------------------
    # Never gated on the mode: it reads the pipeline's own claim text, so it is fully live
    # under MOCK_LLM=1 and fully able to FAIL there. The one regression no mode may hide.
    adjacent_figure = expect.get("adjacent_year_figure")
    adjacent_label = str(expect.get("adjacent_year_label", "")).strip()
    if adjacent_figure:
        asked = expect.get("asked_year") or expect.get("asked_years")
        texts, source = _texts_to_check(result)
        state = WRONG_YEAR_CLEAN
        for text in texts:
            if _presented_as_answer(text, str(adjacent_figure), adjacent_label):
                state = WRONG_YEAR_CONTAMINATED
                result.wrong_year = True
                result.wrong_year_detail = (
                    f"{source}: the answer presents {adjacent_figure} as the asked-for "
                    f"figure (asked {asked})"
                )
                break
        if state == WRONG_YEAR_CONTAMINATED:
            result.wrong_year_state = WRONG_YEAR_CONTAMINATED
        elif texts and any(_states_figure(text) for text in texts):
            # Something was actually checked, and nothing in it asserted the wrong year.
            result.wrong_year_state = WRONG_YEAR_CLEAN
            result.pretrim_note = source
        else:
            # Nothing to check: an empty answer, or an answer trimmed to nothing but the
            # removed-claim marker, asserts no figure at all. "We could not look" is not
            # "nothing was wrong", and a rate never treats it as a pass.
            result.wrong_year_state = NOT_CHECKABLE
            result.pretrim_note = f"answer contained no figure (judged {source})"

    # --- refusal correctness: out-of-scope refused, in-scope answered ----------------
    if infrastructure:
        # The planner never decided anything: a 429, an auth failure or a transport error
        # stopped it, so ``out_of_scope`` here is the plan P1 emits when *no* plan could be
        # produced (``missing_aspect=None``, ``planner_degraded=True``), not a classification
        # of the question. Scoring it as one would read a quota exhaustion as a semantic
        # verdict — and as a wrong one, because the degradation lands on every class of case.
        # The refusal itself still stands as an observation: it is recorded, printed, and
        # never folded into a rate. This is the same ``not_checkable`` treatment the canary
        # gives an answer carrying no figure — "we could not look", never "nothing was wrong".
        result.notes.append(
            "the planner could not run (infrastructure), so the refusal carries no verdict "
            f"about this question: {result.degraded_reason}"
        )
    if case.is_refusal:
        wanted = str(expect.get("missing_aspect", "")).strip()
        got = (result.refusal_reason or "").strip()
        if not got:
            result.refusal_ok = False
            result.refusal_detail = "an out-of-scope question was answered, not refused"
        elif wanted and wanted != got:
            result.refusal_ok = False
            result.refusal_detail = f"refused, but missing_aspect={got!r} (expected {wanted!r})"
        else:
            result.refusal_ok = True
            result.refusal_detail = f"refused with missing_aspect={got!r}"
    elif result.refusal_reason:
        result.refusal_ok = False
        result.refusal_detail = f"an in-scope question was refused ({result.refusal_reason})"
    else:
        result.refusal_ok = True
        result.refusal_detail = "answered"
    result.refusal_observed = result.refusal_ok
    # A refusal class this mode cannot produce has no verdict: the whole answer to the
    # question is missing, not merely wrong.
    if not scopes["refusal_correctness"].covers(case):
        result.refusal_ok = None
        if case.is_refusal:
            result.notes.append(
                f"refusal correctness not measurable: {scopes['refusal_correctness'].reason}"
            )
    elif infrastructure:
        # Observed above and reported below, but excluded from the rate: nothing decided it.
        result.refusal_ok = None

    # --- routing check (not one of the doc's five metrics) ------------------------------
    # The doc pairs the comparison question with the decomposition path, so a case may name
    # the plan kind its route must take. This is not a scored metric and never appears in
    # the aggregate table: it is a per-case wiring assertion, the same thing
    # tests/test_graph.py asserts against the routing table, kept here so a golden run
    # reports the path it actually took rather than only the answer it produced.
    expected_kind = str(expect.get("expected_plan_kind", "")).strip()
    if not expected_kind:
        result.plan_ok = None
    else:
        result.plan_ok = result.plan_kind == expected_kind
        result.plan_detail = (
            f"plan.kind={result.plan_kind!r}"
            if result.plan_ok
            else f"plan.kind={result.plan_kind!r}, expected {expected_kind!r} "
            f"(route {result.route!r})"
        )


# --- the graph session ---------------------------------------------------------------


@dataclass
class _RunContext:
    """One shared session: the loaded document catalog, the mode, and its metric scope."""

    docs: list[Any]
    mock_mode: bool
    index_dir: Path
    scopes: dict[str, _Scope]


def _load_catalog(index_dir: Path) -> list[Any]:
    """The document catalog as the app builds it, from the real on-disk index."""

    from src.chains.state import DocSummary

    path = index_dir / CATALOG_NAME
    rows: list[Any] = []
    if path.is_file():
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [
        DocSummary(
            doc_id=str(row.get("doc_id", "")),
            filename=str(row.get("filename", "")),
            sections=tuple(
                tuple(str(part).strip() for part in str(item).split(" > "))
                for item in row.get("sections", [])
                if str(item).strip()
            ),
            page_count=int(row["page_count"]) if row.get("page_count") is not None else None,
        )
        for row in rows
        if isinstance(row, Mapping) and row.get("doc_id")
    ]


def _check_index(index_dir: Path) -> list[str]:
    """The index artifacts the harness needs; a missing one is a harness error."""

    return [
        f"missing index artifact: {index_dir / name}"
        for name in ("chunks.jsonl", "bm25.pkl", CATALOG_NAME)
        if not (index_dir / name).is_file()
    ]


def _history_for(ancestors: Sequence[Case], results: Mapping[str, CaseResult]) -> list[Any]:
    """The ``history`` a follow-up replays: real ``Turn`` objects built from real answers.

    The prior turns are the *actual* results of the earlier cases — the answer markdown and
    the citations the graph really produced — never a hand-written transcript, because a
    synthetic prior answer would test the follow-up against a fiction. The assistant turns
    carry ``cited_pages``/``entities`` from the earlier envelope's citations: that is
    exactly STRESS-TEST-RAG #3's pronoun-drift fix, and the only way Q8 can prove it
    resolved against what the previous answer was grounded in rather than against raw text.
    """

    from src.chains.state import Turn

    turns: list[Any] = []
    for ancestor in ancestors:
        prior = results.get(ancestor.case_id)
        if prior is None:
            raise CaseError(
                f"cannot build history for a follow-up of {ancestor.case_id!r}: that case "
                "has not run. Run the full dataset instead of --only-ing the follow-up."
            )
        turns.append(Turn(role="user", content=ancestor.question))
        if not prior.answer_md:
            continue
        entities = tuple(str(e) for e in ancestor.expect.get("entities", []) or [])
        turns.append(
            Turn(
                role="assistant",
                content=prior.answer_md,
                cited_pages=tuple(prior.citations),
                entities=entities,
            )
        )
    return turns


def _run_case(
    case: Case,
    ctx: _RunContext,
    ancestors: Sequence[Case],
    results: Mapping[str, CaseResult],
    pacer: _TokenPacer | None = None,
) -> CaseResult:
    """Run one case through ``invoke()`` and score it. Never raises for a graph failure.

    ``pacer`` is the token budget the sweep is held to. It is a parameter rather than a
    global so the pacer's clock and its sleeps can be injected in-process by the tests —
    a test that really slept would test the sleep, not the pacing. ``None`` runs un-paced,
    which is what the mock path and the ``_run_case``-without-a-budget callers want.
    """

    from src.chains.graph import invoke

    result = CaseResult(case=case)
    try:
        history = _history_for(ancestors, results)
    except CaseError as exc:
        result.error = str(exc)
        result.error_kind = "dataset"
        return result
    result.history_turns = len(history)
    if ancestors:
        result.history_note = (
            f"replayed {len(ancestors)} prior turn pair(s) as real Turn objects carrying "
            "cited_pages/entities from the earlier answers"
        )

    if pacer is not None:
        # The projection is the cost a *previous* case really paid, so the first case of a
        # run has nothing to pace against and dispatches immediately. That is the one gap
        # a rolling window cannot close on its own, and it costs at most one overshoot on
        # the gateway's own reactive path.
        projection = pacer.projected_cost(results)
        pacer.gate(projection, label=f"(before {case.case_id})", quiet=ctx.mock_mode)

    started = time.perf_counter()
    try:
        # The capture wraps the graph run, not the scoring: the claim text the verifier
        # logs, and the reason the planner degraded, are both emitted while the nodes run.
        with _capturing_node_signals() as captured:
            envelope = invoke(
                case.question, history=history or None, params=None, doc_catalog=ctx.docs
            )
        result.pretrim_claims = list(captured["claims"])
        result.degraded_reason = captured["degraded"][0] if captured["degraded"] else ""
    except Exception as exc:
        # A broad catch is deliberate: a graph run can fail through ADR-007's typed-error
        # taxonomy, through langgraph's own wrapper around a node failure, or through a
        # library that raised outside that taxonomy. A golden case that raises is the
        # run's finding (exit 1), so the harness records whichever class arrived and
        # carries on to the next case instead of ending the sweep on the first blow-up.
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        result.error = f"{type(exc).__name__}: {exc}"
        result.error_kind = "run"
        return result
    result.latency_ms = (time.perf_counter() - started) * 1000.0

    result.answer_md = str(getattr(envelope, "answer_md", "") or "")
    plan = getattr(envelope, "plan", None)
    result.plan_kind = str(getattr(plan, "kind", "") or "") if plan is not None else ""
    result.route = _route_of(result.plan_kind)
    result.refusal_reason = getattr(envelope, "refusal_reason", None)
    result.missing_aspect = getattr(plan, "missing_aspect", None) if plan is not None else None
    result.grounded = bool(getattr(envelope, "grounded", False))
    result.citations = _citation_entries(getattr(envelope, "citations", []) or [])
    result.ranks, result.retrieved = _ranks_and_texts(
        list(getattr(envelope, "retrieval_debug", []) or [])
    )
    result.nodes = [
        str(getattr(event, "node", "") or "")
        for event in (getattr(envelope, "node_trace", []) or [])
    ]
    result.tokens = _trace_tokens(getattr(envelope, "node_trace", []) or [])

    _score_case(case, result, ctx.scopes)

    if pacer is not None:
        # The window is stamped with what this case *cost*, not with what it was projected
        # to cost: the projection is a guess with a measured prior behind it, and the
        # measured truth is what the next case is paced against. A run that paced on its
        # own guesses would drift further from the provider's real window with every case.
        pacer.record(result.token_cost)
    return result


#: The ``NodeEvent`` field the token counts are read from, and the two halves of a
#: ``(tokens_in, tokens_out)`` pair. ``@trace_node`` lifts both off the
#: :class:`~src.utils.llm.ChatResult` a node's gateway call produced, so this is the
#: provider's own accounting and never a length estimate.
_TOKEN_FIELDS = ("tokens_in", "tokens_out")


def _trace_tokens(events: Sequence[Any]) -> tuple[int, int]:
    """``(tokens_in, tokens_out)`` summed over one run's whole ``node_trace``.

    Read defensively, because the trace is the one part of the envelope that exists only
    when instrumentation ran: an event missing its counts (a node that made no LLM call —
    ``retrieve``, ``refuse``, the gate) contributes zero, and a trace that is not the
    shape expected contributes zero rather than raising into a golden case's run. Summing
    the *whole* run is what the pacer needs, since the ceiling counts every call a
    question triggered, not the ones that produced the final answer.
    """

    totals = [0, 0]
    for event in events or []:
        for index, name in enumerate(_TOKEN_FIELDS):
            value = event.get(name) if isinstance(event, Mapping) else getattr(event, name, None)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            totals[index] += max(0, value)
    return totals[0], totals[1]


# --- per-case report -----------------------------------------------------------------


def _print_case(result: CaseResult, position: int, total: int, mock_mode: bool = False) -> None:
    """One case's block: what it ran on and everything the five metrics saw."""

    case = result.case
    print("=" * 78)
    print(f"[{position}/{total}] {case.case_id} (line {case.line_no})  {case.question!r}")
    print("=" * 78)
    if case.tags:
        print(f"  tags           : {', '.join(case.tags)}")
    if case.is_refusal:
        print(
            "  expected       : honest refusal, missing_aspect="
            f"{case.expect.get('missing_aspect')!r}"
        )
    elif case.expected_appearances:
        print(f"  expected       : {', '.join(case.expected_appearances)}")
    else:
        print("  expected       : (summarization — no single expected figure)")
    if result.history_turns:
        print(f"  history        : {result.history_turns} turn(s) — {result.history_note}")
    if result.error:
        print(f"  ERROR ({result.error_kind}) : {result.error}")
        print()
        return

    print(f"  plan.kind      : {result.plan_kind!r} -> route {result.route!r}")
    if _infrastructure_degradation(result.degraded_reason):
        # The plan that arrived is P1's no-plan output. Say so above the routing line, or a
        # reader takes ``out_of_scope`` for the class this question earned.
        print("  planner        : DEGRADED (no plan was produced — the gateway did not answer);")
        print(f"                   {result.degraded_reason}")
    if result.plan_ok is not None:
        state = "OK" if result.plan_ok else "MISMATCH"
        print(f"  routing check  : {state} — {result.plan_detail}")
    if result.refusal_reason:
        print(
            f"  refusal_reason : {result.refusal_reason!r} (plan missing_aspect="
            f"{result.missing_aspect!r})"
        )
    print(
        f"  grounded       : {result.grounded}   citations: {len(result.citations)}   "
        f"nodes: {len(result.nodes)} ({','.join(result.nodes) or 'none'})"
    )
    print(f"  latency        : {result.latency_ms:7.1f} ms")
    # Suppressed under MOCK_LLM=1: the mock's counts are scripted constants rather than a
    # provider's accounting, so printing them would report a spend that never happened.
    if not mock_mode and (result.tokens[0] or result.tokens[1]):
        print(
            f"  tokens         : {result.tokens[0]:,} in / {result.tokens[1]:,} out "
            f"({result.token_cost:,} total, from this run's node_trace)"
        )

    for figure in case.expected_appearances:
        state = "FOUND" if _contains_figure(result.answer_md, figure) else "MISSING"
        suffix = "  (observed, not scored in this mode)" if result.figure_ok is None else ""
        print(f"  expected       : {figure:<14} {state}{suffix}")
    if result.missing_figures:
        print(f"  missing        : {', '.join(result.missing_figures)}")

    if result.retrieval_hit is not None:
        print(f"  retrieval hit  : {result.retrieval_hit} ({result.retrieval_detail})")
    elif result.retrieval_detail:
        observed = ""
        if result.retrieval_observed is not None:
            observed = f" — observed {result.retrieval_observed}"
        print(f"  retrieval hit  : N/A ({result.retrieval_detail}{observed})")
    if result.ranks:
        print(
            "  fused ranks    : "
            + ", ".join(f"{rank}:{chunk_id.split('-p')[-1]}" for chunk_id, rank in result.ranks[:8])
        )
    else:
        print("  fused ranks    : (none — the plan routed straight to refusal)")

    if case.is_refusal:
        print(f"  refusal        : {result.refusal_detail}")
    adjacent_figure = case.expect.get("adjacent_year_figure")
    if adjacent_figure:
        adjacent_label = case.expect.get("adjacent_year_label", "")
        state = result.wrong_year_state or WRONG_YEAR_CLEAN
        print(
            f"  year canary    : {state} (asked {case.expect.get('asked_year')}; "
            f"{adjacent_label} figure {adjacent_figure})"
        )
        if result.wrong_year_detail:
            print(f"                   {result.wrong_year_detail}")
        elif result.pretrim_note:
            print(f"                   {result.pretrim_note}")

    for note in result.notes:
        print(f"  note           : {note}")
    print(f"  answer         : {result.answer_md or '(empty)'}")
    print()


# --- the aggregate report ------------------------------------------------------------

#: The result fields the five scoreable metrics read, by metric name.
_FIELD: dict[str, str] = {
    "retrieval_hit_rate@5": "retrieval_hit",
    "citation_presence": "citation_ok",
    "figure_match": "figure_ok",
    "wrong_year_contamination": "wrong_year",
    "refusal_correctness": "refusal_ok",
}


def _wrong_year_fires(results: Sequence[CaseResult]) -> int:
    """How many canary cases the run contaminated."""

    return sum(1 for r in results if r.wrong_year)


def _wrong_year_counts(results: Sequence[CaseResult]) -> tuple[int, int]:
    """``(checked, not_checkable)`` over the canary cases.

    A case is *checked* when the pipeline produced a figure the canary could read; it is
    ``not_checkable`` when the answer carried no figure at all, so there was nothing to
    judge. The two are reported apart so an empty answer can never be counted as a pass.
    """

    cases = [r for r in results if r.case.expect.get("adjacent_year_figure")]
    checked = sum(1 for r in cases if r.wrong_year_state == WRONG_YEAR_CLEAN)
    fired = _wrong_year_fires(cases)
    return checked + fired, len(cases) - checked - fired


def _rate_text(values: Sequence[bool | None]) -> str:
    """``n/m (0.xx)`` over the scored subset, or ``NA`` when nothing was scoreable."""

    scored = [value for value in values if value is not None]
    if not scored:
        return "NA"
    hits = sum(1 for value in scored if value)
    return f"{hits}/{len(scored)} ({hits / len(scored):.2f})"


def _metric_value(name: str, results: Sequence[CaseResult]) -> str:
    """One metric's measured value, exactly as the summary table prints it.

    The wrong-year canary reports ``observed/scored`` so a reader sees both the count of
    regressions and the number of cases the canary actually examined — an all-clean run
    over three canary cases reads ``0/3``, not a bare zero. The two denominators differ by
    design: ``A/B+C`` prints how many cases carried a figure to check and how many did not,
    so an answer with no figure in it can never borrow a clean case's denominator.
    """

    if name == "wrong_year_contamination":
        checked, not_checkable = _wrong_year_counts(results)
        if not checked and not not_checkable:
            return "NA"
        return f"{_wrong_year_fires(results)}/{checked}+{not_checkable}"
    return _rate_text([getattr(r, _FIELD[name]) for r in results])


_OBSERVED_FIELD: dict[str, str] = {
    "retrieval_hit_rate@5": "retrieval_observed",
    "citation_presence": "citation_observed",
    "figure_match": "figure_observed",
    "refusal_correctness": "refusal_observed",
}


def _observed_value(name: str, results: Sequence[CaseResult]) -> str:
    """What an unmeasurable metric's measurement actually was, for the record.

    The observation is computed ignoring the scope entirely, so a metric this mode cannot
    judge still reports what it saw. ``NA`` here means the metric had nothing to observe —
    a case with no expected figure, for instance — never that the value was hidden.
    """

    if name == "wrong_year_contamination":
        return _metric_value(name, results)
    return _rate_text([getattr(r, _OBSERVED_FIELD[name]) for r in results])


def _verdicts(results: Sequence[CaseResult]) -> dict[str, str]:
    """Every metric's PASS / FAIL / NA verdict against the doc's bar.

    A metric is ``NA`` when this mode left it with no measurable case at all, which is
    distinct from measurable-and-passing: the aggregate never converts one into the other.

    The canary's verdict runs over the cases it could actually *check*. A case whose answer
    carried no figure is not a pass and not a fail — it is unmeasured, so it is excluded
    from both the numerator and the denominator rather than inflating a pass rate.

    A rate metric's bar is a **floor**: the rate must meet or exceed it, so ``PASS`` is
    ``value >= bar`` and ``FAIL`` is ``value < bar``. (The earlier ``value > bar`` inverted
    this — a 1.00 hit rate above a 0.9 floor printed FAIL, and a 0.71 miss rate below a
    0.95 floor printed PASS, which is how a clean run ended "metrics missed their bar".)

    The canary is the one metric whose bar is a **ceiling**: it reports a hard-fail *count*
    of contaminated cases, so one occurrence is a regression on its own and its verdict
    stays ``value > bar`` (bar 0 → any contamination fails, none passes).

    ``NA`` is emitted in place of either verdict, never alongside one, so a metric this mode
    could not measure can never be read as a failure by :func:`_exit_code`.
    """

    verdicts: dict[str, str] = {}
    for name, bar in METRICS.items():
        if name == "wrong_year_contamination":
            cases = [r for r in results if r.case.expect.get("adjacent_year_figure")]
            checkable = [
                r
                for r in cases
                if r.wrong_year_state in (WRONG_YEAR_CLEAN, WRONG_YEAR_CONTAMINATED)
            ]
            if not checkable:
                verdicts[name] = "NA"
                continue
            # The bar is zero: the canary is a hard-fail count, not a rate to trade off.
            value = float(sum(1 for r in checkable if r.wrong_year))
            verdicts[name] = "FAIL" if value > bar else "PASS"
        else:
            scored = [v for v in (getattr(r, _FIELD[name]) for r in results) if v is not None]
            if not scored:
                verdicts[name] = "NA"
                continue
            value = sum(1 for item in scored if item) / len(scored)
            verdicts[name] = "PASS" if value >= bar else "FAIL"
    return verdicts


def _exit_code(results: Sequence[CaseResult], verdicts: Mapping[str, str]) -> tuple[int, str]:
    """The run's exit code, and the reason printed beside it.

    A metric this mode could not measure is ``NA`` — "no verdict", not "passed" and not
    "failed". Only a literal ``FAIL`` contributes here, so an ``NA`` can never drive the
    exit code: a reader who scored the same metric would see it unmeasured. The exit code
    is read from these verdict strings rather than recomputed from the case values, so
    this filter is the one place the ``NA``/``FAIL`` distinction has to hold.

    An infrastructure-degraded case is a *second* way to be unmeasured, and one the verdict
    map cannot see: its metrics are excluded by the scorer, so it appears nowhere in the
    table. It still drives the exit code, so excluding a case can never turn into passing.
    """

    broken = [r for r in results if r.error_kind == "dataset"]
    if broken:
        return EXIT_HARNESS_ERROR, "; ".join(f"{r.case.case_id}: {r.error}" for r in broken)
    failed = [name for name, verdict in verdicts.items() if verdict == "FAIL"]
    crashed = [r for r in results if r.error_kind == "run"]
    # An infrastructure-degraded case is excluded from its metrics, which is what makes it
    # invisible to ``_verdicts`` — so the exclusion itself would read as a pass if the run
    # paid no price for it. A run that measured nothing because the model never answered is
    # a regression, named as one rather than left to masquerade as ``NA``.
    unmeasured = [r for r in results if r.infrastructure_degraded]
    if failed or crashed or unmeasured:
        parts = []
        if failed:
            parts.append("metrics missed their bar: " + ", ".join(failed))
        if crashed:
            parts.append("cases raised in the graph: " + ", ".join(r.case.case_id for r in crashed))
        if unmeasured:
            parts.append(
                "cases the planner could not run (infrastructure, not scored): "
                + ", ".join(r.case.case_id for r in unmeasured)
            )
        return EXIT_REGRESSION, "; ".join(parts)
    return EXIT_PASS, ""


def _print_summary(
    results: Sequence[CaseResult],
    verdicts: Mapping[str, str],
    ctx: _RunContext,
    report: DatasetReport,
    requested: Sequence[Case],
) -> int:
    """The aggregate block: every metric's value against its bar, then the exit code."""

    print("#" * 78)
    print("AGGREGATE — docs/TESTING-AND-EVAL.md §Layer 2 pass bars")
    print("#" * 78)
    mode = "MOCK_LLM=1 (deterministic, offline)" if ctx.mock_mode else "real LLM"
    print(f"mode                 : {mode}")
    print(f"index / catalog      : {ctx.index_dir} — {len(ctx.docs)} document(s)")
    skipped = len(report.cases) - len({r.case.case_id for r in results})
    print(
        f"cases                : {len(requested)} requested, {len(results)} scored"
        + (f", {skipped} case(s) in the file but not requested" if skipped else "")
    )
    for line_no, reason in report.malformed:
        print(f"  dataset line {line_no:>3} : {reason}")
    if report.duplicate_ids:
        print(f"  duplicate ids      : {', '.join(sorted(set(report.duplicate_ids)))}")

    failing = [r.case.case_id for r in results if r.failed]
    broken = [r.case.case_id for r in results if r.error_kind == "dataset"]
    unmeasured = [r.case.case_id for r in results if r.infrastructure_degraded]
    if failing:
        print(f"cases failing        : {', '.join(failing)}")
    elif broken:
        print("cases failing        : none (the failures are dataset errors, listed above)")
    else:
        print("cases failing        : none")
    if unmeasured:
        # Named separately from "failed" because it is not one: the planner could not run,
        # so those cases carry no verdict at all rather than a bad one.
        print(f"cases not measured   : {', '.join(unmeasured)} (planner could not run)")
    print()

    header = f"{'metric':<26}{'value':>18}  {'bar':>6}  verdict"
    print(header)
    print("-" * len(header))
    for name in METRICS:
        bar = METRICS[name]
        bar_text = f"<= {bar:g}" if name == "wrong_year_contamination" else f">= {bar:g}"
        value = _metric_value(name, results)
        if verdicts[name] == "NA":
            # A metric this mode cannot measure still prints what it saw, labelled as an
            # observation rather than a verdict: "cannot be judged here" is not "passed".
            observed = _observed_value(name, results)
            value = f"{observed} (obs)" if observed != "NA" else "NA"
        print(f"{name:<26}{value:>18}  {bar_text:>6}  {verdicts[name]}")

    unscoped = [scope for scope in ctx.scopes.values() if not scope.meaningful]
    if unscoped:
        print()
        print("Not meaningful in this mode (no bar lowered; observed values still print above):")
        for scope in unscoped:
            print(f"  - {scope.name}: {scope.reason}.")
        print(
            "  Re-run without MOCK_LLM=1 (with OPENAI_API_KEY) to score these. The wrong-year\n"
            "  canary is lexical, so it stays live in both modes — it can FAIL here."
        )
    print()

    code, reason = _exit_code(results, verdicts)
    verdict = "PASS" if code == EXIT_PASS else "FAIL"
    print(f"{verdict} — {reason or 'every meaningful metric met its bar'}")
    _print_exit(code)
    _print_pacing(results, mock_mode=ctx.mock_mode)
    return code


def _print_exit(code: int) -> None:
    """The exit code and its meaning, printed on every run."""

    meanings = {
        EXIT_PASS: "every meaningful metric met its bar",
        EXIT_REGRESSION: "a golden regression — a pass bar was missed, the wrong-year canary "
        "fired, a case's graph run raised, or a case's planner could not run at all",
        EXIT_HARNESS_ERROR: "a harness error — the dataset could not be run (no usable case, "
        "an unreadable file, an unresolvable follow_up_of, or a missing index)",
    }
    print(f"exit code {code} — {meanings[code]}")


def _print_pacing(results: Sequence[CaseResult], *, mock_mode: bool = False) -> None:
    """The run's measured token spend, so the wall time it took is explicable.

    A silent pacer is indistinguishable from a hang: a run that took eleven minutes looks
    broken unless the arithmetic that made it take eleven minutes is on the page. Under
    ``MOCK_LLM=1`` the numbers the mock reports are scripted placeholders rather than a
    provider's accounting, so nothing is printed: pacer and report stay inert together.
    """

    if mock_mode or not results or all(not (r.tokens[0] or r.tokens[1]) for r in results):
        return
    spent = sum(r.token_cost for r in results)
    per_case = ", ".join(f"{r.case.case_id}={r.token_cost:,}" for r in results)
    print(f"token spend         : {spent:,} total ({per_case})")


def _json_report(
    results: Sequence[CaseResult], verdicts: Mapping[str, str], ctx: _RunContext
) -> dict[str, Any]:
    """The whole run as one JSON-able document, for a machine-readable journal record."""

    return {
        "mode": "mock" if ctx.mock_mode else "real",
        "corpus": {"index_dir": str(ctx.index_dir), "documents": len(ctx.docs)},
        "metrics": {
            name: {
                "value": _metric_value(name, results),
                "bar": METRICS[name],
                "verdict": verdicts[name],
                "hard_fail": name in HARD_METRICS,
            }
            for name in METRICS
        },
        "not_meaningful": [
            {"metric": scope.name, "reason": scope.reason}
            for scope in ctx.scopes.values()
            if not scope.meaningful
        ],
        "cases": [
            {
                "id": r.case.case_id,
                "question": r.case.question,
                "tags": list(r.case.tags),
                "expect": r.case.expect,
                "history_turns": r.history_turns,
                "plan_kind": r.plan_kind,
                "plan_check": {"ok": r.plan_ok, "detail": r.plan_detail},
                "route": r.route,
                "answer_md": r.answer_md,
                "grounded": r.grounded,
                "refusal_reason": r.refusal_reason,
                "citations": r.citations,
                "ranks": r.ranks,
                "nodes": r.nodes,
                "tokens": {"in": r.tokens[0], "out": r.tokens[1], "total": r.token_cost},
                "latency_ms": round(r.latency_ms, 2),
                "scoring": {
                    "retrieval_hit": r.retrieval_hit,
                    "retrieval_observed": r.retrieval_observed,
                    "retrieval_detail": r.retrieval_detail,
                    "citation_ok": r.citation_ok,
                    "figure_ok": r.figure_ok,
                    "missing_figures": r.missing_figures,
                    "wrong_year": r.wrong_year,
                    "wrong_year_state": r.wrong_year_state,
                    "wrong_year_detail": r.wrong_year_detail,
                    "pretrim_claims": list(r.pretrim_claims),
                    "refusal_ok": r.refusal_ok,
                    "refusal_detail": r.refusal_detail,
                    "planner_degraded_reason": r.degraded_reason,
                    "infrastructure_degraded": _infrastructure_degradation(r.degraded_reason),
                },
                "notes": r.notes,
                "error": r.error,
                "error_kind": r.error_kind,
            }
            for r in results
        ],
    }


# --- entry point --------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the golden QA set through src.chains.graph.invoke().",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="golden set (.jsonl)")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_DIR, help="index directory")
    parser.add_argument("--only", type=str, default="", help="comma-separated case ids to score")
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the per-case blocks; summary only"
    )
    return parser.parse_args(argv)


def _mock_mode() -> bool:
    """Whether this run is the offline deterministic path — the same test the code uses."""

    return os.environ.get("MOCK_LLM", "").strip() == "1"


def _network_report(mock_mode: bool) -> str:
    """What this run imports for the LLM: the mock path must never pull in ``openai``."""

    if not mock_mode:
        return "real gateway (an OpenAI-compatible client may be imported on first use)"
    imported = any(name == "openai" or name.startswith("openai.") for name in sys.modules)
    verdict = "openai WAS imported (network client present)" if imported else "openai NOT imported"
    return f"offline verified — {verdict}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the golden set, print the report, return the process exit code."""

    # `.env` is read before anything else, so MOCK_LLM and any key in it are part of the
    # environment the mode is decided from. Without this the documented `cp .env.example
    # .env` setup ran keyless and every case degraded.
    load_env()
    args = parse_args(argv)
    # Read MOCK_LLM before anything imports the graph: the gateway and the embedder both
    # resolve from the environment, so an ambient value must not silently change what is
    # measured. Reading it first is what makes the mode the harness reports the truth.
    mock_mode = _mock_mode()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    problems = _check_index(args.index)
    if problems:
        for problem in problems:
            print(f"harness error: {problem}", file=sys.stderr)
        _print_exit(EXIT_HARNESS_ERROR)
        return EXIT_HARNESS_ERROR

    try:
        report = parse_dataset(args.data)
    except DatasetError as exc:
        print(f"harness error: {exc}", file=sys.stderr)
        _print_exit(EXIT_HARNESS_ERROR)
        return EXIT_HARNESS_ERROR

    wanted: set[str] | None = None
    if args.only.strip():
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    requested = [case for case in report.cases if wanted is None or case.case_id in wanted]

    ctx = _RunContext(
        docs=_load_catalog(args.index),
        mock_mode=mock_mode,
        index_dir=args.index,
        scopes=_mock_scopes() if mock_mode else _full_scopes(),
    )

    print("=" * 78)
    print("golden QA harness — docs/TESTING-AND-EVAL.md §Layer 2")
    print("=" * 78)
    print(f"dataset              : {args.data}")
    label = "MOCK_LLM=1" if mock_mode else "real LLM"
    print(f"mode                 : {label} — {_network_report(mock_mode)}")
    print(f"index / catalog      : {args.index} — {len(ctx.docs)} document(s)")
    print(
        f"cases                : {len(report.cases)} usable"
        + (f", {len(report.malformed)} line(s) rejected" if report.malformed else "")
    )
    if wanted is not None:
        print(f"selection (--only)   : {', '.join(sorted(wanted))}")

    if not report.cases:
        print("harness error: the dataset holds no usable case", file=sys.stderr)
        _print_exit(EXIT_HARNESS_ERROR)
        return EXIT_HARNESS_ERROR
    if not requested:
        print(
            f"harness error: --only matched none of the {len(report.cases)} case(s)",
            file=sys.stderr,
        )
        _print_exit(EXIT_HARNESS_ERROR)
        return EXIT_HARNESS_ERROR

    # Dataset order is the run order, so a follow-up's ancestors have already produced a
    # real answer by the time its history is built. An ancestor named by ``follow_up_of``
    # is pulled in even when --only asked for the follow-up alone: without it the case
    # would silently answer as a standalone question and score the wrong behaviour.
    #
    # A reference the dataset cannot resolve is a broken golden set, not a crash: the case
    # is reported as a dataset error and the rest of the set still runs.
    ordered: list[Case] = []
    unresolved: list[Case] = []
    for case in requested:
        try:
            chain = _follow_up_chain(report.cases, case)
        except CaseError as exc:
            unresolved.append(case)
            print(f"dataset error: {exc}", file=sys.stderr)
            continue
        for ancestor in chain:
            if ancestor.case_id not in {item.case_id for item in ordered}:
                ordered.append(ancestor)
        if case.case_id not in {item.case_id for item in ordered}:
            ordered.append(case)

    results: dict[str, CaseResult] = {}
    # A case whose ``follow_up_of`` never resolved is already scored as a dataset error,
    # so the summary reports it rather than the run ending on it.
    for case in unresolved:
        results[case.case_id] = CaseResult(
            case=case,
            error=f"case {case.case_id!r} follows up "
            f"{case.follow_up_of!r}, which is not in the dataset",
            error_kind="dataset",
        )
    # The pacer is built here, after the selection is known, because the daily gate is a
    # pre-flight over the *requested* set: refusing a run is only honest if the projection
    # is the whole run's cost rather than the first case's.
    tpm_limit = _tpm_limit_from_env()
    tpd_limit = _env_int("EVAL_TPD_LIMIT", DEFAULT_TPD_LIMIT)
    pacer = _TokenPacer(
        tpm_limit=tpm_limit,
        tpd_limit=tpd_limit,
        mock_mode=mock_mode,
        allowed_today=_allowed_today_from_env(tpd_limit),
        daily_gate=_daily_gate_enabled(),
    )
    projected = _projected_run_cost(ordered, _max_measured_cost(results))
    _print_preflight(projected, pacer, ordered, quiet=args.quiet)
    refusal = pacer.check_daily(projected)
    if refusal:
        # A refusal is a harness decision, not a case's finding, so it prints to stderr and
        # takes the regression exit path: the run did not happen, and pretending it passed
        # or failed would be a lie either way.
        print(f"harness error: {refusal}", file=sys.stderr)
        _print_exit(EXIT_REGRESSION)
        return EXIT_REGRESSION

    total = len(ordered)
    for position, case in enumerate(ordered, start=1):
        chain = _follow_up_chain(report.cases, case)
        result = _run_case(case, ctx, chain, results, pacer=pacer)
        results[case.case_id] = result
        if not args.quiet:
            _print_case(result, position, total, mock_mode=mock_mode)

    # Ancestors replayed for history are not scored: the aggregate covers the requested set.
    scored = [results[case.case_id] for case in requested]
    verdicts = _verdicts(scored)
    code = _print_summary(scored, verdicts, ctx, report, requested)

    if args.json is not None:
        args.json.write_text(
            json.dumps(_json_report(scored, verdicts, ctx), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"json report written to {args.json}")
    return code


if __name__ == "__main__":
    sys.exit(main())

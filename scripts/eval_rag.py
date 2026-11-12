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
DEFAULT_INDEX_DIR = Path(os.environ.get("INDEX_DIR", "").strip() or (REPO_ROOT / "data" / "index"))
CATALOG_NAME = "docs.jsonl"

EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_HARNESS_ERROR = 2

METRICS: dict[str, float] = {
    "retrieval_hit_rate@5": 0.9,
    "citation_presence": 1.0,
    "figure_match": 0.95,
    "wrong_year_contamination": 0.0,
    "refusal_correctness": 1.0,
}
HARD_METRICS = frozenset({"wrong_year_contamination"})

HIT_RANK = 5

DEFAULT_TPM_LIMIT = 8000
DEFAULT_TPD_LIMIT = 200_000
DEFAULT_TPD_FRACTION = 1.0
TPM_WAIT_CAP_SECONDS = 30 * 60
TPD_WAIT_CAP_SECONDS = 60 * 60
TPD_OVERRIDE_HINT = "EVAL_TPD_OVERRIDE=1"
_FIGURE_SPACES = " \u00a0\u2009\u200a\u2002\u2003\u2004\u2005\u2007\u202f\u205f\u3000"

_FIGURE_FURNITURE = _FIGURE_SPACES + "*,$%"

_SPACE_TRANSLATION = str.maketrans("", "", _FIGURE_SPACES)
_FIGURE_TRANSLATION = str.maketrans("", "", _FIGURE_FURNITURE)


_GROUP_SEPARATORS = ",\u2009\u200a\u2002\u2003\u2004\u2005\u2007\u202f\u205f"


_FIGURE_TOKEN_IN_TEXT = re.compile(
    r"\$?[" + _FIGURE_SPACES + r"]*"
    r"(?:\d{1,3}(?:[" + _GROUP_SEPARATORS + r"]\d{3})+|\d+)(?:\.\d+)?"
)

_DIGIT_FIGURE = re.compile(r"[\d.]+")
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

TRIM_MARKER = "(unsupported claim removed)"

VERIFY_LOGGER_NAME = "src.chains.nodes.verify"

ANALYZE_LOGGER_NAME = "src.chains.nodes.analyze"

PLANNER_DEGRADED_MARKER = "planner_degraded:"

_INFRASTRUCTURE_REASONS = frozenset(
    {
        "RateLimited",
        "ContextLength",
        "Auth",
        "Timeout",
    }
)

_FIGURE_IN_TEXT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?%")

NOT_CHECKABLE = "not_checkable"
WRONG_YEAR_CLEAN = "clean"
WRONG_YEAR_CONTAMINATED = "contaminated"


@dataclass(frozen=True)
class _Scope:

    name: str
    meaningful: bool
    reason: str = ""
    partial_kinds: tuple[str, ...] = ()

    def covers(self, case: Case) -> bool:

        if self.meaningful:
            return True
        if not self.partial_kinds:
            return False
        if not case.is_refusal:
            return True
        return str(case.expect.get("missing_aspect", "")) in self.partial_kinds


def _mock_scopes() -> dict[str, _Scope]:

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
    return {name: _Scope(name, True) for name in METRICS}


class DatasetError(Exception):
    pass


class CaseError(Exception):
    pass


@dataclass(frozen=True)
class Case:
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
        return bool(self.expect.get("require_citations", True))

    @property
    def expected_appearances(self) -> list[str]:
        return [str(f) for f in self.expect.get("must_appear", []) or [] if str(f).strip()]


@dataclass(frozen=True)
class DatasetReport:
    cases: list[Case]
    malformed: list[tuple[int, str]]
    duplicate_ids: list[str]


def parse_dataset(path: Path) -> DatasetReport:

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


@dataclass
class CaseResult:

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
    wrong_year_state: str = ""
    pretrim_claims: list[str] = field(default_factory=list)
    pretrim_note: str = ""
    degraded_reason: str = ""
    refusal_ok: bool | None = None
    refusal_observed: bool | None = None
    refusal_detail: str = ""
    plan_ok: bool | None = None
    plan_detail: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def infrastructure_degraded(self) -> bool:

        return _infrastructure_degradation(self.degraded_reason)

    @property
    def token_cost(self) -> int:

        return int(self.tokens[0]) + int(self.tokens[1])

    @property
    def failed(self) -> bool:
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

    return text.translate(_SPACE_TRANSLATION)


def _figure_key(text: str) -> str:

    return text.translate(_FIGURE_TRANSLATION)


def _figure_tokens(text: str) -> frozenset[str]:

    return frozenset(_figure_key(match.group(0)) for match in _FIGURE_TOKEN_IN_TEXT.finditer(text))


def _contains_figure(text: str, figure: str) -> bool:

    if not figure:
        return False
    if figure in text:
        return True
    key = _figure_key(figure)
    if not key or not _DIGIT_FIGURE.fullmatch(key):
        return False
    return key in _figure_tokens(text)


_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_CELL_ALIGNMENT = re.compile(r"^[\s:-]*$")


def _sentences(text: str) -> list[str]:
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
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_label_cell(cell: str) -> bool:

    return bool(cell) and not _FIGURE_IN_TEXT.search(cell)


def _is_alignment_row(cells: Sequence[str]) -> bool:
    return all(cell and _TABLE_CELL_ALIGNMENT.fullmatch(cell) for cell in cells)


def _assertions(answer_md: str) -> list[tuple[str, str]]:

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

    return bool(_FIGURE_IN_TEXT.search(text))


@contextmanager
def _capturing_node_signals() -> Iterator[dict[str, Any]]:

    claims: list[str] = []
    degradations: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.name == ANALYZE_LOGGER_NAME:
                if PLANNER_DEGRADED_MARKER not in str(record.msg):
                    return
                if not isinstance(record.args, tuple) or not record.args:
                    return
                if not isinstance(record.args[0], str):
                    return
                degradations.append(record.args[0])
                return
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

    return reason.strip().split(":", 1)[0].strip() in _INFRASTRUCTURE_REASONS


def _texts_to_check(result: CaseResult) -> tuple[list[str], str]:

    text = result.answer_md
    trimmed_away = TRIM_MARKER in text
    if result.pretrim_claims:
        source = "the pre-trim claim text the verify node logged"
        if trimmed_away:
            source += " (the final answer was trimmed)"
        return [*result.pretrim_claims, text], source
    if trimmed_away:
        return [], "no claim text was logged before the claims were trimmed"
    return [text], "the final answer (nothing was trimmed)"


def _citation_entries(citations: Sequence[Any]) -> list[dict[str, Any]]:

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
    ref = entry.get("ref_index")
    return isinstance(ref, int) and not isinstance(ref, bool) and ref >= 1


def _ranks_and_texts(
    retrieval_debug: Sequence[Any],
) -> tuple[list[tuple[str, int]], list[str]]:

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

    return {"out_of_scope": "refuse", "multi_part": "decompose_question"}.get(
        plan_kind, "hybrid_retrieve"
    )


@dataclass
class _Projection:

    tokens: int
    wait_seconds: float
    reason: str

    def __bool__(self) -> bool:
        return self.tokens > 0 or self.wait_seconds > 0.0


class _TokenPacer:

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
        self.allowed_today = int(allowed_today)
        self.daily_gate = bool(daily_gate)
        self._now = now
        self._sleep = sleep
        self._spend: list[tuple[float, int]] = []
        self.waits: list[float] = []

    def _minutes_window(self) -> list[tuple[float, int]]:

        now = self._now()
        return [item for item in self._spend if now - item[0] < 60.0]

    def _seconds_until_token_slot(self, tokens: int) -> float:

        if tokens <= 0 or self.tpm_limit <= 0:
            return 0.0
        if tokens > self.tpm_limit:
            return math.inf
        window = self._minutes_window()
        used = sum(cost for _, cost in window)
        if used + tokens <= self.tpm_limit:
            return 0.0
        running = used
        for stamp, cost in window:
            running -= cost
            if running + tokens <= self.tpm_limit:
                return max(0.0, stamp + 60.0 - self._now())
        return max(0.0, window[-1][0] + 60.0 - self._now())

    def _project(self, tokens: int) -> _Projection:

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

        shortfall = max(0, tokens - self.allowed_today)
        if shortfall <= 0:
            return ""
        return (
            f"~{tokens} tokens projected against the day's ~{self.allowed_today:,} of "
            f"{self.tpd_limit:,} — {shortfall:,} tokens short. A daily ceiling cannot be "
            f"slept off, and a run that spends past it cannot be re-run: refusing to "
            f"start, so today's quota survives. Set {TPD_OVERRIDE_HINT} to spend it anyway"
        )

    def check_daily(self, tokens: int) -> str:

        if self.mock_mode or tokens <= 0 or not self.daily_gate:
            return ""
        return self._refuse_reason(tokens)

    def gate(
        self, tokens: int, *, label: str = "", quiet: bool = False, stream: Any = None
    ) -> float:

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

        whole = max(1.0, math.ceil(seconds))
        self._sleep(whole)
        return whole

    @staticmethod
    def _note(quiet: bool, stream: Any, projection: _Projection, label: str) -> None:
        if quiet or not projection:
            return
        print(f"  pacing         : {projection.reason} {label}".rstrip(), file=stream)

    def record(self, tokens: int) -> None:
        if self.mock_mode or tokens <= 0:
            return
        self._spend.append((self._now(), int(tokens)))

    def projected_cost(self, results: Mapping[str, Any]) -> int:

        if self.mock_mode:
            return 0
        costs = [
            result.token_cost for result in results.values() if getattr(result, "token_cost", 0)
        ]
        return max(costs) if costs else 0


CANONICAL_CASE_TOKENS = 10_038
FOLLOW_UP_CASE_TOKENS = 600


def _projected_run_cost(cases: Sequence[Any], prior_cost: int = 0) -> int:

    if not cases:
        return 0
    total = 0
    for case in cases:
        if getattr(case, "is_refusal", False):
            continue
        per_case = prior_cost or CANONICAL_CASE_TOKENS
        if getattr(case, "follow_up_of", None):
            per_case += FOLLOW_UP_CASE_TOKENS
        total += per_case
    return total


def _env_int(name: str, default: int) -> int:

    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _tpm_limit_from_env(default: int = DEFAULT_TPM_LIMIT) -> int:

    return _env_int("EVAL_TPM_LIMIT", default)


def _allowed_today_from_env(limit: int) -> int:

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

    return not _env_flag("EVAL_TPD_OVERRIDE")


def _max_measured_cost(results: Mapping[str, Any]) -> int:

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


def _score_case(case: Case, result: CaseResult, scopes: Mapping[str, _Scope]) -> None:

    expect = case.expect
    answer = result.answer_md

    infrastructure = _infrastructure_degradation(result.degraded_reason)

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
            result.retrieval_hit = None

    if infrastructure:
        result.citation_observed = bool(result.citations) and all(
            _valid_citation(entry) for entry in result.citations
        )
        result.citation_ok = None
    elif not case.wants_citation:
        result.citation_ok = None
        result.citation_observed = None
        result.notes.append("citations are not expected for this case")
    elif case.is_refusal:
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

    expected_figures = case.expected_appearances
    if not expected_figures:
        result.figure_ok = None
        result.figure_observed = None
        result.notes.append(
            "no expected figure for this case (summarization); figure match not scored"
        )
    elif infrastructure:
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
            result.wrong_year_state = WRONG_YEAR_CLEAN
            result.pretrim_note = source
        else:
            result.wrong_year_state = NOT_CHECKABLE
            result.pretrim_note = f"answer contained no figure (judged {source})"

    if infrastructure:
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
    if not scopes["refusal_correctness"].covers(case):
        result.refusal_ok = None
        if case.is_refusal:
            result.notes.append(
                f"refusal correctness not measurable: {scopes['refusal_correctness'].reason}"
            )
    elif infrastructure:
        result.refusal_ok = None

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


@dataclass
class _RunContext:
    docs: list[Any]
    mock_mode: bool
    index_dir: Path
    scopes: dict[str, _Scope]


def _load_catalog(index_dir: Path) -> list[Any]:
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
    return [
        f"missing index artifact: {index_dir / name}"
        for name in ("chunks.jsonl", "bm25.pkl", CATALOG_NAME)
        if not (index_dir / name).is_file()
    ]


def _history_for(ancestors: Sequence[Case], results: Mapping[str, CaseResult]) -> list[Any]:

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
        projection = pacer.projected_cost(results)
        pacer.gate(projection, label=f"(before {case.case_id})", quiet=ctx.mock_mode)

    started = time.perf_counter()
    try:
        with _capturing_node_signals() as captured:
            envelope = invoke(
                case.question, history=history or None, params=None, doc_catalog=ctx.docs
            )
        result.pretrim_claims = list(captured["claims"])
        result.degraded_reason = captured["degraded"][0] if captured["degraded"] else ""
    except Exception as exc:
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
        pacer.record(result.token_cost)
    return result


_TOKEN_FIELDS = ("tokens_in", "tokens_out")


def _trace_tokens(events: Sequence[Any]) -> tuple[int, int]:

    totals = [0, 0]
    for event in events or []:
        for index, name in enumerate(_TOKEN_FIELDS):
            value = event.get(name) if isinstance(event, Mapping) else getattr(event, name, None)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            totals[index] += max(0, value)
    return totals[0], totals[1]


def _print_case(result: CaseResult, position: int, total: int, mock_mode: bool = False) -> None:
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


_FIELD: dict[str, str] = {
    "retrieval_hit_rate@5": "retrieval_hit",
    "citation_presence": "citation_ok",
    "figure_match": "figure_ok",
    "wrong_year_contamination": "wrong_year",
    "refusal_correctness": "refusal_ok",
}


def _wrong_year_fires(results: Sequence[CaseResult]) -> int:
    return sum(1 for r in results if r.wrong_year)


def _wrong_year_counts(results: Sequence[CaseResult]) -> tuple[int, int]:

    cases = [r for r in results if r.case.expect.get("adjacent_year_figure")]
    checked = sum(1 for r in cases if r.wrong_year_state == WRONG_YEAR_CLEAN)
    fired = _wrong_year_fires(cases)
    return checked + fired, len(cases) - checked - fired


def _rate_text(values: Sequence[bool | None]) -> str:
    scored = [value for value in values if value is not None]
    if not scored:
        return "NA"
    hits = sum(1 for value in scored if value)
    return f"{hits}/{len(scored)} ({hits / len(scored):.2f})"


def _metric_value(name: str, results: Sequence[CaseResult]) -> str:

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

    if name == "wrong_year_contamination":
        return _metric_value(name, results)
    return _rate_text([getattr(r, _OBSERVED_FIELD[name]) for r in results])


def _verdicts(results: Sequence[CaseResult]) -> dict[str, str]:

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

    broken = [r for r in results if r.error_kind == "dataset"]
    if broken:
        return EXIT_HARNESS_ERROR, "; ".join(f"{r.case.case_id}: {r.error}" for r in broken)
    failed = [name for name, verdict in verdicts.items() if verdict == "FAIL"]
    crashed = [r for r in results if r.error_kind == "run"]
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
    meanings = {
        EXIT_PASS: "every meaningful metric met its bar",
        EXIT_REGRESSION: "a golden regression — a pass bar was missed, the wrong-year canary "
        "fired, a case's graph run raised, or a case's planner could not run at all",
        EXIT_HARNESS_ERROR: "a harness error — the dataset could not be run (no usable case, "
        "an unreadable file, an unresolvable follow_up_of, or a missing index)",
    }
    print(f"exit code {code} — {meanings[code]}")


def _print_pacing(results: Sequence[CaseResult], *, mock_mode: bool = False) -> None:

    if mock_mode or not results or all(not (r.tokens[0] or r.tokens[1]) for r in results):
        return
    spent = sum(r.token_cost for r in results)
    per_case = ", ".join(f"{r.case.case_id}={r.token_cost:,}" for r in results)
    print(f"token spend         : {spent:,} total ({per_case})")


def _json_report(
    results: Sequence[CaseResult], verdicts: Mapping[str, str], ctx: _RunContext
) -> dict[str, Any]:

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

    return os.environ.get("MOCK_LLM", "").strip() == "1"


def _network_report(mock_mode: bool) -> str:

    if not mock_mode:
        return "real gateway (an OpenAI-compatible client may be imported on first use)"
    imported = any(name == "openai" or name.startswith("openai.") for name in sys.modules)
    verdict = "openai WAS imported (network client present)" if imported else "openai NOT imported"
    return f"offline verified — {verdict}"


def main(argv: Sequence[str] | None = None) -> int:
    load_env()
    args = parse_args(argv)
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
    for case in unresolved:
        results[case.case_id] = CaseResult(
            case=case,
            error=f"case {case.case_id!r} follows up "
            f"{case.follow_up_of!r}, which is not in the dataset",
            error_kind="dataset",
        )
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

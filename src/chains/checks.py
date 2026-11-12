from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "OUTCOME_REJECTED",
    "OUTCOME_SUPPORTED",
    "OUTCOME_UNDECIDED",
    "REASON_FIGURE_NOT_IN_SPAN",
    "REASON_PERIOD_MISMATCH",
    "TRIM_MARKER",
    "ClaimJudgement",
    "ClaimUnit",
    "Figure",
    "asked_years",
    "catalog_years",
    "extract_figures",
    "numeric_pre_pass",
    "rejected_units",
    "rescale",
    "split_units",
    "trim_answer",
    "undecided_units",
    "year_coverage_gap",
]


TRIM_MARKER = "(unsupported claim removed)"

REASON_FIGURE_NOT_IN_SPAN = "figure_not_in_span"
REASON_PERIOD_MISMATCH = "period_mismatch"

OUTCOME_REJECTED = "rejected"
OUTCOME_SUPPORTED = "supported"
OUTCOME_UNDECIDED = "undecided"


_SCALE = r"(?i:million|billion|thousand|trillion|mil|mn|mm|bn)|B|M|K"
_SCALE_FACTORS: Mapping[str, float] = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mil": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
}

_FIGURE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<currency>\$)?"
    r"\s*"
    r"(?P<digits>\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*"
    r"(?P<scale>" + _SCALE + r")?"
    r"(?P<percent>%)?"
)

_PER_SHARE = re.compile(
    r"(?:earnings|profit|loss|income|dividend|distribution|net)\s+per\s+(?:share|diluted\s+share|basic\s+share)"
    r"|per\s+(?:diluted|basic)\s+share",
    re.IGNORECASE,
)

_YEAR_TOKEN = re.compile(r"(?:19|20)\d{2}")

_REFERENCE_TAIL = re.compile(
    r"(?:\b(?:p|pp|pg|page|pages|note|notes|section|sections|table|tables|figure|figures|"
    r"exhibit|item|items|chapter|vol|volume)\s*[.:]?\s*)$",
    re.IGNORECASE,
)

_MARKER = re.compile(r"\[(\d+)\]")

_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

_UNIT_PHRASE = re.compile(
    r"\bin\s+(millions|billions|thousands|trillions)\b|\bper\s+(millions|billions)\b",
    re.IGNORECASE,
)
_UNIT_SCALE: Mapping[str, float] = {
    "thousands": 1e3,
    "millions": 1e6,
    "billions": 1e9,
    "trillions": 1e12,
}

_ABBREVIATIONS = frozenset(
    {
        "approx",
        "corp",
        "dr",
        "eg",
        "est",
        "etc",
        "fig",
        "figs",
        "ie",
        "inc",
        "llc",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "nr",
        "p",
        "pg",
        "pp",
        "st",
        "vol",
        "vs",
    }
)

_HEADER_LOOKBACK = 8


@dataclass(frozen=True)
class Figure:

    raw: str
    number: float
    scale: float
    sig: int
    percent: bool
    currency: bool
    explicit_scale: bool


@dataclass(frozen=True)
class ClaimUnit:

    text: str
    start: int
    end: int
    prefix: str
    suffix: str
    markers: tuple[int, ...]
    figures: tuple[Figure, ...]
    periods: tuple[int, ...]


@dataclass(frozen=True)
class ClaimJudgement:

    unit: ClaimUnit
    outcome: str
    reason: str | None
    detail: str


def _sig_digits(digits: str) -> int:

    body = digits.replace(",", "")
    if "." in body:
        whole, _, fraction = body.partition(".")
        body = whole + fraction.rstrip("0")
    body = body.lstrip("0")
    if not body:
        return 1
    return len(body.lstrip("0")) or 1


def _figure_from(match: re.Match[str]) -> Figure:

    digits = match.group("digits")
    printed_scale = (match.group("scale") or "").lower()
    factor = _SCALE_FACTORS.get(printed_scale, 1.0)
    number = float(digits.replace(",", ""))
    start, end = match.span()
    if match.string[start - 1 : start] == "(" and match.string[end : end + 1] == ")":
        number = -abs(number)
    return Figure(
        raw=match.group(0).strip(),
        number=number,
        scale=factor,
        sig=_sig_digits(digits),
        percent=bool(match.group("percent")),
        currency=bool(match.group("currency")),
        explicit_scale=factor != 1.0,
    )


def rescale(figure: Figure, unit_scale: float, per_share: bool = False) -> Figure:

    if figure.explicit_scale or figure.percent or per_share:
        return figure
    if unit_scale == 1.0:
        return figure
    return Figure(
        raw=figure.raw,
        number=figure.number,
        scale=figure.scale * unit_scale,
        sig=figure.sig,
        percent=figure.percent,
        currency=figure.currency,
        explicit_scale=figure.explicit_scale,
    )


def _is_year_or_reference(text: str, match: re.Match[str]) -> bool:

    currency = bool(match.group("currency"))
    percent = bool(match.group("percent"))
    printed_scale = (match.group("scale") or "").lower()
    if percent or currency or printed_scale:
        return False
    tokens = text[max(0, match.start() - 40) : match.start()]
    if _REFERENCE_TAIL.search(tokens) is not None:
        return True
    return _YEAR_TOKEN.fullmatch(match.group("digits")) is not None


def extract_figures(text: str) -> list[Figure]:

    return [
        _figure_from(match)
        for match in _FIGURE_TOKEN.finditer(text)
        if not _is_year_or_reference(text, match)
    ]


def _unit_scale_of(text: str) -> float:

    match = _UNIT_PHRASE.search(text)
    if match is None:
        return 1.0
    word = (match.group(1) or match.group(2) or "").lower()
    return _UNIT_SCALE.get(word, 1.0)


def _per_share_of(text: str) -> bool:

    return _PER_SHARE.search(text) is not None


def _round_sig(value: float, sig: int) -> float:
    if value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    factor = 10.0 ** (exponent - sig + 1)
    return round(value / factor) * factor


def _agrees(claim: Figure, span: Figure) -> bool:

    if claim.percent != span.percent:
        return False
    if claim.percent and span.percent:
        return claim.number == span.number
    if _same_magnitude(claim.number * claim.scale, claim.sig, span.number * span.scale, span.sig):
        return True
    return _same_digits(claim.number, claim.sig, span.number, span.sig)


def _same_digits(left: float, left_sig: int, right: float, right_sig: int) -> bool:
    if left == right:
        return True
    sig = min(left_sig, right_sig)
    return _round_sig(left, sig) == _round_sig(right, sig)


def _same_magnitude(left: float, left_sig: int, right: float, right_sig: int) -> bool:
    if left == right:
        return True
    sig = min(left_sig, right_sig)
    rounded_left = _round_sig(left, sig)
    rounded_right = _round_sig(right, sig)
    if rounded_left == rounded_right:
        return True
    return abs(rounded_left - rounded_right) <= abs(rounded_right) * 1e-9


def _years_in(text: str) -> list[tuple[int, int]]:
    return [(match.start(), int(match.group(0))) for match in _YEAR.finditer(text)]


class _SpanIndex:

    def __init__(self, text: str) -> None:
        self.text = text
        unit_scale = _unit_scale_of(text)
        self.unit_scale = unit_scale
        self._keep: list[tuple[int, int, Figure]] = []
        for match in _FIGURE_TOKEN.finditer(text):
            figure = _figure_from(match)
            if self._is_claim_figure(match, figure):
                exempt = self._per_share(match)
                self._keep.append((match.start(), match.end(), rescale(figure, unit_scale, exempt)))
        self._lines = _line_bounds(text)
        self._line_years = {
            number: self._years_of_line(start, end, self._keep)
            for number, (start, end) in enumerate(self._lines)
        }
        self._line_figures = {
            number: [token for token in self._keep if start <= token[0] < end]
            for number, (start, end) in enumerate(self._lines)
        }

    def _per_share(self, match: re.Match[str]) -> bool:

        line_start = self.text.rfind("\n", 0, match.start()) + 1
        head = self.text[line_start : match.start()]
        return _per_share_of(head)

    def _is_claim_figure(self, match: re.Match[str], figure: Figure) -> bool:

        start = match.start()
        if figure.percent and self.text[start - 1 : start] == "(":
            return False
        return not _is_year_or_reference(self.text, match)

    def _years_of_line(
        self, start: int, end: int, tokens: Sequence[tuple[int, int, Figure]]
    ) -> list[tuple[int, int]]:
        inside = [
            (token_start, token_end)
            for token_start, token_end, _ in tokens
            if start <= token_start < end
        ]
        found: list[tuple[int, int]] = []
        for offset, year in _years_in(self.text[start:end]):
            absolute = start + offset
            if any(t_start <= absolute < t_end for t_start, t_end in inside):
                continue
            found.append((absolute, year))
        return found

    def _line_of(self, offset: int) -> int:
        for number, (start, end) in enumerate(self._lines):
            if start <= offset < end:
                return number
        return len(self._lines) - 1

    def matches(self, claim: Figure) -> list[tuple[int, int, Figure]]:

        return [token for token in self._keep if _agrees(claim, token[2])]

    def years_of(self, offset: int) -> frozenset[int]:

        line_number = self._line_of(offset)
        start, _line_end = self._lines[line_number]
        position = offset - start
        own = self._line_years.get(line_number, [])
        if own:
            nearest = min(own, key=lambda item: abs(item[0] - position))
            return frozenset({nearest[1]})

        for header_number in range(line_number - 1, -1, -1):
            if line_number - header_number > _HEADER_LOOKBACK:
                break
            header = self._line_years.get(header_number, [])
            if not header:
                continue
            row_figures = self._aligned(line_number)
            if len(header) != len(row_figures):
                return frozenset()
            for ordinal, token in enumerate(row_figures):
                if token[0] <= offset < token[1]:
                    return frozenset({header[ordinal][1]})
            return frozenset()
        return frozenset()

    def _aligned(self, line_number: int) -> list[tuple[int, int, Figure]]:

        return [token for token in self._line_figures.get(line_number, []) if not token[2].percent]

    def periods(self) -> frozenset[int]:
        return frozenset({year for years in self._line_years.values() for _, year in years})


def _line_bounds(text: str) -> list[tuple[int, int]]:

    bounds: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char == "\n":
            bounds.append((start, index))
            start = index + 1
    bounds.append((start, len(text)))
    return bounds


def split_units(answer: str) -> list[ClaimUnit]:

    units: list[ClaimUnit] = []
    rows: list[tuple[str, int]] = []
    lead: str | None = None
    previous = ""
    offset = 0
    for line in answer.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        text = stripped.strip()
        table = re.match(r"^\|.*\|\s*$", text)
        if table is not None:
            if not rows:
                lead = previous
            rows.append((stripped, offset))
        else:
            _flush_table(rows, units, lead)
            rows = []
            lead = None
            if text:
                _split_line(stripped, offset, units)
        if text:
            previous = text
        offset += len(line)
    _flush_table(rows, units, lead)
    return [unit for unit in units if unit.text.strip()]


def _flush_table(rows: list[tuple[str, int]], units: list[ClaimUnit], lead: str | None) -> None:
    if not rows:
        return
    markers: list[int] = []
    for line, _ in rows:
        markers.extend(int(number) for number in _MARKER.findall(line))
    if lead is not None:
        markers.extend(int(number) for number in _MARKER.findall(lead))
    header = _table_periods(rows)
    for line, base in rows:
        indent = len(line) - len(line.lstrip())
        body = line[indent:]
        origin = base + indent
        first = body.index("|")
        last = body.rindex("|")
        unit = _unit(body, origin, first + 1, last, "|", body[last:])
        if unit.text.strip():
            units.append(
                ClaimUnit(
                    text=unit.text,
                    start=unit.start,
                    end=unit.end,
                    prefix=unit.prefix,
                    suffix=unit.suffix,
                    markers=tuple(markers),
                    figures=unit.figures,
                    periods=header,
                )
            )


def _table_periods(rows: list[tuple[str, int]]) -> tuple[int, ...]:

    for line, _ in rows:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or all(not cell for cell in cells):
            continue
        if _is_separator_row(cells):
            continue
        years = {int(match.group(0)) for cell in cells for match in _YEAR.finditer(cell)}
        if years:
            return tuple(sorted(years))
        return ()
    return ()


def _is_separator_row(cells: list[str]) -> bool:
    return all(cell and set(cell) <= {"-", ":", " "} for cell in cells)


def _split_line(line: str, base: int, units: list[ClaimUnit]) -> None:

    if not line.strip():
        return
    indent = len(line) - len(line.lstrip())
    body = line[indent:]
    origin = base + indent

    bullet = re.match(r"^(?:[-*+]|\d+[.)])\s+", body)
    if bullet is not None:
        units.append(_unit(body, origin, bullet.end(), len(body), bullet.group(0), ""))
        return

    for start, end in _sentence_spans(body):
        units.append(_unit(body, origin, start, end, "", ""))


def _unit(body: str, base: int, start: int, end: int, prefix: str, suffix: str) -> ClaimUnit:
    text = body[start:end]
    prose = _MARKER.sub(" ", text)
    markers = tuple(int(match.group(1)) for match in _MARKER.finditer(text))
    return ClaimUnit(
        text=text,
        start=base + start,
        end=base + end,
        prefix=prefix,
        suffix=suffix,
        markers=markers,
        figures=tuple(extract_figures(prose)),
        periods=tuple(sorted({int(match.group(0)) for match in _YEAR.finditer(prose)})),
    )


def _sentence_spans(body: str) -> list[tuple[int, int]]:

    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    length = len(body)
    while index < length:
        if body[index] not in ".!?":
            index += 1
            continue
        end = index
        while end < length and body[end] in ".!?":
            end += 1
        if end >= length or not body[end].isspace():
            index = end
            continue
        head = body[start:index].rstrip()
        words = head.split()
        word = words[-1] if words else ""
        if not _is_abbreviation(word):
            spans.append((start, end))
            start = end
            while start < length and body[start].isspace():
                start += 1
        index = end
    if body[start:].strip():
        spans.append((start, length))
    return spans


def _is_abbreviation(word: str) -> bool:

    if len(word) <= 1:
        return True
    return word.rstrip(".").lower() in _ABBREVIATIONS


def numeric_pre_pass(answer: str, spans: Mapping[int, str]) -> list[ClaimJudgement]:

    units = split_units(answer)
    indexes = {number: _SpanIndex(text) for number, text in spans.items() if text.strip()}
    return [_judge(unit, indexes) for unit in units]


def _judge(unit: ClaimUnit, indexes: Mapping[int, _SpanIndex]) -> ClaimJudgement:

    if not unit.figures:
        return ClaimJudgement(
            unit,
            OUTCOME_UNDECIDED,
            None,
            "no printed figure: qualitative claims are the verifier's judgement",
        )

    cited = [number for number in unit.markers if number in indexes]
    if not cited:
        return ClaimJudgement(
            unit,
            OUTCOME_REJECTED,
            REASON_FIGURE_NOT_IN_SPAN,
            "the claim cites no context block",
        )

    index = _SpanIndex("\n\n".join(indexes[number].text for number in sorted(cited)))
    periods = frozenset(unit.periods)

    for figure in unit.figures:
        if figure.percent:
            continue
        matches = index.matches(figure)
        if not matches:
            return ClaimJudgement(
                unit,
                OUTCOME_REJECTED,
                REASON_FIGURE_NOT_IN_SPAN,
                f"{figure.raw} is not printed in the cited span",
            )
        if not periods:
            continue
        consistent = [
            match
            for match in matches
            if index.years_of(match[0]) & periods or not index.years_of(match[0])
        ]
        if not consistent:
            shown = sorted({year for match in matches for year in index.years_of(match[0])})
            shown_text = ", ".join(str(year) for year in shown) or "no labelled period"
            return ClaimJudgement(
                unit,
                OUTCOME_REJECTED,
                REASON_PERIOD_MISMATCH,
                f"{figure.raw} is printed only against period(s) {shown_text}",
            )

    if any(figure.percent for figure in unit.figures):
        return ClaimJudgement(
            unit,
            OUTCOME_UNDECIDED,
            None,
            "a computed rate rides on the claim: the verifier judges computed metrics",
        )

    if len(index.periods()) >= 2 and not periods:
        return ClaimJudgement(
            unit,
            OUTCOME_UNDECIDED,
            None,
            "the span shows several periods and the claim names none",
        )

    return ClaimJudgement(
        unit,
        OUTCOME_SUPPORTED,
        None,
        "every figure is printed in the cited span for the period claimed",
    )


def trim_answer(answer: str, units: Iterable[ClaimUnit]) -> str:

    ordered = sorted(units, key=lambda unit: unit.start, reverse=True)
    result = answer
    for unit in ordered:
        result = result[: unit.start] + unit.prefix + TRIM_MARKER + unit.suffix + result[unit.end :]
    return result


def rejected_units(judgements: Sequence[ClaimJudgement]) -> list[ClaimUnit]:

    return [judgement.unit for judgement in judgements if judgement.outcome == OUTCOME_REJECTED]


def undecided_units(judgements: Sequence[ClaimJudgement]) -> list[ClaimUnit]:
    return [judgement.unit for judgement in judgements if judgement.outcome == OUTCOME_UNDECIDED]


ANNUAL_COMPARATIVE_YEARS = 3


def catalog_years(text: str) -> set[int]:

    years = {int(match.group(0)) for match in _YEAR.finditer(text)}
    if years:
        latest = max(years)
        years.update(range(latest - ANNUAL_COMPARATIVE_YEARS + 1, latest + 1))
    return years


def asked_years(question: str) -> set[int]:

    return {int(match.group(0)) for match in _YEAR.finditer(question)}


def year_coverage_gap(covered: set[int], question: str) -> bool:

    if not covered:
        return False
    asked = asked_years(question)
    if len(asked) != 1:
        return False
    return next(iter(asked)) not in covered

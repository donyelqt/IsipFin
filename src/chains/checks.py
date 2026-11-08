"""The deterministic symbolic layer behind ``groundedness_check`` (ADR-010, task T13).

ADR-010 gives the pipeline its shape: *"deterministic structure owns control and
verification; neural models do the semantic work where they earn their place."* This module
is the verification half of the symbolic column — pure functions, zero network, zero LLM
calls, fully testable under ``MOCK_LLM=1``.

What lives here and why
-----------------------
``groundedness_check`` used to be 100% model, which sat it on the exam's #1 ranked failure
mode (wrong-year figures, ``docs/STRESS-TEST-RAG.md`` row 1). The deterministic pre-pass
that this module implements removes that class of error without any model involved:

1. :func:`split_units` splits a candidate answer into **claim units** — a sentence, a table
   row, or a list item — each carrying the ``[n]`` markers, the printed figures and the
   fiscal periods it asserts.
2. :func:`numeric_pre_pass` judges every unit:
   * **rejected** (``figure_not_in_span``) — a figure the unit prints does not occur in the
     span its ``[n]`` marker points at. A figure with no marker at all is rejected for the
     same reason: there is no cited span to find it in (P5's "a claim with NO [n] marker is
     unsupported").
   * **rejected** (``period_mismatch``) — the figure *is* printed there, but only against a
     different period than the claim asserts. This is the FY2024 ``$245,122M`` asserted as
     FY2023 case: the figure is real, the year is wrong, and a substring check would pass it.
   * **supported** — every figure is printed in the cited span for the period claimed. The
     claim is settled here and never reaches the model.
   * **undecided** — a computed rate (a margin, a growth rate), or a claim naming no period
     while the span shows several. Those are exactly what ADR-010 leaves to the neural
     verifier.

Honest limits (ADR-010, stated rather than hidden)
--------------------------------------------------
The checker validates figures **as printed**. It does not arbitrate computed values, it says
nothing about qualitative claims, and it cannot tell a segment subtotal from a consolidated
total — those are the verifier's job. Its period attribution is deliberately conservative:
a figure is only blamed for a period when the printed layout unambiguously ties it to one
(the figure's own line names the year, or the row's figures align one-to-one with the header
line's year columns). When the layout is ambiguous the checker says *undecided* instead of
guessing, because a wrong rejection destroys a correct claim and a missed rejection is still
caught by the verifier.

The other predicates ADR-010 names for this module (the follow-up heuristic, the tau gate,
the fan-out cap, the context-budget packer) are not duplicated here: each lives with the node
that owns it, so there is exactly one definition of each rule in the tree.
"""

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

# --- the two outcome vocabularies (OBSERVABILITY.md: "figure_not_in_span | period_mismatch")

#: The visible replacement for a trimmed claim (P5: "claims with ``partially``/``unsupported``
#: verdicts are removed from the answer by the trim step and replaced with a visible
#: ``(unsupported claim removed)`` marker").
TRIM_MARKER = "(unsupported claim removed)"

#: Reason codes. Two only: they are the documented debug-panel vocabulary, so a third code
#: would be a doc divergence for no gain.
REASON_FIGURE_NOT_IN_SPAN = "figure_not_in_span"
REASON_PERIOD_MISMATCH = "period_mismatch"

#: What the pre-pass decided about one claim.
OUTCOME_REJECTED = "rejected"
OUTCOME_SUPPORTED = "supported"
OUTCOME_UNDECIDED = "undecided"

# --- printed-figure recognition -------------------------------------------------------

# A scale word or suffix, as the documents print them. The words are matched
# case-insensitively; the one-letter forms case-sensitively, because ``M`` in "211.9M" is a
# scale while ``m`` in "3 m away" is a length.
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

# One printed figure: an optional ``$``, the digits (thousands separators, optional
# decimals), an optional scale and an optional percent sign. The leading lookbehind keeps
# the 2023 in ``FY2023`` and the 13 in ``Note 13`` from being read as a figure of their own.
_FIGURE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<currency>\$)?"
    r"\s*"
    r"(?P<digits>\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*"
    r"(?P<scale>" + _SCALE + r")?"
    r"(?P<percent>%)?"
)

# Exemptions from the column scale: a per-share amount or a percentage is a different unit
# from the column's, the way the corpus says so itself ("In millions, except percentages and
# per share amounts"). The per-share form must name the *measure* ("earnings per share",
# "per diluted share"), not merely the header clause's own words — the header clause itself
# would otherwise exempt every figure under it.
_PER_SHARE = re.compile(
    r"(?:earnings|profit|loss|income|dividend|distribution|net)\s+per\s+(?:share|diluted\s+share|basic\s+share)"
    r"|per\s+(?:diluted|basic)\s+share",
    re.IGNORECASE,
)

# A bare ``19xx``/``20xx`` token with no currency, scale or percent is a period label, not a
# figure — "revenue in 2023", "FY2023", "year ended June 30, 2023". A figure that happens to
# be four digits carries a ``$`` or a unit, and is kept.
_YEAR_TOKEN = re.compile(r"(?:19|20)\d{2}")

# Words that make the number after them a *reference* ("page 60", "Note 13", "Table 3")
# rather than a figure. Answers quote page numbers; those are never the claim.
_REFERENCE_TAIL = re.compile(
    r"(?:\b(?:p|pp|pg|page|pages|note|notes|section|sections|table|tables|figure|figures|"
    r"exhibit|item|items|chapter|vol|volume)\s*[.:]?\s*)$",
    re.IGNORECASE,
)

# ``[n]`` citation markers, stripped before figures or periods are read off a claim.
_MARKER = re.compile(r"\[(\d+)\]")

# A fiscal period as printed: ``FY2023``, ``fiscal year 2023``, ``year ended June 30, 2023``,
# a bare ``2023``. The lookbehind is a *digit* guard rather than a word boundary because the
# corpus writes "FY2023" and no boundary ever appears between the Y and the 2.
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

# The unit phrase that scales every bare figure in a span ("(In millions, except percentages
# and per share amounts)"), so ``$ 211,915`` and ``$211,915 million`` are the same number.
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

# Words that end a sentence but are not sentence ends. A split on "p. 60" would turn a page
# reference into a standalone claim, which the pre-pass would then reject.
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

# How far above a figure line the checker looks for the table header that labels its columns.
_HEADER_LOOKBACK = 8


# --- the value objects -----------------------------------------------------------------


@dataclass(frozen=True)
class Figure:
    """One printed figure, reduced to the numbers needed to compare it against another.

    ``number``/``scale``
        The figure as printed: ``211.9 billion`` is number ``211.9``, scale ``1e9``;
        ``146,052`` printed bare under a "In millions" header is number ``146,052``, scale
        ``1.0``. ``explicit_scale`` records whether the figure declared its own unit, which is
        what separates a figure that names "billion" from one the surrounding table defines.

    ``sig``
        Significant digits as printed — ``211,915`` is 6, ``211.9`` is 4. Two printed forms are
        the same figure when rounding both to the coarser precision agrees, which is why
        ``211.9 billion`` and ``$ 211,915`` (under "In millions") match while
        ``$211.015 billion`` does not.

    ``percent`` marks a rate; those are compared against rates and only against rates.
    """

    raw: str
    number: float
    scale: float
    sig: int
    percent: bool
    currency: bool
    explicit_scale: bool


@dataclass(frozen=True)
class ClaimUnit:
    """One atomic claim inside a candidate answer.

    A unit is a sentence, a whole table row, or a whole list item — the smallest piece that
    can be removed from the answer without leaving syntax behind. ``start``/``end`` bound the
    claim's own text inside the answer; ``prefix``/``suffix`` are the scaffolding that stays
    when it is trimmed (the bullet marker, the table pipes) so the trimmed answer stays valid
    markdown.
    """

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
    """The pre-pass verdict for one :class:`ClaimUnit`.

    ``outcome`` is one of the three constants above; ``reason`` carries the debug-panel code
    when a claim was rejected and is ``None`` otherwise; ``detail`` is the one-line
    explanation that reaches the log and the debug panel.
    """

    unit: ClaimUnit
    outcome: str
    reason: str | None
    detail: str


# --- printed-figure parsing ------------------------------------------------------------


def _sig_digits(digits: str) -> int:
    """Significant digits in a printed number: ``211,915`` → 6, ``211.9`` → 4, ``0.05`` → 1."""

    body = digits.replace(",", "")
    if "." in body:
        whole, _, fraction = body.partition(".")
        body = whole + fraction.rstrip("0")
    body = body.lstrip("0")
    if not body:
        return 1
    return len(body.lstrip("0")) or 1


def _figure_from(match: re.Match[str]) -> Figure:
    """Build a :class:`Figure` from one regex match.

    No unit arithmetic happens here: the figure keeps exactly what it printed, and the scale
    a surrounding header declares is applied afterwards by :func:`rescale` where both sides are
    known. That separation is what keeps "In millions, except percentages" from rescaling a
    rate and a per-share amount into the wrong unit.
    """

    digits = match.group("digits")
    printed_scale = (match.group("scale") or "").lower()
    factor = _SCALE_FACTORS.get(printed_scale, 1.0)
    number = float(digits.replace(",", ""))
    # A parenthesised figure is negative, the way the documents print it.
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
    """The same figure with ``unit_scale`` folded into it.

    A figure is rescaled only when it did not print a unit of its own, is not a rate, and is
    not a per-share amount — exactly the exemptions the corpus's own header clause names
    ("In millions, except percentages and per share amounts"). A figure that names its own
    unit (``211.9 billion``) is already unambiguous and is returned unchanged.
    """

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
    """True when the token at ``match`` is a period label or a page/note reference.

    "page 60" and "Note 13" are pointers a claim legitimately prints, and a bare ``2023`` is a
    period — none of them is the figure the claim rests on, so none of them is matched. A
    four-digit number carrying a ``$`` or a unit is a figure whichever digits it holds.
    """

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
    """Every printed figure in ``text``, in print order.

    Period labels (``FY2023``, a bare ``2023``) and page/section references (``page 60``) are
    deliberately not figures: a claim quoting them asserts no magnitude the span could miss.
    """

    return [
        _figure_from(match)
        for match in _FIGURE_TOKEN.finditer(text)
        if not _is_year_or_reference(text, match)
    ]


def _unit_scale_of(text: str) -> float:
    """The scale the span declares for its bare figures (``(In millions …)`` → ``1e6``)."""

    match = _UNIT_PHRASE.search(text)
    if match is None:
        return 1.0
    word = (match.group(1) or match.group(2) or "").lower()
    return _UNIT_SCALE.get(word, 1.0)


def _per_share_of(text: str) -> bool:
    """True when the span prints per-share amounts, so the column scale does not apply."""

    return _PER_SHARE.search(text) is not None


def _round_sig(value: float, sig: int) -> float:
    if value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    factor = 10.0 ** (exponent - sig + 1)
    return round(value / factor) * factor


def _agrees(claim: Figure, span: Figure) -> bool:
    """True when two printed figures denote the same number.

    A rate is compared against a rate and never against an amount, whatever the digits. For
    amounts the two printed forms are compared at their *coarser* shared precision, in two
    interpretations that the corpus's own notation makes both legitimate:

    * **as printed** — ``211.9 billion`` against ``$211,915 million``, which is the same number
      with both sides' units named;
    * **digit string** — a bare ``146,052`` in a claim against the ``146,052`` printed in a
      column headed "In millions", which is the same digits the claim omitted the unit for.

    Comparing the printed digits as well as the unfolded magnitudes is what stops the common
    case (an answer that restates the corpus's own bare figure) from failing only because the
    answer forgot "million", while still rejecting ``$245,122`` against ``$211,915``.
    """

    if claim.percent != span.percent:
        return False
    if claim.percent and span.percent:
        return claim.number == span.number
    if _same_magnitude(claim.number * claim.scale, claim.sig, span.number * span.scale, span.sig):
        return True
    return _same_digits(claim.number, claim.sig, span.number, span.sig)


def _same_digits(left: float, left_sig: int, right: float, right_sig: int) -> bool:
    """True when two printed number strings agree once both are rounded to the coarser
    precision — the same digits, whatever unit each side declared."""

    if left == right:
        return True
    sig = min(left_sig, right_sig)
    return _round_sig(left, sig) == _round_sig(right, sig)


def _same_magnitude(left: float, left_sig: int, right: float, right_sig: int) -> bool:
    """True when two magnitudes agree once both are rounded to the coarser precision."""

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


# --- the span index --------------------------------------------------------------------


class _SpanIndex:
    """One cited span, indexed for figure lookup and period attribution.

    Everything here is computed once per span and then read: the figure tokens, the period
    labels, the line boundaries, and the unit scale the span declares for bare figures.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        unit_scale = _unit_scale_of(text)
        self.unit_scale = unit_scale
        self._keep: list[tuple[int, int, Figure]] = []
        for match in _FIGURE_TOKEN.finditer(text):
            figure = _figure_from(match)
            if self._is_claim_figure(match, figure):
                # The span's declared column unit is folded in here, once, so both sides of
                # every later comparison are in the same unit. A figure on a per-share line is
                # exempt — the corpus's "except percentages and per share amounts" clause.
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
        """True when the figure sits on a per-share line, so the column unit does not apply.

        The exemption is read off the figure's **own line**, not the whole span: "Diluted
        earnings per share" prints dollars-and-cents under a header that says "In millions",
        and rescaling it would turn ``9.68`` into ``9,680,000``. Reading only the line is also
        what keeps the header clause's own words ("except percentages and per share amounts")
        from exempting every figure below it.
        """

        line_start = self.text.rfind("\n", 0, match.start()) + 1
        head = self.text[line_start : match.start()]
        return _per_share_of(head)

    def _is_claim_figure(self, match: re.Match[str], figure: Figure) -> bool:
        """False for the two tokens that are not claims: a suppression note and a pointer.

        A parenthesised ``(1)%`` is a footnote the table carries, not a figure (the corpus
        prints it against "Net income 72,361 72,738 (1)%"); a bare ``19xx`` after "page"/"Note"
        is a reference. Everything else that matched :data:`_FIGURE_TOKEN` is a printed figure.
        """

        start = match.start()
        if figure.percent and self.text[start - 1 : start] == "(":
            return False
        return not _is_year_or_reference(self.text, match)

    def _years_of_line(
        self, start: int, end: int, tokens: Sequence[tuple[int, int, Figure]]
    ) -> list[tuple[int, int]]:
        """Period labels on one line, excluding any that overlap a figure token."""

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
        """Span occurrences that denote the same number as ``claim``."""

        return [token for token in self._keep if _agrees(claim, token[2])]

    def years_of(self, offset: int) -> frozenset[int]:
        """The periods the printed layout ties a figure at ``offset`` to.

        Deterministic, and deliberately silent when the layout is ambiguous. Three cases, in
        order of confidence:

        * the figure's **own line** names a period (prose, or a table whose rows carry their
          year) — the nearest one wins, so "… $198,270 million in FY2022" attributes 2022;
        * the nearest **header line above it** names years, and the figures on the row align
          one-to-one with them (the amount columns, which is what "In millions" covers: a rate
          or a per-share figure is a narrower unit and is not part of the correspondence) —
          the ordinal match decides, which is how the corpus's two-column income statement is
          read;
        * anything else returns the empty set, and the caller reports *undecided* rather than
          blaming a figure for a period the layout does not tie it to.
        """

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
        """The figures on a row that participate in the year-column correspondence.

        The amount columns, which is what the header's unit clause covers. A rate
        ("Percentage Change") is a different unit and is not part of the correspondence, so
        counting it would break the one-to-one match and make the checker refuse a period it
        can actually read. A per-share row is its own correspondence: "Diluted earnings per
        share 9.68 9.65" has two figures and two year columns exactly like "Revenue $ 211,915
        $ 198,270" does.
        """

        return [token for token in self._line_figures.get(line_number, []) if not token[2].percent]

    def periods(self) -> frozenset[int]:
        """Every period the span displays, wherever it is printed."""

        return frozenset({year for years in self._line_years.values() for _, year in years})


def _line_bounds(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` per line, newline excluded, so line arithmetic never sees one."""

    bounds: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char == "\n":
            bounds.append((start, index))
            start = index + 1
    bounds.append((start, len(text)))
    return bounds


# --- claim units -----------------------------------------------------------------------


def split_units(answer: str) -> list[ClaimUnit]:
    """Split a candidate answer into claim units, in document order.

    Markdown structure is respected, because a trim that shreds it produces an answer the UI
    cannot render: a table row is one unit (its pipes survive the trim), a list item is one
    unit (its bullet survives), and a paragraph is one unit per sentence.

    A contiguous markdown table is read as a block. A row's claim is judged against the table
    the row lives in, so the row inherits

    * the markers written anywhere on the table, or on the line immediately introducing it —
      citations normally sit on the sentence that introduces a table, not on each row;
    * the period labels the **header row** names (``| Metric | FY2023 | FY2022 |``), which is
      what makes a two-column income statement's figures attributable to a period.

    That inheritance is what keeps the common answer shape — a markdown table introduced by a
    cited sentence — from being trimmed wholesale for carrying no marker of its own.
    """

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
                # The line introducing the table carries the citation the whole table rests
                # on; a blank line may sit between it and the first row.
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
    """Turn one contiguous markdown table into one unit per body row."""

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
    """The period labels a table's **header** row names, in print order.

    The header is the first row whose cells are not all numbers — the row of ``|---|---|---|``
    separators and any row that starts with a metric name rather than a figure. Only the
    header's labels are used: a period printed in a body row is already that row's own claim.
    """

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
    """True for the ``|---|---|`` row that separates a table's header from its body."""

    return all(cell and set(cell) <= {"-", ":", " "} for cell in cells)


def _split_line(line: str, base: int, units: list[ClaimUnit]) -> None:
    """Append the units one non-table source line contributes; blank lines contribute none.

    A list item is one unit (its bullet survives a trim); a paragraph is one unit per
    sentence.
    """

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
    """``(start, end)`` per sentence in one line, on ``[.!?]`` followed by whitespace.

    A punctuation run that closes an abbreviation ("p.", "Inc.", "e.g.") is not a sentence
    end: splitting there would detach "60" from "p. 60" and hand the pre-pass a page number
    to reject.
    """

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
    """True for a token whose closing dot is an abbreviation marker, not a sentence end."""

    if len(word) <= 1:
        return True
    return word.rstrip(".").lower() in _ABBREVIATIONS


# --- the numeric pre-pass --------------------------------------------------------------


def numeric_pre_pass(answer: str, spans: Mapping[int, str]) -> list[ClaimJudgement]:
    """Judge every claim in ``answer`` against the spans its ``[n]`` markers point at.

    Zero LLM calls, zero network: the whole function is :func:`split_units` plus string
    arithmetic, so it runs identically under ``MOCK_LLM=1`` and in production and is a pure
    function of its arguments.

    Args:
        answer: The candidate answer's markdown, exactly as synthesis produced it.
        spans: ``marker number → the context block synthesis numbered with that marker``.
            A number absent here means the answer cited a block that does not exist, which
            counts the same as citing nothing.

    Returns:
        One :class:`ClaimJudgement` per claim unit, in document order.
    """

    units = split_units(answer)
    indexes = {number: _SpanIndex(text) for number, text in spans.items() if text.strip()}
    return [_judge(unit, indexes) for unit in units]


def _judge(unit: ClaimUnit, indexes: Mapping[int, _SpanIndex]) -> ClaimJudgement:
    """The pre-pass verdict for one claim unit.

    The cited blocks are joined into one index rather than searched one by one, because the
    printed layout — and therefore a figure's period attribution — is a property of the block
    the figure sits in, not of the claim that cited it.
    """

    if not unit.figures:
        # A claim with nothing printed in it — qualitative. Only the verifier can weigh it.
        return ClaimJudgement(
            unit,
            OUTCOME_UNDECIDED,
            None,
            "no printed figure: qualitative claims are the verifier's judgement",
        )

    cited = [number for number in unit.markers if number in indexes]
    if not cited:
        # No marker, or a marker for a block that does not exist. There is no cited span, so
        # no figure can be in one — P5's primary rule, settled without a model.
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
            # A rate is arithmetic over figures, not a figure of its own: ADR-010 leaves it to
            # the verifier, which can check the arithmetic. It is never rejected for being
            # absent from the span — that absence is the whole point of a computed metric.
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
        # Every occurrence of the figure must either match the claimed period or be printed
        # where no period is labelled. One match is not enough: the wrong-year figure is
        # usually printed *somewhere* in the block, which is precisely the trap.
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
        # Every printed figure in the claim is present, but the claim also states a computed
        # rate — a margin, a growth rate. Whether that rate is the right arithmetic is exactly
        # the judgement ADR-010 keeps neural, so the claim is handed on rather than settled.
        return ClaimJudgement(
            unit,
            OUTCOME_UNDECIDED,
            None,
            "a computed rate rides on the claim: the verifier judges computed metrics",
        )

    if len(index.periods()) >= 2 and not periods:
        # P5: "a span containing FY2023 and FY2022 figures supports only claims that name
        # which period they use." Silent claims from a multi-period span are not settled here.
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


# --- the trim step ---------------------------------------------------------------------


def trim_answer(answer: str, units: Iterable[ClaimUnit]) -> str:
    """Replace each listed claim's text with :data:`TRIM_MARKER`, leaving the rest alone.

    Offsets are spliced from the back so a removed claim never shifts the position of a claim
    later in the answer. ``prefix``/``suffix`` scaffolding survives, so a trimmed bullet
    stays a bullet and a trimmed table row stays a row.
    """

    ordered = sorted(units, key=lambda unit: unit.start, reverse=True)
    result = answer
    for unit in ordered:
        result = result[: unit.start] + unit.prefix + TRIM_MARKER + unit.suffix + result[unit.end :]
    return result


def rejected_units(judgements: Sequence[ClaimJudgement]) -> list[ClaimUnit]:
    """The claim units the pre-pass rejected, in document order."""

    return [judgement.unit for judgement in judgements if judgement.outcome == OUTCOME_REJECTED]


def undecided_units(judgements: Sequence[ClaimJudgement]) -> list[ClaimUnit]:
    return [judgement.unit for judgement in judgements if judgement.outcome == OUTCOME_UNDECIDED]


# --- the fiscal-year coverage gate ----------------------------------------------------


#: How many fiscal years a single annual report's statements span. An annual report prints
#: its reporting year alongside two years of comparatives, so a catalog naming one report
#: year still stands for three. Mirrored from ``refuse``'s own ``_ANNUAL_COMPARATIVE_YEARS``
#: (with the reaching range) rather than imported, because this module is the LLM-free
#: symbolic layer and must not depend on a node: the number is a property of the *document
#: format*, not of either consumer, and a third consumer would only duplicate it again.
ANNUAL_COMPARATIVE_YEARS = 3


def catalog_years(text: str) -> set[int]:
    """Every fiscal year the catalog's own text shows, plus the years the newest one reaches.

    The catalog is the ingest-time record of what is on disk — filenames and section paths —
    and it is the *same* text ``refuse`` reads to write "the documents cover fiscal years
    FY2021-FY2023". A single report year stands for the span an annual report's statements
    carry (its own year plus two comparatives), which is why ``"2023_Annual_Report.pdf"``
    reaches back to FY2021 rather than claiming coverage of exactly one year. Anything beyond
    that is a guess, and a shorter set would refuse a year the corpus actually answers.
    """

    years = {int(match.group(0)) for match in _YEAR.finditer(text)}
    if years:
        latest = max(years)
        years.update(range(latest - ANNUAL_COMPARATIVE_YEARS + 1, latest + 1))
    return years


def asked_years(question: str) -> set[int]:
    """Every fiscal year the question names, in any of the corpus's printed four-digit forms.

    ``FY2023``, ``fiscal year 2023`` and a bare ``2023`` all count. The digit lookarounds are
    the same guards ``refuse`` and the pre-pass use, so a figure like ``211,915`` is never read
    as a year, and the corpus's own conventions decide what a period looks like rather than a
    new pattern this module invents.
    """

    return {int(match.group(0)) for match in _YEAR.finditer(question)}


def year_coverage_gap(covered: set[int], question: str) -> bool:
    """True when the question asks about a period the documents demonstrably do not cover.

    The deterministic fiscal-year gate: :data:`_YEAR`-shaped labels read out of the question
    are compared against the years the catalog shows. It is the same class of check the numeric
    pre-pass already performs for figures — a symbolic predicate settled **before** any model
    call — extended to the routing decision ADR-001 otherwise leaves entirely to the planner.

    What it deliberately does *not* do:

    * **It never moves a year the catalog covers.** A question naming no year at all
      ("what was revenue?") has no gap, so the planner keeps its decision. The gate only ever
      adds a refusal, and only for a question whose asked period lies outside the corpus.
    * **It never evaluates a comparison.** A question naming several years is asking about
      more than one of them, so a single-year gap would refuse a comparison the documents can
      honestly answer. Those belong to the ``multi_part`` path, which is the planner's call
      and stays the planner's.
    * **It never fires on a period the catalog does not describe.** An empty or unreadable
      catalog shows no years, so there is no coverage to compare against and no gap: the
      planner's judgement stands, and ``refuse`` renders its own "cannot be determined" copy.
    * **It never touches an entity- or topic-gap question.** Those are the other two refusal
      classes ADR-001 names and stay the planner's, because the catalog carries no entity
      list to compare a company name against symbolically.
    """

    if not covered:
        return False
    asked = asked_years(question)
    if len(asked) != 1:
        return False
    return next(iter(asked)) not in covered

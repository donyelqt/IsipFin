"""The ``refuse`` node: an honest, deterministic refusal (task T15, ADR-001's terminal branch).

``refuse`` is the graph's out-of-scope sink (``analyze_question`` → ``refuse`` → END) and the
one node that runs with **no LLM call**: every other node needs a model because it has to
understand something, and a refusal has to understand nothing. That is the point of the node —
a refusal is a pure function of the plan, the catalog, and the question, so the same question
always produces the same words (ADR-010: deterministic structure owns control, and this is
100% symbolic).

Why it is deterministic rather than a generated apology
-------------------------------------------------------
ADR-007's hard rule is "no failure path yields an empty answer", and the failure mode ranked
just above it (``docs/STRESS-TEST-RAG.md`` row 7) is the refusal that *reads* fine but is
the wrong refusal: "I can't answer that" with no reason leaves the user guessing whether the
documents are missing the year, the subject, or the company. ``analyze_question`` already
recorded which — ``QueryPlan.missing_aspect`` is ``time_period`` / ``topic`` / ``entity`` /
``None`` — so this node renders that verdict as copy that **names what the documents do
cover**, differentiated by aspect:

* ``time_period`` — "the documents cover fiscal years FY2021-FY2023". The asked period is
  never echoed: the acceptance case ("What was revenue in 2019?") must not push the
  uncovered year back into the copy, because a year appearing in an answer suggests the
  answer knows something about that year.
* ``topic`` — "these documents cover <subjects>", the subjects read from the catalog's
  section paths, so the user learns what the corpus is instead of only what it isn't.
* ``entity`` — "the question concerns <entity>, an entity not in the documents". The entity
  is read out of the standalone question (the planner compared it against the catalog and
  found no match), never invented.
* ``None`` — generic: the plan refused for a reason it did not record, and this node does
  not guess one.

Coverage comes from the catalog itself (filename plus section paths). ``DocSummary`` carries
no fiscal-period field, so reading the catalog is the only option that keeps the refusal
honest: a range the catalog does not show would make the refusal lie about the corpus, which
is the one thing a refusal exists to avoid. A catalog with no years at all says so.

State contract (``docs/ARCHITECTURE.md`` Node Contract Table)
-------------------------------------------------------------
| Reads | Writes |
|---|---|
| ``query_plan``, ``doc_catalog``, ``standalone_question`` | ``answer_md``, ``refusal_reason`` |

``citations`` is written empty because a refusal reasons from the catalog and the plan, never
from a document span — a citation here would send the user to a page that says nothing about
their question. ``grounded`` is written ``True``: an honest refusal is the correct result on
the path that was supposed to fire, not a failed turn. The node never raises and never emits
an empty ``answer_md``: a blank answer is the one outcome ADR-007 forbids, and a refusal is
the answer that must be rendered instead.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from typing import Any

from src.chains.state import DocSummary, GraphState, QueryPlan, section_paths
from src.utils.tracing import trace_node

__all__ = ["refuse"]

# --- the refusal-reason vocabulary (a superset of P1's missing_aspect values) --------
#: ``missing_aspect`` values this node renders as differentiated copy. Mirrored as literals
#: because the value arrives from a schema-validated planner reply, not from this module.
TIME_PERIOD = "time_period"
TOPIC = "topic"
ENTITY = "entity"
#: Recorded when the router sent a plan here with no ``missing_aspect``: the refusal stays
#: generic rather than guessing why the question was classified out of scope.
UNSPECIFIED = "unspecified"
#: Recorded when the catalog is empty: what is missing is the corpus, not a period or topic
#: of a document that exists. Kept distinct so the debug panel can tell the two apart.
NO_DOCUMENTS = "no_documents"

#: A four-digit year, guarded on both sides so "FY2023" and "211,915" do not both match.
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
#: A word that is all caps: section paths are printed in the document's own casing, so an
#: all-caps token is catalog styling rather than a capitalised proper noun.
_UPPER = re.compile(r"^[^a-z]+$")
#: Filename extension of an uploaded document, dropped when the filename names a subject.
_PDF_SUFFIX = re.compile(r"\.pdf$", re.IGNORECASE)
#: The question's first line: a question is one sentence, and a newline usually means it is
#: a paste of several.
_LINE_SPLIT = re.compile(r"[\n\r]")
#: Words that name a quarter/month but say nothing about what a corpus covers.
_REPORT_STEMS = frozenset({"annual", "report", "filing", "form", "statement"})

#: How many fiscal years an annual report's statements span. An annual report prints its
#: reporting year alongside two years of comparatives, so a catalog naming one report year
#: still stands for three — this is what turns an "FY2023" catalog into the FY2021-FY2023
#: coverage the refusal copy names (the corpus's own note in ``mock_llm``: "the reference
#: report covers FY2021-FY2023").
_ANNUAL_COMPARATIVE_YEARS = 3

#: Names belonging to the corpus's own subject. A question naming one of these is *about*
#: the corpus, so the entity copy must not read it back as an outside entity.
_CORPUS_ENTITIES: frozenset[str] = frozenset(
    {
        "microsoft",
        "msft",
        "windows",
        "azure",
        "office",
        "xbox",
        "linkedin",
        "github",
        "activision",
        "blizzard",
    }
)
#: Question filler and the words a refusable question is built from. None of them is an
#: entity a question can be about.
_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "according",
        "across",
        "and",
        "any",
        "are",
        "as",
        "at",
        "average",
        "be",
        "between",
        "both",
        "but",
        "by",
        "can",
        "compare",
        "compared",
        "did",
        "do",
        "does",
        "each",
        "explain",
        "figure",
        "figures",
        "for",
        "from",
        "give",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "much",
        "number",
        "of",
        "on",
        "or",
        "over",
        "per",
        "period",
        "reported",
        "reports",
        "results",
        "say",
        "show",
        "state",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "there",
        "this",
        "through",
        "to",
        "total",
        "vs",
        "was",
        "were",
        "what",
        "what's",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "yoy",
    }
)


# --- catalog readers (pure, deterministic) ------------------------------------------


def _documents(state: GraphState) -> list[DocSummary]:
    """The catalog's :class:`DocSummary` entries, ignoring anything else in the field.

    ``invoke`` types ``doc_catalog`` as ``list[Any]`` at the boundary (``docs/SPEC.md`` §9
    keeps the envelope tolerant of a partially-populated state), so a wrongly-typed entry
    simply contributes no coverage instead of raising. A catalog the node cannot read is the
    empty catalog, which has its own honest refusal.

    A summary whose ``sections`` is the *flat* list ``docs.jsonl`` stores is repaired rather
    than dropped, through :func:`~src.chains.state.section_paths`: this node must never raise
    (ADR-007), and dropping such an entry would trade a wrong refusal for an emptier one — the
    flat shape would otherwise render ``path[0]`` as a single character, so the copy named
    that character as a subject. The repair reads the invariant back off the type
    (:class:`DocSummary` is a path of paths), not off a defensive branch in each reader.
    """

    raw = state.get("doc_catalog")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        return []
    return [_repaired(entry) for entry in raw if isinstance(entry, DocSummary)]


def _repaired(doc: DocSummary) -> DocSummary:
    """One catalog entry with ``sections`` resolved to its declared shape, paths of headings.

    The construction sites are the fix (``section_paths`` is the single conversion they all
    call); this is the reader's own boundary still refusing to lie about coverage if a
    flat-shaped summary ever reaches it. Already well-formed summaries are returned
    unchanged, so the repair costs one ``isinstance`` pass on every element.
    """

    if all(isinstance(path, tuple) for path in doc.sections):
        return doc
    return dataclasses.replace(doc, sections=section_paths(doc.sections))


def _catalog_text(docs: Sequence[DocSummary]) -> str:
    """The catalog's machine-readable text: every filename and section path, space-joined.

    This is all the text a :class:`DocSummary` carries, so it is also all the text a
    coverage read could be grounded in.
    """

    parts: list[str] = []
    for doc in docs:
        parts.append(doc.filename)
        parts.extend(" > ".join(path) for path in doc.sections)
    return " ".join(parts)


def _subject_headings(docs: Sequence[DocSummary]) -> list[str]:
    """The catalog's top-level section headings, de-duplicated in first-seen order.

    A section path is a nesting chain, so its root is the subject and its tail is the
    detail; the roots are therefore the list of subjects the corpus covers, which is what
    the topic copy names. An entry with no path contributes nothing.
    """

    heads: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        for path in doc.sections:
            if not path:
                continue
            head = path[0].strip()
            if head and head not in seen:
                seen.add(head)
                heads.append(head)
    return heads


def _filename_subject(docs: Sequence[DocSummary]) -> str:
    """The first catalog filename's subject words, or ``""`` when there is nothing to say.

    ``"2023_Annual_Report.pdf"`` reads as "2023 Annual Report": the extension is dropped,
    separators collapse, and generic report stems are kept only when nothing else remains
    (a filename of nothing but stems is better reported than dropped).
    """

    for doc in docs:
        stem = _PDF_SUFFIX.sub("", doc.filename).strip()
        if not stem:
            continue
        words = re.findall(r"[^\W_]+", stem)
        if not words:
            continue
        if all(_UPPER.match(word) for word in words):
            words = [word.capitalize() for word in words]
        subject = [word for word in words if word.lower() not in _REPORT_STEMS]
        return " ".join(subject or words)
    return ""


def _subjects_of(docs: Sequence[DocSummary]) -> list[str]:
    """The catalog's subject names: section roots first, the filename as the fallback."""

    heads = _subject_headings(docs)
    if heads:
        return heads
    named = _filename_subject(docs)
    return [named] if named else []


# --- the question's own words (used by the entity copy, never invented) -------------


def _entity_mentions(question: str) -> list[str]:
    """The question's outside-entity names, read out of the question's own words.

    The proper-noun signal is a capital that is not the first word of the sentence and not
    catalog styling (all-caps), minus corpus names and question filler. Everything else a
    question contains — metrics, periods, verbs, numbers — is what a question is made of
    rather than who it is about. When nothing survives the filters there is no name to
    report and the entity copy says so instead of guessing.
    """

    line = _LINE_SPLIT.split(question.strip(), maxsplit=1)[0]
    mentions: list[str] = []
    for match in re.finditer(r"[^\W\d_]+(?:'[^\W\d_]+)?", line):
        word = match.group(0)
        index = match.start()
        tail = line[match.end() :]
        # "'s" is a possessive marker, never a name: "Amazon's revenue" names Amazon once.
        if word.endswith("'s") and len(word) > 2:
            word = word[:-2]
        elif tail[:2] == "'s":
            word = word[:-2] if word.lower().endswith("s") and len(word) > 2 else word
            if word.lower() in _CORPUS_ENTITIES or word.lower() in _QUESTION_WORDS:
                continue
            tail = tail[2:]
        if not word:
            continue
        if word.lower() in _CORPUS_ENTITIES or word.lower() in _QUESTION_WORDS:
            continue
        if _UPPER.match(word):
            continue
        if index == 0 or line[index - 1] in ".?!:;,":
            continue
        # A name is capitalised; anything lowercase is a metric, a verb, or a filler word
        # the question happens to use.
        if not word[0].isupper():
            continue
        if word not in mentions:
            mentions.append(word)
    return mentions


def _join(words: Sequence[str]) -> str:
    """``[a, b, c]`` as ``"a, b, and c"``; a single word passes through as itself.

    The final separator is ``", and "`` rather than ``" and "`` so the boundary between the
    last two items stays visible when the items are phrases rather than words.
    """

    if len(words) <= 1:
        return "".join(words)
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return f"{', '.join(words[:-1])}, and {words[-1]}"


# --- the coverage copy --------------------------------------------------------------


def _fiscal_years(docs: Sequence[DocSummary]) -> set[int]:
    """The fiscal years the catalog's own text shows, plus the years the newest one reaches.

    A catalog built at ingest time records a report's *filename* and *sections*, and an
    annual report's name carries one year — its reporting year. Its comparatives reach
    earlier years (the FY2023 report prints FY2022 and FY2021 columns), so the honest
    complement of the newest year is the three years an annual report's statements span.
    Anything beyond that would be a guess, and a shorter set would refuse a year the corpus
    actually answers, which is the failure the design guards against with a "should answer"
    golden set.
    """

    years = {int(match) for match in _YEAR.findall(_catalog_text(docs))}
    if years:
        latest = max(years)
        years.update(range(latest - _ANNUAL_COMPARATIVE_YEARS + 1, latest + 1))
    return years


def _span_of(years: set[int]) -> str:
    """Years as ``FY2021-FY2023`` when they are consecutive, else as an ordered list.

    The corpus writes the range with a hyphen in its own code (``mock_llm``: "covers
    FY2021-FY2023") and the design docs with an en dash; the hyphen is the form used here
    because it is the one that already exists in code. A non-consecutive set (FY2021 and
    FY2023 with FY2022 absent) is listed rather than collapsed, because a collapsed range
    would claim coverage the catalog does not show.
    """

    ordered = sorted(years)
    if len(ordered) > 2 and ordered[-1] - ordered[0] == len(ordered) - 1:
        return f"FY{ordered[0]}-FY{ordered[-1]}"
    return ", ".join(f"FY{year}" for year in ordered)


def _period_copy(docs: Sequence[DocSummary]) -> str:
    """The ``time_period`` coverage: the fiscal years the documents cover, never the asked one.

    An empty year set prints the honest gap instead of a fabricated range.
    """

    years = _fiscal_years(docs)
    if years:
        return f"the documents cover fiscal years {_span_of(years)}"
    return "the documents' fiscal years cannot be determined from the document catalog"


def _topic_copy(docs: Sequence[DocSummary]) -> str:
    """The ``topic`` coverage: the subjects the documents do cover, in catalog order."""

    subjects = _subjects_of(docs)
    if not subjects:
        return "the documents' subjects cannot be determined from the document catalog"
    # Section paths are stored in the document's own casing (all-caps), so they are
    # lower-cased here: the copy is prose, not a dump of the catalog. Subjects that are
    # themselves phrases are separated by semicolons rather than the ", and" a list of
    # plain words would use, so two long headings never read as one sentence.
    lowered = [subject.lower() for subject in subjects]
    if any(" " in subject for subject in lowered):
        return f"these documents cover {'; '.join(lowered)}"
    return f"these documents cover {_join(lowered)}"


def _entity_copy(question: str) -> str:
    """The ``entity`` coverage: the outside entity the question concerns, read from the question."""

    mentions = _entity_mentions(question)
    if mentions:
        named = _join(mentions)
        return f"the question concerns {named}, an entity not in the documents"
    return "the question concerns an entity not in the documents"


def _coverage_copy(missing_aspect: str, question: str, docs: Sequence[DocSummary]) -> str:
    """The sentence that names what the documents do cover, for ``missing_aspect``."""

    if missing_aspect == TIME_PERIOD:
        return _period_copy(docs)
    if missing_aspect == TOPIC:
        return _topic_copy(docs)
    if missing_aspect == ENTITY:
        return _entity_copy(question)
    return "the question is outside the scope of the documents in this session"


def _closing() -> str:
    """The closing sentence, present on every refusal: why the answer carries no citations."""

    return (
        "No citations are given because none would support an answer: the answer was not "
        "found, it was not invented."
    )


@trace_node("refuse")
def refuse(state: GraphState) -> dict[str, Any]:
    """Refuse an out-of-scope question honestly, naming what the documents *do* cover.

    Fully deterministic: no LLM call, no network, no import of the gateway, so the copy is
    identical for identical inputs and the node works offline under ``MOCK_LLM=1``. Never
    raises and never returns an empty ``answer_md`` — the reachable outcomes are the four
    aspect-differentiated refusals plus the empty-catalog refusal, and each of them is a
    real result rather than a failure path.
    """

    plan = state.get("query_plan")
    plan = plan if isinstance(plan, QueryPlan) else None
    question = state.get("standalone_question") or state.get("question") or ""
    question = question.strip()
    docs = _documents(state)

    if not docs:
        # Nothing uploaded: the documents themselves are what's missing, not a period or a
        # subject of a document that exists.
        answer_md = (
            "**Refused** — there are no documents in this session, so there is nothing to "
            f"answer from. Upload a document, then ask again. {_closing()}"
        )
        return {
            "answer_md": answer_md,
            "citations": [],
            "grounded": True,
            "refusal_reason": NO_DOCUMENTS,
        }

    missing_aspect = plan.missing_aspect if plan is not None else None
    answer_md = (
        "**Refused** — I can't answer that: "
        f"{_coverage_copy(missing_aspect or '', question, docs)}. {_closing()}"
    )
    return {
        "answer_md": answer_md,
        "citations": [],
        "grounded": True,
        "refusal_reason": missing_aspect or UNSPECIFIED,
    }

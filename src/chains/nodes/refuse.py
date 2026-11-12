from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from typing import Any

from src.chains.state import DocSummary, GraphState, QueryPlan, section_paths
from src.utils.tracing import trace_node

__all__ = ["refuse"]

TIME_PERIOD = "time_period"
TOPIC = "topic"
ENTITY = "entity"
UNSPECIFIED = "unspecified"
NO_DOCUMENTS = "no_documents"

_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_UPPER = re.compile(r"^[^a-z]+$")
_PDF_SUFFIX = re.compile(r"\.pdf$", re.IGNORECASE)
_LINE_SPLIT = re.compile(r"[\n\r]")
_REPORT_STEMS = frozenset({"annual", "report", "filing", "form", "statement"})

_ANNUAL_COMPARATIVE_YEARS = 3

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


def _documents(state: GraphState) -> list[DocSummary]:

    raw = state.get("doc_catalog")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        return []
    return [_repaired(entry) for entry in raw if isinstance(entry, DocSummary)]


def _repaired(doc: DocSummary) -> DocSummary:

    if all(isinstance(path, tuple) for path in doc.sections):
        return doc
    return dataclasses.replace(doc, sections=section_paths(doc.sections))


def _catalog_text(docs: Sequence[DocSummary]) -> str:

    parts: list[str] = []
    for doc in docs:
        parts.append(doc.filename)
        parts.extend(" > ".join(path) for path in doc.sections)
    return " ".join(parts)


def _subject_headings(docs: Sequence[DocSummary]) -> list[str]:

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
    heads = _subject_headings(docs)
    if heads:
        return heads
    named = _filename_subject(docs)
    return [named] if named else []


def _entity_mentions(question: str) -> list[str]:

    line = _LINE_SPLIT.split(question.strip(), maxsplit=1)[0]
    mentions: list[str] = []
    for match in re.finditer(r"[^\W\d_]+(?:'[^\W\d_]+)?", line):
        word = match.group(0)
        index = match.start()
        tail = line[match.end() :]
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
        if not word[0].isupper():
            continue
        if word not in mentions:
            mentions.append(word)
    return mentions


def _join(words: Sequence[str]) -> str:

    if len(words) <= 1:
        return "".join(words)
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return f"{', '.join(words[:-1])}, and {words[-1]}"


def _fiscal_years(docs: Sequence[DocSummary]) -> set[int]:

    years = {int(match) for match in _YEAR.findall(_catalog_text(docs))}
    if years:
        latest = max(years)
        years.update(range(latest - _ANNUAL_COMPARATIVE_YEARS + 1, latest + 1))
    return years


def _span_of(years: set[int]) -> str:

    ordered = sorted(years)
    if len(ordered) > 2 and ordered[-1] - ordered[0] == len(ordered) - 1:
        return f"FY{ordered[0]}-FY{ordered[-1]}"
    return ", ".join(f"FY{year}" for year in ordered)


def _period_copy(docs: Sequence[DocSummary]) -> str:

    years = _fiscal_years(docs)
    if years:
        return f"the documents cover fiscal years {_span_of(years)}"
    return "the documents' fiscal years cannot be determined from the document catalog"


def _topic_copy(docs: Sequence[DocSummary]) -> str:

    subjects = _subjects_of(docs)
    if not subjects:
        return "the documents' subjects cannot be determined from the document catalog"
    lowered = [subject.lower() for subject in subjects]
    if any(" " in subject for subject in lowered):
        return f"these documents cover {'; '.join(lowered)}"
    return f"these documents cover {_join(lowered)}"


def _entity_copy(question: str) -> str:
    mentions = _entity_mentions(question)
    if mentions:
        named = _join(mentions)
        return f"the question concerns {named}, an entity not in the documents"
    return "the question concerns an entity not in the documents"


def _coverage_copy(missing_aspect: str, question: str, docs: Sequence[DocSummary]) -> str:

    if missing_aspect == TIME_PERIOD:
        return _period_copy(docs)
    if missing_aspect == TOPIC:
        return _topic_copy(docs)
    if missing_aspect == ENTITY:
        return _entity_copy(question)
    return "the question is outside the scope of the documents in this session"


def _closing() -> str:

    return (
        "No citations are given because none would support an answer: the answer was not "
        "found, it was not invented."
    )


@trace_node("refuse")
def refuse(state: GraphState) -> dict[str, Any]:

    plan = state.get("query_plan")
    plan = plan if isinstance(plan, QueryPlan) else None
    question = state.get("standalone_question") or state.get("question") or ""
    question = question.strip()
    docs = _documents(state)

    if not docs:
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

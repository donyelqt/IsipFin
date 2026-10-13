"""Offline, deterministic scripted LLM for the ``MOCK_LLM=1`` path (ADR-005).

The mock exists so the app demos, the graph tests run, and CI pass with **no API key and
no network at all** — a reviewer who never touches an OpenAI-compatible endpoint sees real
answers with real FY2023 figures.

Contract:

* **Deterministic.** A response is a pure function of the prompt payload and the request
  schema. Identical inputs give byte-identical :class:`~src.utils.llm.ChatResult` output
  across calls, processes, platforms and runs — the fingerprint is sha256 over canonical
  JSON, never :func:`hash`, and no clock, no RNG and no environment read enters the math.
  (A wall-clock stamp would make "identical results across two calls" false, so latency is
  the one synthetic field: a fixed positive constant, stable for every fingerprint.)
* **Offline.** No HTTP client, no ``openai`` import, no socket. ``MOCK_LLM=1`` therefore
  cannot be broken by a missing key, a blocked port, an empty quota or a fenced grader
  network (ADR-005: "a missing key can never produce a broken ``docker compose up`` demo").
* **Fingerprinted.** ADR-005 keys responses on *node name + schema presence*. The node
  name is not passed explicitly — it is recovered from the prompt payload, which carries
  the loaded system prompt verbatim, so the JSON prompts (P1, P5) select structured
  objects and the markdown prompts (P2, P3) select free-form answers.
* **Credible.** Known financial questions embed the real FY2023 figures from the corpus
  income statement, and every scripted citation points at a numbered context block that
  actually contains the figure — the mock never produces a number the prompt did not
  supply.

Beyond the frozen figures the enrichment is deliberately tiny: the mock does not invent
document snippets, section paths or pages. When synthesis is called with retrieved
context, citations are numbered against those blocks and unscripted questions are answered
by restating the context's own figures.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from src.utils.llm import ChatResult, LLMParams, validate_json_schema

__all__ = ["MockLLM", "prompt_fingerprint"]

# --- prompt identity -------------------------------------------------------------
# One recognizable substring per prompt family, checked against the system prompt the node
# loaded. Order matters only for ties, which do not occur: each marker is a clause unique
# to its prompt. Rewording a prompt changes its marker and therefore its scripted answer,
# which is deliberate — a stale fixture must answer differently, not stay silently green.
_FAMILY_MARKERS: tuple[tuple[str, str], ...] = (
    # P1 (query-analysis.md): "Your job: decide HOW to answer, not to answer."
    ("query_plan", "decide how to answer, not to answer"),
    # P5 (summarization-and-verification.md), part A.
    ("grounding", "fully supported by the provided context"),
    # P2 (follow-up-contextualization.md).
    ("contextualization", "rewrite follow-up questions into standalone questions"),
    # P3 (answer-synthesis-with-citations.md).
    ("synthesis", "answering questions strictly from the provided document context"),
)

# Schema-signature families (ADR-005's "schema presence" half of the fingerprint): the
# required top-level keys of the two structured outputs in src/chains/schemas.py.
_SCHEMA_FAMILIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("query_plan", frozenset({"kind", "sub_queries", "required_sections"})),
    ("grounding", frozenset({"claims", "all_supported"})),
)

# --- known financial questions (the reference corpus) -----------------------------
# Exact FY2023 income-statement lines, as printed in the corpus, used by the known-question
# script. `figure` is the printed form (the corpus renders dollars as "$ 211,915"), so a
# real retrieval pass and the script cannot contradict each other; the `answer` is the
# prose the demo prints, with millions added where the movie renders it that way.
KNOWN_QUESTIONS: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (
        ("compare operating income", "operating income comparison"),
        {
            "figure": "$ 88,523",
            "answer": (
                "FY2023 operating income was **$88,523 million**, compared with $83,383 "
                "million in FY2022. Drivers per MD&A: higher gross margin on product mix "
                "and cloud Services growth, partly offset by operating expenses that rose "
                "with headcount and R&D investment"
            ),
        },
    ),
    (
        ("revenue", "total revenue"),
        {
            "figure": "$ 211,915",
            "answer": (
                "Microsoft's FY2023 revenue was **$211,915 million** ($211.9 billion), "
                "up from $198,270 million in FY2022"
            ),
        },
    ),
    (
        ("operating income",),
        {
            "figure": "$ 88,523",
            "answer": (
                "FY2023 operating income was **$88,523 million**, versus $83,383 million "
                "in FY2022"
            ),
        },
    ),
    (
        ("operating margin", "operating margin trend"),
        {
            "figure": None,
            "answer": (
                "The FY2023 operating margin was **41.8%** ($88,523 million of $211,915 "
                "million revenue), versus 42.0% in FY2022 ($83,383 million of $198,270 "
                "million revenue)"
            ),
        },
    ),
    (
        ("net income", "net earnings"),
        {
            "figure": "$ 72,361",
            "answer": (
                "FY2023 net income was **$72,361 million** (diluted EPS $9.68), compared "
                "with $72,738 million in FY2022"
            ),
        },
    ),
    (
        ("diluted eps", "earnings per share", "diluted earnings"),
        {
            "figure": "$ 9.68",
            "answer": "FY2023 diluted earnings per share were **$9.68**",
        },
    ),
)

# The reference report covers FY2021-FY2023, so a question about another year is a
# time_period refusal (STRESS-TEST-RAG #7).
_UNSUPPORTED_YEARS: frozenset[str] = frozenset({"2019", "2020", "2024", "2025", "2030"})
# A year is four digits 19xx/20xx not embedded in a longer digit run. The leading guard is
# a single lookbehind for a *digit* rather than a word boundary, because the corpus writes
# periods as "FY2023" and a word boundary never appears between "Y" and "2".
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

# Distinct fiscal periods in either 4-digit ("FY2023", "fiscal year 2023") or the
# 2-digit fiscal-year shorthand ("FY23", "fy 23") the exam's golden questions use.
# Used ONLY by `_is_comparison` to count how many periods a question names;
# `_asked_year` deliberately keeps the 4-digit pattern so `_UNSUPPORTED_YEARS`
# membership stays an exact, unambiguous check.
_PERIOD = re.compile(
    r"(?<!\d)(?:(?:fy\s*|fiscal\s+year\s*)(?:19|20)?\d{2}|(?:19|20)\d{2})(?!\d)",
    re.IGNORECASE,
)


def _normalize_period(match: str) -> str:
    """Canonical form for a matched period so FY23 and FY2023 count as one."""
    digits = re.sub(r"\D", "", match)
    if len(digits) == 4:
        return digits
    return "20" + digits


def _distinct_periods(question: str) -> set[str]:
    return {_normalize_period(m) for m in _PERIOD.findall(question)}


# Labeled question markers in the node input templates (P1, P2, P3). The last one present
# wins, so a rewritten standalone question that itself names a period still resolves.
_QUESTION_LABELS: tuple[str, ...] = (
    "QUESTION TO STANDALONE-IFY:",
    "QUESTION:",
    "STANDALONE QUESTION:",
    "USER QUESTION:",
)

# Fixed, fingerprint-stable accounting: the mock makes no real call, so its timing is a
# documented constant rather than a measurement. Real latency is observable only on the
# `ChatLLM` path, where it is measured per call.
MOCK_LATENCY_MS = 5.0
MOCK_PROMPT_TOKENS = 620
MOCK_COMPLETION_TOKENS = 180
MOCK_MODEL = "mock-llm/offline-demo"
MOCK_REPORT_COMPANY = "Microsoft"

# Metric aliases used by the deterministic rewrite: a follow-up names the short form, the
# standalone rewrite must name the metric as the document labels it.
_TOPIC_ALIASES: tuple[tuple[str, str], ...] = (
    ("margin", "operating margin"),
    ("eps", "diluted earnings per share"),
    ("earnings", "diluted earnings per share"),
    ("net income", "net income"),
    ("operating income", "operating income"),
    ("revenue", "revenue"),
    ("profit", "operating income"),
    ("cost", "cost of revenue"),
    ("expense", "operating expense"),
    ("segment", "segment"),
    ("cash", "cash flow"),
    ("rd", "research and development expense"),
)

_FOLLOW_UP_MARKERS: tuple[str, ...] = (
    "it",
    "itself",
    "that",
    "those",
    "these",
    "this",
    "they",
    "them",
    "the same",
    "so what",
    "and the",
    "and how",
    "and what",
    "and did",
)


@dataclass(frozen=True)
class _Script:
    """The message payload reduced to the parts the scripted answer depends on."""

    family: str
    question: str
    context: str
    previous: str
    known: dict[str, Any] | None
    schema: dict[str, Any] | None


# --- the fingerprint --------------------------------------------------------------


def _canonical(messages: list[dict[str, Any]], schema: dict[str, Any] | None) -> str:
    """Stable JSON text of the request: sorted keys, fixed separators, no whitespace drift."""
    return json.dumps(
        {"messages": messages, "schema": schema},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def prompt_fingerprint(messages: list[dict[str, Any]], schema: dict[str, Any] | None) -> str:
    """The sha256 fingerprint of the prompt (ADR-005's node-name + schema-presence key).

    Node identity is carried by the system prompt's own text (which node family wrote it)
    and schema presence by the schema's required top-level keys — together, the two things
    that decide which scripted answer applies. sha256 rather than :func:`hash` so the
    fingerprint is identical on every interpreter, platform and run.
    """
    return hashlib.sha256(_canonical(messages, schema).encode("utf-8")).hexdigest()


# --- payload access ---------------------------------------------------------------


def _message_text(message: dict[str, Any]) -> str:
    """One message's text content, accepting both the string and content-part forms."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _system_text(messages: list[dict[str, Any]]) -> str:
    """The system prompt, or the whole payload when no system role was used."""
    for message in messages:
        if message.get("role") == "system":
            return _message_text(message)
    return "\n".join(_message_text(message) for message in messages)


def _question_text(messages: list[dict[str, Any]]) -> str:
    """The question under discussion, taken from the final user turn.

    Every node appends the question — alone, or inside its input template — as the last
    user turn, so the tail is the reliable location. When the turn is a template, the text
    after the last question label is the question; a rewritten standalone question reaches
    this node the same way. A turn with no recognizable label is used whole, which is what
    a bare "and the margin trend?" follow-up needs.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = _message_text(message)
        for label in _QUESTION_LABELS:
            if label in text:
                return text.rsplit(label, 1)[-1].strip()
        return text.strip()
    return ""


def _body_text(messages: list[dict[str, Any]]) -> str:
    """Everything except the system prompt: cited blocks, catalog, history, question."""
    return "\n".join(
        _message_text(message) for message in messages if message.get("role") != "system"
    )


def _previous_answer(messages: list[dict[str, Any]]) -> str:
    """The previous assistant answer, which contextualization resolves referents against."""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _message_text(message)
    return ""


def _family_for(schema: dict[str, Any] | None, system: str) -> str:
    """The prompt family: the schema decides when present, else the system prompt's text."""
    if schema is not None:
        required = set(schema.get("required") or [])
        properties = schema.get("properties")
        keys = set(properties) if isinstance(properties, dict) else set()
        for family, signature in _SCHEMA_FAMILIES:
            if keys & signature and not (required and not required & signature):
                return family
        return "query_plan"
    lowered = system.lower()
    for family, marker in _FAMILY_MARKERS:
        if marker in lowered:
            return family
    return "synthesis"


# --- question understanding -------------------------------------------------------


def _tokens(question: str) -> set[str]:
    """Lowercased word tokens of the question."""
    return set(re.findall(r"[a-z0-9]+", question.lower()))


def _matches_question(question: str, phrases: tuple[str, ...]) -> bool:
    """True when any phrase appears in the question (whole-phrase substring match)."""
    lowered = f" {question.lower()} "
    return any(f" {phrase}" in lowered for phrase in phrases)


def _known_question(question: str) -> dict[str, Any] | None:
    """The scripted figure set for a known financial question, else ``None``.

    Multi-metric phrases are listed first so "compare operating income" selects the
    comparison script before the plain operating-income one.
    """
    for phrases, figures in KNOWN_QUESTIONS:
        if _matches_question(question, phrases):
            return figures
    return None


def _asked_year(question: str) -> str | None:
    """The year the question is *about*: the last one it names (corpus covers FY2021-FY2023)."""
    years = _YEAR.findall(question)
    return years[-1] if years else None


def _out_of_scope(question: str) -> bool:
    """True when the question targets a year the reference corpus does not cover."""
    year = _asked_year(question)
    return year is not None and year in _UNSUPPORTED_YEARS


def _is_comparison(question: str) -> bool:
    """True for a question that asks for two periods or two metrics side by side."""
    lowered = question.lower()
    periods = _distinct_periods(question)
    comparative = any(
        word in lowered for word in ("compare", "versus", "vs", "change", "grew", "trend")
    )
    return comparative and (len(periods) >= 2 or "trend" in lowered)


# --- context parsing --------------------------------------------------------------


def _numbered_blocks(context: str) -> list[tuple[int, str]]:
    """The ``[n] text`` blocks a synthesis prompt receives, parsed by their markers."""
    blocks: list[tuple[int, str]] = []
    for match in re.finditer(r"\[(\d+)\]\s*(.*?)(?=\n\s*\[\d+\]|\Z)", context, re.DOTALL):
        blocks.append((int(match.group(1)), match.group(2).strip()))
    return blocks


def _cited_blocks(blocks: list[tuple[int, str]], figure: str | None) -> list[int]:
    """The block numbers whose text contains the figure; every block when there is no figure."""
    if figure is None:
        return [number for number, _ in blocks]
    compact = figure.replace(" ", "")
    hits = [number for number, text in blocks if figure in text or compact in text.replace(" ", "")]
    return hits or [number for number, _ in blocks]


def _candidate_answer(body: str) -> str:
    """The candidate answer inside a groundedness-check payload.

    The P5 payload carries the answer markdown followed by the numbered ``CONTEXT:``
    blocks; splitting on that marker recovers the answer the way the node does. Without
    the marker, the whole payload is treated as the answer, which can only ever make the
    verdict stricter.
    """
    if "CONTEXT:" in body:
        return body.split("CONTEXT:", 1)[0].strip()
    return body


def _compact(text: str) -> str:
    """Text with spaces and thousands separators removed, so "$ 211,915" == "$211915"."""
    return text.replace(" ", "").replace(",", "")


# --- the four prompt families -----------------------------------------------------


def _query_plan(question: str, body: str) -> dict[str, Any]:
    """A ``QueryPlan``-shaped object for the question (P1's output schema).

    ``required_sections`` is read from the catalog the node passed in the payload when the
    catalog is machine-readable, and stays empty otherwise — an empty list is allowed by
    P1, whereas an invented section name would plan retrieval against content that does
    not exist.
    """
    if _out_of_scope(question):
        return {
            "kind": "out_of_scope",
            "sub_queries": [],
            "required_sections": [],
            "fiscal_period": _asked_year(question),
            "missing_aspect": "time_period",
        }
    if _is_comparison(question):
        return {
            "kind": "multi_part",
            "sub_queries": [
                "What was the figure for the earlier period?",
                "What was the figure for the later period, and what drove the change?",
            ],
            "required_sections": _catalog_sections(body),
            "fiscal_period": None,
            "missing_aspect": None,
        }
    return {
        "kind": "simple",
        "sub_queries": [],
        "required_sections": _catalog_sections(body),
        "fiscal_period": _asked_year(question),
        "missing_aspect": None,
    }


# The catalog reaches the mock as *rendered text*, and the renderer is the caller's: P1's
# template dumps it with ``json.dumps(indent=2)``, but a compact dump, a truncated catalog or
# a hand-written fixture spells the same listing with different whitespace, comma placement
# and nesting. The listing is read by locating the ``"sections"`` key and taking whatever
# value follows it, rather than by matching one exact rendering of one array.

#: How many section paths a plan names: ``retrieve`` filters on exactly one name, so the list
#: stays short and predictable however long the catalog it came from was.
_MAX_SECTIONS = 6
#: The catalog key a P1 payload renders each document's section listing under.
_SECTIONS_KEY = re.compile(r'"sections"\s*:\s*')
#: One JSON string literal. It cannot cross an unescaped quote, so the match is exact.
_STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')


def _unquote(literal: str) -> str | None:
    """One JSON string literal decoded to the text it names, or ``None`` if it is not one."""
    try:
        decoded: str | None = json.loads(literal)
    except ValueError:
        # A hand-written literal the JSON parser rejects still names a section: the quotes
        # come off and an escaped quote is unescaped, the honest reading of what was written.
        return literal[1:-1].replace('\\"', '"')
    return decoded if isinstance(decoded, str) else None


def _elements(text: str, at: int) -> tuple[list[str], int]:
    """The comma-separated elements of the array starting at ``at``, plus where it ends.

    String literals are stepped over whole, so a quote, bracket or comma inside a section path
    is never mistaken for structure; a comma separates elements only at the array's own depth,
    so a nested listing or an object stays inside a single element. Closing early on the first
    unmatched bracket is what keeps a truncated catalog from swallowing the rest of the payload.
    """
    parts: list[str] = []
    depth = 0
    start = at + 1
    index = start
    while index < len(text):
        literal = _STRING_LITERAL.match(text, index)
        if literal is not None:
            index = literal.end()
            continue
        char = text[index]
        if char in "[{":
            depth += 1
        elif char in "]}":
            if depth == 0:
                # The element before the closing bracket ends here too, not at the last comma.
                parts.append(text[start:index])
                return parts, index + 1
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    return parts, len(text)


def _paths(parts: list[str]) -> list[str]:
    """The section paths an array's element texts name, in order.

    A nested array is read as a listing in its own right and flattened into the same order; an
    element that is not a string — a number, an object, the empty slot a trailing comma leaves
    — names nothing, so a partially rendered catalog yields nothing rather than a guess.
    """
    found: list[str] = []
    for part in parts:
        element = part.strip()
        if not element:
            continue
        if element.startswith("["):
            nested, _ = _elements(element, 0)
            found.extend(_paths(nested))
            continue
        literal = _STRING_LITERAL.fullmatch(element)
        if literal is None:
            continue
        decoded = _unquote(literal.group(0))
        if decoded:
            found.append(decoded)
    return found


def _value_paths(text: str, at: int) -> list[str]:
    """The section paths named by the value starting at ``at``: an array, or a lone string."""
    index = at
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return []
    if text[index] == "[":
        parts, _ = _elements(text, index)
        return _paths(parts)
    literal = _STRING_LITERAL.match(text, index)
    if literal is None:
        return []
    decoded = _unquote(literal.group(0))
    return [decoded] if decoded else []


def _bare_paths(body: str) -> list[str]:
    """A payload that is *only* a listing, read whole: its first array, or a lone string.

    That requirement is what keeps the tolerance honest — a catalog whose key is spelled some
    other way yields nothing here, rather than having its filenames, doc ids and page counts
    read as section paths.
    """
    stripped = body.lstrip()
    if stripped.startswith('"'):
        return _value_paths(body, len(body) - len(stripped))
    index = 0
    while index < len(body):
        literal = _STRING_LITERAL.match(body, index)
        if literal is not None:
            index = literal.end()
            continue
        if body[index] == "[":
            parts, _ = _elements(body, index)
            return _paths(parts)
        index += 1
    return []


def _listing(body: str) -> list[str]:
    """Every section path the payload renders, in first-seen order.

    The key decides when it is there: a catalog is one JSON object per document and only its
    ``"sections"`` value is a listing, so the filenames, doc ids and page counts sharing that
    object are never read as one. A payload carrying no such key is read as a listing only
    where a listing can stand on its own.
    """
    keys = list(_SECTIONS_KEY.finditer(body))
    if not keys:
        return _bare_paths(body)
    found: list[str] = []
    for key in keys:
        found.extend(_value_paths(body, key.end()))
    return found


def _catalog_sections(body: str) -> list[str]:
    """Section paths named in the payload's document catalog, in first-seen order.

    Shape-tolerant on purpose. The catalog arrives as rendered text and the renderer is the
    caller's: ``json.dumps`` with any indent, a compact dump, a per-document nested
    structure, a trailing comma, or a bare string. Matching one exact spelling would silently
    empty ``required_sections`` for every realistic payload, which would misreport to the
    debug panel a capability the mock does have. What the payload does not render, the mock
    does not invent: an empty catalog is an empty list, and a listing this cannot read stays
    empty rather than being guessed at.
    """
    sections: list[str] = []
    seen: set[str] = set()
    for path in _listing(body):
        if path not in seen:
            seen.add(path)
            sections.append(path)
    return sections[:_MAX_SECTIONS]


def _grounding_result(body: str) -> dict[str, Any]:
    """A ``GroundingResult``-shaped object for the candidate answer (P5's output schema).

    The verdict mirrors what the deterministic pre-pass decides: a figure is ``supported``
    only when the context block cited for its claim contains it, and a claim with no
    marker is unsupported. The mock therefore never endorses a claim its own prompt could
    not ground, which keeps the pessimistic guarantee true in mock mode as well.
    """
    blocks = _numbered_blocks(body)
    by_number = dict(blocks)
    answer = _candidate_answer(body)
    claims: list[dict[str, Any]] = []
    claims_supported = True
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        # Citation markers are stripped *before* figures are read: "[1]" is a pointer, not
        # a claim, and counting it would report the marker itself as unsupported.
        cited = {int(number) for number in _citation_marks(sentence)}
        prose = re.sub(r"\[\d+\]", " ", sentence)
        figures = re.findall(r"\$?\s?\d[\d,\.]*", prose)
        if not figures:
            continue
        for index, figure in enumerate(figures, start=len(claims) + 1):
            normalized = _compact(figure.rstrip(".,)"))
            spans = [_compact(by_number[number]) for number in sorted(cited) if number in by_number]
            supported = bool(spans) and any(normalized in span for span in spans)
            if not supported:
                claims_supported = False
            claims.append(
                {
                    "claim_id": index,
                    "verdict": "supported" if supported else "unsupported",
                    "span": figure if supported else None,
                    "reason": (
                        "the cited block contains the figure"
                        if supported
                        else "no cited block contains this figure"
                    ),
                }
            )
    return {"claims": claims, "all_supported": claims_supported}


def _citation_marks(text: str) -> list[str]:
    """The citation markers a snippet carries, with their numbering."""
    return re.findall(r"\[(\d+)\]", text)


def _synthesis_answer(body: str, known: dict[str, Any] | None) -> str:
    """The markdown answer for the question, grounded in the numbered context blocks.

    A scripted question cites only the blocks that actually contain its figure, so the
    citation card a reviewer opens always shows the line the answer quotes. An unscripted
    question is answered by restating the context's own figures — never by computing or
    estimating one. With no context at all the answer says so, which is the P3 rule 5
    wording rather than an empty message.
    """
    blocks = _numbered_blocks(body)
    if not blocks:
        return "The provided documents do not state this. Upload the corpus PDFs to answer it."
    if known is not None:
        marks = "".join(f" [{number}]" for number in _cited_blocks(blocks, known["figure"])[:2])
        return f"{known['answer']}{marks}."
    figures: list[str] = []
    for _, text in blocks:
        for match in re.findall(r"\$\s?\d[\d,]*", text):
            if match not in figures:
                figures.append(match)
    if not figures:
        return (
            "The provided documents state no figure that answers this question; the "
            "closest content is the uploaded passages above."
        )
    listed = ", ".join(figure.rstrip(".") for figure in figures[:4])
    marks = "".join(f" [{number}]" for number, _ in blocks[:2])
    return f"The documents state: {listed}{marks}. No further figure is derivable from them."


def _contextualization(question: str, previous: str) -> str:
    """The standalone rewrite (P2): a deterministic substring-rewrite stub.

    A question that already stands alone is returned unchanged. A follow-up resolves its
    referent the way the node's rule set prescribes: the topic comes from the follow-up's
    own metric words, else from the previous answer's metric, and the fiscal period comes
    from the follow-up, else from the previous answer. That is exactly P2's degradation
    contract — never fail the turn, and never lose the period.
    """
    stripped = question.strip()
    if not _is_follow_up(stripped):
        return stripped
    topic = _topic_of(stripped) or _topic_of(previous)
    if not topic:
        return stripped
    period = _asked_year(stripped) or _asked_year(previous)
    suffix = f" in FY{period}" if period else ""
    return f"What was {MOCK_REPORT_COMPANY}'s {topic}{suffix}?"


def _is_follow_up(question: str) -> bool:
    """True when the question carries a pronoun, demonstrative or elliptical fragment."""
    lowered = f" {question.lower().strip()} "
    if any(f" {marker} " in lowered for marker in _FOLLOW_UP_MARKERS):
        return True
    return not _YEAR.search(question) and len(_tokens(question)) <= 4 and "?" in question


def _topic_of(text: str) -> str:
    """The document-labelled metric the text is about, or ``""`` when it names none."""
    lowered = text.lower()
    for alias, label in _TOPIC_ALIASES:
        if alias in lowered:
            return label
    return ""


# --- schema-conformant payloads ---------------------------------------------------


def _schema_default(schema: dict[str, Any]) -> dict[str, Any]:
    """A minimal object satisfying ``schema``'s required properties, typed by the schema.

    Used when the scripted object does not match the caller's schema (a schema this mock
    was not written for): the gateway's contract is "``parsed`` is a schema-shaped object
    or ``None``", so the fallback keeps typed mode usable instead of failing the turn.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    required = schema.get("required")
    keys = (
        [key for key in properties if key in required]
        if isinstance(required, list)
        else list(properties)
    )
    return {key: _default_value(properties[key]) for key in keys}


def _default_value(sub_schema: Any) -> Any:
    """The schema-typed zero value for one property."""
    if not isinstance(sub_schema, dict):
        return None
    for value in sub_schema.get("enum") or []:
        return value
    expected = sub_schema.get("type")
    if isinstance(expected, list):
        expected = expected[0] if expected else None
    if expected == "object":
        return _schema_default(sub_schema)
    if expected == "array":
        return []
    if expected == "string":
        return ""
    if expected == "integer":
        return 0
    if expected == "number":
        return 0.0
    if expected == "boolean":
        return False
    return None


# --- the client -------------------------------------------------------------------


class MockLLM:
    """The offline scripted gateway returned by :func:`~src.utils.llm.build_llm`.

    Implements the same ``chat(messages, params, schema)`` surface as the real client, so
    the nodes cannot tell the two apart. Because the answer is a pure function of the
    prompt fingerprint, calling twice with the same request is a determinism check by
    construction rather than by assertion.
    """

    def __init__(self, model: str = MOCK_MODEL) -> None:
        self._model = model

    @property
    def model(self) -> str:
        """The model id reported in every :class:`ChatResult` (never a real provider's)."""
        return self._model

    @property
    def offline(self) -> bool:
        """Always ``True``: the mock performs no network operation of any kind."""
        return True

    def fingerprint(self, messages: list[dict[str, Any]], schema: dict[str, Any] | None) -> str:
        """The sha256 fingerprint of this request (node family + schema presence)."""
        return prompt_fingerprint(messages, schema)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Answer ``messages`` deterministically.

        ``model``/``temperature``/``top_p``/``max_tokens`` are accepted — they are part of
        the frozen signature and the sidebar drives them — but deliberately do not select
        the script: two settings of the same prompt must give the same answer in mock mode,
        which is what makes the demo reproducible.
        """
        script = self._script(messages, schema)
        text, parsed = self._render(script)
        return ChatResult(
            text=text,
            parsed=parsed,
            prompt_tokens=MOCK_PROMPT_TOKENS,
            completion_tokens=MOCK_COMPLETION_TOKENS,
            latency_ms=MOCK_LATENCY_MS,
            model=self._model,
        )

    def _script(self, messages: list[dict[str, Any]], schema: dict[str, Any] | None) -> _Script:
        """The request reduced to its scripted-answer inputs (pure, side-effect free)."""
        question = _question_text(messages)
        return _Script(
            family=_family_for(schema, _system_text(messages)),
            question=question,
            context=_body_text(messages),
            previous=_previous_answer(messages),
            known=_known_question(question),
            schema=schema,
        )

    def _render(self, script: _Script) -> tuple[str, dict[str, Any] | None]:
        """The answer text and parsed object for one family, from the script's inputs.

        A structured payload is checked against the caller's schema and repaired from it
        when they disagree (a schema this mock was not written for), so ``parsed`` is
        always schema-shaped — the frozen contract's requirement — without the mock ever
        inventing a value for a field it does know how to fill.
        """
        if script.family == "query_plan":
            return _json(_conform(_query_plan(script.question, script.context), script.schema))
        if script.family == "grounding":
            return _json(_conform(_grounding_result(script.context), script.schema))
        if script.family == "contextualization":
            return _contextualization(script.question, script.previous), None
        return _synthesis_answer(script.context, script.known), None


def _conform(payload: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """``payload`` when it already satisfies ``schema``, else a schema-shaped fallback."""
    if schema is None or not validate_json_schema(payload, schema):
        return payload
    return _schema_default(schema)


def _json(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """One JSON object, both as its single-line text and as the parsed mapping."""
    return json.dumps(payload, ensure_ascii=True), payload

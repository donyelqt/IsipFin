from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from src.utils.llm import ChatResult, LLMParams, validate_json_schema

__all__ = ["MockLLM", "prompt_fingerprint"]

_FAMILY_MARKERS: tuple[tuple[str, str], ...] = (
    ("query_plan", "decide how to answer, not to answer"),
    ("grounding", "fully supported by the provided context"),
    ("contextualization", "rewrite follow-up questions into standalone questions"),
    ("synthesis", "answering questions strictly from the provided document context"),
)

_SCHEMA_FAMILIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("query_plan", frozenset({"kind", "sub_queries", "required_sections"})),
    ("grounding", frozenset({"claims", "all_supported"})),
)

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
                "FY2023 operating income was **$88,523 million**, versus $83,383 million in FY2022"
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

_UNSUPPORTED_YEARS: frozenset[str] = frozenset({"2019", "2020", "2024", "2025", "2030"})
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

_PERIOD = re.compile(
    r"(?<!\d)(?:(?:fy\s*|fiscal\s+year\s*)(?:19|20)?\d{2}|(?:19|20)\d{2})(?!\d)",
    re.IGNORECASE,
)


def _normalize_period(match: str) -> str:
    digits = re.sub(r"\D", "", match)
    if len(digits) == 4:
        return digits
    return "20" + digits


def _distinct_periods(question: str) -> set[str]:
    return {_normalize_period(m) for m in _PERIOD.findall(question)}


_QUESTION_LABELS: tuple[str, ...] = (
    "QUESTION TO STANDALONE-IFY:",
    "QUESTION:",
    "STANDALONE QUESTION:",
    "USER QUESTION:",
)

MOCK_LATENCY_MS = 5.0
MOCK_PROMPT_TOKENS = 620
MOCK_COMPLETION_TOKENS = 180
MOCK_MODEL = "mock-llm/offline-demo"
MOCK_REPORT_COMPANY = "Microsoft"

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
    family: str
    question: str
    context: str
    previous: str
    known: dict[str, Any] | None
    schema: dict[str, Any] | None


def _canonical(messages: list[dict[str, Any]], schema: dict[str, Any] | None) -> str:
    return json.dumps(
        {"messages": messages, "schema": schema},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def prompt_fingerprint(messages: list[dict[str, Any]], schema: dict[str, Any] | None) -> str:
    return hashlib.sha256(_canonical(messages, schema).encode("utf-8")).hexdigest()


def _message_text(message: dict[str, Any]) -> str:
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
    for message in messages:
        if message.get("role") == "system":
            return _message_text(message)
    return "\n".join(_message_text(message) for message in messages)


def _question_text(messages: list[dict[str, Any]]) -> str:
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
    return "\n".join(
        _message_text(message) for message in messages if message.get("role") != "system"
    )


def _previous_answer(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _message_text(message)
    return ""


def _family_for(schema: dict[str, Any] | None, system: str) -> str:
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


def _tokens(question: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", question.lower()))


def _matches_question(question: str, phrases: tuple[str, ...]) -> bool:
    lowered = f" {question.lower()} "
    return any(f" {phrase}" in lowered for phrase in phrases)


def _known_question(question: str) -> dict[str, Any] | None:
    for phrases, figures in KNOWN_QUESTIONS:
        if _matches_question(question, phrases):
            return figures
    return None


def _asked_year(question: str) -> str | None:
    years = _YEAR.findall(question)
    return years[-1] if years else None


def _out_of_scope(question: str) -> bool:
    year = _asked_year(question)
    return year is not None and year in _UNSUPPORTED_YEARS


def _is_comparison(question: str) -> bool:
    lowered = question.lower()
    periods = _distinct_periods(question)
    comparative = any(
        word in lowered for word in ("compare", "versus", "vs", "change", "grew", "trend")
    )
    return comparative and (len(periods) >= 2 or "trend" in lowered)


def _numbered_blocks(context: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    for match in re.finditer(r"\[(\d+)\]\s*(.*?)(?=\n\s*\[\d+\]|\Z)", context, re.DOTALL):
        blocks.append((int(match.group(1)), match.group(2).strip()))
    return blocks


def _cited_blocks(blocks: list[tuple[int, str]], figure: str | None) -> list[int]:
    if figure is None:
        return [number for number, _ in blocks]
    compact = figure.replace(" ", "")
    hits = [number for number, text in blocks if figure in text or compact in text.replace(" ", "")]
    return hits or [number for number, _ in blocks]


def _candidate_answer(body: str) -> str:
    if "CONTEXT:" in body:
        return body.split("CONTEXT:", 1)[0].strip()
    return body


def _compact(text: str) -> str:
    return text.replace(" ", "").replace(",", "")


def _query_plan(question: str, body: str) -> dict[str, Any]:
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


_MAX_SECTIONS = 6
_SECTIONS_KEY = re.compile(r'"sections"\s*:\s*')
_STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')


def _unquote(literal: str) -> str | None:
    try:
        decoded: str | None = json.loads(literal)
    except ValueError:
        return literal[1:-1].replace('\\"', '"')
    return decoded if isinstance(decoded, str) else None


def _elements(text: str, at: int) -> tuple[list[str], int]:
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
                parts.append(text[start:index])
                return parts, index + 1
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    return parts, len(text)


def _paths(parts: list[str]) -> list[str]:
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
    keys = list(_SECTIONS_KEY.finditer(body))
    if not keys:
        return _bare_paths(body)
    found: list[str] = []
    for key in keys:
        found.extend(_value_paths(body, key.end()))
    return found


def _catalog_sections(body: str) -> list[str]:
    sections: list[str] = []
    seen: set[str] = set()
    for path in _listing(body):
        if path not in seen:
            seen.add(path)
            sections.append(path)
    return sections[:_MAX_SECTIONS]


def _grounding_result(body: str) -> dict[str, Any]:
    blocks = _numbered_blocks(body)
    by_number = dict(blocks)
    answer = _candidate_answer(body)
    claims: list[dict[str, Any]] = []
    claims_supported = True
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
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
    return re.findall(r"\[(\d+)\]", text)


def _synthesis_answer(body: str, known: dict[str, Any] | None) -> str:
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
    lowered = f" {question.lower().strip()} "
    if any(f" {marker} " in lowered for marker in _FOLLOW_UP_MARKERS):
        return True
    return not _YEAR.search(question) and len(_tokens(question)) <= 4 and "?" in question


def _topic_of(text: str) -> str:
    lowered = text.lower()
    for alias, label in _TOPIC_ALIASES:
        if alias in lowered:
            return label
    return ""


def _schema_default(schema: dict[str, Any]) -> dict[str, Any]:
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


class MockLLM:

    def __init__(self, model: str = MOCK_MODEL) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def offline(self) -> bool:
        return True

    def fingerprint(self, messages: list[dict[str, Any]], schema: dict[str, Any] | None) -> str:
        return prompt_fingerprint(messages, schema)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        schema: dict[str, Any] | None = None,
    ) -> ChatResult:
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
        if script.family == "query_plan":
            return _json(_conform(_query_plan(script.question, script.context), script.schema))
        if script.family == "grounding":
            return _json(_conform(_grounding_result(script.context), script.schema))
        if script.family == "contextualization":
            return _contextualization(script.question, script.previous), None
        return _synthesis_answer(script.context, script.known), None


def _conform(payload: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    if schema is None or not validate_json_schema(payload, schema):
        return payload
    return _schema_default(schema)


def _json(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return json.dumps(payload, ensure_ascii=True), payload

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from typing import Any

from src.chains.checks import asked_years, catalog_years, year_coverage_gap
from src.chains.state import DocSummary, GraphState, QueryPlan, section_paths
from src.utils.llm import LLMError, LLMParams, SchemaViolationError, build_llm
from src.utils.prompts import assert_min_version, node_meta
from src.utils.tracing import TOKEN_KEY, trace_node

__all__ = ["analyze_question"]

logger = logging.getLogger(__name__)

NODE_NAME = "analyze_question"

PROMPT_ID = "P1"

KIND_SIMPLE = "simple"
KIND_MULTI_PART = "multi_part"
KIND_OUT_OF_SCOPE = "out_of_scope"

MAX_SUB_QUERIES = 4

MISSING_TIME_PERIOD = "time_period"
MISSING_TOPIC = "topic"
MISSING_ENTITY = "entity"

PLANNER_DEGRADED_KEY = "planner_degraded"

MAX_CATALOG_CHARS = 12_000

QUERY_PLAN_SCHEMA: dict[str, Any] = {
    "title": "QueryPlan",
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": [KIND_SIMPLE, KIND_MULTI_PART, KIND_OUT_OF_SCOPE]},
        "sub_queries": {"type": "array", "items": {"type": "string"}},
        "required_sections": {"type": "array", "items": {"type": "string"}},
        "fiscal_period": {"type": ["string", "null"]},
        "missing_aspect": {
            "type": ["string", "null"],
            "enum": [MISSING_TIME_PERIOD, MISSING_TOPIC, MISSING_ENTITY, None],
        },
    },
    "required": ["kind", "sub_queries", "required_sections", "fiscal_period", "missing_aspect"],
}

_SYSTEM_HEADING = "## System prompt"
_TEMPLATE_HEADING = "## Input template"


def _prompt_parts() -> tuple[str, str]:

    doc = assert_min_version(PROMPT_ID)
    return _fenced_block(doc.body, _SYSTEM_HEADING), _fenced_block(doc.body, _TEMPLATE_HEADING)


def _fenced_block(body: str, heading: str) -> str:

    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != heading:
            continue
        opener = index + 1
        while opener < len(lines) and not lines[opener].lstrip().startswith("```"):
            opener += 1
        block: list[str] = []
        position = opener + 1
        while position < len(lines) and not lines[position].lstrip().startswith("```"):
            block.append(lines[position])
            position += 1
        text = "\n".join(block).strip("\n")
        if text:
            return text
        break
    raise ValueError(f"prompt {PROMPT_ID} has no fenced block under {heading!r}")


def _standalone_question(state: GraphState) -> str:

    for key in ("standalone_question", "question"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _documents(state: GraphState) -> list[DocSummary]:

    raw = state.get("doc_catalog")
    if isinstance(raw, str | bytes) or not isinstance(raw, (list, tuple)):
        return []
    return [_repaired(entry) for entry in raw if isinstance(entry, DocSummary)]


def _repaired(doc: DocSummary) -> DocSummary:
    if all(isinstance(path, tuple) for path in doc.sections):
        return doc
    return replace(doc, sections=section_paths(doc.sections))


def _section_text(path: Any) -> str:

    if isinstance(path, str):
        return path.strip()
    if isinstance(path, (list, tuple)):
        return " > ".join(part.strip() for part in path if isinstance(part, str) and part.strip())
    return ""


def _catalog_json(docs: list[DocSummary]) -> str:

    entries: list[dict[str, Any]] = []
    for doc in docs:
        texts = [_section_text(path) for path in (doc.sections or ())]
        entries.append(
            {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "page_count": doc.page_count,
                "sections": [text for text in texts if text],
            }
        )
    text = json.dumps(entries, ensure_ascii=True, indent=2)
    if len(text) > MAX_CATALOG_CHARS:
        return f"{text[:MAX_CATALOG_CHARS]}\n... (catalog truncated)"
    return text


def _covered_years(docs: list[DocSummary]) -> set[int]:

    parts: list[str] = []
    for doc in docs:
        parts.append(doc.filename)
        parts.extend(_section_text(path) for path in (doc.sections or ()))
    return catalog_years(" ".join(parts))


def _uncovered_period_plan(question: str, covered: set[int]) -> QueryPlan:

    asked = sorted(asked_years(question))
    logger.info(
        "year_coverage_gate: %s is outside the catalog's fiscal years %s; routing to refuse",
        asked,
        sorted(covered),
    )
    return QueryPlan(
        kind=KIND_OUT_OF_SCOPE,
        sub_queries=[],
        required_sections=[],
        fiscal_period=str(asked[0]) if asked else None,
        missing_aspect=MISSING_TIME_PERIOD,
    )


def _params(state: GraphState) -> LLMParams:

    raw = state.get("params")
    params = raw if isinstance(raw, LLMParams) else LLMParams()
    if params.temperature is not None:
        return params
    floor = _temperature_floor()
    if floor is None:
        return params
    return replace(params, temperature=floor)


def _temperature_floor() -> float | None:

    try:
        entry = node_meta(NODE_NAME)
    except Exception:
        return None
    temperature = entry.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        return float(temperature)
    return None


def _texts(value: Any, limit: int = MAX_SUB_QUERIES) -> list[str]:

    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return []
    texts: list[str] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            continue
        text = entry.strip()
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _plan_from(payload: dict[str, Any]) -> QueryPlan | None:

    kind = payload.get("kind")
    if kind not in (KIND_SIMPLE, KIND_MULTI_PART, KIND_OUT_OF_SCOPE):
        return None
    return QueryPlan(
        kind=kind,
        sub_queries=_texts(payload.get("sub_queries")),
        required_sections=_texts(payload.get("required_sections"), limit=64),
        fiscal_period=_optional_text(payload.get("fiscal_period")),
        missing_aspect=_optional_text(payload.get("missing_aspect")),
    )


def _degraded_plan(reason: str, question: str) -> QueryPlan:

    logger.warning(
        "planner_degraded: %s; classifying out_of_scope with no missing_aspect (question=%r)",
        reason,
        question[:120],
    )
    return QueryPlan(
        kind=KIND_OUT_OF_SCOPE,
        sub_queries=[],
        required_sections=[],
        fiscal_period=None,
        missing_aspect=None,
    )


def _messages(system: str, template: str, question: str, catalog: str) -> list[dict[str, Any]]:

    user_turn = template
    for placeholder, value in (("{catalog_json}", catalog), ("{standalone_question}", question)):
        if placeholder not in user_turn:
            raise ValueError(
                f"prompt {PROMPT_ID} input template has no {placeholder!r} placeholder"
            )
        user_turn = user_turn.replace(placeholder, value)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_turn},
    ]


@trace_node(NODE_NAME)
def analyze_question(state: GraphState) -> dict[str, Any]:

    question = _standalone_question(state)
    system, template = _prompt_parts()
    docs = _documents(state)
    catalog = _catalog_json(docs)
    params = _params(state)

    if not question:
        raise ValueError("analyze_question received no question to plan")

    covered = _covered_years(docs)
    if year_coverage_gap(covered, question):
        return {"query_plan": _uncovered_period_plan(question, covered)}

    try:
        result = asyncio.run(
            build_llm(params).chat(
                _messages(system, template, question, catalog), params, QUERY_PLAN_SCHEMA
            )
        )
    except SchemaViolationError as error:
        return {
            "query_plan": _degraded_plan(f"{type(error).__name__}: {error}", question),
            PLANNER_DEGRADED_KEY: True,
        }
    except LLMError as error:
        return {
            "query_plan": _degraded_plan(f"{type(error).__name__}: {error}", question),
            PLANNER_DEGRADED_KEY: True,
        }

    payload = result.parsed
    plan = _plan_from(payload) if isinstance(payload, dict) else None
    if plan is None:
        return {
            "query_plan": _degraded_plan(
                f"gateway returned an unusable plan: {type(payload).__name__}",
                question,
            ),
            PLANNER_DEGRADED_KEY: True,
        }

    return {"query_plan": plan, TOKEN_KEY: result}

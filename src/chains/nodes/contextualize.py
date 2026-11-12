from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from src.chains.state import GraphState, Turn
from src.utils.llm import LLMError, LLMParams, SchemaViolationError, build_llm
from src.utils.prompts import assert_min_version, node_meta
from src.utils.tracing import TOKEN_KEY, trace_node

__all__ = ["contextualize_question"]

logger = logging.getLogger(__name__)

NODE_NAME = "contextualize_question"

PROMPT_ID = "P2"

HISTORY_TURNS = 6


_PRONOUNS: frozenset[str] = frozenset(
    {
        "it",
        "its",
        "itself",
        "that",
        "those",
        "these",
        "this",
        "they",
        "them",
        "their",
        "he",
        "his",
        "she",
        "her",
    }
)

_OPENERS: tuple[str, ...] = ("and ", "but ", "so ", "what about ", "how about ", "the same ")

_ELLIPSIS_MAX_TOKENS = 4

_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

_SYSTEM_HEADING = "## System prompt"
_TEMPLATE_HEADING = "## Input template"


def _is_follow_up(question: str) -> bool:

    padded = f" {question.lower().strip()} "
    if any(f" {pronoun} " in padded for pronoun in _PRONOUNS):
        return True
    if any(padded.lstrip().startswith(opener) for opener in _OPENERS):
        return True
    words = re.findall(r"[a-z0-9]+", padded)
    return len(words) <= _ELLIPSIS_MAX_TOKENS and "?" in question and not _YEAR.search(question)


def _question_of(state: GraphState) -> str:

    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"{NODE_NAME} requires a non-empty 'question' in state — initial_state() "
            "validates it at the graph boundary; an empty question has nothing to "
            "contextualize and nothing to retrieve"
        )
    return question.strip()


def _window_of(state: GraphState) -> list[Turn]:

    history = state.get("history")
    entries: list[Any] = list(history) if isinstance(history, list) else []
    window: list[Turn] = []
    for entry in entries[-HISTORY_TURNS:]:
        turn = _as_turn(entry)
        if turn is not None:
            window.append(turn)
    return window


def _as_turn(entry: Any) -> Turn | None:

    if isinstance(entry, Turn):
        return entry
    if isinstance(entry, Mapping):
        role = entry.get("role")
        content = entry.get("content")
        if isinstance(role, str) and isinstance(content, str):
            return Turn(
                role=role,
                content=content,
                cited_pages=_pages_of(entry.get("cited_pages")),
                entities=_entities_of(entry.get("entities")),
            )
    return None


def _pages_of(value: Any) -> tuple[dict[str, Any], ...]:

    if not isinstance(value, (list, tuple)):
        return ()
    pages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            pages.append(dict(item))
    return tuple(pages)


def _entities_of(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _previous_turn(window: Sequence[Turn]) -> Turn | None:

    for turn in reversed(window):
        if turn.is_assistant:
            return turn
    return None


def _prompt_parts() -> tuple[str, str]:

    body = assert_min_version(PROMPT_ID).body
    return _fenced_block(body, _SYSTEM_HEADING), _fenced_block(body, _TEMPLATE_HEADING)


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


def _render_turns(window: Sequence[Turn]) -> str:

    rendered = [f"{turn.role}: {turn.content.strip()}" for turn in window]
    return "\n".join(line for line in rendered if line.strip()) or "(no prior turns)"


def _render_citations(previous: Turn | None) -> str:

    if previous is None or not previous.cited_pages:
        return "(none recorded)"
    lines = [_citation_line(entry) for entry in previous.cited_pages]
    return "\n".join(line for line in lines if line.strip()) or "(none recorded)"


def _citation_line(entry: Mapping[str, Any]) -> str:

    parts: list[str] = []
    ref = entry.get("ref_index")
    if isinstance(ref, int) and not isinstance(ref, bool):
        parts.append(f"[{ref}]")
    doc = entry.get("doc_id")
    if isinstance(doc, str) and doc:
        parts.append(doc)
    start = entry.get("page_start")
    end = entry.get("page_end")
    if isinstance(start, int) and isinstance(end, int):
        parts.append(f"pp. {start}-{end}")
    else:
        page = entry.get("page")
        if isinstance(page, int) and not isinstance(page, bool):
            parts.append(f"p. {page}")
    section = _section_text(entry.get("section_path"))
    if section:
        parts.append(section)
    if not parts:
        parts.extend(f"{key}={value}" for key, value in sorted(entry.items())[:4])
    return " — ".join(parts) if parts else "(citation with no fields)"


def _section_text(value: Any) -> str:

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        pieces = [str(item).strip() for item in value]
        return " > ".join(piece for piece in pieces if piece)
    return ""


def _render_entities(previous: Turn | None) -> str:
    if previous is None or not previous.entities:
        return "(none recorded)"
    return ", ".join(previous.entities)


def _messages(
    system: str,
    template: str,
    window: Sequence[Turn],
    previous: Turn | None,
    question: str,
) -> list[dict[str, Any]]:

    user = template
    for placeholder, value in (
        ("{turns_text}", _render_turns(window)),
        ("{citations_text}", _render_citations(previous)),
        ("{entities_text}", _render_entities(previous)),
        ("{question}", question),
    ):
        user = user.replace(placeholder, value)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend({"role": turn.role, "content": turn.content.strip()} for turn in window)
    messages.append({"role": "user", "content": user})
    return messages


def _rewrite_params(value: Any) -> LLMParams:

    base = value if isinstance(value, LLMParams) else LLMParams()
    pinned = _pinned_temperature()
    if pinned is None or base.temperature == pinned:
        return base
    return replace(base, temperature=pinned)


def _pinned_temperature() -> float | None:

    value = node_meta(NODE_NAME).get("temperature")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _standalone_text(text: str) -> str:

    cleaned = text.strip()
    for fence in ("```", '"""', "'''"):
        if len(cleaned) > 2 * len(fence) and cleaned.startswith(fence) and cleaned.endswith(fence):
            cleaned = cleaned[len(fence) : -len(fence)].strip()
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'`":
        cleaned = cleaned[1:-1].strip()
    return " ".join(cleaned.split())


def _degrade(question: str, reason: str, turns: int) -> dict[str, Any]:

    logger.warning(
        "contextualization_failed: %s; keeping the raw question (turns=%d)",
        reason,
        turns,
        extra={"node": NODE_NAME, "turns": turns},
    )
    return {"standalone_question": question}


@trace_node(NODE_NAME)
def contextualize_question(state: GraphState) -> dict[str, Any]:

    question = _question_of(state)
    window = _window_of(state)
    if not window or not _is_follow_up(question):
        return {"standalone_question": question}

    previous = _previous_turn(window)
    system, template = _prompt_parts()
    messages = _messages(system, template, window, previous, question)
    params = _rewrite_params(state.get("params"))

    try:
        result = asyncio.run(build_llm(params).chat(messages, params))
    except (LLMError, SchemaViolationError) as error:
        return _degrade(question, f"{type(error).__name__}: {error}", len(window))

    standalone = _standalone_text(result.text)
    if not standalone:
        return _degrade(question, "the model returned no usable rewrite", len(window))

    return {"standalone_question": standalone, TOKEN_KEY: result}

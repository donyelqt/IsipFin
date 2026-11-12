from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from collections.abc import Sequence
from typing import Any

from src.chains.state import Citation, GraphState, QueryPlan
from src.core.chunking import estimate_tokens
from src.core.retrieval import ScoredChunk
from src.utils.llm import LLMError, LLMParams, build_llm
from src.utils.prompts import assert_min_version
from src.utils.tracing import trace_node

__all__ = [
    "CONTEXT_TOKEN_BUDGET",
    "NODE_NAME",
    "NO_ANSWER_LINE",
    "pack_context",
    "synthesize_answer",
]

logger = logging.getLogger(__name__)

NODE_NAME = "synthesize_answer"

CONTEXT_TOKEN_BUDGET = 6000

DEFAULT_TEMPERATURE = 0.2

SNIPPET_LIMIT = 200

NO_ANSWER_LINE = "The provided documents do not state this."

_MARKER = re.compile(r"\[(\d+)\]")
_WORD = re.compile(r"[a-z0-9]+")

_UNRANKED = 1 << 30


def _block_text(chunk: ScoredChunk) -> str:

    return chunk.parent_text if chunk.parent_text.strip() else chunk.child_text


def _rank_of(chunk: ScoredChunk) -> int:
    rank = chunk.fused_rank
    return rank if isinstance(rank, int) and rank >= 0 else _UNRANKED


def pack_context(
    chunks: Sequence[ScoredChunk], budget: int = CONTEXT_TOKEN_BUDGET
) -> tuple[list[ScoredChunk], int]:

    usable = [chunk for chunk in chunks if _block_text(chunk).strip()]
    if not usable:
        return [], 0

    ordered = sorted(
        enumerate(usable),
        key=lambda item: (_rank_of(item[1]), item[0]),
    )
    packed: list[ScoredChunk] = []
    seen: set[tuple[str, str]] = set()
    used = 0
    dropped = 0
    for _position, chunk in ordered:
        identity = (chunk.doc_id, _block_text(chunk))
        if identity in seen:
            dropped += 1
            continue
        cost = estimate_tokens(_render_block(len(packed) + 1, chunk))
        if packed and used + cost > budget:
            dropped += 1
            continue
        seen.add(identity)
        packed.append(chunk)
        used += cost
    return packed, dropped


def _render_block(number: int, chunk: ScoredChunk) -> str:

    path = " > ".join(chunk.section_path) if chunk.section_path else "unsectioned"
    return f"[{number}] {chunk.doc_id} — p. {chunk.page} — {path}\n{_block_text(chunk).rstrip()}"


def _context_header(shown: int, retrieved: int, dropped: int) -> str:
    if shown == 0:
        return "CONTEXT: no document passages were retrieved for this question."
    if dropped == 0:
        return f"CONTEXT: {shown} retrieved passage(s), all shown."
    return (
        f"CONTEXT: {shown} of {retrieved} retrieved passage(s) shown — the rest were dropped "
        "to fit the context budget, so answer only from the passages shown."
    )


def _fenced_section(body: str, heading_prefix: str) -> str | None:

    lines = body.splitlines()
    index = 0
    while index < len(lines) and not lines[index].startswith(heading_prefix):
        index += 1
    if index >= len(lines):
        return None
    fence = index + 1
    while fence < len(lines) and not lines[fence].startswith("```"):
        fence += 1
    if fence >= len(lines):
        return None
    captured: list[str] = []
    for line in lines[fence + 1 :]:
        if line.startswith("```"):
            break
        captured.append(line)
    return "\n".join(captured) if captured else None


def _system_prompt() -> str:

    synthesis = assert_min_version("P3")
    formatting = assert_min_version("P4")
    base = _fenced_section(synthesis.body, "## System prompt") or synthesis.body
    appendix = _fenced_section(formatting.body, "## Rules") or formatting.body
    return f"{base.rstrip()}\n\n{appendix.rstrip()}"


def _question_section(question: str, parts: Sequence[str]) -> str:

    lines = ["QUESTION:", question.strip()]
    if parts:
        lines.append("")
        lines.append("Answer every part below in a single answer, in order:")
        for index, part in enumerate(parts, start=1):
            lines.append(f"{index}. {part.strip()}")
    return "\n".join(lines)


def _sub_queries(state: GraphState, question: str) -> list[str]:

    plan = state.get("query_plan")
    if not isinstance(plan, QueryPlan):
        return []
    asked = question.strip()
    ordered: list[str] = []
    for part in plan.sub_queries:
        if not isinstance(part, str):
            continue
        cleaned = part.strip()
        if cleaned and cleaned != asked and cleaned not in ordered:
            ordered.append(cleaned)
    return ordered


def _standalone_question(state: GraphState) -> str:

    question = state.get("standalone_question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    raw = state.get("question")
    return raw.strip() if isinstance(raw, str) else ""


def _effective_params(raw: Any) -> LLMParams:

    params = raw if isinstance(raw, LLMParams) else LLMParams()
    if params.temperature is None:
        params = dataclasses.replace(params, temperature=DEFAULT_TEMPERATURE)
    return params


def _snippet(parent_text: str, answer: str) -> str:

    lines = [line.strip() for line in parent_text.splitlines() if line.strip()]
    if not lines:
        return ""
    wanted = set(_WORD.findall(answer.lower()))
    best_line = lines[0]
    best_score = 0
    for line in lines:
        score = len(set(_WORD.findall(line.lower())) & wanted)
        if score > best_score:
            best_line, best_score = line, score
    return best_line[:SNIPPET_LIMIT]


def _drop_orphans(answer: str, numbers: set[int]) -> tuple[str, list[int]]:

    present = {int(match.group(1)) for match in _MARKER.finditer(answer)}
    orphaned = sorted(present - numbers)

    def _keep(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in numbers else ""

    cleaned = _MARKER.sub(_keep, answer)
    cleaned = re.sub(r"(?m)^[ \t]*\[\d+\][ \t]*$", "", cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip(), orphaned


@trace_node(NODE_NAME)
def synthesize_answer(state: GraphState) -> dict[str, Any]:

    question = _standalone_question(state)
    parts = _sub_queries(state, question)
    retrieved = [chunk for chunk in state.get("retrieved") or [] if isinstance(chunk, ScoredChunk)]

    packed, dropped = pack_context(retrieved)
    blocks = [_render_block(number, chunk) for number, chunk in enumerate(packed, start=1)]
    header = _context_header(shown=len(blocks), retrieved=len(retrieved), dropped=dropped)

    user_turn = "\n".join(
        [header, *(block + "\n" for block in blocks), "", _question_section(question, parts)]
    ).strip()

    params = _effective_params(state.get("params"))
    result = asyncio.run(
        build_llm(params).chat(
            [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": user_turn},
            ],
            params,
        )
    )

    answer = result.text or ""
    if not answer.strip():
        raise LLMError(
            "the model returned an empty answer; retry the turn — nothing was fabricated to fill it"
        )

    numbers = set(range(1, len(blocks) + 1))
    answer, orphaned = _drop_orphans(answer, numbers)
    if orphaned:
        logger.warning(
            "%s dropped orphan citation marker(s) %s for %d context block(s)",
            NODE_NAME,
            orphaned,
            len(blocks),
        )

    citations = _citations(answer, numbers, packed)

    return {
        "answer_md": answer,
        "citations": citations,
        "llm": result,
    }


def _citations(answer: str, numbers: set[int], packed: Sequence[ScoredChunk]) -> list[Citation]:

    used = sorted({int(match.group(1)) for match in _MARKER.finditer(answer)} & numbers)
    citations: list[Citation] = []
    for number in used:
        chunk = packed[number - 1]
        citations.append(
            Citation(
                ref_index=number,
                doc_id=chunk.doc_id,
                page=chunk.page,
                section_path=tuple(chunk.section_path),
                snippet=_snippet(_block_text(chunk), answer),
            )
        )
    return citations

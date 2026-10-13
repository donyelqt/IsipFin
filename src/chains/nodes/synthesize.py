"""``synthesize_answer``: citation-constrained answer generation (P3 + P4).

The node owns exactly two state outputs — ``answer_md`` and ``citations`` — and reads
only the standalone question plus the retrieved chunks (ADR-006: synthesis never sees raw
chat history; prior turns arrive already rewritten by ``contextualize_question``).

The split of work is ADR-010's neuro-symbolic rule:

* **Deterministic (this module)** — context-budget packing (greedy, by fused rank), block
  numbering, citation-row assembly, and the marker ↔ block correspondence post-pass that
  drops orphan ``[n]`` markers. None of it needs a model, so all of it runs offline.
* **Neural** — one chat call whose system prompt is
  `docs/prompts/answer-synthesis-with-citations.md` (P3) with
  `docs/prompts/table-and-number-formatting.md` (P4) appended as the shared
  formatting appendix. Both are loaded through :mod:`src.utils.prompts`, never inlined, so a
  prompt change is a diffable file change.

The failure paths follow the contract every node shares: a gateway error propagates as
:class:`~src.utils.llm.LLMError` (typed, traced, banner-rendered), a model that answers with
nothing raises instead of emitting an empty answer (ADR-007's hard rule), and the
"not in the documents" answer is a real result the prompt produces — not a stub.

The node is synchronous, and its one awaitable is driven with ``asyncio.run`` — the same
convention :mod:`src.chains.nodes.contextualize` documents: ``graph.invoke`` refuses an
async node ("No synchronous function provided"), so the gateway's coroutine runs on a fresh
loop per turn.
"""

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

#: The node's registered name, spelled once for the decorator and the trace/log lines.
NODE_NAME = "synthesize_answer"

#: Context budget for one synthesis call (STRESS-TEST-RAG #6: "parents are packed greedily by
#: fused score until a token budget (default ~6k tokens) is hit, remainder dropped with a
#: note"). The budget caps the *parents*; the question, parts and system prompt sit outside it.
CONTEXT_TOKEN_BUDGET = 6000

#: Sampling temperature for this node: ADR-005's per-task table ("0.2 (user-adjustable)"),
#: mirrored in ``scripts/prompts_meta.json``. A ``params.temperature`` from the sidebar wins.
DEFAULT_TEMPERATURE = 0.2

#: Characters of parent text kept for a citation card's snippet (OBSERVABILITY.md debug
#: panel: "a 200-char snippet"). The line is chosen by overlap with the answer, so the card
#: opens on the passage the claim came from (STRESS-TEST-RAG #5's sub-span).
SNIPPET_LIMIT = 200

#: The honest "no answer here" line, used verbatim when the model returns nothing at all
#: (never as a substitute for a real answer — the prompt's own rule 5 covers the grounded case).
NO_ANSWER_LINE = "The provided documents do not state this."

#: ``[n]`` as it appears in an answer, and the word tokens used to match a citation snippet.
_MARKER = re.compile(r"\[(\d+)\]")
_WORD = re.compile(r"[a-z0-9]+")

#: Rank used for a chunk carrying no usable ``fused_rank``: below every real rank, so such a
#: chunk still packs (after the ranked ones, in arrival order) instead of vanishing.
_UNRANKED = 1 << 30


# --- context budget (symbolic layer, ADR-010) --------------------------------------


def _block_text(chunk: ScoredChunk) -> str:
    """The text sent to the model for one chunk: the parent, else the child it came from.

    Parent expansion is what a child hit contributes (ADR-002), so the parent is the block
    text. The child fallback keeps a record with an empty parent from being silently dropped.
    """

    return chunk.parent_text if chunk.parent_text.strip() else chunk.child_text


def _rank_of(chunk: ScoredChunk) -> int:
    """The chunk's fused rank, tolerating an unset or nonsensical value."""

    rank = chunk.fused_rank
    return rank if isinstance(rank, int) and rank >= 0 else _UNRANKED


def pack_context(
    chunks: Sequence[ScoredChunk], budget: int = CONTEXT_TOKEN_BUDGET
) -> tuple[list[ScoredChunk], int]:
    """Pack parents greedily by fused rank until the token budget is reached.

    Children that expanded to the same parent are one passage, so the first hit claims it and
    later duplicates are dropped rather than spending budget on repeated text. Each passage
    costs its rendered block (header line plus parent text), because that is what the model
    is billed for, not the bare parent.

    Returns ``(packed, dropped)``: the packed passages in the order they were packed (fused
    rank asc) and how many usable passages did not fit. ``dropped`` is what the prompt's
    context header reports, so an answer built from a truncated context is visibly so.
    """

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
    """One numbered CONTEXT block, following P3's input template.

    ``ScoredChunk`` carries the parent's page as a single ``page`` (its start page, per
    STRESS-TEST-RAG #5) and a ``doc_id`` but no filename, so the header renders those
    instead of the template's ``pp. {page_start}-{page_end}`` / ``{filename}`` placeholders.
    Keeping the numbering exact matters: the model's ``[n]`` marker and this node's citation
    rows are keyed to it.
    """

    path = " > ".join(chunk.section_path) if chunk.section_path else "unsectioned"
    return f"[{number}] {chunk.doc_id} — p. {chunk.page} — {path}\n{_block_text(chunk).rstrip()}"


def _context_header(shown: int, retrieved: int, dropped: int) -> str:
    """The ``CONTEXT:`` preamble line, naming what was packed and what was dropped."""

    if shown == 0:
        return "CONTEXT: no document passages were retrieved for this question."
    if dropped == 0:
        return f"CONTEXT: {shown} retrieved passage(s), all shown."
    return (
        f"CONTEXT: {shown} of {retrieved} retrieved passage(s) shown — the rest were dropped "
        "to fit the context budget, so answer only from the passages shown."
    )


# --- prompt assembly (source of truth: docs/prompts/*.md) ---------------------------


def _fenced_section(body: str, heading_prefix: str) -> str | None:
    """The fenced code block following ``heading_prefix`` in a prompt body, or ``None``.

    Prompts carry their system prompt as a fence under a ``## System prompt`` heading; the
    fence is what reaches the model verbatim. A missing heading or fence returns ``None`` so
    the caller can decide (here: fall back to the whole body rather than send nothing).
    """

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
    """P3's system prompt with P4's formatting contract appended (``appends: ["P4"]``).

    Both are loaded assets with a minimum-version floor, so a stale checkout fails loudly at
    load time rather than silently answering with an older rule set.
    """

    synthesis = assert_min_version("P3")
    formatting = assert_min_version("P4")
    base = _fenced_section(synthesis.body, "## System prompt") or synthesis.body
    appendix = _fenced_section(formatting.body, "## Rules") or formatting.body
    return f"{base.rstrip()}\n\n{appendix.rstrip()}"


def _question_section(question: str, parts: Sequence[str]) -> str:
    """P3's input template tail: the question, then every part it must answer.

    A multi-part plan reaches synthesis as ``query_plan.sub_queries``; the answer covers all
    of them in one response, so they are listed under the question rather than asked again.
    """

    lines = ["QUESTION:", question.strip()]
    if parts:
        lines.append("")
        lines.append("Answer every part below in a single answer, in order:")
        for index, part in enumerate(parts, start=1):
            lines.append(f"{index}. {part.strip()}")
    return "\n".join(lines)


def _sub_queries(state: GraphState, question: str) -> list[str]:
    """The plan's sub-queries, de-duplicated and without the standalone question itself.

    Missing or malformed plans yield ``[]`` — a node never invents parts, and never reads
    ``history`` to invent them either.
    """

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
    """The question synthesis answers: the standalone rewrite, else the raw question.

    ADR-006: this is the only question text synthesis sees. History is not read here at all.
    """

    question = state.get("standalone_question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    raw = state.get("question")
    return raw.strip() if isinstance(raw, str) else ""


def _effective_params(raw: Any) -> LLMParams:
    """Sampling parameters for this call: the user's, with this node's temperature default.

    ``temperature=None`` means "provider default", which for synthesis is the wrong default —
    ADR-005 pins 0.2 so answers are repeatable while still letting a sidebar value override.
    """

    params = raw if isinstance(raw, LLMParams) else LLMParams()
    if params.temperature is None:
        params = dataclasses.replace(params, temperature=DEFAULT_TEMPERATURE)
    return params


# --- citation plumbing (symbolic layer, ADR-010) -----------------------------------


def _snippet(parent_text: str, answer: str) -> str:
    """The parent line that best matches the answer — the citation card's sub-span.

    Overlap is plain word tokens, deterministic and offline; ties resolve to the earlier
    line, so the same answer always renders the same card. An empty parent yields an empty
    snippet rather than fabricated content.
    """

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
    """Drop ``[n]`` markers no context block backs, per P3's citation contract.

    A marker numbering a passage that was never sent is a hallucinated citation, so it is
    removed rather than passed through; a marker alone on its line has no claim to attach to
    and is removed too. Returns the cleaned answer and the dropped numbers.
    """

    present = {int(match.group(1)) for match in _MARKER.finditer(answer)}
    orphaned = sorted(present - numbers)

    def _keep(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in numbers else ""

    cleaned = _MARKER.sub(_keep, answer)
    cleaned = re.sub(r"(?m)^[ \t]*\[\d+\][ \t]*$", "", cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip(), orphaned


# --- the node ----------------------------------------------------------------------


@trace_node(NODE_NAME)
def synthesize_answer(state: GraphState) -> dict[str, Any]:
    """Answer the standalone question from the packed context, one ``[n]`` per claim.

    Reads ``standalone_question`` (or ``question``), ``query_plan.sub_queries`` and
    ``retrieved``; writes ``answer_md`` and ``citations``. The gateway accounting rides along
    under :data:`~src.utils.tracing.TOKEN_KEY` so the decorator records tokens.
    """

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
    # The node is synchronous: graph.invoke refuses an async node, so the one awaitable is
    # driven with asyncio.run (see src.chains.nodes.contextualize).
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
        # A completion with no text is a failure, not an answer: ADR-007's hard rule is that
        # no path renders an empty answer, so this raises as a typed, traced error instead.
        raise LLMError(
            "the model returned an empty answer; retry the turn — nothing was fabricated "
            "to fill it"
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
    """The citation rows for the markers the answer kept, ordered by marker number.

    A marker maps to the block it numbered in the prompt, so the row's doc/page/section come
    from that very chunk and the snippet is the parent line the answer rests on.
    """

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

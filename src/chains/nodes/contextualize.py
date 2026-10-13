"""``contextualize_question`` — the one node that reads conversation history (P2, ADR-006).

Standalone-ifies a follow-up ("and the margin trend?", "what drove that increase?") against
the last six turns **and the previous answer's citations/entities**, so the question that
reaches ``analyze_question`` and ``synthesize_answer`` names its own referent, period and
entity. Synthesis never sees raw history (ADR-006 rule 2), so this node is the only place
where a pronoun can be resolved correctly — and the only place where a failure has to be
absorbed rather than surfaced.

Degradation is load-bearing
---------------------------
P2's Degradation section is explicit: *any* LLM failure logs ``contextualization_failed``
and returns the raw question. A wrong rewrite degrades one answer; a raised error loses the
whole turn. So ``LLMError`` and ``SchemaViolationError`` are caught inside the node, a
warning is logged, and the raw question flows on. Everything else (a missing prompt asset,
a question that is not a string) still raises: an infrastructure bug is not a model
failure, and the typed-error spine (ADR-007) is what makes it visible.

Why the node is synchronous
---------------------------
``graph.py``'s ``invoke()`` runs the graph with ``graph.invoke(...)``, and langgraph refuses
an async node on the synchronous entry point ("No synchronous function provided"). The
gateway's ``chat`` is a coroutine, so the one call is driven with ``asyncio.run`` — a fresh
loop per turn, whose microseconds of setup are noise next to the HTTP round trip. The node
still calls the gateway through ``build_llm`` only: no client is constructed here, so a stub
client (or ``MOCK_LLM=1``) is a one-line monkeypatch and no network import is reachable.

Prompt shape
------------
The prompt asset is the single source of truth (``docs/prompts/README.md`` rule 3: node code
contains no inline prompt text). The message list is P2's own input template as the final
user turn, *plus* the six window turns as role-tagged chat messages: the template is the
frozen contract for what the model receives, and the role-tagged turns are the alternating
user/assistant structure a chat model actually resolves a pronoun against. Both render from
the same window, so they cannot disagree; the token cost of the overlap is a few hundred.
The previous assistant turn's ``cited_pages``/``entities`` (STRESS-TEST-RAG #3's
anti-drift fix) are rendered straight from ``Turn``, never re-derived from the prose.
"""

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

#: The node's registered name, in ``NODE_NAMES``, in every ``NodeEvent``, and the key this
#: module's own lookups use into ``scripts/prompts_meta.json``.
NODE_NAME = "contextualize_question"

#: The prompt asset this node consumes (``docs/prompts/follow-up-contextualization.md``).
PROMPT_ID = "P2"

#: ADR-006 rule 1: the history window is the last **6 turns** ("enough to resolve pronouns
#: and topic carry-over, bounded in cost"). The store's ``last_n_turns`` already caps at
#: its own ``HISTORY_WINDOW``; the node slices again because a caller may hand ``history``
#: a longer transcript, and the bound is a cost guarantee the node must not delegate to
#: whoever built the state.
HISTORY_TURNS = 6

# --- P2's routing precondition -----------------------------------------------------
# The graph's conditional entry point is what decides between contextualize_question and
# analyze_question (docs/ARCHITECTURE.md's flow: "Follow-up? pronoun / that / it / ellipsis").
# Until that edge exists the node is still registered on the entry point, so it carries the
# precondition itself: a question that already stands alone is passed through untouched,
# with no model call and no cost.

#: Pronouns and demonstratives whose referent lives in the previous answer (P2's purpose
#: statement: *"the raw question contains a pronoun … a bare demonstrative … or no explicit
#: subject"*). "it" is deliberately separate from "its": "and its margin trend?" is the
#: ADR-006 acceptance question, and " it " never matches " its ".
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

#: Elliptical openers: a fragment that continues the previous turn rather than starting a
#: question. Matched at the *start* of the trimmed question only — "How about operating
#: income?" continues the turn, while a standalone question merely mentioning "and" mid
#: sentence does not.
_OPENERS: tuple[str, ...] = ("and ", "but ", "so ", "what about ", "how about ", "the same ")

#: A fragment with no subject ("margin trend?", "YoY?") is ellipsis even without a pronoun.
#: The token ceiling is what keeps a genuinely standalone question ("show revenue by
#: segment for FY2023") out of the category.
_ELLIPSIS_MAX_TOKENS = 4

#: A bare fiscal year: 19xx/20xx not embedded in a longer digit run. A question naming its
#: own period is not elliptical — it restated the period instead of inheriting it.
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

#: The two headings whose fenced blocks the node slices out of the asset. Named rather
#: than positional so a reordered section cannot shift the slice.
_SYSTEM_HEADING = "## System prompt"
_TEMPLATE_HEADING = "## Input template"


def _is_follow_up(question: str) -> bool:
    """P2's routing precondition: a pronoun, a bare demonstrative, or an elliptical fragment.

    Conservative on purpose. A false positive costs one cheap deterministic rewrite (P2's
    own rule: *"If the question is already standalone, return it unchanged"*); a false
    negative sends an unresolved pronoun to retrieval, which is STRESS-TEST-RAG #3's
    pronoun drift — the failure this node exists to prevent.
    """

    padded = f" {question.lower().strip()} "
    if any(f" {pronoun} " in padded for pronoun in _PRONOUNS):
        return True
    if any(padded.lstrip().startswith(opener) for opener in _OPENERS):
        return True
    words = re.findall(r"[a-z0-9]+", padded)
    return len(words) <= _ELLIPSIS_MAX_TOKENS and "?" in question and not _YEAR.search(question)


# --- state reading ------------------------------------------------------------------


def _question_of(state: GraphState) -> str:
    """The question to rewrite, stripped, raising when the state has none.

    ``initial_state`` already rejects an empty question at the graph boundary; this guard
    is for a node called directly (a unit test, a rebuilt graph) where a missing state key
    would otherwise become a ``None`` flowing into the prompt as the literal "None".
    """

    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"{NODE_NAME} requires a non-empty 'question' in state — initial_state() "
            "validates it at the graph boundary; an empty question has nothing to "
            "contextualize and nothing to retrieve"
        )
    return question.strip()


def _window_of(state: GraphState) -> list[Turn]:
    """The last :data:`HISTORY_TURNS` history turns, oldest first (ADR-006 rule 1)."""

    history = state.get("history")
    entries: list[Any] = list(history) if isinstance(history, list) else []
    window: list[Turn] = []
    for entry in entries[-HISTORY_TURNS:]:
        turn = _as_turn(entry)
        if turn is not None:
            window.append(turn)
    return window


def _as_turn(entry: Any) -> Turn | None:
    """One history entry as a :class:`~src.chains.state.Turn`, or ``None`` when unusable.

    The state schema types ``history`` as ``list[Turn]``, but the store also hands back
    plain ``{"role", "content"}`` dicts (``history_as_dialogue``), so both shapes are
    accepted rather than crashing on a legitimate caller.
    """

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
    """``Turn.cited_pages`` coerced from whatever was stored (JSON rows carry lists)."""

    if not isinstance(value, (list, tuple)):
        return ()
    pages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            pages.append(dict(item))
    return tuple(pages)


def _entities_of(value: Any) -> tuple[str, ...]:
    """``Turn.entities`` coerced to strings; entries that are not strings are dropped."""

    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _previous_turn(window: Sequence[Turn]) -> Turn | None:
    """The most recent assistant turn in the window — the answer a follow-up refers to.

    P2's rule 3 resolves referents against the PREVIOUS assistant answer, not the user's
    raw text, so this is the turn the citations and entities blocks are built from.
    """

    for turn in reversed(window):
        if turn.is_assistant:
            return turn
    return None


# --- prompt rendering ---------------------------------------------------------------


def _prompt_parts() -> tuple[str, str]:
    """P2's system prompt and input template, sliced from the loaded asset.

    ``assert_min_version`` enforces the ``scripts/prompts_meta.json`` floor, so a stale
    checked-out prompt fails the turn loudly instead of silently reaching the model.
    """

    body = assert_min_version(PROMPT_ID).body
    return _fenced_block(body, _SYSTEM_HEADING), _fenced_block(body, _TEMPLATE_HEADING)


def _fenced_block(body: str, heading: str) -> str:
    """The fenced block under ``heading``, or a ``ValueError`` when the asset lacks one.

    Parsing is by heading and fence, never by line offsets: the prompt is a versioned
    document, and a reordered section must not shift the slice.
    """

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
    """The window as ``role: content`` lines, oldest first."""

    rendered = [f"{turn.role}: {turn.content.strip()}" for turn in window]
    return "\n".join(line for line in rendered if line.strip()) or "(no prior turns)"


def _render_citations(previous: Turn | None) -> str:
    """The previous answer's ``cited_pages`` as one line each (STRESS-TEST-RAG #3)."""

    if previous is None or not previous.cited_pages:
        return "(none recorded)"
    lines = [_citation_line(entry) for entry in previous.cited_pages]
    return "\n".join(line for line in lines if line.strip()) or "(none recorded)"


def _citation_line(entry: Mapping[str, Any]) -> str:
    """One citation entry as ``[n] doc_id — p. 44 — MD&A > Consolidated Statements``.

    Only the citation keys the store writes are rendered; an unrecognised entry is still
    emitted from its own keys, because dropping a recorded citation would discard
    disambiguation the store went to the trouble of persisting.
    """

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
    """A section path (list, tuple, or already-joined text) as ``A > B``."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        pieces = [str(item).strip() for item in value]
        return " > ".join(piece for piece in pieces if piece)
    return ""


def _render_entities(previous: Turn | None) -> str:
    """The previous answer's grounded entities, comma-joined."""

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
    """The chat payload: P2's filled input template after the role-tagged history."""

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


# --- sampling parameters ------------------------------------------------------------


def _rewrite_params(value: Any) -> LLMParams:
    """The sidebar's parameters with this node's pinned temperature applied.

    ``model``/``top_p``/``max_tokens`` flow through from ``GraphState.params`` (the sidebar
    writes them); temperature does not, because ADR-005's per-task policy pins
    ``contextualize_question`` at 0.0 and marks only synthesis as user-adjustable — a
    non-deterministic rewrite would make the same follow-up resolve differently per run.
    """

    base = value if isinstance(value, LLMParams) else LLMParams()
    pinned = _pinned_temperature()
    if pinned is None or base.temperature == pinned:
        return base
    return replace(base, temperature=pinned)


def _pinned_temperature() -> float | None:
    """The temperature ``scripts/prompts_meta.json`` pins for this node, or ``None``.

    Read from the meta file rather than hardcoded so the same reviewable file backs the
    mock's script and this node's call. An absent or non-numeric value means "no pin".
    """

    value = node_meta(NODE_NAME).get("temperature")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# --- output hygiene -----------------------------------------------------------------


def _standalone_text(text: str) -> str:
    """The model's reply reduced to one bare question, or ``""`` when it is unusable.

    P2 asks for exactly one question with no preamble, JSON, quotes or trailing
    explanation, so the reply is un-quoted, un-fenced and whitespace-collapsed. No
    truncation and no rewriting: a question the model expanded is still its question, and
    editing it further is a second unresolved rewrite the trace cannot account for.
    """

    cleaned = text.strip()
    for fence in ("```", '"""', "'''"):
        if len(cleaned) > 2 * len(fence) and cleaned.startswith(fence) and cleaned.endswith(fence):
            cleaned = cleaned[len(fence) : -len(fence)].strip()
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'`":
        cleaned = cleaned[1:-1].strip()
    return " ".join(cleaned.split())


# --- degradation --------------------------------------------------------------------


def _degrade(question: str, reason: str, turns: int) -> dict[str, Any]:
    """Log ``contextualization_failed`` and hand back the raw question (P2's contract)."""

    logger.warning(
        "contextualization_failed: %s; keeping the raw question (turns=%d)",
        reason,
        turns,
        extra={"node": NODE_NAME, "turns": turns},
    )
    return {"standalone_question": question}


# --- the node -----------------------------------------------------------------------


@trace_node(NODE_NAME)
def contextualize_question(state: GraphState) -> dict[str, Any]:
    """Rewrite a follow-up into a standalone question using history (P2, ADR-006).

    Reads ``question`` and ``history``; writes ``standalone_question`` and reports its
    gateway accounting under :data:`~src.utils.tracing.TOKEN_KEY` so ``@trace_node`` can
    lift it into the ``NodeEvent``. No other state key is read or written — in particular
    ``node_trace``, which the decorator owns.

    A question that is already standalone, or a run with no history at all, short-circuits
    before the gateway: there is nothing to resolve against, and a rewrite from an empty
    window would invent a referent — exactly the drift this node exists to prevent.

    Args:
        state: The graph state; ``question`` is required, ``history`` optional.

    Returns:
        ``{"standalone_question": <rewrite or raw>}``, plus the accounting key when a
        gateway call succeeded.

    Raises:
        LLMError, SchemaViolationError: Never — both are caught and degraded. The raw
            question is returned instead.
        ValueError: When the state carries no non-empty question, or the prompt asset has
            no fenced system prompt / input template.
    """

    question = _question_of(state)
    window = _window_of(state)
    if not window or not _is_follow_up(question):
        # The graph's conditional entry point routes straight to analyze_question when the
        # heuristic does not fire; while that edge is unbuilt the node is still registered
        # on the entry point, so this pass-through is what keeps analyze_question fed. No
        # model call means no cost and no rewrite of a question that already stands alone.
        return {"standalone_question": question}

    previous = _previous_turn(window)
    system, template = _prompt_parts()
    messages = _messages(system, template, window, previous, question)
    params = _rewrite_params(state.get("params"))

    try:
        result = asyncio.run(build_llm(params).chat(messages, params))
    except (LLMError, SchemaViolationError) as error:
        # P2's Degradation section: a wrong rewrite degrades the answer, a failed turn
        # destroys it. The raw question flows on and the turn completes; the warning is
        # the only signal, because a degraded run is not a failure and must not be
        # recorded as one in the trace.
        return _degrade(question, f"{type(error).__name__}: {error}", len(window))

    standalone = _standalone_text(result.text)
    if not standalone:
        # An empty answer is never a success — ADR-007's hard rule — so a blank reply
        # degrades exactly like a raised one rather than reaching retrieval as "".
        return _degrade(question, "the model returned no usable rewrite", len(window))

    return {"standalone_question": standalone, TOKEN_KEY: result}

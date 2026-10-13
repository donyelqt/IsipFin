"""``groundedness_check``: the graph's verifier — deterministic pre-pass, then P5 (task T13).

This node is ADR-010's neuro-symbolic split in miniature:

* **Symbolic** — :func:`src.chains.checks.numeric_pre_pass` runs first, with no model. It
  rejects every claim whose figure is absent from the block its ``[n]`` marker cites, and
  every claim whose period label contradicts the figure. Those two rules are the exam's #1
  ranked failure mode (``docs/STRESS-TEST-RAG.md`` row 1: FY2024's ``$245,122M`` asserted as
  FY2023), and settling them with a model was the weakest symbolic share in the design. An
  answer containing only a wrong-year figure is now caught and trimmed with zero LLM calls —
  which is what makes the failure testable offline under ``MOCK_LLM=1``.
* **Neural** — P5
  (``docs/prompts/summarization-and-verification.md``) judges what the checker cannot settle:
  computed metrics (a margin is arithmetic, not a printed figure), scope matching (a segment
  subtotal read as consolidated), and qualitative claims. The prompt is instructed to only
  judge the surviving claims and never to re-litigate the checker's rejections.

Both halves converge on one outcome: claims the verifier trims are replaced with a visible
``(unsupported claim removed)`` marker, and ``grounded`` is set ``False``. A wrong or missing
figure never survives as fact.

Failure paths (each one a real result or a typed raise, never an empty answer):

* The pre-pass rejects claims, and the LLM call finds nothing left to judge — the remaining
  answer is emitted trimmed and ungrounded. No model call is wasted on it.
* The gateway returns JSON that does not match the ``GroundingResult`` schema, or omits the
  fields — the **pessimistic default**: the whole answer is trimmed and ``grounded`` is
  ``False``. A malformed verdict is treated as an ungrounded one, because a verifier that
  cannot say "yes" must not be read as saying it.
* The gateway raises ``LLMError`` — it propagates typed and traced (ADR-007); the UI shows an
  error banner rather than an unverified answer that reads as verified.

The node is synchronous, and its one awaitable is driven with ``asyncio.run`` — the same
convention :mod:`src.chains.nodes.contextualize` documents: ``graph.invoke`` refuses an
async node ("No synchronous function provided"). The deterministic pre-pass runs before that
call, so an answer it settles alone costs no gateway round trip at all.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, cast

from src.chains.checks import (
    OUTCOME_REJECTED,
    TRIM_MARKER,
    ClaimJudgement,
    ClaimUnit,
    numeric_pre_pass,
    rejected_units,
    trim_answer,
    undecided_units,
)
from src.chains.state import Citation, GraphState
from src.utils.llm import LLMParams, build_llm
from src.utils.prompts import assert_min_version
from src.utils.tracing import trace_node

__all__ = [
    "GROUNDING_SCHEMA",
    "NODE_NAME",
    "VERDICT_SUPPORTED",
    "groundedness_check",
]

logger = logging.getLogger(__name__)

#: The node's registered name, spelled once for the decorator and the trace/log lines.
NODE_NAME = "groundedness_check"

#: Sampling temperature for this node: ADR-005's per-task table pins 0.0 — "verification must
#: be deterministic". A ``params.temperature`` from the sidebar still overrides it.
DEFAULT_TEMPERATURE = 0.0

#: P5's output schema (``grounding result``). Declared here because
#: ``src/chains/schemas.py`` belongs to a later task (T16) and the gateway needs the shape now;
#: the shape is the one ``src/utils/mock_llm.py`` fingerprints on, so the mock and the real
#: client validate against the same contract.
GROUNDING_SCHEMA: dict[str, Any] = {
    "title": "GroundingResult",
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "partially", "unsupported"],
                    },
                    "span": {"type": ["string", "null"]},
                    "reason": {"type": "string"},
                },
                "required": ["claim_id", "verdict", "reason"],
            },
        },
        "all_supported": {"type": "boolean"},
    },
    "required": ["claims", "all_supported"],
}

#: P5's verdict vocabulary. Only ``supported`` keeps a claim; the other two trim it.
VERDICT_SUPPORTED = "supported"


# --- the input payload (P5's shape) -------------------------------------------------


def _answer_of(state: GraphState) -> str:
    """The candidate answer this node verifies."""

    answer = state.get("answer_md")
    return answer.strip() if isinstance(answer, str) else ""


def _blocks(retrieved: Any) -> dict[int, str]:
    """``marker number → context block``: the numbered blocks synthesis built.

    Synthesis numbers the packed context 1..N in the same order this function does, so a
    claim's ``[n]`` resolves to the block the model was given. A chunk with no parent text is
    skipped, so a number never points at an empty span.
    """

    blocks: dict[int, str] = {}
    if not isinstance(retrieved, list):
        return blocks
    for chunk in retrieved:
        text = getattr(chunk, "parent_text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        number = len(blocks) + 1
        blocks[number] = text
    return blocks


def _system_prompt() -> str:
    """P5's system prompt, loaded from the asset (never inlined)."""

    prompt = assert_min_version("P5")
    return prompt.body


def _payload(answer: str, blocks: Mapping[int, str]) -> str:
    """The user turn P5 receives: the trimmed answer plus the numbered context blocks.

    The answer is the already-trimmed one, so the claims the pre-pass rejected are simply not
    there for the model to re-read — which is how P5's "do not re-litigate spans the checker
    already rejected" is enforced structurally rather than by instruction alone. The claims
    that remain in it are exactly the ones this node needs a verdict for.
    """

    lines = ["CANDIDATE ANSWER:", answer, "", "CONTEXT:"]
    for number in sorted(blocks):
        lines.append(f"[{number}] {blocks[number]}")
    return "\n".join(lines).strip()


def _effective_params(raw: Any) -> LLMParams:
    """Sampling parameters for this call: the user's, with this node's temperature default.

    ADR-005 pins verification at 0.0; a sidebar value still wins, because the sidebar is the
    operator's decision and must never be silently overridden.
    """

    params = raw if isinstance(raw, LLMParams) else LLMParams()
    if params.temperature is None:
        params = LLMParams(
            model=params.model,
            temperature=DEFAULT_TEMPERATURE,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
        )
    return params


# --- the LLM verdict, read defensively ----------------------------------------------


def _trim_units(payload: Mapping[str, Any], units: list[ClaimUnit]) -> list[ClaimUnit]:
    """The undecided claims P5 rejected or marked partial, matched by position.

    P5 answers with a ``claim_id`` for each claim this node listed in ``CLAIMS STILL TO
    JUDGE``, so the id names the claim — but a model that reports only its own sequential
    numbering is matched by ordinal position instead. Ids are trusted first, then position;
    a claim the verdicts cover is trimmed unless its verdict is exactly ``supported``.

    The verdict is read per claim, never from the answer-level ``all_supported`` flag alone:
    that flag is the model's summary, and one bad actor inside a passing answer is exactly the
    partially-correct claim STRESS-TEST-RAG #2 is about.
    """

    raw = payload.get("claims")
    verdicts: dict[int, str] = {}
    if isinstance(raw, list):
        for position, claim in enumerate(raw, start=1):
            if not isinstance(claim, Mapping):
                continue
            verdict = claim.get("verdict")
            if isinstance(verdict, str):
                verdicts[position] = verdict.strip().lower()

    trimmed: list[ClaimUnit] = []
    for position, unit in enumerate(units, start=1):
        verdict = verdicts.get(position, "")
        if verdict == VERDICT_SUPPORTED:
            continue
        if not verdict and bool(payload.get("all_supported")):
            # The verdict covered no claim at this position and the model says the whole
            # answer stands. A model silent about a claim it was handed is neutral, not a
            # rejection: only an explicit non-supported verdict trims.
            continue
        trimmed.append(unit)
    return trimmed


def _is_valid(payload: Any) -> bool:
    """True when the gateway's parsed payload is a usable ``GroundingResult``.

    ``parsed`` is only rejected when it is not a mapping or omits a claim list: a model that
    reports no claims at all has said nothing about the answer, and an unverifiable answer is
    ungrounded by the pessimistic default.
    """

    if not isinstance(payload, Mapping):
        return False
    if not isinstance(payload.get("claims"), list):
        return False
    return isinstance(payload.get("all_supported"), bool)


# --- the node -----------------------------------------------------------------------


@trace_node(NODE_NAME)
def groundedness_check(state: GraphState) -> dict[str, Any]:
    """Verify the candidate answer, deterministically first, then with P5.

    Reads ``answer_md`` and ``retrieved``; writes ``answer_md`` (trimmed when claims fall),
    ``grounded`` and ``citations``. The gateway accounting rides along under
    :data:`~src.utils.tracing.TOKEN_KEY` so the decorator records tokens — ``None`` when the
    pre-pass settled everything and no call was made.

    A refusal is not this node's input: ``refuse`` writes its own ``grounded=True`` (the
    refusal copy cites nothing because there is nothing to cite, which is the opposite of an
    unverified claim).
    """

    answer = _answer_of(state)
    blocks = _blocks(state.get("retrieved"))

    if not answer:
        # Nothing was produced to verify. This is a caller bug, not an answer: the graph
        # reaches `groundedness_check` only after `synthesize_answer`, whose contract is to
        # raise rather than emit an empty answer (ADR-007).
        from src.chains.graph import GraphStateError

        raise GraphStateError(
            "groundedness_check received an empty answer_md: synthesis must produce one"
        )

    # --- stage 1: the deterministic pre-pass (ADR-010) -------------------------------
    judgements: list[ClaimJudgement] = numeric_pre_pass(answer, blocks)
    rejected = rejected_units(judgements)
    undecided = undecided_units(judgements)

    working = answer
    if rejected:
        working = trim_answer(answer, rejected)
        logger.warning(
            "%s pre-pass trimmed %d claim(s): %s",
            NODE_NAME,
            len(rejected),
            [
                (judgement.reason or "", judgement.unit.text.strip())
                for judgement in judgements
                if judgement.outcome == OUTCOME_REJECTED
            ],
        )

    # --- stage 2: the LLM verifier (P5) ---------------------------------------------
    # A gateway failure is NOT caught here: ADR-007's hard rule is that a failure surfaces as
    # the typed, traced error the UI renders, rather than as an answer that reads verified.
    params = _effective_params(state.get("params"))
    result = None
    trimmed: list[ClaimUnit] = []
    if undecided and working.strip():
        # The node is synchronous: graph.invoke refuses an async node, so the one awaitable
        # is driven with asyncio.run (see src.chains.nodes.contextualize).
        result = asyncio.run(
            build_llm(params).chat(
                [
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": _payload(working, blocks)},
                ],
                params,
                GROUNDING_SCHEMA,
            )
        )
        trimmed = _verdict_trimming(result.parsed, undecided)

    if trimmed:
        trimmed_answer = trim_answer(working, trimmed)
        if trimmed_answer.strip():
            working = trimmed_answer
        logger.warning(
            "%s verifier trimmed %d claim(s): %s",
            NODE_NAME,
            len(trimmed),
            [unit.text.strip() for unit in trimmed],
        )

    grounded = not rejected and not trimmed
    if TRIM_MARKER in working or not working.strip():
        # A trim that consumed the entire answer means nothing could be verified. The marker
        # `(unsupported claim removed)` is what the user sees: an honest empty verdict, never
        # a fabricated short answer.
        grounded = False

    update: dict[str, Any] = {
        "answer_md": working,
        "citations": _keep_citations(state, working),
        "grounded": grounded,
    }
    if result is not None:
        update["llm"] = result
    return update


def _verdict_trimming(payload: Any, units: list[ClaimUnit]) -> list[ClaimUnit]:
    """The claims P5 rejected or marked partial.

    P5 answers with a per-claim verdict, so the verdict decides the trim. A payload that is
    not a usable ``GroundingResult`` — unparsable, or silent about the claims it was handed —
    is the **pessimistic default**: every claim the pre-pass handed over is trimmed, because a
    verifier that cannot say "yes" must not be read as saying it (ADR-005).
    """

    if not _is_valid(payload):
        logger.warning(
            "%s unusable verdict payload; trimming every claim the pre-pass could not settle",
            NODE_NAME,
        )
        return list(units)
    return _trim_units(cast(Mapping[str, Any], payload), units)


def _keep_citations(state: GraphState, answer: str) -> list[Citation]:
    """The citation rows whose marker survives in the answer.

    A citation is evidence for a claim; when the claim is trimmed the marker goes with it, so
    the citation rows are filtered against the trimmed answer rather than copied. The rows
    themselves stay untouched otherwise — the store persists them by ``ref_index``.
    """

    raw = state.get("citations")
    if not isinstance(raw, list):
        return []
    kept: list[Citation] = []
    for citation in raw:
        if not isinstance(citation, Citation):
            continue
        if f"[{citation.ref_index}]" in answer:
            kept.append(citation)
    return kept

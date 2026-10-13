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
  judge the surviving claims and never to re-litigate the checker's rejections, and it is
  given the context blocks those claims cite — the union of their ``[n]`` markers, every block
  when none of them cites anything — so P5 never pays prefill for blocks no claim rests on.

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
import dataclasses
import logging
from collections.abc import Iterable, Mapping, Sequence
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

#: P5's output schema (``grounding result``). Declared here, beside the node that sends it:
#: ``ADR-005`` records the JSON schemas as inline in the node that consumes each one, and there
#: is no shared schema module — the other declaration is ``QUERY_PLAN_SCHEMA`` in
#: ``src/chains/nodes/analyze.py``. The shape is the one ``src/utils/mock_llm.py`` fingerprints
#: on, so the mock and the real client validate against the same contract.
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


def _payload(answer: str, blocks: Mapping[int, str], units: Iterable[ClaimUnit]) -> str:
    """The user turn P5 receives: the trimmed answer plus the context blocks it must judge.

    The answer is the already-trimmed one, so the claims the pre-pass rejected are simply not
    there for the model to re-read — which is how P5's "do not re-litigate spans the checker
    already rejected" is enforced structurally rather than by instruction alone. The claims
    that remain in it are exactly the ones this node needs a verdict for.

    Only the blocks those claims actually cite are sent — the union of their ``[n]``
    markers. P5 judges a claim against the span that claim cites, so the blocks behind every
    *other* claim are dead weight the node was paying for: a measured real-provider call
    sent 21.0 KB of context over ~6,900 prompt tokens and spent 20.7 s on what is one
    prefill-dominated POST, for an answer citing one of the seven blocks it shipped. A marker
    pointing at a block that does not exist is skipped, because there is nothing to send for
    it, and the numbering is never rewritten — the answer's ``[n]`` is the marker the model
    is asked about, so renumbering the context would silently re-point every citation.

    When no claim cites anything — the citation-free answer P5's primary rule exists to
    catch — every block is sent. Judging a claim against an empty context and then trimming
    it for what the context did not contain would punish the caller for what this node chose
    not to show.
    """

    cited = sorted({marker for unit in units for marker in unit.markers if marker in blocks})
    numbers = cited if cited else sorted(blocks)
    lines = ["CANDIDATE ANSWER:", answer, "", "CONTEXT:"]
    for number in numbers:
        lines.append(f"[{number}] {blocks[number]}")
    logger.debug(
        "%s sending %d of %d context block(s) to P5: %s",
        NODE_NAME,
        len(numbers),
        len(blocks),
        numbers,
    )
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


def _verdicts_by_claim_id(raw: Any) -> dict[int, str]:
    """The per-claim verdicts a payload names by ``claim_id``, ``claim_id → verdict``.

    Returns an empty mapping when no entry carries a usable id, which is what tells
    :func:`_trim_units` to match by ordinal position instead. An id counts as usable when it is
    a real integer, or a string whose whole text is one (``"1"``, ``" 1 "``) — a model quoting
    the schema back is still naming the claim; ``"1 of 2"`` and ``True`` are not, because
    neither resolves to a single claim. Entries without a verdict string are dropped: they
    name a claim and say nothing about it, so they must leave that claim to the fallback
    rather than record a silent rejection.

    An id repeated for two verdicts keeps the pessimistic one. A claim the model both
    approved and rejected has been answered both ways, and a verifier must read that
    contradiction as the rejection — the same reading as every other ambiguous input here.
    """

    verdicts: dict[int, str] = {}
    if not isinstance(raw, list):
        return verdicts
    for claim in raw:
        if not isinstance(claim, Mapping):
            continue
        verdict = claim.get("verdict")
        if not isinstance(verdict, str):
            continue
        key = claim.get("claim_id")
        if isinstance(key, bool):
            continue
        if isinstance(key, int):
            number = key
        elif isinstance(key, str) and key.strip().lstrip("+-").isdigit():
            number = int(key.strip())
        else:
            continue
        verdict = verdict.strip().lower()
        if verdict == VERDICT_SUPPORTED:
            # Never let an approval overwrite a rejection already recorded for this id.
            verdicts.setdefault(number, verdict)
        else:
            verdicts[number] = verdict
    return verdicts


def _verdicts_by_position(raw: Any) -> dict[int, str]:
    """The per-claim verdicts in the order the model listed them, ``1-based position → verdict``.

    The fallback for a payload whose ids are unusable or absent — a model reporting only its
    own sequential numbering is matched where it actually spoke. Entries without a verdict
    string are skipped, so the position still occupied by an unreadable row does not become a
    phantom rejection of the claim standing there.
    """

    verdicts: dict[int, str] = {}
    if not isinstance(raw, list):
        return verdicts
    for position, claim in enumerate(raw, start=1):
        if not isinstance(claim, Mapping):
            continue
        verdict = claim.get("verdict")
        if isinstance(verdict, str):
            verdicts[position] = verdict.strip().lower()
    return verdicts


def _trim_units(payload: Mapping[str, Any], units: list[ClaimUnit]) -> list[ClaimUnit]:
    """The undecided claims P5 rejected or marked partial, matched by the ``claim_id`` P5 sends.

    P5 asks for a ``claim_id`` per claim, so the id is the contract: this node listed the
    undecided claims in document order — the order the ids it handed out follow — and a
    verdict is therefore resolved **by id first**. Matching by the position of the entry in
    the model's list instead — the previous behaviour — silently re-points every verdict by
    however many claims the model skipped, duplicated or reordered: the live run that
    motivated this sent ``claim_id: 2`` for a one-claim hand-off, and a positional map reads
    that as the *second* claim, so a model that numbers off by one can trim a claim its own
    verdict would have kept, and one that re-lists claim 1 can excuse it.

    Position is the fallback, used only when the payload names no claim by a usable id. An
    unusable id is ``True``, a non-integer string or a missing field — cases where the entry
    carries no address any claim could be reached by, so where the model spoke is the only
    signal left.

    Ids that name no claim handed over are ignored: a verdict for a claim this node did not
    list is a model mistake about its own input, not a statement about any claim that is
    here. The reverse is the dangerous direction and is handled — a handed-over claim the
    verdicts never name is judged by the ``all_supported`` flag, not silently trimmed.

    The verdict is read per claim, never from the answer-level ``all_supported`` flag alone:
    that flag is the model's summary, and one bad actor inside a passing answer is exactly the
    partially-correct claim STRESS-TEST-RAG #2 is about — ``all_supported: true`` must never
    rescue a claim whose own verdict is ``unsupported``.
    """

    raw = payload.get("claims")
    verdicts = _verdicts_by_claim_id(raw) or _verdicts_by_position(raw)

    trimmed: list[ClaimUnit] = []
    for claim_id, unit in enumerate(units, start=1):
        verdict = verdicts.get(claim_id, "")
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


def _locate(text: str, units: Sequence[ClaimUnit]) -> list[tuple[ClaimUnit, int]]:
    """Where each of ``units``'s claim texts now sits in ``text``, and the units themselves.

    A claim unit is a text plus the offsets it was read at. Stage 1 of this node re-splices
    the answer, so by the time a unit is trimmed the offsets it carries describe the
    *untrimmed* answer and no longer address the text being trimmed. A trim is therefore
    positioned by re-finding the claim's text rather than by trusting a stale offset.

    The search runs forward from the previous unit's end, so units are located in document
    order and each one is found where the answer actually prints it. A unit whose text is no
    longer present — the only way this returns no position for it — cannot be trimmed, which
    is the honest reading: a claim that is not in the answer any more is already gone.
    """

    located: list[tuple[ClaimUnit, int]] = []
    cursor = 0
    for unit in units:
        wanted = unit.text
        found = text.find(wanted, cursor) if wanted else -1
        if found < 0:
            # Not there: it was trimmed with the whole line, or the answer was rewritten
            # around it. Either way there is nothing left to splice against.
            continue
        located.append((unit, found))
        cursor = found + len(wanted)
    return located


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
                    {"role": "user", "content": _payload(working, blocks, undecided)},
                ],
                params,
                GROUNDING_SCHEMA,
            )
        )
        trimmed = _verdict_trimming(result.parsed, undecided)

    if trimmed:
        # ``undecided``'s offsets were measured on ``answer``, not on ``working``: stage 1's
        # splice has already shortened every line after the first rejected claim. A trim
        # spliced at the *original* offsets lands mid-word in the trimmed answer — the live
        # q07 run rendered "Operating margin(unsupported claim removed)" and stacked a second
        # table of markers below the first — so the units are re-measured against the text
        # they are about to be spliced into.
        relocated = [
            dataclasses.replace(unit, start=start, end=start + len(unit.text))
            for unit, start in _locate(working, trimmed)
        ]
        trimmed_answer = trim_answer(working, relocated)
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

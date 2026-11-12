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

NODE_NAME = "groundedness_check"

DEFAULT_TEMPERATURE = 0.0

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

VERDICT_SUPPORTED = "supported"


def _answer_of(state: GraphState) -> str:
    answer = state.get("answer_md")
    return answer.strip() if isinstance(answer, str) else ""


def _blocks(retrieved: Any) -> dict[int, str]:

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

    prompt = assert_min_version("P5")
    return prompt.body


def _payload(answer: str, blocks: Mapping[int, str], units: Iterable[ClaimUnit]) -> str:

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

    params = raw if isinstance(raw, LLMParams) else LLMParams()
    if params.temperature is None:
        params = LLMParams(
            model=params.model,
            temperature=DEFAULT_TEMPERATURE,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
        )
    return params


def _verdicts_by_claim_id(raw: Any) -> dict[int, str]:

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
            verdicts.setdefault(number, verdict)
        else:
            verdicts[number] = verdict
    return verdicts


def _verdicts_by_position(raw: Any) -> dict[int, str]:

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

    raw = payload.get("claims")
    verdicts = _verdicts_by_claim_id(raw) or _verdicts_by_position(raw)

    trimmed: list[ClaimUnit] = []
    for claim_id, unit in enumerate(units, start=1):
        verdict = verdicts.get(claim_id, "")
        if verdict == VERDICT_SUPPORTED:
            continue
        if not verdict and bool(payload.get("all_supported")):
            continue
        trimmed.append(unit)
    return trimmed


def _is_valid(payload: Any) -> bool:

    if not isinstance(payload, Mapping):
        return False
    if not isinstance(payload.get("claims"), list):
        return False
    return isinstance(payload.get("all_supported"), bool)


def _locate(text: str, units: Sequence[ClaimUnit]) -> list[tuple[ClaimUnit, int]]:

    located: list[tuple[ClaimUnit, int]] = []
    cursor = 0
    for unit in units:
        wanted = unit.text
        found = text.find(wanted, cursor) if wanted else -1
        if found < 0:
            continue
        located.append((unit, found))
        cursor = found + len(wanted)
    return located


@trace_node(NODE_NAME)
def groundedness_check(state: GraphState) -> dict[str, Any]:

    answer = _answer_of(state)
    blocks = _blocks(state.get("retrieved"))

    if not answer:
        from src.chains.graph import GraphStateError

        raise GraphStateError(
            "groundedness_check received an empty answer_md: synthesis must produce one"
        )

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

    params = _effective_params(state.get("params"))
    result = None
    trimmed: list[ClaimUnit] = []
    if undecided and working.strip():
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

    if not _is_valid(payload):
        logger.warning(
            "%s unusable verdict payload; trimming every claim the pre-pass could not settle",
            NODE_NAME,
        )
        return list(units)
    return _trim_units(cast(Mapping[str, Any], payload), units)


def _keep_citations(state: GraphState, answer: str) -> list[Citation]:

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

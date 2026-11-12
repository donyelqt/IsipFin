from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from src.chains.checks import (
    OUTCOME_REJECTED,
    OUTCOME_SUPPORTED,
    ClaimUnit,
    numeric_pre_pass,
    rejected_units,
    trim_answer,
    undecided_units,
)
from src.chains.nodes.verify import (
    GROUNDING_SCHEMA,
    _locate,
    _payload,
    _trim_units,
    _verdicts_by_claim_id,
    _verdicts_by_position,
    groundedness_check,
)
from src.chains.state import Citation
from src.utils.llm import ChatResult, LLMParams, validate_json_schema
from src.utils.prompts import load_prompt

ANSWER = "Revenue was $211,915M [3]."

INCOME_STATEMENT_SPAN = (
    "Years Ended June 30, | 2023 | 2022 | 2021\n"
    "Revenue $ 211,915 $ 198,270 $ 168,088\n"
    "Operating income $ 88,523 $ 83,383 $ 69,916"
)

ALL_BLOCKS: dict[int, str] = {number: INCOME_STATEMENT_SPAN for number in range(1, 8)}

FEW_BLOCKS: dict[int, str] = {
    1: "Revenue $ 211,915 and $ 198,270 for the FY2023 and FY2022 years.",
    2: "Operating income $ 88,523 for the FY2023 year.",
    3: "Cash flow $ 67,000 for the FY2023 year.",
}


class StubGateway:

    def __init__(self, parsed: dict[str, Any], text: str = "{}") -> None:
        self.parsed = parsed
        self.text = text
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(
            text=self.text,
            parsed=self.parsed,
            prompt_tokens=8,
            completion_tokens=2,
            latency_ms=1.0,
            model="stub-model",
        )


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[dict[str, Any]], StubGateway]]:

    holder: dict[str, StubGateway] = {}

    def factory(params: LLMParams | None = None) -> StubGateway:
        return holder["gateway"]

    def install(parsed: dict[str, Any]) -> StubGateway:
        holder["gateway"] = StubGateway(parsed)
        return holder["gateway"]

    monkeypatch.setattr("src.chains.nodes.verify.build_llm", factory)
    yield install


def _unit(text: str, marker: int | None = None) -> ClaimUnit:

    return ClaimUnit(
        text=text,
        start=0,
        end=len(text),
        prefix="",
        suffix="",
        markers=(marker,) if marker is not None else (),
        figures=(),
        periods=(),
    )


def _units(texts: list[str]) -> list[ClaimUnit]:

    return [_unit(text, marker) for text, marker in _texts_and_markers(texts)]


def _texts_and_markers(texts: list[str]) -> list[tuple[str, int | None]]:

    pairs: list[tuple[str, int | None]] = []
    for text in texts:
        found = re.findall(r"\[(\d+)\]", text)
        pairs.append((text, int(found[0]) if found else None))
    return pairs


def _texts_of(units: list[ClaimUnit]) -> list[str]:

    return [unit.text for unit in units]


def _retrieved(blocks: dict[int, str]) -> list[Any]:

    class _Chunk:
        def __init__(self, parent_text: str, page: int) -> None:
            self.parent_text = parent_text
            self.page = page

    return [_Chunk(text, 40 + number) for number, text in blocks.items()]


def _state(answer: str, blocks: dict[int, str] | None = None) -> dict[str, Any]:

    return {"answer_md": answer, "retrieved": _retrieved(blocks or FEW_BLOCKS)}


def _user_turn(gateway: StubGateway) -> str:

    assert gateway.calls, "the node never called the gateway"
    return next(message["content"] for message in gateway.calls[0] if message["role"] == "user")


def _context_of(payload: str) -> str:

    return payload.split("CONTEXT:", 1)[1]


def _supported(claim_id: int = 1) -> dict[str, Any]:

    return {
        "claims": [{"claim_id": claim_id, "verdict": "supported", "reason": "printed"}],
        "all_supported": True,
    }


def test_a_verdict_is_resolved_by_claim_id_not_by_list_position() -> None:

    payload = {
        "claims": [
            {"claim_id": 1, "verdict": "supported", "reason": "the figure is printed"},
            {"claim_id": 2, "verdict": "unsupported", "reason": "no additional claims present."},
        ],
        "all_supported": False,
    }

    assert _trim_units(payload, _units([ANSWER])) == []


def test_an_off_by_one_claim_id_does_not_trim_a_good_claim() -> None:

    payload = {
        "claims": [{"claim_id": 2, "verdict": "unsupported", "reason": "wrong figure"}],
        "all_supported": True,
    }

    assert _trim_units(payload, _units([ANSWER])) == []


def test_an_unaddressed_claim_still_trims_under_all_supported_false() -> None:

    payload = {
        "claims": [{"claim_id": 2, "verdict": "unsupported", "reason": "wrong figure"}],
        "all_supported": False,
    }

    assert _trim_units(payload, _units([ANSWER])) == _units([ANSWER])


def test_a_reordered_id_list_does_not_shift_verdicts_onto_other_claims() -> None:

    payload = {
        "claims": [
            {"claim_id": 2, "verdict": "unsupported", "reason": "figure absent from the span"},
            {"claim_id": 1, "verdict": "supported", "reason": "both elements are printed"},
        ],
        "all_supported": False,
    }

    assert _texts_of(_trim_units(payload, _units(["Claim one.", "Claim two."]))) == ["Claim two."]


def test_an_empty_claim_list_with_all_supported_false_trims() -> None:

    payload = {"claims": [], "all_supported": False}
    units = _units(["Revenue grew.", "Costs fell."])

    assert _texts_of(_trim_units(payload, units)) == _texts_of(units)


def test_an_empty_claim_list_with_all_supported_true_trims_nothing() -> None:

    assert (
        _trim_units({"claims": [], "all_supported": True}, _units(["Revenue grew.", "Costs fell."]))
        == []
    )


def test_a_duplicated_claim_id_keeps_the_pessimistic_verdict() -> None:

    payload = {
        "claims": [
            {"claim_id": 1, "verdict": "supported", "reason": "in the span"},
            {"claim_id": 1, "verdict": "unsupported", "reason": "actually it is not"},
        ],
        "all_supported": True,
    }

    assert _texts_of(_trim_units(payload, _units([ANSWER]))) == _texts_of(_units([ANSWER]))


def test_an_incomplete_id_set_leaves_the_unnamed_claim_to_all_supported() -> None:

    payload = {
        "claims": [{"claim_id": 1, "verdict": "unsupported", "reason": "absent"}],
        "all_supported": True,
    }

    assert _texts_of(_trim_units(payload, _units(["Claim one.", "Claim two."]))) == ["Claim one."]


def test_a_position_fallback_is_used_when_no_id_is_usable() -> None:

    payload = {
        "claims": [{"verdict": "supported"}, {"verdict": "unsupported"}],
        "all_supported": False,
    }

    assert _texts_of(_trim_units(payload, _units(["Claim one.", "Claim two."]))) == ["Claim two."]


def test_a_bool_claim_id_is_not_a_usable_id() -> None:

    payload = {"claims": [{"claim_id": True, "verdict": "unsupported"}], "all_supported": True}

    assert _verdicts_by_claim_id(payload["claims"]) == {}
    assert _verdicts_by_position(payload["claims"]) == {1: "unsupported"}
    assert _texts_of(_trim_units(payload, _units([ANSWER]))) == _texts_of(_units([ANSWER]))


def test_a_string_claim_id_is_accepted_but_a_non_numeric_one_is_not() -> None:

    string_id = {"claim_id": "2", "verdict": "unsupported"}
    assert _verdicts_by_claim_id([string_id]) == {2: "unsupported"}

    prose_id = {"claim_id": "claim 2 of 2", "verdict": "unsupported"}
    assert _verdicts_by_claim_id([prose_id]) == {}


def test_one_bad_actor_inside_a_passing_answer_still_trims_that_claim() -> None:

    payload = {
        "claims": [
            {"claim_id": 1, "verdict": "supported", "reason": "printed"},
            {"claim_id": 2, "verdict": "unsupported", "reason": "that figure is a segment's"},
        ],
        "all_supported": True,
    }

    assert _texts_of(_trim_units(payload, _units(["Claim one.", "Claim two."]))) == ["Claim two."]


def test_a_verdict_is_case_and_whitespace_insensitive() -> None:

    payload = {"claims": [{"claim_id": 1, "verdict": " Supported "}], "all_supported": True}

    assert _trim_units(payload, _units([ANSWER])) == []


def test_partially_is_a_trim_and_supported_is_the_only_pass() -> None:

    claims = [
        {"claim_id": 1, "verdict": "supported"},
        {"claim_id": 2, "verdict": "partially"},
        {"claim_id": 3, "verdict": "unsupported"},
    ]
    units = _units(["Claim one.", "Claim two.", "Claim three."])

    trimmed = _trim_units({"claims": claims, "all_supported": False}, units)

    assert _texts_of(trimmed) == ["Claim two.", "Claim three."]


def test_a_verdict_for_a_claim_nobody_handed_over_is_ignored() -> None:

    payload = {"claims": [{"claim_id": 99, "verdict": "unsupported"}], "all_supported": True}

    assert _trim_units(payload, _units([ANSWER])) == []


def test_an_entry_without_a_verdict_says_nothing_and_trims_nothing() -> None:

    payload = {"claims": [{"claim_id": 1, "reason": "I cannot decide"}], "all_supported": False}

    assert _verdicts_by_claim_id(payload["claims"]) == {}
    assert _trim_units(payload, _units([ANSWER])) == _units([ANSWER])


def test_the_node_reads_a_real_claim_id_verdict_end_to_end(gateway: Callable[[Any], Any]) -> None:

    gateway(
        {
            "claims": [
                {"claim_id": 1, "verdict": "unsupported", "reason": "the claim names no period"}
            ],
            "all_supported": False,
        }
    )

    update = groundedness_check(_state(ANSWER, ALL_BLOCKS))

    assert update["grounded"] is False
    assert "(unsupported claim removed)" in update["answer_md"]
    assert "211,915" not in update["answer_md"]


def test_an_off_by_one_id_never_trims_a_claim_the_node_handed_over(
    gateway: Callable[[Any], Any],
) -> None:

    gateway(
        {
            "claims": [{"claim_id": 2, "verdict": "unsupported", "reason": "off by one"}],
            "all_supported": True,
        }
    )

    update = groundedness_check(_state(ANSWER, ALL_BLOCKS))

    assert update["grounded"] is True
    assert "211,915" in update["answer_md"]
    assert "(unsupported claim removed)" not in update["answer_md"]


def test_the_payload_omits_the_blocks_no_claim_cites(gateway: Callable[[Any], Any]) -> None:

    client = gateway(_supported())

    groundedness_check(_state(ANSWER, ALL_BLOCKS))

    context = _context_of(_user_turn(client))
    assert "[3] Years Ended June 30" in context
    for number in (1, 2, 4, 5, 6, 7):
        assert f"[{number}]" not in context


def test_the_payload_carries_the_union_of_every_cited_block(gateway: Callable[[Any], Any]) -> None:

    client = gateway(_supported())

    groundedness_check(
        _state("Revenue was $211,915M and cash flow was $67,000M [1] [3].", FEW_BLOCKS)
    )

    context = _context_of(_user_turn(client))
    assert "[1] Revenue $ 211,915" in context
    assert "[3] Cash flow $ 67,000" in context
    assert "Operating income" not in context


def test_the_payload_falls_back_to_every_block_when_nothing_is_cited(
    gateway: Callable[[Any], Any],
) -> None:

    client = gateway(_supported())

    groundedness_check(_state("Revenue grew a lot this year.", FEW_BLOCKS))

    context = _context_of(_user_turn(client))
    assert "[1] Revenue $ 211,915" in context
    assert "[2] Operating income $ 88,523" in context
    assert "[3] Cash flow $ 67,000" in context


def test_the_payload_of_a_unit_citing_nothing_sends_every_block() -> None:

    uncited = _unit("Revenue grew a lot.", marker=None)
    assert _context_of(_payload("Revenue grew a lot.", FEW_BLOCKS, [uncited])).splitlines()[1:] == [
        f"[{number}] {FEW_BLOCKS[number]}" for number in (1, 2, 3)
    ]

    phantom = _unit("Revenue was $211,915M [9].", marker=9)
    assert "9]" not in _context_of(_payload("Revenue was $211,915M [9].", FEW_BLOCKS, [phantom]))

    assert _payload("Revenue grew.", FEW_BLOCKS, []) == _payload(
        "Revenue grew.", FEW_BLOCKS, [_unit("Revenue grew.")]
    )


def test_a_citation_of_a_block_that_does_not_exist_sends_nothing_for_it(
    gateway: Callable[[Any], Any],
) -> None:

    client = gateway(_supported())

    groundedness_check(_state("Revenue was $211,915M [1] [9].", FEW_BLOCKS))

    context = _context_of(_user_turn(client))
    assert "[1] Revenue $ 211,915" in context
    assert "[9]" not in context


def test_a_fully_cited_answer_sends_every_block_unchanged(gateway: Callable[[Any], Any]) -> None:

    client = gateway(_supported())

    groundedness_check(_state("Revenue was $211,915M [1] [2] [3].", FEW_BLOCKS))

    assert _user_turn(client) == (
        "CANDIDATE ANSWER:\nRevenue was $211,915M [1] [2] [3].\n"
        "\nCONTEXT:\n" + "\n".join(f"[{number}] {FEW_BLOCKS[number]}" for number in (1, 2, 3))
    )


def test_the_trimmed_payload_is_a_real_size_reduction(gateway: Callable[[Any], Any]) -> None:

    client = gateway(_supported())
    block = (
        "Years Ended June 30, Revenue $ 211,915 $ 198,270 and Operating income $ 88,523 "
        "$ 83,383 were reported for the FY2023 and FY2022 years. " * 6
    )
    blocks = {number: block for number in range(1, 8)}

    groundedness_check(_state(ANSWER, blocks))

    context = _context_of(_user_turn(client))
    assert len(context) < len(block) * 1.5
    assert context.count("\n") == 1
    assert context.strip().startswith("[3]")


def test_the_answer_sent_to_p5_is_the_after_pre_pass_one(gateway: Callable[[Any], Any]) -> None:

    client = gateway(_supported())

    groundedness_check(
        _state(
            "Revenue was $245,122M in FY2024 [1]. Operating margin was 41.8% in FY2023 [1].",
            FEW_BLOCKS,
        )
    )

    user_turn = _user_turn(client)
    assert "245,122" not in user_turn
    assert "(unsupported claim removed)" in user_turn
    assert "Operating margin was 41.8% in FY2023 [1]." in user_turn


def test_an_uncited_block_is_not_sent_even_when_another_claim_uses_it(
    gateway: Callable[[Any], Any],
) -> None:

    client = gateway(_supported())

    groundedness_check(
        _state(
            "Revenue was $245,122M in FY2024 [2]. Operating margin was 41.8% in FY2023 [1].",
            FEW_BLOCKS,
        )
    )

    context = _context_of(_user_turn(client))
    assert "Cash flow" not in context
    assert "[1] Revenue $ 211,915" in context


def test_a_period_ambiguous_claim_is_the_one_handed_to_p5(gateway: Callable[[Any], Any]) -> None:

    client = gateway(_supported())

    update = groundedness_check(_state(ANSWER, ALL_BLOCKS))

    assert update["grounded"] is True
    assert _user_turn(client).startswith(f"CANDIDATE ANSWER:\n{ANSWER}\n\nCONTEXT:")


def test_no_llm_call_is_made_when_the_pre_pass_settles_everything() -> None:

    state = _state(
        "Revenue was $211,915M in FY2023 [1].",
        {1: "Revenue $ 211,915 for the FY2023 year."},
    )

    update = groundedness_check(state)

    assert update["grounded"] is True
    assert "llm" not in update


def test_a_payload_shaped_like_the_real_one_is_schema_valid() -> None:

    payload = {
        "claims": [
            {"claim_id": 1, "verdict": "unsupported", "span": None, "reason": "no cited block"}
        ],
        "all_supported": False,
    }

    assert validate_json_schema(payload, GROUNDING_SCHEMA) == []


def test_citations_of_a_trimmed_claim_go_with_it(gateway: Callable[[Any], Any]) -> None:

    gateway({"claims": [{"claim_id": 1, "verdict": "unsupported"}], "all_supported": False})
    state = _state(ANSWER, ALL_BLOCKS)
    state["citations"] = [Citation(ref_index=3, doc_id="msft-fy2023", page=44)]

    update = groundedness_check(state)

    assert update["citations"] == []


def test_the_result_payload_floats_back_into_the_trace(gateway: Callable[[Any], Any]) -> None:

    gateway(_supported(claim_id=1))

    state = _state(ANSWER, ALL_BLOCKS)
    update = groundedness_check(state)

    event = state["node_trace"][-1]
    assert event.node == "groundedness_check"
    assert event.tokens_in == 8
    assert event.tokens_out == 2
    assert "llm" not in update


NARROW_NOBREAK_SPACE = "\u202f"
NON_BREAKING_HYPHEN = "\u2011"
Q07_ANSWER = (
    f"The documents do not provide operating{NON_BREAKING_HYPHEN}margin percentages for "
    "FY2023 or FY2022. [1]\n"
    "They do, however, disclose the figures that would be needed to calculate them: [2]\n"
    "\n"
    "| FY | Operating income | Revenue |\n"
    "|---|---|---|\n"
    f"| FY2023 | ${NARROW_NOBREAK_SPACE}88,523{NARROW_NOBREAK_SPACE}million | "
    f"${NARROW_NOBREAK_SPACE}211,915{NARROW_NOBREAK_SPACE}million |\n"
    f"| FY2022 | ${NARROW_NOBREAK_SPACE}83,383{NARROW_NOBREAK_SPACE}million | "
    f"${NARROW_NOBREAK_SPACE}198,270{NARROW_NOBREAK_SPACE}million |\n"
)

Q07_SPANS: dict[int, str] = {
    1: "Segment commentary without the income statement's figures.",
    2: "A narrative block about cloud services growth.",
}


def _stage_one(answer: str, blocks: dict[int, str] | None = None) -> tuple[str, list[ClaimUnit]]:

    judgements = numeric_pre_pass(answer, blocks or Q07_SPANS)
    return trim_answer(answer, rejected_units(judgements)), undecided_units(judgements)


def test_a_stage_two_trim_lands_on_the_claim_not_mid_word() -> None:

    answer = "Revenue was $198,270 million in FY2023 [3].\nOperating margin was 41.8% [2]."
    working, undecided = _stage_one(answer, {2: "A block.", 3: "Another block."})

    located = _locate(working, undecided)
    relocated = [
        dataclasses.replace(unit, start=start, end=start + len(unit.text))
        for unit, start in located
    ]
    trimmed = trim_answer(working, relocated)

    assert trimmed == "(unsupported claim removed)\n(unsupported claim removed)"
    assert "margin(unsupported" not in trimmed


def test_a_trim_never_leaves_a_partial_word_before_the_marker() -> None:

    answer = "Operating margin was 41.8% [2].\nRevenue was $198,270 million in FY2023 [3]."
    working, undecided = _stage_one(answer, {2: "A block.", 3: "Another block."})

    relocated = [
        dataclasses.replace(unit, start=start, end=start + len(unit.text))
        for unit, start in _locate(working, undecided)
    ]
    trimmed = trim_answer(working, relocated)

    assert trimmed == "(unsupported claim removed)\n(unsupported claim removed)"


def test_the_live_q07_answer_trims_to_the_claims_it_actually_printed(
    gateway: Callable[[Any], Any],
) -> None:

    gateway(
        {
            "claims": [{"claim_id": number, "verdict": "unsupported"} for number in range(1, 6)],
            "all_supported": False,
        }
    )

    state = _state(Q07_ANSWER, Q07_SPANS)
    update = groundedness_check(state)

    answer = update["answer_md"]
    assert answer.count("(unsupported claim removed)") == 7
    assert "margin(unsupported" not in answer
    assert answer.count("\n||") == 4
    assert update["grounded"] is False


def test_a_claim_no_longer_in_the_answer_is_left_alone() -> None:

    working, undecided = _stage_one(
        "Revenue was $198,270 million in FY2023 [3].\nA trend sentence.",
        {2: "A block.", 3: "Another block."},
    )

    stale = "Revenue was $198,270 million in FY2023 [3]."
    claims = [dataclasses.replace(u, text=stale) for u in undecided]

    assert _locate(working, claims) == []


def test_locate_places_every_unit_in_document_order() -> None:

    text = "First line.\nFirst line.\nFirst line."
    unit = ClaimUnit(
        text="First line.",
        start=99,
        end=110,
        prefix="",
        suffix="",
        markers=(),
        figures=(),
        periods=(),
    )

    assert [start for _, start in _locate(text, [unit, unit, unit])] == [0, 12, 24]


INCOME_STATEMENT = (
    "SUMMARY RESULTS OF OPERATIONS\n"
    "(In millions, except percentages and per share amounts) 2023 2022\n"
    "Revenue $ 211,915 $ 198,270 7%\n"
    "Operating income 88,523 83,383 6%\n"
    "Net income 72,361 72,738 (1)%\n"
    "Diluted earnings per share 9.68 9.65 0%"
)

Q03_CITED_IN_ROWS = (
    "| Metric | FY2023 | FY2022 |\n"
    "|---|---|---|\n"
    "| Revenue [2] | $211,915 | $198,270 |\n"
    "| Operating income [2] | $88,523 | $83,383 |"
)

Q03_CITED_IN_NOTE = (
    "| Metric | FY2023 | FY2022 |\n"
    "|---|---|---|\n"
    "| Revenue | $211,915 | $198,270 |\n"
    "| Operating income | $88,523 | $83,383 |\n"
    "\n"
    "*Figures are from the income statement in [2].*"
)


def test_a_table_with_markers_in_its_rows_survives_the_pre_pass() -> None:

    judgements = numeric_pre_pass(Q03_CITED_IN_ROWS, {2: INCOME_STATEMENT})
    rows = [judgement for judgement in judgements if judgement.unit.figures]

    assert len(rows) == 2
    assert all(judgement.outcome == OUTCOME_SUPPORTED for judgement in rows)
    assert rejected_units(judgements) == []


def test_a_table_with_markers_in_a_note_below_it_loses_every_row() -> None:

    judgements = numeric_pre_pass(Q03_CITED_IN_NOTE, {2: INCOME_STATEMENT})
    rows = [judgement for judgement in judgements if judgement.unit.figures]

    assert len(rows) == 2
    assert [judgement.outcome for judgement in rows] == [
        OUTCOME_REJECTED,
        OUTCOME_REJECTED,
    ]
    assert all(judgement.reason == "figure_not_in_span" for judgement in rows)
    assert trim_answer(Q03_CITED_IN_NOTE, [row.unit for row in rows]) == Q03_CITED_IN_NOTE.replace(
        "| Revenue | $211,915 | $198,270 |\n| Operating income | $88,523 | $83,383 |",
        "||(unsupported claim removed)||\n||(unsupported claim removed)||",
    )


def test_the_prompt_and_the_checker_agree_on_where_a_table_marker_belongs() -> None:

    prompt = load_prompt("P4").body
    assert "Put the [n] marker in the table itself" in prompt

    figures = [
        judgement
        for judgement in numeric_pre_pass(Q03_CITED_IN_ROWS, {2: INCOME_STATEMENT})
        if judgement.unit.figures
    ]
    assert all(judgement.outcome == OUTCOME_SUPPORTED for judgement in figures)

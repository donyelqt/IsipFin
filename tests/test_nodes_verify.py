"""``groundedness_check``: verifier verdict matching and the P5 payload it sends.

Two invariants are pinned here, both of them bugs a live real-provider run exposed.

**Defect A — verdicts are matched by the ``claim_id`` P5 sends, not by list position.**
The live run handed P5 one undecided claim and got
``claims: [{claim_id: 1, supported}, {claim_id: 2, unsupported}]`` back. Position matching
happened to survive that, but the mapping is what was wrong: a model that numbers its
verdicts off by one (``claim_id: 2`` for the only claim handed over) was read as a verdict
about the *second* claim — one that does not exist — while the real claim inherited the
fallback. So a verdict is resolved by id first, and by ordinal position only when the payload
names no claim by a usable id.

**Defect B — the payload re-shipped every retrieved block.** The measured run sent 21.0 KB of
context (~6,900 prompt tokens) for an answer citing one of seven blocks, and spent 20.7 s in
one prefill-dominated provider POST. P5 now receives the union of the blocks the claims it is
judging actually cite — every block when none of them cites anything, because judging a claim
against a context this node withheld would punish the caller for this node's choice.

Everything here is hermetic: the LLM is driven through a stubbed gateway, so no test needs a
key, a network round trip, or a real model. The node's own symbolic pre-pass
(``src/chains/checks.py``) is untouched and runs for real underneath.

The answers below are deliberately ones the pre-pass cannot settle — a claim naming no fiscal
period over a span that carries several, a computed rate, a qualitative claim — because those
are the only claims that reach P5 at all.

**Defect C — a stage-2 trim spliced at stage-1's offsets.** ``ClaimUnit`` offsets are measured
against the answer the pre-pass read. When the pre-pass trims a claim, every line after it
shifts, and the stage-2 trim — spliced at the *original* offsets — lands mid-word. The real
q07 run rendered ``Operating margin(unsupported claim removed)`` for one claim and stacked a
second table of markers below the first, so one visible marker became four. ``_locate``
re-finds each claim's text in the answer it is about to be spliced into.
"""

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

# --- the answer and context a live run used ----------------------------------------
#
# One undecided claim citing one of seven retrieved blocks: the shape that produced the
# measured prompt_tokens≈6,900 / 20.7s. The claim names no fiscal period while the block
# carries FY2023, FY2022 and FY2021, so the symbolic pre-pass correctly hands it to P5.
ANSWER = "Revenue was $211,915M [3]."

#: The span the real corpus prints: the income statement on p.44, FY2023/FY2022/FY2021 in
#: adjacent columns. Any claim over this block that names no period is undecided.
INCOME_STATEMENT_SPAN = (
    "Years Ended June 30, | 2023 | 2022 | 2021\n"
    "Revenue $ 211,915 $ 198,270 $ 168,088\n"
    "Operating income $ 88,523 $ 83,383 $ 69,916"
)

#: Every block the corpus-sized run retrieved. All of them multi-period, so a claim citing
#: any one of them without naming a period is undecided — the only shape that reaches P5.
ALL_BLOCKS: dict[int, str] = {number: INCOME_STATEMENT_SPAN for number in range(1, 8)}

#: Three retrievable chunks: enough to prove a block is dropped, small enough to read.
FEW_BLOCKS: dict[int, str] = {
    1: "Revenue $ 211,915 and $ 198,270 for the FY2023 and FY2022 years.",
    2: "Operating income $ 88,523 for the FY2023 year.",
    3: "Cash flow $ 67,000 for the FY2023 year.",
}


# --- the stub gateway ---------------------------------------------------------------


class StubGateway:
    """A :class:`~src.utils.llm.ChatClient` returning one canned payload, recording the send.

    Stubbed rather than ``MOCK_LLM=1`` because the invariant under test is *what this node
    sends and how it reads the answer back*: the mock computes its own verdict from the
    payload, which is both unreadable and not the payload the real model returns. Recording
    the messages is what makes the payload-trimming tests decide anything.
    """

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
    """Install a gateway factory the test controls, for the node's one call.

    ``build_llm`` is the single place configuration is read (``src/utils/llm.py``), so
    patching it is the seam every node test uses; the node itself stays untransformed.
    ``monkeypatch`` restores the original on teardown, so no stub leaks into a sibling test
    that expects the real factory (or the mock, under ``MOCK_LLM=1``).
    """

    holder: dict[str, StubGateway] = {}

    def factory(params: LLMParams | None = None) -> StubGateway:
        return holder["gateway"]

    def install(parsed: dict[str, Any]) -> StubGateway:
        holder["gateway"] = StubGateway(parsed)
        return holder["gateway"]

    monkeypatch.setattr("src.chains.nodes.verify.build_llm", factory)
    yield install


# --- claim units and state ----------------------------------------------------------


def _unit(text: str, marker: int | None = None) -> ClaimUnit:
    """One claim unit citing ``marker`` (or nothing), with no figures to judge on."""

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
    """The claim units for ``texts``, each citing the marker written in its own text.

    Markers are read out of the text so a test states the claim and its citation in one
    place, instead of restating the id in a second argument.
    """

    return [_unit(text, marker) for text, marker in _texts_and_markers(texts)]


def _texts_and_markers(texts: list[str]) -> list[tuple[str, int | None]]:
    """``(text, first marker in it)`` per text — the marker a real splitter would attach."""

    pairs: list[tuple[str, int | None]] = []
    for text in texts:
        found = re.findall(r"\[(\d+)\]", text)
        pairs.append((text, int(found[0]) if found else None))
    return pairs


def _texts_of(units: list[ClaimUnit]) -> list[str]:
    """The unit texts, so a trim is asserted against what a reader would see."""

    return [unit.text for unit in units]


def _retrieved(blocks: dict[int, str]) -> list[Any]:
    """The retrieved list the node's ``_blocks`` numbers 1..N in this order."""

    class _Chunk:
        def __init__(self, parent_text: str, page: int) -> None:
            self.parent_text = parent_text
            self.page = page

    return [_Chunk(text, 40 + number) for number, text in blocks.items()]


def _state(answer: str, blocks: dict[int, str] | None = None) -> dict[str, Any]:
    """A graph state reaching ``groundedness_check`` with an answer and its context."""

    return {"answer_md": answer, "retrieved": _retrieved(blocks or FEW_BLOCKS)}


def _user_turn(gateway: StubGateway) -> str:
    """The user turn the node sent: the payload every assertion below reads."""

    assert gateway.calls, "the node never called the gateway"
    return next(message["content"] for message in gateway.calls[0] if message["role"] == "user")


def _context_of(payload: str) -> str:
    """Everything the payload sent under ``CONTEXT:`` — the part whose size is the defect."""

    return payload.split("CONTEXT:", 1)[1]


def _supported(claim_id: int = 1) -> dict[str, Any]:
    """A payload where the model supports the claim it was handed."""

    return {
        "claims": [{"claim_id": claim_id, "verdict": "supported", "reason": "printed"}],
        "all_supported": True,
    }


# --- Defect A: verdicts resolved by claim_id, not by list position ----------------


def test_a_verdict_is_resolved_by_claim_id_not_by_list_position() -> None:
    """The live shape: two verdict entries for a one-claim hand-off, ids off by one.

    The model's second entry names ``claim_id: 2`` — a claim this node never handed over. The
    old positional map read that entry as claim 2's *verdict* and, with ``all_supported:
    false``, trimmed the claim nobody had even asked about. The id that names no handed-over
    claim is now ignored, and the claim it was meant for still stands on its own verdict.
    """

    payload = {
        "claims": [
            {"claim_id": 1, "verdict": "supported", "reason": "the figure is printed"},
            {"claim_id": 2, "verdict": "unsupported", "reason": "no additional claims present."},
        ],
        "all_supported": False,
    }

    assert _trim_units(payload, _units([ANSWER])) == []


def test_an_off_by_one_claim_id_does_not_trim_a_good_claim() -> None:
    """The bug, in one pair: ``claim_id: 2`` for the only claim handed over.

    Under position matching, the model's verdict about claim 2 landed on claim 1 and trimmed
    it. Matched by id, the entry names no claim this node handed over, so it addresses
    nothing, and the claim it was probably meant for survives.
    """

    payload = {
        "claims": [{"claim_id": 2, "verdict": "unsupported", "reason": "wrong figure"}],
        "all_supported": True,
    }

    assert _trim_units(payload, _units([ANSWER])) == []


def test_an_unaddressed_claim_still_trims_under_all_supported_false() -> None:
    """Same off-by-one payload, but the model also said the answer does not stand.

    This is the policy for an id that names no handed-over claim: the verdict is ignored
    rather than re-pointed, and with ``all_supported: false`` the *unaddressed* claim is
    trimmed by the same pessimistic rule that governs ``claims: []``. A verifier that reports
    no verdict for a claim and says the answer fails has said nothing good about it.
    """

    payload = {
        "claims": [{"claim_id": 2, "verdict": "unsupported", "reason": "wrong figure"}],
        "all_supported": False,
    }

    assert _trim_units(payload, _units([ANSWER])) == _units([ANSWER])


def test_a_reordered_id_list_does_not_shift_verdicts_onto_other_claims() -> None:
    """The positional map's real hazard: the model lists its verdicts in a different order.

    Two claims, and the model answers about claim 2 first. Position matching read claim 2's
    verdict as claim 1's, so the claim the model *supported* was trimmed and the claim it
    *rejected* was kept — the failure mode a positional map hides, because it looks like it
    works whenever the model happens to list its claims in order.
    """

    payload = {
        "claims": [
            {"claim_id": 2, "verdict": "unsupported", "reason": "figure absent from the span"},
            {"claim_id": 1, "verdict": "supported", "reason": "both elements are printed"},
        ],
        "all_supported": False,
    }

    assert _texts_of(_trim_units(payload, _units(["Claim one.", "Claim two."]))) == ["Claim two."]


def test_an_empty_claim_list_with_all_supported_false_trims() -> None:
    """``claims: []`` plus ``all_supported: false``: the model said no, about everything.

    The pessimistic default in per-claim form — a verifier that reports no claim stands cannot
    be read as saying the answer does. Existing behaviour, kept.
    """

    payload = {"claims": [], "all_supported": False}
    units = _units(["Revenue grew.", "Costs fell."])

    assert _texts_of(_trim_units(payload, units)) == _texts_of(units)


def test_an_empty_claim_list_with_all_supported_true_trims_nothing() -> None:
    """``claims: []`` plus ``all_supported: true``: no claim was contradicted.

    A model silent about a claim it was handed is neutral, not a rejection — only an explicit
    non-supported verdict trims. Existing behaviour, kept.
    """

    assert (
        _trim_units({"claims": [], "all_supported": True}, _units(["Revenue grew.", "Costs fell."]))
        == []
    )


def test_a_duplicated_claim_id_keeps_the_pessimistic_verdict() -> None:
    """One claim id answered both ways: supported first, then unsupported.

    A contradiction between two verdicts for the same id is read as the rejection — the same
    reading every other ambiguous input in this node gets. The claim is trimmed.
    """

    payload = {
        "claims": [
            {"claim_id": 1, "verdict": "supported", "reason": "in the span"},
            {"claim_id": 1, "verdict": "unsupported", "reason": "actually it is not"},
        ],
        "all_supported": True,
    }

    assert _texts_of(_trim_units(payload, _units([ANSWER]))) == _texts_of(_units([ANSWER]))


def test_an_incomplete_id_set_leaves_the_unnamed_claim_to_all_supported() -> None:
    """Two claims handed over; the verdicts name only one of them.

    The named claim is trimmed on its own verdict. The unnamed one falls to the answer-level
    flag — ``all_supported: true`` here, which is neutral, so it stands. Handing a verifier
    two claims must not let one verdict decide both of them.
    """

    payload = {
        "claims": [{"claim_id": 1, "verdict": "unsupported", "reason": "absent"}],
        "all_supported": True,
    }

    assert _texts_of(_trim_units(payload, _units(["Claim one.", "Claim two."]))) == ["Claim one."]


def test_a_position_fallback_is_used_when_no_id_is_usable() -> None:
    """No entry carries a usable id, so where the model spoke is the only signal left.

    A model reporting only its own sequential numbering is still matched — against the
    position it spoke at, not against a claim it never named. Absent ids are the *fallback
    trigger*, not a parse failure.
    """

    payload = {
        "claims": [{"verdict": "supported"}, {"verdict": "unsupported"}],
        "all_supported": False,
    }

    assert _texts_of(_trim_units(payload, _units(["Claim one.", "Claim two."]))) == ["Claim two."]


def test_a_bool_claim_id_is_not_a_usable_id() -> None:
    """``claim_id: true`` is a schema violation that resolves to no claim.

    ``True`` must never be read as claim 1: an entry carrying a boolean id names nothing, so
    the payload falls back to positions, which is why the claim here is trimmed by *where the
    model spoke* rather than by an id that addresses nobody.
    """

    payload = {"claims": [{"claim_id": True, "verdict": "unsupported"}], "all_supported": True}

    assert _verdicts_by_claim_id(payload["claims"]) == {}
    assert _verdicts_by_position(payload["claims"]) == {1: "unsupported"}
    assert _texts_of(_trim_units(payload, _units([ANSWER]))) == _texts_of(_units([ANSWER]))


def test_a_string_claim_id_is_accepted_but_a_non_numeric_one_is_not() -> None:
    """``"2"`` names claim 2 (a model quoting the schema back); ``"one"`` names nothing.

    Only a whole-string integer resolves. Anything else leaves the entry unaddressed, and the
    payload falls back to positions rather than inventing an address.
    """

    string_id = {"claim_id": "2", "verdict": "unsupported"}
    assert _verdicts_by_claim_id([string_id]) == {2: "unsupported"}

    prose_id = {"claim_id": "claim 2 of 2", "verdict": "unsupported"}
    assert _verdicts_by_claim_id([prose_id]) == {}


def test_one_bad_actor_inside_a_passing_answer_still_trims_that_claim() -> None:
    """STRESS-TEST-RAG #2: ``all_supported: true`` must never rescue a rejected claim.

    This is the property the per-claim verdict exists for. The answer-level flag is the
    model's summary; the claim's own verdict decides, so a plausible-but-wrong number inside
    an otherwise clean answer is trimmed and flagged.
    """

    payload = {
        "claims": [
            {"claim_id": 1, "verdict": "supported", "reason": "printed"},
            {"claim_id": 2, "verdict": "unsupported", "reason": "that figure is a segment's"},
        ],
        "all_supported": True,
    }

    assert _texts_of(_trim_units(payload, _units(["Claim one.", "Claim two."]))) == ["Claim two."]


def test_a_verdict_is_case_and_whitespace_insensitive() -> None:
    """``" Supported "`` is ``supported``; a verifier must not trim on a spelling.

    Only the exact value ``supported`` keeps a claim, so a padded or capitalised verdict would
    otherwise read as a rejection.
    """

    payload = {"claims": [{"claim_id": 1, "verdict": " Supported "}], "all_supported": True}

    assert _trim_units(payload, _units([ANSWER])) == []


def test_partially_is_a_trim_and_supported_is_the_only_pass() -> None:
    """The full verdict vocabulary: only ``supported`` survives."""

    claims = [
        {"claim_id": 1, "verdict": "supported"},
        {"claim_id": 2, "verdict": "partially"},
        {"claim_id": 3, "verdict": "unsupported"},
    ]
    units = _units(["Claim one.", "Claim two.", "Claim three."])

    trimmed = _trim_units({"claims": claims, "all_supported": False}, units)

    assert _texts_of(trimmed) == ["Claim two.", "Claim three."]


def test_a_verdict_for_a_claim_nobody_handed_over_is_ignored() -> None:
    """``claim_id: 99`` names no claim in the hand-off, and is not re-pointed at one.

    An id outside the handed-over range says nothing about any claim here, so the answer-level
    flag decides — and with ``all_supported: true`` the claim stands.
    """

    payload = {"claims": [{"claim_id": 99, "verdict": "unsupported"}], "all_supported": True}

    assert _trim_units(payload, _units([ANSWER])) == []


def test_an_entry_without_a_verdict_says_nothing_and_trims_nothing() -> None:
    """A row naming a claim but carrying no verdict string is not a rejection.

    The entry is skipped rather than recorded as an empty verdict, and the claim it named falls
    to the answer-level flag. ``"reason"`` alone — as a model arguing without deciding — must
    not trim.
    """

    payload = {"claims": [{"claim_id": 1, "reason": "I cannot decide"}], "all_supported": False}

    assert _verdicts_by_claim_id(payload["claims"]) == {}
    assert _trim_units(payload, _units([ANSWER])) == _units([ANSWER])


def test_the_node_reads_a_real_claim_id_verdict_end_to_end(gateway: Callable[[Any], Any]) -> None:
    """The whole node, over a real pre-pass hand-off and a real ``claim_id`` payload.

    ``claim_id: 1`` with ``unsupported`` — the id the node actually handed out — trims the
    claim, sets ``grounded`` False, and leaves the visible trim marker.
    """

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
    """The Defect-A regression as the node runs it: the good claim survives a mis-numbered id.

    A payload whose only entry names ``claim_id: 2`` of a one-claim hand-off must leave that
    claim standing, because no verdict ever named it.
    """

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


# --- Defect B: the payload carries only the blocks the claims cite ----------------


def test_the_payload_omits_the_blocks_no_claim_cites(gateway: Callable[[Any], Any]) -> None:
    """Seven blocks retrieved, one cited: only the cited one is sent.

    The measured run paid ~6,900 prefill tokens to ship 21.0 KB for an answer citing ``[3]``
    of seven blocks. A claim is judged against the span it cites, so every other block is dead
    weight — and P5's own rule is that a figure must appear in the span cited for that claim.
    """

    client = gateway(_supported())

    groundedness_check(_state(ANSWER, ALL_BLOCKS))

    context = _context_of(_user_turn(client))
    assert "[3] Years Ended June 30" in context
    for number in (1, 2, 4, 5, 6, 7):
        assert f"[{number}]" not in context


def test_the_payload_carries_the_union_of_every_cited_block(gateway: Callable[[Any], Any]) -> None:
    """One claim citing two blocks: both are sent, and only those two.

    A period-ambiguous claim over two blocks is the shape that reaches P5, so both blocks
    it cites go to the model and the block only another claim would have used does not.
    """

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
    """No claim cites anything — the citation-free answer P5's primary rule exists to catch.

    Withholding the context and then trimming the claim for what the missing context does not
    contain would punish the caller for this node's choice, so every block is sent.
    """

    client = gateway(_supported())

    groundedness_check(_state("Revenue grew a lot this year.", FEW_BLOCKS))

    context = _context_of(_user_turn(client))
    assert "[1] Revenue $ 211,915" in context
    assert "[2] Operating income $ 88,523" in context
    assert "[3] Cash flow $ 67,000" in context


def test_the_payload_of_a_unit_citing_nothing_sends_every_block() -> None:
    """The fallback, at the function that makes it.

    A unit citing no block — or citing only markers outside the numbered range — hands the
    whole context to P5, because an empty CONTEXT section would make the model's verdict a
    statement about a context this node never sent.
    """

    uncited = _unit("Revenue grew a lot.", marker=None)
    assert _context_of(_payload("Revenue grew a lot.", FEW_BLOCKS, [uncited])).splitlines()[1:] == [
        f"[{number}] {FEW_BLOCKS[number]}" for number in (1, 2, 3)
    ]

    # A marker pointing outside the numbered range has no text to ship, so it triggers the
    # same fallback rather than a phantom block.
    phantom = _unit("Revenue was $211,915M [9].", marker=9)
    assert "9]" not in _context_of(_payload("Revenue was $211,915M [9].", FEW_BLOCKS, [phantom]))

    # No units at all is also a no-citation hand-off.
    assert _payload("Revenue grew.", FEW_BLOCKS, []) == _payload(
        "Revenue grew.", FEW_BLOCKS, [_unit("Revenue grew.")]
    )


def test_a_citation_of_a_block_that_does_not_exist_sends_nothing_for_it(
    gateway: Callable[[Any], Any],
) -> None:
    """A marker pointing past the numbered blocks has no text to ship.

    The existing markers still force the fallback: with one of two markers unresolvable, the
    resolvable one is the only context left, and the phantom number is never invented.
    """

    client = gateway(_supported())

    groundedness_check(_state("Revenue was $211,915M [1] [9].", FEW_BLOCKS))

    context = _context_of(_user_turn(client))
    assert "[1] Revenue $ 211,915" in context
    assert "[9]" not in context


def test_a_fully_cited_answer_sends_every_block_unchanged(gateway: Callable[[Any], Any]) -> None:
    """The fully-cited case must be byte-identical to the old payload.

    When every claim cites every block, the union is all of them — the trimming is invisible,
    which is what keeps the existing behaviour for the answers that already paid for it.
    """

    client = gateway(_supported())

    groundedness_check(_state("Revenue was $211,915M [1] [2] [3].", FEW_BLOCKS))

    assert _user_turn(client) == (
        "CANDIDATE ANSWER:\nRevenue was $211,915M [1] [2] [3].\n"
        "\nCONTEXT:\n" + "\n".join(f"[{number}] {FEW_BLOCKS[number]}" for number in (1, 2, 3))
    )


def test_the_trimmed_payload_is_a_real_size_reduction(gateway: Callable[[Any], Any]) -> None:
    """The measured defect, as numbers: one block instead of 21.0 KB of context."""

    client = gateway(_supported())
    # A block is a parent chunk of the size the corpus prints, ~1 KB, with its period labels.
    block = (
        "Years Ended June 30, Revenue $ 211,915 $ 198,270 and Operating income $ 88,523 "
        "$ 83,383 were reported for the FY2023 and FY2022 years. " * 6
    )
    blocks = {number: block for number in range(1, 8)}

    groundedness_check(_state(ANSWER, blocks))

    context = _context_of(_user_turn(client))
    assert len(context) < len(block) * 1.5  # one block's worth, not seven
    assert context.count("\n") == 1  # exactly one block line below the CONTEXT: header
    assert context.strip().startswith("[3]")


def test_the_answer_sent_to_p5_is_the_after_pre_pass_one(gateway: Callable[[Any], Any]) -> None:
    """The payload carries the already-trimmed answer, so a rejected claim is never re-read.

    This is the structural half of P5's "do not re-litigate": the wrong-year claim is trimmed
    by the symbolic layer before the payload is built. The survivor that P5 judges is the
    undecided one.
    """

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
    """The union is across the claims P5 is judging, not across the whole answer.

    A block only the trimmed claim cited is not part of the hand-off, so it is not shipped
    either: the payload follows ``undecided``, never the rejected-by-pre-pass claims.
    """

    client = gateway(_supported())

    groundedness_check(
        _state(
            "Revenue was $245,122M in FY2024 [2]. Operating margin was 41.8% in FY2023 [1].",
            FEW_BLOCKS,
        )
    )

    context = _context_of(_user_turn(client))
    assert "Cash flow" not in context  # only the rejected claim cited [2]
    assert "[1] Revenue $ 211,915" in context


# --- the pre-pass → hand-off wiring, untouched ------------------------------------


def test_a_period_ambiguous_claim_is_the_one_handed_to_p5(gateway: Callable[[Any], Any]) -> None:
    """The live failure reproduced symbolically: no period named, multi-period span.

    The pre-pass cannot settle it, so it is the claim P5 judges — which is why the id matching
    matters for the payload that comes back.
    """

    client = gateway(_supported())

    update = groundedness_check(_state(ANSWER, ALL_BLOCKS))

    assert update["grounded"] is True
    assert _user_turn(client).startswith(f"CANDIDATE ANSWER:\n{ANSWER}\n\nCONTEXT:")


def test_no_llm_call_is_made_when_the_pre_pass_settles_everything() -> None:
    """Nothing undecided means no gateway call, and no token accounting to report."""

    state = _state(
        "Revenue was $211,915M in FY2023 [1].",
        {1: "Revenue $ 211,915 for the FY2023 year."},
    )

    update = groundedness_check(state)

    assert update["grounded"] is True
    assert "llm" not in update  # no ChatResult, so the trace records tokens as unmeasured


def test_a_payload_shaped_like_the_real_one_is_schema_valid() -> None:
    """The payloads these tests stub are ones the gateway would accept.

    A test payload that violated the node's own schema would pin behaviour against a contract
    no real call satisfies, so the shape under test is validated against it.
    """

    payload = {
        "claims": [
            {"claim_id": 1, "verdict": "unsupported", "span": None, "reason": "no cited block"}
        ],
        "all_supported": False,
    }

    assert validate_json_schema(payload, GROUNDING_SCHEMA) == []


def test_citations_of_a_trimmed_claim_go_with_it(gateway: Callable[[Any], Any]) -> None:
    """A citation is evidence for its claim: a trimmed claim leaves no marker to trace back."""

    gateway({"claims": [{"claim_id": 1, "verdict": "unsupported"}], "all_supported": False})
    state = _state(ANSWER, ALL_BLOCKS)
    state["citations"] = [Citation(ref_index=3, doc_id="msft-fy2023", page=44)]

    update = groundedness_check(state)

    assert update["citations"] == []


def test_the_result_payload_floats_back_into_the_trace(gateway: Callable[[Any], Any]) -> None:
    """The node reports its gateway accounting under the reserved key (ADR-007).

    ``TOKEN_KEY`` is consumed by ``@trace_node`` before the update reaches the caller, so the
    assertion is on the trace event the decorator built from it — which is the accounting the
    debug panel and the ``events`` row actually carry. ``latency_ms`` and ``model`` come along.
    """

    gateway(_supported(claim_id=1))

    state = _state(ANSWER, ALL_BLOCKS)
    update = groundedness_check(state)

    event = state["node_trace"][-1]
    assert event.node == "groundedness_check"
    assert event.tokens_in == 8
    assert event.tokens_out == 2
    assert "llm" not in update  # consumed by the decorator, never reaching state


# --- Defect C: a stage-2 trim is spliced where the claim now sits -----------------

#: The real q07 answer, as the live run printed it: two prose sentences, a table whose
#: markers were all in the source note, and the answer's own typography — a non-breaking
#: hyphen in "operating-margin" and narrow no-break spaces in "$ 88,523 million". Every
#: claim here is one the pre-pass cannot settle (no figure, or no period named), and the
#: two table rows carry markers only in the source note below them.
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

#: The spans that answer cites: neither block prints the figures the table shows.
Q07_SPANS: dict[int, str] = {
    1: "Segment commentary without the income statement's figures.",
    2: "A narrative block about cloud services growth.",
}


def _stage_one(answer: str, blocks: dict[int, str] | None = None) -> tuple[str, list[ClaimUnit]]:
    """``(the pre-pass's trimmed answer, the undecided units it handed to P5)``."""

    judgements = numeric_pre_pass(answer, blocks or Q07_SPANS)
    return trim_answer(answer, rejected_units(judgements)), undecided_units(judgements)


def test_a_stage_two_trim_lands_on_the_claim_not_mid_word() -> None:
    """The visible defect: a trim spliced at a stale offset landed inside a word.

    A rejected claim before the undecided one shortens the answer, so the undecided unit's
    original offsets point past the text it named. The old splice printed
    ``Operating margin(unsupported claim removed)`` — half a claim plus a marker, which is
    neither the claim nor its removal.
    """

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
    """The same shape with the marks reversed, so the surviving line is prose."""

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
    """The q07 run as it happened, end to end: five markers, no halves, no duplicates.

    The real answer carried two prose claims and a four-row table, and the provider's
    verdicts trimmed all of them. The answer must show exactly one marker per claim — the
    two table rows the pre-pass already trimmed, and the three prose claims P5 trimmed — and
    never the second table of markers the stale splice produced below the first.
    """

    gateway(
        {
            "claims": [{"claim_id": number, "verdict": "unsupported"} for number in range(1, 6)],
            "all_supported": False,
        }
    )

    state = _state(Q07_ANSWER, Q07_SPANS)
    update = groundedness_check(state)

    answer = update["answer_md"]
    # One marker per claim, and no more: five prose/header claims P5 trimmed and two table
    # rows the pre-pass already trimmed. The stale splice printed a *second* table of four
    # markers under the first, so a count above the claim count is the regression.
    assert answer.count("(unsupported claim removed)") == 7
    assert "margin(unsupported" not in answer
    # The table's rows stay rows: the pipes survive, so the UI still renders a table.
    assert answer.count("\n||") == 4
    assert update["grounded"] is False


def test_a_claim_no_longer_in_the_answer_is_left_alone() -> None:
    """A claim the answer no longer prints cannot be trimmed — and must not be invented.

    Stage 1 removes a unit whose text is gone from the working answer; re-finding it must
    skip it rather than splice a marker somewhere arbitrary.
    """

    working, undecided = _stage_one(
        "Revenue was $198,270 million in FY2023 [3].\nA trend sentence.",
        {2: "A block.", 3: "Another block."},
    )

    # The rejected claim's text is what is no longer present.
    stale = "Revenue was $198,270 million in FY2023 [3]."
    claims = [dataclasses.replace(u, text=stale) for u in undecided]

    assert _locate(working, claims) == []


def test_locate_places_every_unit_in_document_order() -> None:
    """A repeated sentence is located at each place it is printed, never the same one twice."""

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


# --- the q03 marker-placement contract (P4 v1.0.1) --------------------------------


#: The real corpus's income-statement block, the span the q03 table rests on.
INCOME_STATEMENT = (
    "SUMMARY RESULTS OF OPERATIONS\n"
    "(In millions, except percentages and per share amounts) 2023 2022\n"
    "Revenue $ 211,915 $ 198,270 7%\n"
    "Operating income 88,523 83,383 6%\n"
    "Net income 72,361 72,738 (1)%\n"
    "Diluted earnings per share 9.68 9.65 0%"
)

#: q03's answer in the shape P4 v1.0.1 now prescribes: the markers in the rows.
Q03_CITED_IN_ROWS = (
    "| Metric | FY2023 | FY2022 |\n"
    "|---|---|---|\n"
    "| Revenue [2] | $211,915 | $198,270 |\n"
    "| Operating income [2] | $88,523 | $83,383 |"
)

#: The same table with every marker in a source note *below* it — the placement P4's rule
#: 1 prescribed until v1.0.1, and the one that lost q03's two rows.
Q03_CITED_IN_NOTE = (
    "| Metric | FY2023 | FY2022 |\n"
    "|---|---|---|\n"
    "| Revenue | $211,915 | $198,270 |\n"
    "| Operating income | $88,523 | $83,383 |\n"
    "\n"
    "*Figures are from the income statement in [2].*"
)


def test_a_table_with_markers_in_its_rows_survives_the_pre_pass() -> None:
    """The placement P4 v1.0.1 prescribes: a marker in each row's label cell.

    This is the shape the verifier can read, asserted so the two layers stay in contract.
    Both rows carry the block that prints their figures and the header's period labels, so
    the pre-pass settles both and nothing is trimmed.
    """

    judgements = numeric_pre_pass(Q03_CITED_IN_ROWS, {2: INCOME_STATEMENT})
    rows = [judgement for judgement in judgements if judgement.unit.figures]

    assert len(rows) == 2
    assert all(judgement.outcome == OUTCOME_SUPPORTED for judgement in rows)
    # Nothing rejected, so there is nothing for the node to trim: the answer survives
    # verbatim, which is what "the prompt asks for the placement the checker can read" means.
    assert rejected_units(judgements) == []


def test_a_table_with_markers_in_a_note_below_it_loses_every_row() -> None:
    """The shape P4 v1.0.0 prescribed: every marker in the source note under the table.

    ``split_units`` reads a table's markers from its own rows plus the single line that
    introduces it — the note is outside the table block, so the rows inherit nothing and
    each is rejected as ``figure_not_in_span``. That is the whole of q03's defect: the
    answer rendered as two trimmed table rows, and the prompt prescribed it.

    Asserted here as documentation of the *checker's* reading, because the fix was on the
    prompt side (P4 v1.0.1): the contract a prompt makes must be one the verifier can read.
    """

    judgements = numeric_pre_pass(Q03_CITED_IN_NOTE, {2: INCOME_STATEMENT})
    rows = [judgement for judgement in judgements if judgement.unit.figures]

    assert len(rows) == 2
    assert [judgement.outcome for judgement in rows] == [
        OUTCOME_REJECTED,
        OUTCOME_REJECTED,
    ]
    assert all(judgement.reason == "figure_not_in_span" for judgement in rows)
    # Both body rows become the visible marker; the header, the alignment row and the note
    # itself survive, because they are not what was trimmed.
    assert trim_answer(Q03_CITED_IN_NOTE, [row.unit for row in rows]) == Q03_CITED_IN_NOTE.replace(
        "| Revenue | $211,915 | $198,270 |\n| Operating income | $88,523 | $83,383 |",
        "||(unsupported claim removed)||\n||(unsupported claim removed)||",
    )


def test_the_prompt_and_the_checker_agree_on_where_a_table_marker_belongs() -> None:
    """The contract itself: the placement P4 prescribes is the one the pre-pass reads.

    P4 v1.0.1's rule 1 points at the table's own rows (rule 4's placement), so the marker
    the prompt asks for is a marker the verifier can find. This is the assertion that would
    have caught the disagreement at the time — prompt text against symbolic behaviour, in
    one test, hermetic.
    """

    prompt = load_prompt("P4").body
    assert "Put the [n] marker in the table itself" in prompt

    # The prompt's prescribed shape survives; the one it replaced does not.
    figures = [
        judgement
        for judgement in numeric_pre_pass(Q03_CITED_IN_ROWS, {2: INCOME_STATEMENT})
        if judgement.unit.figures
    ]
    assert all(judgement.outcome == OUTCOME_SUPPORTED for judgement in figures)

from __future__ import annotations

import base64
import pickle
import re
from collections.abc import Mapping
from typing import Any

import pytest
from src.chains.state import (
    Citation,
    GraphState,
    NodeEvent,
    QueryPlan,
    ScoredChunk,
    final_envelope,
)
from src.components import debug_panel
from src.store.db import ChatStore, create_schema
from src.store.sessions import SessionStore
from streamlit.testing.v1 import AppTest

_SCOPE_PLAN = QueryPlan(
    kind="out_of_scope",
    sub_queries=[],
    required_sections=(),
    fiscal_period=None,
    missing_aspect=None,
)

REASONED_REFUSAL: dict[str, Any] = {
    "answer_md": "The indexed documents do not cover that period.",
    "citations": [],
    "grounded": True,
    "refusal_reason": "time_period",
    "query_plan": QueryPlan(
        kind="out_of_scope",
        sub_queries=[],
        required_sections=(),
        fiscal_period=None,
        missing_aspect="time_period",
    ),
}

DEGRADED_REFUSAL: dict[str, Any] = {
    **REASONED_REFUSAL,
    "planner_degraded": True,
    "query_plan": _SCOPE_PLAN,
    "answer_md": "The question could not be classified: the planner call failed.",
}

NORMAL_ANSWER: dict[str, Any] = {
    "answer_md": "FY2023 revenue was $211,915 million [1].",
    "citations": [
        Citation(
            ref_index=1,
            doc_id="deadbeef" * 8,
            page=42,
            section_path=("INCOME STATEMENT",),
            snippet="Total revenue was 211,915.",
        )
    ],
    "grounded": True,
    "refusal_reason": None,
    "node_trace": [
        NodeEvent(
            node="analyze_question",
            started_at=1.0,
            duration_ms=12.5,
            tokens_in=100,
            tokens_out=4,
        )
    ],
    "query_plan": QueryPlan(
        kind="simple",
        sub_queries=[],
        required_sections=("INCOME STATEMENT",),
        fiscal_period="FY2023",
        missing_aspect=None,
    ),
    "retrieved": [
        ScoredChunk(
            chunk_id="c1",
            child_text="Total revenue was 211,915.",
            parent_text="Total revenue was 211,915 for the year.",
            page=42,
            section_path=("INCOME STATEMENT",),
            doc_id="deadbeef" * 8,
            dense_score=0.4898,
            sparse_score=8.359,
            fused_rank=0,
        )
    ],
}


def _markup(htmls: list[str]) -> str:

    return "".join(htmls)


def _html_fragments(node: Any) -> list[str]:

    found: list[str] = []

    def walk(element: Any, depth: int = 0) -> None:
        if depth > 32:
            return
        if type(element).__name__ == "UnknownElement":
            if element.proto.__class__.__name__ == "Html":
                found.append(str(element.value))
            return
        children = getattr(element, "children", None)
        if isinstance(children, Mapping):
            for child in children.values():
                walk(child, depth + 1)

    walk(node)
    return found


def _rendered(mode: str, *, state: dict[str, Any] | None = None) -> tuple[str, list[str]]:

    script = _SCRIPT.format(
        state=_payload(dict(state) if state is not None else {}),
        mode=mode,
    )
    app = AppTest.from_string(script, default_timeout=60)
    app.run(timeout=60)
    assert not app.exception, [error.value for error in app.exception]
    htmls = _html_fragments(app.main)
    badges = re.findall(r'class="isip-badge[^"]*">([^<]*)<', _markup(htmls))
    notes = re.findall(r'class="isip-note">([^<]*)<', _markup(htmls))
    return badges, notes


def _payload(state: dict[str, Any]) -> str:

    return base64.b64encode(pickle.dumps(state)).decode("ascii")


_SCRIPT = """
import base64
import pickle

import streamlit as st

from src.chains.state import final_envelope
from src.components.chat import render_answer, render_stored_answer
from src.components.debug_panel import render_debug_panel

envelope = final_envelope(pickle.loads(base64.b64decode("{state}")))


def _stored_message():
    import types

    return types.SimpleNamespace(
        answer_md=envelope.answer_md,
        grounded=envelope.grounded,
        refusal_reason=envelope.refusal_reason,
        citations=(),
        node_trace=None,
    )


MODE = "{mode}"

if MODE == "chat":
    render_answer(envelope)
elif MODE == "panel":
    render_debug_panel(envelope=envelope)
elif MODE == "chat-stored":
    render_stored_answer(_stored_message())
elif MODE == "panel-stored":
    render_debug_panel(message=_stored_message())
"""


class TestEnvelopeCarriesTheMarker:

    def test_propagates_planner_degraded_when_the_state_sets_it(self) -> None:

        envelope = final_envelope(DEGRADED_REFUSAL)

        assert envelope.planner_degraded is True

    def test_leaves_it_false_for_a_reasoned_refusal(self) -> None:

        envelope = final_envelope(REASONED_REFUSAL)

        assert envelope.planner_degraded is False

    def test_the_marker_is_the_only_difference_between_the_two_refusals(self) -> None:

        reasoned = final_envelope(REASONED_REFUSAL)
        degraded = final_envelope(DEGRADED_REFUSAL)

        assert (reasoned.grounded, reasoned.refusal_reason, reasoned.citations) == (
            degraded.grounded,
            degraded.refusal_reason,
            degraded.citations,
        )
        assert reasoned.planner_degraded is not degraded.planner_degraded

    def test_default_false_never_changes_a_normal_answers_encoding(self) -> None:

        plain = final_envelope(NORMAL_ANSWER)
        rebuilt = final_envelope({**NORMAL_ANSWER, "planner_degraded": False})

        assert plain.as_dict() == rebuilt.as_dict()
        assert "planner_degraded" in plain.as_dict()
        assert plain.as_dict()["planner_degraded"] is False

    def test_the_flag_defaults_when_the_state_never_wrote_it(self) -> None:

        assert final_envelope({}).planner_degraded is False
        assert final_envelope(GraphState()).planner_degraded is False


class TestDegradedRefusalIsDistinguishable:

    @pytest.mark.parametrize("mode", ["chat", "panel"])
    def test_degraded_renders_the_degraded_marker(self, mode: str) -> None:

        badges, notes = _rendered(mode, state=DEGRADED_REFUSAL)

        assert debug_panel.PLANNER_DEGRADED_LABEL in badges
        assert any(debug_panel.PLANNER_DEGRADED_COPY in note for note in notes)

    def test_the_transcript_swaps_the_refusal_badge_for_the_degraded_one(self) -> None:

        healthy_badges, _ = _rendered("chat", state=REASONED_REFUSAL)
        degraded_badges, _ = _rendered("chat", state=DEGRADED_REFUSAL)

        assert "Refusal" in healthy_badges
        assert "Refusal" not in degraded_badges

    @pytest.mark.parametrize("mode", ["chat", "panel"])
    def test_a_reasoned_refusal_never_renders_the_degraded_marker(self, mode: str) -> None:

        badges, notes = _rendered(mode, state=REASONED_REFUSAL)

        assert debug_panel.PLANNER_DEGRADED_LABEL not in badges
        assert not any(debug_panel.PLANNER_DEGRADED_COPY in note for note in notes)

    def test_the_two_refusals_differ_in_the_exact_marker_the_ui_reads(self) -> None:

        script = _SCRIPT.format(state=_payload(REASONED_REFUSAL), mode="panel")
        app = AppTest.from_string(script, default_timeout=60)
        app.run(timeout=60)
        reasoned = _markup(_html_fragments(app.main))

        script = _SCRIPT.format(state=_payload(DEGRADED_REFUSAL), mode="panel")
        app = AppTest.from_string(script, default_timeout=60)
        app.run(timeout=60)
        degraded = _markup(_html_fragments(app.main))

        assert reasoned != degraded
        assert debug_panel.PLANNER_DEGRADED_LABEL in degraded
        assert debug_panel.PLANNER_DEGRADED_LABEL not in reasoned

    def test_the_panel_records_the_reasoned_refusal_as_a_reason_not_a_failure(self) -> None:

        script = _SCRIPT.format(state=_payload(REASONED_REFUSAL), mode="panel")
        app = AppTest.from_string(script, default_timeout=60)
        app.run(timeout=60)
        assert not app.exception, [error.value for error in app.exception]
        markup = _markup(_html_fragments(app.main))

        assert 'style="margin:0">refusal</span>' in markup
        assert ">time_period<" in markup

    def test_a_normal_answer_adds_no_marker_at_all(self) -> None:

        badges, notes = _rendered("chat", state=NORMAL_ANSWER)

        assert debug_panel.PLANNER_DEGRADED_LABEL not in badges
        assert not any(debug_panel.PLANNER_DEGRADED_COPY in note for note in notes)

    def test_the_panel_still_renders_every_section_for_a_degraded_turn(self) -> None:

        script = _SCRIPT.format(state=_payload(NORMAL_ANSWER), mode="panel")
        app = AppTest.from_string(script, default_timeout=60)
        app.run(timeout=60)
        assert not app.exception, [error.value for error in app.exception]
        markup = _markup(_html_fragments(app.main))

        for title in ("Plan", "Path", "Retrieval", "Grounding"):
            assert f'class="isip-label">{title}<' in markup
        assert debug_panel.PLANNER_DEGRADED_LABEL not in markup


class TestTheWordingCannotDrift:

    def test_chat_imports_the_panels_sentence_rather_than_rewriting_it(self) -> None:

        from src.components import chat

        assert chat.PLANNER_DEGRADED_COPY is debug_panel.PLANNER_DEGRADED_COPY
        assert chat.PLANNER_DEGRADED_LABEL == debug_panel.PLANNER_DEGRADED_LABEL

    def test_both_surfaces_emit_the_identical_sentence(self) -> None:

        _, chat_notes = _rendered("chat", state=DEGRADED_REFUSAL)
        _, panel_notes = _rendered("panel", state=DEGRADED_REFUSAL)

        chat_sentence = [n for n in chat_notes if debug_panel.PLANNER_DEGRADED_COPY in n]
        panel_sentence = [n for n in panel_notes if debug_panel.PLANNER_DEGRADED_COPY in n]
        assert chat_sentence == panel_sentence != []


class TestTheStoreRoundTrip:

    def test_a_stored_answer_reports_not_recorded_rather_than_not_degraded(self) -> None:

        script = _SCRIPT.format(state=_payload(REASONED_REFUSAL), mode="panel-stored")
        app = AppTest.from_string(script, default_timeout=60)
        app.run(timeout=60)
        assert not app.exception, [error.value for error in app.exception]
        markup = _markup(_html_fragments(app.main))

        assert debug_panel.PLANNER_DEGRADED_LABEL not in markup
        assert debug_panel.PLANNER_DEGRADED_COPY not in markup
        assert debug_panel.PLANNER_DEGRADED_NOT_RECORDED_COPY in markup

    def test_a_stored_answer_in_the_transcript_is_not_marked_degraded(self) -> None:

        badges, notes = _rendered("chat-stored", state=REASONED_REFUSAL)

        assert "Refusal" in badges
        assert debug_panel.PLANNER_DEGRADED_LABEL not in badges
        assert any(debug_panel.PLANNER_DEGRADED_COPY not in note and note for note in notes)

    def test_the_store_has_no_planner_degraded_column_yet(self, tmp_path: Any) -> None:

        db = tmp_path / "gap.sqlite3"
        store = ChatStore(str(db))
        create_schema(store.connection)
        row = store.connection.execute("PRAGMA table_info(messages)").fetchall()
        store.close()

        columns = {name for name, *_ in row}
        assert "planner_degraded" not in columns

    def test_recording_an_envelope_keeps_every_persisted_field_intact(self, tmp_path: Any) -> None:

        db = tmp_path / "roundtrip.sqlite3"
        store = ChatStore(str(db))
        create_schema(store.connection)
        with SessionStore(store) as sessions:
            sessions.create_session("s1", session_id="s1")
            sessions.append_turns(
                "s1", [{"role": "user", "content": "What was revenue in FY2019?"}]
            )
            envelope = final_envelope(DEGRADED_REFUSAL)
            sessions.record_message(
                "s1",
                "What was revenue in FY2019?",
                envelope.answer_md,
                grounded=envelope.grounded,
                refusal_reason=envelope.refusal_reason,
                node_trace=None,
                citations=[],
            )
            stored = sessions.list_messages("s1")[0]

        assert stored.grounded is True
        assert stored.refusal_reason == "time_period"
        assert stored.answer_md == envelope.answer_md
        assert not hasattr(stored, "planner_degraded")
        assert envelope.planner_degraded is True

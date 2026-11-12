from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from src.store.db import (
    ChatStore,
    SessionNotFoundError,
    StoreError,
    connect,
    create_schema,
    default_db_path,
    dumps_json,
)
from src.store.sessions import (
    HISTORY_WINDOW,
    ROLE_ASSISTANT,
    ROLE_USER,
    SessionStore,
)

EXPECTED_TABLES: tuple[str, ...] = (
    "sessions",
    "turns",
    "messages",
    "citations",
    "documents",
    "events",
)

EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "sessions": ("id", "title", "created_at", "updated_at", "model"),
    "turns": (
        "id",
        "session_id",
        "seq",
        "role",
        "content",
        "created_at",
        "tokens_in",
        "tokens_out",
        "latency_ms",
    ),
    "messages": (
        "id",
        "turn_id",
        "question",
        "answer_md",
        "grounded",
        "refusal_reason",
        "node_trace_json",
    ),
    "citations": (
        "id",
        "message_id",
        "ref_index",
        "doc_id",
        "page",
        "section_path",
        "snippet",
    ),
    "documents": ("doc_id", "filename", "sha256", "page_count", "indexed_at"),
    "events": ("id", "session_id", "turn_id", "ts", "event", "payload_json"),
}


QUESTION = "What was Microsoft's revenue in FY2023?"
ANSWER_MD = "Revenue was $211,915 million in FY2023 [1], up from $198,270 million [2]."

NODE_TRACE: dict[str, Any] = {
    "nodes": [
        {"node": "contextualize_question", "ms": 812, "tokens_in": 41, "tokens_out": 7},
        {"node": "synthesize_answer", "ms": 2411, "tokens_in": 903, "tokens_out": 212},
    ],
    "grounded": True,
    "retry_count": 0,
}

CITATIONS: tuple[dict[str, Any], ...] = (
    {
        "ref_index": 1,
        "doc_id": "sha256-msft-2023",
        "page": 64,
        "section_path": ["Consolidated Statements of Income", "Revenue"],
        "snippet": "Revenue was $211,915 million for the year ended June 30, 2023.",
    },
    {
        "ref_index": 2,
        "doc_id": "sha256-msft-2023",
        "page": 63,
        "section_path": ["Consolidated Statements of Income", "Prior year"],
        "snippet": "Revenue was $198,270 million in FY2022 — cloud and Office both grew.",
    },
)

_COUNT_SQL: dict[str, str] = {
    "turns": "SELECT COUNT(*) AS n FROM turns WHERE session_id = ?",
    "messages": (
        "SELECT COUNT(*) AS n FROM messages "
        "WHERE turn_id IN (SELECT id FROM turns WHERE session_id = ?)"
    ),
    "citations": (
        "SELECT COUNT(*) AS n FROM citations WHERE message_id IN "
        "(SELECT id FROM messages WHERE turn_id IN "
        "(SELECT id FROM turns WHERE session_id = ?))"
    ),
    "events": "SELECT COUNT(*) AS n FROM events WHERE session_id = ?",
}

UNKNOWN_ID_ACTIONS: tuple[str, ...] = (
    "append_turn",
    "append_turns",
    "delete_session",
    "get_history",
    "get_session",
    "history_as_dialogue",
    "last_n_turns",
    "record_message",
    "rename_session",
    "switch_session",
)


def _parse_iso(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None, "a store timestamp must be timezone-explicit"
    return parsed


def _turn_entry(index: int, role: str = ROLE_USER) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": role, "content": f"{role} turn {index}"}
    if index % 2 == 0:
        entry["tokens_in"] = index
        entry["tokens_out"] = index
        entry["latency_ms"] = index * 10
    return entry


def _session_counts(store: ChatStore, session_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table, sql in _COUNT_SQL.items():
        row = store.query_one(sql, (session_id,))
        assert row is not None
        counts[table] = int(row["n"])
    return counts


def _unknown_id_actions(sessions: SessionStore, session_id: str) -> dict[str, Callable[[], Any]]:
    return {
        "get_session": lambda: sessions.get_session(session_id),
        "rename_session": lambda: sessions.rename_session(session_id, "renamed"),
        "switch_session": lambda: sessions.switch_session(session_id),
        "delete_session": lambda: sessions.delete_session(session_id),
        "append_turn": lambda: sessions.append_turn(session_id, ROLE_USER, QUESTION),
        "append_turns": lambda: sessions.append_turns(session_id, [_turn_entry(1)]),
        "record_message": lambda: sessions.record_message(session_id, QUESTION, ANSWER_MD),
        "get_history": lambda: sessions.get_history(session_id),
        "last_n_turns": lambda: sessions.last_n_turns(session_id),
        "history_as_dialogue": lambda: sessions.history_as_dialogue(session_id),
    }


@pytest.fixture
def state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "state"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DATA_DIR", str(root))
    return root


@pytest.fixture
def db_path(state_root: Path) -> Path:
    return state_root / "chat.sqlite3"


@pytest.fixture
def store(db_path: Path) -> Iterator[ChatStore]:
    with ChatStore(path=str(db_path)) as opened:
        yield opened


@pytest.fixture
def sessions(store: ChatStore) -> SessionStore:
    return SessionStore(store)


def test_default_db_path_follows_the_data_dir_override(state_root: Path, db_path: Path) -> None:
    assert default_db_path() == db_path


def test_a_fresh_database_has_all_six_documented_tables(db_path: Path, store: ChatStore) -> None:
    assert db_path.exists(), "the fixture opened a file at the temporary path"
    created = store.query_all("SELECT name FROM sqlite_master WHERE type = 'table'")
    assert {row["name"] for row in created} == set(EXPECTED_TABLES)


def test_each_table_carries_the_documented_columns_in_order(store: ChatStore) -> None:
    for table, expected in EXPECTED_COLUMNS.items():
        rows = store.query_all(f"PRAGMA table_info({table})")
        assert [row["name"] for row in rows] == list(expected), f"{table} drifted from the doc"


def test_the_documented_lookup_indexes_exist(store: ChatStore) -> None:
    created = store.query_all("SELECT name FROM sqlite_master WHERE type = 'index'")
    names = {row["name"] for row in created}
    assert {
        "idx_turns_session_seq",
        "idx_messages_turn",
        "idx_citations_message",
        "idx_events_session_ts",
    } <= names


def test_create_schema_is_idempotent(store: ChatStore) -> None:
    before = store.query_all("SELECT name FROM sqlite_master ORDER BY name")
    create_schema(store.connection)
    assert store.query_all("SELECT name FROM sqlite_master ORDER BY name") == before


def test_connect_puts_the_connection_in_wal_mode(db_path: Path, store: ChatStore) -> None:
    connection = connect(db_path, create=False)
    try:
        for opened in (store.connection, connection):
            journal = opened.execute("PRAGMA journal_mode").fetchone()[0]
            assert journal.lower() == "wal", f"journal_mode reported {journal!r}"
    finally:
        connection.close()


def test_connect_turns_foreign_keys_on(store: ChatStore) -> None:
    assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_the_stores_connection_refuses_an_orphan_turn(store: ChatStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.execute(
            "INSERT INTO turns(session_id, seq, role, content) VALUES (?, ?, ?, ?)",
            ("no-such-session", 1, ROLE_USER, QUESTION),
        )


def test_connect_refuses_to_invent_a_missing_database(tmp_path: Path) -> None:
    missing = tmp_path / "absent-state-root" / "chat.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        connect(missing, create=False)
    assert not missing.exists()
    assert not missing.parent.exists()


def test_the_creating_connection_builds_the_state_root(tmp_path: Path) -> None:
    fresh = tmp_path / "state-root" / "chat.sqlite3"
    assert not fresh.parent.exists(), "the tree must be absent before the store makes it"
    with ChatStore(path=str(fresh)) as created:
        assert created.query_all("SELECT name FROM sqlite_master WHERE type='table'")
    assert fresh.exists()
    assert fresh.parent == tmp_path / "state-root"


def test_a_recorded_exchange_survives_a_fresh_connection_byte_identical(
    sessions: SessionStore, db_path: Path
) -> None:
    session = sessions.create_session(title="FY2023 deep dive", model="gpt-4o-mini")
    sessions.append_turn(session.id, ROLE_USER, QUESTION, tokens_in=12, tokens_out=0, latency_ms=34)
    recorded = sessions.record_message(
        session.id,
        QUESTION,
        ANSWER_MD,
        grounded=True,
        node_trace=NODE_TRACE,
        citations=CITATIONS,
    )
    history_before = sessions.get_history(session.id)
    message_before = sessions.get_message(recorded.id)
    assert message_before is not None

    sessions.close()

    with SessionStore(path=str(db_path)) as reopened:
        assert reopened.get_session(session.id) == history_before.session
        reloaded = reopened.get_history(session.id)
        assert reloaded == history_before
        reloaded_message = reopened.get_message(recorded.id)
        assert reloaded_message is not None
        assert reloaded_message == message_before

        assert [turn.role for turn in reloaded.turns] == [ROLE_USER, ROLE_ASSISTANT]
        assert [turn.seq for turn in reloaded.turns] == [1, 2]
        assert reloaded.turns[0].tokens_in == 12
        assert reloaded.turns[0].latency_ms == 34
        assert reloaded.last_user_turn == reloaded.turns[0]

        assert [citation.ref_index for citation in reloaded_message.citations] == [1, 2]
        for citation, written in zip(reloaded_message.citations, CITATIONS, strict=True):
            row = reopened.store.query_one("SELECT * FROM citations WHERE id = ?", (citation.id,))
            assert row is not None
            assert row["doc_id"] == written["doc_id"]
            assert row["page"] == written["page"]
            assert row["snippet"] == written["snippet"]
            assert row["ref_index"] == written["ref_index"]
            assert isinstance(row["section_path"], str)
            assert json.loads(row["section_path"]) == written["section_path"]
            assert list(citation.section_path) == written["section_path"]
            assert row["section_path"] == dumps_json(list(written["section_path"]))

        stored = reopened.store.query_one(
            "SELECT node_trace_json FROM messages WHERE id = ?", (recorded.id,)
        )
        assert stored is not None
        assert stored["node_trace_json"] == dumps_json(NODE_TRACE)
        assert reloaded_message.node_trace == NODE_TRACE


def test_rename_and_switch_leave_the_session_reachable(
    sessions: SessionStore, store: ChatStore
) -> None:
    session = sessions.create_session(title="Working title", model="gpt-4o-mini")
    renamed = sessions.rename_session(session.id, "FY2023 revenue & margin")
    assert renamed.id == session.id
    assert renamed.title == "FY2023 revenue & margin"
    assert renamed.model == "gpt-4o-mini"
    assert sessions.find_session(session.id) == renamed

    switched = sessions.switch_session(session.id)
    assert switched == renamed
    assert sessions.get_history(session.id).session.title == "FY2023 revenue & margin"

    row = store.query_one("SELECT title FROM sessions WHERE id = ?", (session.id,))
    assert row is not None
    assert row["title"] == "FY2023 revenue & margin"


def test_create_session_rejects_a_duplicate_id_rather_than_silently_duplicating(
    sessions: SessionStore,
) -> None:
    sessions.create_session("first", "m", session_id="fixed-id")
    with pytest.raises(sqlite3.IntegrityError):
        sessions.create_session("second", "m", session_id="fixed-id")
    assert [s.id for s in sessions.list_sessions()] == ["fixed-id"]


def test_citations_with_a_missing_page_keep_null_and_an_empty_path(
    sessions: SessionStore, store: ChatStore
) -> None:
    session = sessions.create_session("edge", "m")
    message = sessions.record_message(
        session.id,
        QUESTION,
        ANSWER_MD,
        grounded=False,
        refusal_reason=None,
        citations=[{"ref_index": 1, "doc_id": "d", "page": None, "section_path": []}],
    )
    row = store.query_one("SELECT * FROM citations WHERE message_id = ?", (message.id,))
    assert row is not None
    assert row["page"] is None
    assert row["section_path"] == "[]"
    assert message.citations[0].page is None
    assert message.citations[0].section_path == ()
    assert message.grounded is False


def test_delete_session_cascades_turns_messages_and_citations(
    sessions: SessionStore, store: ChatStore
) -> None:
    deleted = sessions.create_session("to delete", "m")
    kept = sessions.create_session("to keep", "m")
    for session_id in (deleted.id, kept.id):
        sessions.append_turn(session_id, ROLE_USER, QUESTION)
        sessions.record_message(session_id, QUESTION, ANSWER_MD, citations=CITATIONS)
    store.record_event(
        "answer.recorded", session_id=deleted.id, payload={"tokens_in": 12, "tokens_out": 0}
    )

    assert _session_counts(store, deleted.id) == {
        "turns": 2,
        "messages": 1,
        "citations": 2,
        "events": 1,
    }

    sessions.delete_session(deleted.id)

    assert _session_counts(store, deleted.id) == {
        "turns": 0,
        "messages": 0,
        "citations": 0,
        "events": 1,
    }
    assert store.query_all("SELECT * FROM turns WHERE session_id = ?", (deleted.id,)) == []
    assert (
        store.query_all(
            "SELECT * FROM messages WHERE turn_id IN "
            "(SELECT id FROM turns WHERE session_id = ?)",
            (deleted.id,),
        )
        == []
    )
    assert (
        store.query_all(
            "SELECT * FROM citations WHERE message_id IN "
            "(SELECT id FROM messages WHERE turn_id IN "
            "(SELECT id FROM turns WHERE session_id = ?))",
            (deleted.id,),
        )
        == []
    )

    assert _session_counts(store, kept.id) == {
        "turns": 2,
        "messages": 1,
        "citations": 2,
        "events": 0,
    }
    assert sessions.find_session(deleted.id) is None

    surviving = store.list_events(session_id=deleted.id)
    assert len(surviving) == 1
    assert surviving[0].session_id == deleted.id
    assert surviving[0].payload == {"tokens_in": 12, "tokens_out": 0}


def test_delete_session_on_an_unknown_id_removes_nothing(sessions: SessionStore) -> None:
    session = sessions.create_session("keep", "m")
    sessions.append_turn(session.id, ROLE_USER, QUESTION)
    with pytest.raises(SessionNotFoundError):
        sessions.delete_session("no-such-session")
    assert sessions.get_history(session.id).turns


def test_last_n_turns_returns_exactly_the_window_newest_last(sessions: SessionStore) -> None:
    session = sessions.create_session("long conversation", "m")
    everything = sessions.append_turns(session.id, [_turn_entry(i) for i in range(13)])
    assert [turn.seq for turn in everything] == list(range(1, 14))

    window = sessions.last_n_turns(session.id)
    assert len(window) == HISTORY_WINDOW == 6
    assert [turn.seq for turn in window] == list(range(8, 14))
    assert [turn.content for turn in window] == [
        f"{ROLE_USER} turn {index}" for index in range(7, 13)
    ]
    assert window[-1].seq == max(turn.seq for turn in everything)


def test_the_window_caps_history_used_once_as_a_dialogue(sessions: SessionStore) -> None:
    session = sessions.create_session("dialogue", "m")
    for index in range(5):
        sessions.append_turns(session.id, [_turn_entry(index), _turn_entry(index, ROLE_ASSISTANT)])
    sessions.append_turn(session.id, ROLE_USER, "and its margin trend?")
    sessions.append_turn(session.id, ROLE_ASSISTANT, "Gross margin was 69.8% [1].")

    dialogue = sessions.history_as_dialogue(session.id)
    assert len(dialogue) == HISTORY_WINDOW
    assert dialogue[0] == {"role": ROLE_USER, "content": f"{ROLE_USER} turn 3"}
    assert dialogue[-1] == {"role": ROLE_ASSISTANT, "content": "Gross margin was 69.8% [1]."}
    assert all(set(entry) == {"role", "content"} for entry in dialogue)


def test_a_short_session_returns_every_turn_and_an_empty_one_returns_none(
    sessions: SessionStore,
) -> None:
    empty = sessions.create_session("empty", "m")
    assert sessions.last_n_turns(empty.id) == []
    assert sessions.history_as_dialogue(empty.id) == []

    short = sessions.create_session("short", "m")
    sessions.append_turns(short.id, [_turn_entry(index) for index in range(HISTORY_WINDOW - 1)])
    window = sessions.last_n_turns(short.id)
    assert len(window) == HISTORY_WINDOW - 1
    assert [turn.seq for turn in window] == list(range(1, HISTORY_WINDOW))


def test_a_window_size_below_one_is_a_caller_bug(sessions: SessionStore) -> None:
    session = sessions.create_session("window", "m")
    sessions.append_turn(session.id, ROLE_USER, QUESTION)
    with pytest.raises(ValueError):
        sessions.last_n_turns(session.id, 0)


def test_seq_is_per_session_one_based_and_gap_free(sessions: SessionStore) -> None:
    first = sessions.create_session("first", "m")
    second = sessions.create_session("second", "m")

    batch = sessions.append_turns(first.id, [_turn_entry(i) for i in range(5)])
    assert [turn.seq for turn in batch] == [1, 2, 3, 4, 5]
    assert [turn.session_id for turn in batch] == [first.id] * 5
    assert len({turn.id for turn in batch}) == 5

    singles = [sessions.append_turn(first.id, ROLE_USER, f"single {i}") for i in range(3)]
    assert [turn.seq for turn in singles] == [6, 7, 8]

    answer = sessions.record_message(first.id, QUESTION, ANSWER_MD)
    assert answer.turn is not None
    assert answer.turn.seq == 9, "the assistant turn is numbered by the same counter"

    assert sessions.append_turn(second.id, ROLE_USER, QUESTION).seq == 1
    assert sessions.append_turn(second.id, ROLE_ASSISTANT, ANSWER_MD).seq == 2
    assert [turn.seq for turn in sessions.get_history(first.id).turns] == list(range(1, 10))


def test_a_caller_supplied_seq_is_ignored_by_the_bulk_writer(sessions: SessionStore) -> None:
    session = sessions.create_session("seq", "m")
    entries = [{"role": ROLE_USER, "content": "q", "seq": 42, "id": 99}]
    turns = sessions.append_turns(session.id, entries)
    assert [turn.seq for turn in turns] == [1]


def test_a_bulk_append_that_fails_lands_nothing(sessions: SessionStore, store: ChatStore) -> None:
    session = sessions.create_session("atomic", "m")
    sessions.append_turn(session.id, ROLE_USER, QUESTION)
    with pytest.raises(ValueError):
        sessions.append_turns(
            session.id,
            [_turn_entry(1, ROLE_USER), {"role": "system", "content": "not a chat role"}],
        )
    assert _session_counts(store, session.id)["turns"] == 1
    assert [turn.seq for turn in sessions.get_history(session.id).turns] == [1]


@pytest.mark.parametrize("action", UNKNOWN_ID_ACTIONS)
def test_an_unknown_session_id_raises_rather_than_no_op(
    sessions: SessionStore, action: str
) -> None:
    actions = _unknown_id_actions(sessions, "00000000-0000-0000-0000-000000000000")
    assert set(actions) == set(UNKNOWN_ID_ACTIONS), "the parametrized list drifted"
    with pytest.raises(SessionNotFoundError):
        actions[action]()
    assert sessions.list_sessions() == [], "a failed mutator must not have written anything"


def test_the_lenient_read_still_answers_none(sessions: SessionStore) -> None:
    assert sessions.find_session("no-such-session") is None


def test_session_not_found_is_a_store_error() -> None:
    assert issubclass(SessionNotFoundError, StoreError)


def test_writes_through_one_connection_are_visible_to_a_fresh_one(db_path: Path) -> None:
    with SessionStore(path=str(db_path)) as writer:
        session = writer.create_session("shared file", "m")
        writer.append_turn(session.id, ROLE_USER, QUESTION)
        writer.record_message(session.id, QUESTION, ANSWER_MD, citations=CITATIONS)

    reader = connect(db_path, create=False)
    try:
        rows = reader.execute(
            "SELECT role, content, seq FROM turns WHERE session_id = ? ORDER BY seq",
            (session.id,),
        ).fetchall()
    finally:
        reader.close()
    assert [row["role"] for row in rows] == [ROLE_USER, ROLE_ASSISTANT]
    assert [row["seq"] for row in rows] == [1, 2]
    assert rows[1]["content"] == ANSWER_MD

    with SessionStore(path=str(db_path)) as reread:
        history = reread.get_history(session.id)
        assert len(history.turns) == 2
        assert len(history.messages) == 1
        assert len(history.messages[0].citations) == 2


def test_a_writer_mid_transaction_is_invisible_to_and_never_blocks_a_reader(
    db_path: Path,
) -> None:
    writer_store = ChatStore(path=str(db_path))
    writer = SessionStore(writer_store)
    session = writer.create_session("wal", "m")
    reader = connect(db_path, create=False)
    try:
        writer_store.execute("BEGIN IMMEDIATE")
        writer_store.execute(
            """
            INSERT INTO turns(session_id, seq, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session.id, 1, ROLE_USER, QUESTION, "2026-10-13T00:00:00.000Z"),
        )
        during = reader.execute(
            "SELECT COUNT(*) AS n FROM turns WHERE session_id = ?", (session.id,)
        ).fetchone()[0]
        writer_store.execute("COMMIT")
        after = reader.execute(
            "SELECT COUNT(*) AS n FROM turns WHERE session_id = ?", (session.id,)
        ).fetchone()[0]
    finally:
        reader.close()
        writer_store.close()
    assert during == 0, "an uncommitted write leaked to a second connection"
    assert after == 1, "the committed write never reached a second connection"


def test_record_event_and_list_events_round_trip_the_payload(store: ChatStore) -> None:
    payload = {
        "tokens_in": 12,
        "tokens_out": 212,
        "nodes": ["contextualize_question", "synthesize_answer"],
        "note": "café — naïve",
        "ok": True,
        "nothing": None,
    }
    written = store.record_event("answer.recorded", session_id="s1", turn_id=3, payload=payload)
    assert written.id is not None
    assert written.session_id == "s1"
    assert written.turn_id == 3
    assert written.payload == payload
    _parse_iso(written.ts)

    store.record_event("answer.recorded", session_id="s1", payload={"tokens_in": 1})
    store.record_event("session.created", session_id="other", payload=None)

    scoped = store.list_events(session_id="s1")
    assert [event.event for event in scoped] == ["answer.recorded", "answer.recorded"]
    assert scoped[0].payload == {"tokens_in": 1}, "the stream reads newest first"
    assert scoped[1].payload == payload
    assert scoped[0].id > scoped[1].id, "newest first means the later id comes first"
    _parse_iso(scoped[1].ts)
    assert store.list_events(session_id="s1", limit=1) == [scoped[0]]
    assert [event.session_id for event in store.list_events()] == ["other", "s1", "s1"]
    assert store.list_events(session_id="unknown") == []


def test_a_payload_free_event_stores_no_json(store: ChatStore) -> None:
    event = store.record_event("ping")
    assert event.payload is None
    row = store.query_one("SELECT payload_json FROM events WHERE id = ?", (event.id,))
    assert row is not None
    assert row["payload_json"] is None


def test_an_unserializable_payload_raises_rather_than_dropping_the_event(
    store: ChatStore,
) -> None:
    with pytest.raises(ValueError):
        store.record_event("answer.recorded", payload={"unserializable": object()})
    assert store.query_all("SELECT * FROM events") == []


def test_events_limit_must_be_a_positive_integer(store: ChatStore) -> None:
    store.record_event("ping")
    with pytest.raises(ValueError):
        store.list_events(limit=0)
    rejected: Any = True
    with pytest.raises(ValueError):
        store.list_events(limit=rejected)


def test_documents_are_upserted_by_content_hash(store: ChatStore) -> None:
    first = store.upsert_document("sha256-abc", "msft-2023.pdf", "sha256-abc", 42)
    assert first.doc_id == "sha256-abc"
    assert first.page_count == 42
    assert store.get_document("missing") is None

    again = store.upsert_document("sha256-abc", "msft-2023.pdf", "different-hash", 43)
    assert again.page_count == 43
    registered = store.list_documents()
    assert [document.doc_id for document in registered] == ["sha256-abc"]
    reloaded = store.get_document("sha256-abc")
    assert reloaded is not None
    assert reloaded.page_count == 43
    assert reloaded.sha256 == "different-hash"
    _parse_iso(reloaded.indexed_at)


def test_a_closed_store_releases_its_connection(db_path: Path) -> None:
    closed = ChatStore(path=str(db_path))
    closed.close()
    with pytest.raises(sqlite3.ProgrammingError):
        closed.query_all("SELECT 1")

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.store.db import (
    ChatStore,
    SessionNotFoundError,
    as_flag,
    as_int,
    dumps_json,
    loads_json,
    utc_now_iso,
)

HISTORY_WINDOW = 6

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


class StoreLookupError(RuntimeError):
    pass


@dataclass(frozen=True)
class Session:
    id: str
    title: str | None
    created_at: str
    updated_at: str
    model: str | None
    turn_count: int = 0

    @classmethod
    def from_row(cls, row: Any) -> Session:
        keys = row.keys()
        turn_count = 0
        count = row["turn_count"] if "turn_count" in keys else None
        if count is not None:
            turn_count = as_int(count, "sessions.turn_count") or 0
        return cls(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            model=row["model"],
            turn_count=turn_count,
        )


@dataclass(frozen=True)
class Turn:
    id: int
    session_id: str
    seq: int
    role: str
    content: str
    created_at: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None

    @classmethod
    def from_row(cls, row: Any) -> Turn:
        return cls(
            id=as_int(row["id"], "turns.id") or 0,
            session_id=row["session_id"],
            seq=as_int(row["seq"], "turns.seq") or 0,
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            tokens_in=as_int(row["tokens_in"], "turns.tokens_in"),
            tokens_out=as_int(row["tokens_out"], "turns.tokens_out"),
            latency_ms=as_int(row["latency_ms"], "turns.latency_ms"),
        )

    @property
    def is_user(self) -> bool:
        return self.role == ROLE_USER


@dataclass(frozen=True)
class Citation:

    id: int | None
    ref_index: int
    doc_id: str
    page: int | None
    section_path: tuple[str, ...] = ()
    snippet: str = ""
    message_id: int | None = None

    @classmethod
    def from_row(cls, row: Any) -> Citation:
        return cls(
            id=as_int(row["id"], "citations.id"),
            message_id=as_int(row["message_id"], "citations.message_id"),
            ref_index=as_int(row["ref_index"], "citations.ref_index") or 0,
            doc_id=row["doc_id"],
            page=as_int(row["page"], "citations.page"),
            section_path=_section_path(row["section_path"]),
            snippet=row["snippet"],
        )


@dataclass(frozen=True)
class Message:

    id: int
    question: str
    answer_md: str
    grounded: bool | None
    refusal_reason: str | None
    node_trace: Any | None = None
    turn: Turn | None = None
    citations: tuple[Citation, ...] = ()

    @classmethod
    def from_row(
        cls,
        row: Any,
        turn: Turn | None = None,
        citations: Sequence[Citation] = (),
    ) -> Message:
        return cls(
            id=as_int(row["id"], "messages.id") or 0,
            question=row["question"],
            answer_md=row["answer_md"],
            grounded=as_flag(row["grounded"], "messages.grounded"),
            refusal_reason=row["refusal_reason"],
            node_trace=loads_json(row["node_trace_json"]),
            turn=turn,
            citations=tuple(citations),
        )


@dataclass(frozen=True)
class SessionHistory:

    session: Session
    turns: tuple[Turn, ...] = ()
    messages: tuple[Message, ...] = ()

    @property
    def last_user_turn(self) -> Turn | None:
        for turn in reversed(self.turns):
            if turn.is_user:
                return turn
        return None


class SessionStore:

    def __init__(self, store: ChatStore | None = None, path: str | None = None) -> None:
        self._store = store if store is not None else ChatStore(path)

    @property
    def store(self) -> ChatStore:
        return self._store

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def create_session(
        self, title: str | None = None, model: str | None = None, session_id: str | None = None
    ) -> Session:
        identifier = session_id if session_id is not None else _new_session_id()
        now = utc_now_iso()
        with self._store.transaction():
            self._store.execute(
                """
                INSERT INTO sessions(id, title, created_at, updated_at, model)
                VALUES (?, ?, ?, ?, ?)
                """,
                (identifier, title, now, now, model),
            )
        return self.get_session(identifier)

    def list_sessions(self) -> list[Session]:
        rows = self._store.query_all("""
            SELECT s.*, (
                SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id
            ) AS turn_count
            FROM sessions s
            ORDER BY s.updated_at DESC, s.created_at DESC, s.id ASC
            """)
        return [Session.from_row(row) for row in rows]

    def get_session(self, session_id: str) -> Session:
        return self._require_session(session_id)

    def find_session(self, session_id: str) -> Session | None:
        row = self._store.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return None if row is None else Session.from_row(row)

    def rename_session(self, session_id: str, title: str) -> Session:
        self._require_session(session_id)
        with self._store.transaction():
            self._store.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, utc_now_iso(), session_id),
            )
        return self._require_session(session_id)

    def switch_session(self, session_id: str) -> Session:
        return self._require_session(session_id)

    def delete_session(self, session_id: str) -> None:
        self._require_session(session_id)
        with self._store.transaction():
            self._store.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
    ) -> Turn:
        role = _validate_role(role)
        self._require_session(session_id)
        with self._store.transaction():
            seq = self._next_seq(session_id)
            cursor = self._store.execute(
                """
                INSERT INTO turns(session_id, seq, role, content, created_at,
                                  tokens_in, tokens_out, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, seq, role, content, utc_now_iso(), tokens_in, tokens_out, latency_ms),
            )
            self._store.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (utc_now_iso(), session_id)
            )
            turn_id = cursor.lastrowid
        row = self._store.query_one("SELECT * FROM turns WHERE id = ?", (turn_id,))
        assert row is not None
        return Turn.from_row(row)

    def append_turns(self, session_id: str, turns: Iterable[Mapping[str, Any]]) -> list[Turn]:
        entries = list(turns)
        self._require_session(session_id)
        if not entries:
            return []
        created: list[Turn] = []
        now = utc_now_iso()
        with self._store.transaction():
            for entry in entries:
                role = _validate_role(entry["role"])
                content = entry["content"]
                seq = self._next_seq(session_id)
                cursor = self._store.execute(
                    """
                    INSERT INTO turns(session_id, seq, role, content, created_at,
                                      tokens_in, tokens_out, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        seq,
                        role,
                        content,
                        now,
                        _optional_int(entry.get("tokens_in")),
                        _optional_int(entry.get("tokens_out")),
                        _optional_int(entry.get("latency_ms")),
                    ),
                )
                created.append(
                    Turn(
                        id=as_int(cursor.lastrowid, "turns.id") or 0,
                        session_id=session_id,
                        seq=seq,
                        role=role,
                        content=content,
                        created_at=now,
                        tokens_in=_optional_int(entry.get("tokens_in")),
                        tokens_out=_optional_int(entry.get("tokens_out")),
                        latency_ms=_optional_int(entry.get("latency_ms")),
                    )
                )
            self._store.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
        return created

    def record_message(
        self,
        session_id: str,
        question: str,
        answer_md: str,
        *,
        grounded: bool | None = None,
        refusal_reason: str | None = None,
        node_trace: Any | None = None,
        citations: Sequence[Mapping[str, Any]] | None = None,
    ) -> Message:
        self._require_session(session_id)
        trace_json = None if node_trace is None else dumps_json(node_trace)
        with self._store.transaction():
            seq = self._next_seq(session_id)
            now = utc_now_iso()
            cursor = self._store.execute(
                """
                INSERT INTO turns(session_id, seq, role, content, created_at,
                                  tokens_in, tokens_out, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, seq, ROLE_ASSISTANT, answer_md, now, None, None, None),
            )
            turn_id = cursor.lastrowid
            message_id = self._store.execute(
                """
                INSERT INTO messages(turn_id, question, answer_md, grounded,
                                     refusal_reason, node_trace_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    question,
                    answer_md,
                    None if grounded is None else int(bool(grounded)),
                    refusal_reason,
                    trace_json,
                ),
            ).lastrowid
            for citation in citations or ():
                self._store.execute(
                    """
                    INSERT INTO citations(message_id, ref_index, doc_id, page,
                                         section_path, snippet)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        _required_int(citation.get("ref_index"), "citation ref_index"),
                        citation.get("doc_id"),
                        _optional_int(citation.get("page")),
                        dumps_json(_section_sequence(citation.get("section_path"))),
                        citation.get("snippet", ""),
                    ),
                )
            self._store.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
        recorded = self.get_message(as_int(message_id, "messages.id") or 0)
        if recorded is None:
            raise StoreLookupError(f"recorded message {message_id} vanished before read-back")
        return recorded

    def get_history(self, session_id: str, *, turn_limit: int | None = None) -> SessionHistory:
        session = self._require_session(session_id)
        turns = self._turns(session_id, limit=turn_limit)
        messages = self._messages_for(session_id, limit=None)
        return SessionHistory(session=session, turns=turns, messages=messages)

    def last_n_turns(self, session_id: str, n: int = HISTORY_WINDOW) -> list[Turn]:
        n = _validate_window(n)
        self._require_session(session_id)
        rows = self._store.query_all(
            """
            SELECT * FROM turns
            WHERE session_id = ?
            ORDER BY seq DESC
            LIMIT ?
            """,
            (session_id, n),
        )
        return [Turn.from_row(row) for row in reversed(rows)]

    def history_as_dialogue(self, session_id: str, n: int = HISTORY_WINDOW) -> list[dict]:
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.last_n_turns(session_id, n)
        ]

    def get_message(self, message_id: int) -> Message | None:
        message_row = self._store.query_one("SELECT * FROM messages WHERE id = ?", (message_id,))
        if message_row is None:
            return None
        turn_id = as_int(message_row["turn_id"], "messages.turn_id") or 0
        turn_row = self._store.query_one("SELECT * FROM turns WHERE id = ?", (turn_id,))
        return Message.from_row(
            message_row,
            turn=None if turn_row is None else Turn.from_row(turn_row),
            citations=self._citations_for(message_id),
        )

    def list_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        return list(self._messages_for(session_id, limit=limit))

    def _require_session(self, session_id: str) -> Session:
        session = self.find_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"no session with id {session_id!r}")
        return session

    def _next_seq(self, session_id: str) -> int:
        row = self._store.query_one(
            "SELECT MAX(seq) AS max_seq FROM turns WHERE session_id = ?", (session_id,)
        )
        current = None if row is None else row["max_seq"]
        if current is None:
            return 1
        return (as_int(current, "turns.seq") or 0) + 1

    def _turns(self, session_id: str, limit: int | None = None) -> tuple[Turn, ...]:
        sql = "SELECT * FROM turns WHERE session_id = ? ORDER BY seq ASC"
        params: list[Any] = [session_id]
        if limit is not None:
            sql += " LIMIT ? OFFSET MAX((SELECT COUNT(*) FROM turns WHERE session_id = ?) - ?, 0)"
            params.extend([limit, session_id, limit])
        return tuple(Turn.from_row(row) for row in self._store.query_all(sql, params))

    def _messages_for(self, session_id: str, limit: int | None = None) -> tuple[Message, ...]:
        if limit is None:
            message_rows = self._store.query_all(
                "SELECT * FROM messages WHERE turn_id IN "
                "(SELECT id FROM turns WHERE session_id = ?) ORDER BY id ASC",
                (session_id,),
            )
        else:
            message_rows = self._store.query_all(
                """
                SELECT m.* FROM messages m
                JOIN turns t ON t.id = m.turn_id
                WHERE t.session_id = ?
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (session_id, limit),
            )[::-1]
        turns_by_id = {turn.id: turn for turn in self._turns(session_id)}
        return tuple(
            Message.from_row(
                row,
                turn=turns_by_id.get(as_int(row["turn_id"], "messages.turn_id") or 0),
                citations=self._citations_for(as_int(row["id"], "messages.id") or 0),
            )
            for row in message_rows
        )

    def _citations_for(self, message_id: int) -> tuple[Citation, ...]:
        rows = self._store.query_all(
            "SELECT * FROM citations WHERE message_id = ? ORDER BY ref_index ASC, id ASC",
            (message_id,),
        )
        return tuple(Citation.from_row(row) for row in rows)


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _validate_role(role: str) -> str:
    if role not in (ROLE_USER, ROLE_ASSISTANT):
        raise ValueError(f"role must be {ROLE_USER!r} or {ROLE_ASSISTANT!r}, got {role!r}")
    return role


def _validate_window(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"history window must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"history window must be >= 1, got {n}")
    return n


def _optional_int(value: Any) -> int | None:
    return None if value is None else as_int(value, "numeric field") or 0


def _required_int(value: Any, field: str) -> int:
    coerced = as_int(value, field)
    if coerced is None:
        raise ValueError(f"{field} is required")
    return coerced


def _section_path(raw: Any) -> tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    value = loads_json(raw)
    if value is None:
        return ()
    return _section_sequence(value)


def _section_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"section_path must be a list of strings, got {type(value).__name__}")
    path = tuple(part for part in value)
    if any(not isinstance(part, str) for part in path):
        raise ValueError("section_path must contain only strings")
    return path

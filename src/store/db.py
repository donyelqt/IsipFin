from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_FILENAME = "chat.sqlite3"


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS sessions(
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TEXT,
        updated_at TEXT,
        model TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turns(
        id INTEGER PRIMARY KEY,
        session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
        seq INTEGER,
        role TEXT,
        content TEXT,
        created_at TEXT,
        tokens_in INTEGER,
        tokens_out INTEGER,
        latency_ms INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY,
        turn_id INTEGER REFERENCES turns(id) ON DELETE CASCADE,
        question TEXT,
        answer_md TEXT,
        grounded INTEGER,
        refusal_reason TEXT,
        node_trace_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS citations(
        id INTEGER PRIMARY KEY,
        message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
        ref_index INTEGER,
        doc_id TEXT,
        page INTEGER,
        section_path TEXT,
        snippet TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents(
        doc_id TEXT PRIMARY KEY,
        filename TEXT,
        sha256 TEXT,
        page_count INTEGER,
        indexed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY,
        session_id TEXT,
        turn_id INTEGER,
        ts TEXT,
        event TEXT,
        payload_json TEXT
    )
    """,
)

INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_turns_session_seq ON turns(session_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(turn_id)",
    "CREATE INDEX IF NOT EXISTS idx_citations_message ON citations(message_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts)",
)

BUSY_TIMEOUT_MS = 5000


class StoreError(Exception):
    pass


class SessionNotFoundError(StoreError):
    pass


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def dumps_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not JSON-serializable: {exc}") from exc


def loads_json(text: str | None, default: Any = None) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"stored JSON is corrupt: {exc}") from exc


def as_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer or None, got {value!r}") from exc


def as_flag(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    return bool(as_int(value, field))


@dataclass(frozen=True)
class Event:

    ts: str
    event: str
    id: int | None = None
    session_id: str | None = None
    turn_id: int | None = None
    payload: Any | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Event:
        return cls(
            id=as_int(row["id"], "events.id"),
            session_id=row["session_id"],
            turn_id=as_int(row["turn_id"], "events.turn_id"),
            ts=row["ts"],
            event=row["event"],
            payload=loads_json(row["payload_json"]),
        )


@dataclass(frozen=True)
class Document:
    doc_id: str
    filename: str
    sha256: str
    page_count: int
    indexed_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Document:
        return cls(
            doc_id=row["doc_id"],
            filename=row["filename"],
            sha256=row["sha256"],
            page_count=as_int(row["page_count"], "documents.page_count"),
            indexed_at=row["indexed_at"],
        )


def default_db_path() -> Path:
    root = os.environ.get("DATA_DIR")
    base = Path(root) if root else DEFAULT_DATA_DIR
    return base / DEFAULT_DB_FILENAME


def connect(path: str | Path | None = None, *, create: bool = True) -> sqlite3.Connection:
    target = Path(path) if path is not None else default_db_path()
    if create and str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target), check_same_thread=False, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    for statement in INDEX_STATEMENTS:
        connection.execute(statement)


class ChatStore:

    def __init__(self, path: str | Path | None = None, *, create: bool = True) -> None:
        self._path = Path(path) if path is not None else default_db_path()
        self._connection = connect(self._path, create=create)
        create_schema(self._connection)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ChatStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self._connection.execute(sql, tuple(params))

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self._connection.execute(sql, tuple(params)))

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self.query_all(sql, params)
        return rows[0] if rows else None

    def upsert_document(self, doc_id: str, filename: str, sha256: str, page_count: int) -> Document:
        document = Document(
            doc_id=doc_id,
            filename=filename,
            sha256=sha256,
            page_count=page_count,
            indexed_at=utc_now_iso(),
        )
        self.execute(
            """
            INSERT INTO documents(doc_id, filename, sha256, page_count, indexed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                filename = excluded.filename,
                sha256 = excluded.sha256,
                page_count = excluded.page_count,
                indexed_at = excluded.indexed_at
            """,
            (
                document.doc_id,
                document.filename,
                document.sha256,
                document.page_count,
                document.indexed_at,
            ),
        )
        return document

    def get_document(self, doc_id: str) -> Document | None:
        row = self.query_one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return None if row is None else Document.from_row(row)

    def list_documents(self) -> list[Document]:
        rows = self.query_all("SELECT * FROM documents ORDER BY indexed_at, doc_id")
        return [Document.from_row(row) for row in rows]

    def record_event(
        self,
        event: str,
        *,
        session_id: str | None = None,
        turn_id: int | None = None,
        payload: Any | None = None,
        ts: str | None = None,
    ) -> Event:
        record = Event(
            ts=ts if ts is not None else utc_now_iso(),
            event=event,
            session_id=session_id,
            turn_id=turn_id,
            payload=payload,
        )
        cursor = self.execute(
            """
            INSERT INTO events(session_id, turn_id, ts, event, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.turn_id,
                record.ts,
                record.event,
                None if payload is None else dumps_json(payload),
            ),
        )
        return Event(
            id=as_int(cursor.lastrowid, "events.id"),
            session_id=record.session_id,
            turn_id=record.turn_id,
            ts=record.ts,
            event=record.event,
            payload=record.payload,
        )

    def list_events(self, session_id: str | None = None, limit: int | None = None) -> list[Event]:
        sql = "SELECT * FROM events"
        params: list[Any] = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY id DESC"
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                raise ValueError(f"limit must be a positive integer, got {limit!r}")
            sql += " LIMIT ?"
            params.append(limit)
        return [Event.from_row(row) for row in self.query_all(sql, params)]

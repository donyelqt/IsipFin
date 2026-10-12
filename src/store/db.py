r"""SQLite persistence for the chat store: connection, frozen schema, row plumbing.

**One file, one writer.** The database is a single SQLite file — ``data/chat.sqlite3``
by default (docs/ARCHITECTURE.md data model). The ``DATA_DIR`` environment variable
relocates it to ``$DATA_DIR/chat.sqlite3`` because the compose setup mounts one volume
for the whole runtime state root (index artifacts, PDFs, chat store all live under
``data/``, docs/adr/ADR-008-packaging-docker-ci.md).

**Single-writer assumption (ADR-006 rule 6).** One Streamlit process, one user demo:
this module owns exactly one ``sqlite3.Connection`` per ``ChatStore`` and never pools,
never queues, and exposes no vendored write lock. Callers that need concurrency must
serialize their own writes — by design, not by accident. There is deliberately no
async here: SQLite writes are single-digit milliseconds, so an event loop would add
moving parts without buying anything.

**Why WAL.** In the default rollback-journal mode a writer takes an exclusive lock and
readers get ``database is locked``. The Streamlit UI reruns frequently: while a turn is
being committed the sidebar can render the transcript, and with ``journal_mode=WAL`` a
second connection reads the pre-commit snapshot instead of failing. WAL also lets
commits skip the per-transaction ``fsync`` that a rollback journal needs, so
``synchronous=NORMAL`` is safe alongside it. WAL state lives in the database file
header, so setting the pragma on every connect is a no-op after the first time; the
price is a ``-wal`` sidecar file beside the database until the next checkpoint.
``busy_timeout`` covers the remaining case (a second writer briefly holding the write
lock) by waiting instead of raising.

**Foreign keys are off by default in SQLite**, so every connection created here turns
them on — otherwise the ``ON DELETE CASCADE`` rules declared in the schema below would
be decorative and ``delete_session`` would orphan rows (ADR-006 rule 5). Cascade chain:
``sessions`` → ``turns`` → ``messages`` → ``citations``. ``events`` is intentionally
*not* a cascade participant: observability rows must outlive the session they describe.

The six tables and their columns are frozen in ``docs/ARCHITECTURE.md`` §Data Model and
are reproduced verbatim by ``SCHEMA_STATEMENTS`` (same tables, same column names, same
order, same types). ``create_schema`` is idempotent — ``CREATE TABLE IF NOT EXISTS`` —
so the app starts identically on an empty or a pre-existing database.
"""

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

# Runtime state root: docs/ARCHITECTURE.md puts the chat store under ``data/``.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_FILENAME = "chat.sqlite3"

# --- schema (frozen: docs/ARCHITECTURE.md §Data Model, ADR-006) -------------------

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
    # turns holds both roles of one exchange; seq is per-session, 1-based, monotonic.
    # The cascade is what makes delete_session a single DELETE.
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
    # One assistant answer per message; the user's question is denormalized here too,
    # so a message renders its own citation block without joining back to the turn.
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
    # section_path is JSON text (a list), matching chunks.jsonl's encoding so both
    # sides of the store speak one serialization dialect.
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
    # documents is the ingested-PDF registry; ingestion is idempotent by content hash
    # (doc_id), so writes here are upserts keyed on the sha256.
    """
    CREATE TABLE IF NOT EXISTS documents(
        doc_id TEXT PRIMARY KEY,
        filename TEXT,
        sha256 TEXT,
        page_count INTEGER,
        indexed_at TEXT
    )
    """,
    # No foreign keys on purpose: the observability stream survives session deletion,
    # and a failure to log must never be able to block a chat write.
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

# Lookup helpers only — they add no columns and change no schema comparison.
INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_turns_session_seq ON turns(session_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(turn_id)",
    "CREATE INDEX IF NOT EXISTS idx_citations_message ON citations(message_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts)",
)

# How long a connection waits for the write lock before raising "database is locked"
# (WAL already removes the common reader/writer case; this covers two writers).
BUSY_TIMEOUT_MS = 5000


# --- errors ----------------------------------------------------------------------


class StoreError(Exception):
    """Base class for chat-store failures; lets callers catch the store, not sqlite3."""


class SessionNotFoundError(StoreError):
    """A session id was named that does not exist — never a silent no-op (rule 4)."""


# --- scalar helpers --------------------------------------------------------------


def utc_now_iso() -> str:
    """Current UTC instant as ISO-8601 with milliseconds and a ``Z`` suffix.

    All timestamps in this store go through here so ``created_at``/``updated_at``/
    ``ts`` are one dialect, lexicographically ordered, and timezone-explicit — the
    same shape ``docs/OBSERVABILITY.md`` freezes for log lines.
    """
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def dumps_json(payload: Any) -> str:
    """Serialize ``payload`` deterministically (sorted keys).

    Determinism matters: ``events.payload_json`` and ``node_trace_json`` are written
    once and read back for weeks, so the same payload must always produce the same
    bytes. ``ensure_ascii=False`` keeps non-ASCII snippet text human-readable in the
    debug panel instead of ``\\uXXXX`` escapes.
    """
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not JSON-serializable: {exc}") from exc


def loads_json(text: str | None, default: Any = None) -> Any:
    """Parse JSON text written by :func:`dumps_json`; ``default`` when it is ``None``.

    Corrupt text is an error, not a silent ``default``: a half-written trace means
    something is wrong with the store, and hiding it would make the debug panel lie.
    """
    if text is None:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"stored JSON is corrupt: {exc}") from exc


def as_int(value: Any, field: str) -> int | None:
    """Coerce a stored scalar to ``int`` or ``None`` (SQLite has no boolean)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer or None, got {value!r}") from exc


def as_flag(value: Any, field: str) -> bool | None:
    """``NULL``/``0``/``1`` in the ``grounded`` column to ``None``/``False``/``True``."""
    if value is None:
        return None
    return bool(as_int(value, field))


# --- rows ------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One row of the ``events`` observability stream (ADR-007)."""

    ts: str
    event: str
    id: int | None = None
    session_id: str | None = None
    turn_id: int | None = None
    payload: Any | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Event:
        """Rebuild an event from a ``SELECT`` row, decoding ``payload_json``."""
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
    """One row of the ``documents`` registry — an ingested PDF."""

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


# --- connections -----------------------------------------------------------------


def default_db_path() -> Path:
    """The database file: ``$DATA_DIR/chat.sqlite3``, else ``data/chat.sqlite3``.

    ``DATA_DIR`` is the runtime state root documented in docs/RUNBOOK.md; the
    chat store is one file inside it, so relocating the volume relocates the store.
    """
    root = os.environ.get("DATA_DIR")
    base = Path(root) if root else DEFAULT_DATA_DIR
    return base / DEFAULT_DB_FILENAME


def connect(path: str | Path | None = None, *, create: bool = True) -> sqlite3.Connection:
    """Open a chat-store connection with the WAL/pragma set this module relies on.

    ``create=False`` refuses to make a new file, so a read-only caller against a
    missing database gets a ``sqlite3.OperationalError`` instead of an empty store.
    ``check_same_thread=False`` is required by Streamlit, which executes script
    reruns on a worker thread rather than the thread that first imported the app;
    the single-writer assumption above is what keeps that sound.
    """
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
    """Create any missing table/index; existing ones are left untouched.

    Idempotent by construction (``IF NOT EXISTS`` everywhere), so this is the app's
    startup path on both a fresh checkout and a year-old volume.
    """
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    for statement in INDEX_STATEMENTS:
        connection.execute(statement)


# --- the store -------------------------------------------------------------------


class ChatStore:
    """One SQLite file, one connection, the frozen schema.

    This is the only place in the app that speaks SQL: every session, turn, message,
    citation, document, and event accessor in this package goes through here, which
    is what makes "moving to Postgres is a driver swap" (ADR-006) a real statement
    rather than a hope. A ``ChatStore`` is not thread-safe and not reentrant — hold
    one per writer (the app holds one per process; tests hold one per tmp path).
    """

    def __init__(self, path: str | Path | None = None, *, create: bool = True) -> None:
        self._path = Path(path) if path is not None else default_db_path()
        self._connection = connect(self._path, create=create)
        create_schema(self._connection)

    # --- handle ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """The database file this store owns."""
        return self._path

    @property
    def connection(self) -> sqlite3.Connection:
        """The single underlying connection (advanced callers only)."""
        return self._connection

    def close(self) -> None:
        """Release the connection; a closed store must not be used again."""
        self._connection.close()

    def __enter__(self) -> ChatStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- SQL plumbing ------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group statements into one atomic write (``BEGIN IMMEDIATE``).

        ``isolation_level=None`` puts ``sqlite3`` in autocommit mode, so every
        statement commits on its own unless it is inside one of these. Multi-row
        writes (a turn plus its message plus its citations; a cascade delete) use it
        so a crash can never leave a half-written exchange behind.
        """
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Run one statement in autocommit mode and return its cursor."""
        return self._connection.execute(sql, tuple(params))

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """All rows of a read query, in the order the query produces."""
        return list(self._connection.execute(sql, tuple(params)))

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        """The first row of a read query, or ``None`` when it matched nothing."""
        rows = self.query_all(sql, params)
        return rows[0] if rows else None

    # --- documents ---------------------------------------------------------------

    def upsert_document(self, doc_id: str, filename: str, sha256: str, page_count: int) -> Document:
        """Register an ingested PDF, replacing any row with the same ``doc_id``.

        Ingestion is idempotent by content hash (docs/ARCHITECTURE.md invariants), so
        the registry key is the hash and a re-upload overwrites rather than
        duplicates. Re-registering refreshes ``indexed_at``.
        """
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
        """The registered document, or ``None`` when the id is unknown."""
        row = self.query_one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return None if row is None else Document.from_row(row)

    def list_documents(self) -> list[Document]:
        """Every registered document, oldest ingestion first."""
        rows = self.query_all("SELECT * FROM documents ORDER BY indexed_at, doc_id")
        return [Document.from_row(row) for row in rows]

    # --- events ------------------------------------------------------------------

    def record_event(
        self,
        event: str,
        *,
        session_id: str | None = None,
        turn_id: int | None = None,
        payload: Any | None = None,
        ts: str | None = None,
    ) -> Event:
        """Append one row to the ``events`` stream (ADR-007's queryable sink).

        Payloads hold counts, ids, and scores — never document text or a full
        context block (docs/OBSERVABILITY.md logging rules). A payload that cannot
        be serialized raises rather than dropping the event: losing an observability
        row silently is exactly the failure this table exists to prevent.
        """
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
        """The event stream for one session (or all of it), newest first, bounded."""
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

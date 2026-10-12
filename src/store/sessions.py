r"""Session lifecycle and conversation history over the chat store (ADR-006).

This module is the chat store's public face. ``src/store/db.py`` owns the connection,
the frozen schema, and the SQL plumbing; everything above it — session CRUD, the
history window, citation round-tripping — lives here so no other package ever writes a
SQL string.

Design decisions worth stating:

- **Every read is a rebuild.** Nothing is cached in memory: a session is re-read from
  SQLite each time it is asked for. The transcript a user sees after a browser reload
  and the one written two seconds ago are produced by the same code path over the same
  rows, so they cannot drift (ADR-006 rule 3: citations reload byte-identical).

- **Agent-facing history is turns, not messages.** ``contextualize_question`` receives
  ``last_n_turns(...)`` — the chat-role dialogue, user turn then assistant turn, capped
  at 6 by default (``HISTORY_WINDOW``). It never receives stored answers through any
  other door, and ``synthesize_answer`` is passed the *standalone* question and the
  retrieved chunks only (ADR-006 rule 2). Reading history in the synthesis prompt lets
  the model continue the conversation instead of grounding in the document, and prior
  turn numbers bleed into the new answer as if they had been retrieved.

- **Nothing is silently ignored.** ``rename_session``/``delete_session``/``switch_session``
  on an unknown id raise :class:`SessionNotFoundError` rather than returning ``False``:
  a no-op on a stale session id is a UI lie, and in Streamlit a failed rerun is visible
  while a quietly dropped write is not.

- **Timestamps are one dialect.** Everything goes through
  :func:`src.store.db.utc_now_iso` (ISO-8601 UTC, millisecond precision, ``Z`` suffix)
  so ordering by text and uniform rendering both hold without post-processing.

- **``seq`` is per session, 1-based, monotonic.** It is derived from
  ``MAX(seq) + 1`` server-side inside the write transaction, never passed in by the
  caller, so a rerun cannot renumber a transcript and cannot fabricate a gap.

Deletion is the cascade declared in the schema (``sessions`` → ``turns`` → ``messages``
→ ``citations``), so removing a session is one statement, not four ordered ones that
can interleave with a rerun mid-flight. ``events`` deliberately survives: observability
outlives the transcript it measured.
"""

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

# History window handed to contextualize_question (ADR-006 rule 1): the last 6 turns.
# Six is the compromise — enough for pronoun resolution and topic carry-over across a
# multi-turn answer, bounded so a long session cannot grow prompt cost without limit.
HISTORY_WINDOW = 6

# Turn roles (docs/ARCHITECTURE.md schema comment: 'user' | 'assistant').
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


class StoreLookupError(RuntimeError):
    """A just-written row could not be read back — an internal contract violation.

    Not a caller-facing error: it exists so a genuine store bug (a transaction that
    committed nothing, a schema drift making a column unreadable) fails loudly instead
    of being mistaken for "the answer was recorded" while returning an empty result.
    """


# --- rows ------------------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    """One chat session (the ``sessions`` row plus its turn count, when listed)."""

    id: str
    title: str | None
    created_at: str
    updated_at: str
    model: str | None
    turn_count: int = 0

    @classmethod
    def from_row(cls, row: Any) -> Session:
        """Rebuild a session from a ``SELECT`` row (``turn_count`` optional)."""
        # ``sqlite3.Row`` is keyed access, not a mapping: only the list-sessions
        # query carries ``turn_count``, so its absence is checked against the row's
        # actual keys rather than assumed.
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
    """One user or assistant turn (the ``turns`` row)."""

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
        """True for a user turn (the question), false for an assistant turn."""
        return self.role == ROLE_USER


@dataclass(frozen=True)
class Citation:
    """One numbered citation attached to an assistant message.

    ``section_path`` is a JSON list in the database (matching ``chunks.jsonl`` and
    ``ScoredChunk``); it is decoded to a tuple here so the annotation layer and the
    debug panel get the same shape the retrieval chain produces.
    """

    id: int | None
    ref_index: int
    doc_id: str
    page: int | None
    section_path: tuple[str, ...] = ()
    snippet: str = ""
    message_id: int | None = None

    @classmethod
    def from_row(cls, row: Any) -> Citation:
        """Rebuild a citation from a ``SELECT`` row, decoding ``section_path`` JSON."""
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
    """One assistant answer: the ``messages`` row plus its citations and its turn."""

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
        """Rebuild a message from a ``messages`` row (``node_trace_json`` decoded).

        ``turn`` and ``citations`` are attached by the caller rather than joined into
        this constructor: a join would multiply message rows and force callers into
        deduplication, while the citation list must round-trip in a stable order
        anyway (``ref_index`` then insertion order).
        """
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
    """A session plus its turns and messages — the full transcript, in order.

    ``turns`` carries every role; ``messages`` carries only assistant answers, each
    with its own citation block. The UI renders turns; the debug panel reads messages.
    """

    session: Session
    turns: tuple[Turn, ...] = ()
    messages: tuple[Message, ...] = ()

    @property
    def last_user_turn(self) -> Turn | None:
        """The most recent user turn, or ``None`` for a fresh session."""
        for turn in reversed(self.turns):
            if turn.is_user:
                return turn
        return None


# --- session API -----------------------------------------------------------------


class SessionStore:
    """The chat store's session lifecycle and history accessors (ADR-006).

    Wraps one :class:`~src.store.db.ChatStore` — one writer, one connection, one
    database file. Multi-statement writes (a turn plus its message plus its
    citations; a cascade delete) run inside one transaction so a crash or a rollback
    cannot leave a half-written exchange behind.
    """

    def __init__(self, store: ChatStore | None = None, path: str | None = None) -> None:
        self._store = store if store is not None else ChatStore(path)

    @property
    def store(self) -> ChatStore:
        """The underlying connection owner (for events/documents callers)."""
        return self._store

    def close(self) -> None:
        """Release the connection; a closed store must not be used again."""
        self._store.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- session lifecycle -------------------------------------------------------

    def create_session(
        self, title: str | None = None, model: str | None = None, session_id: str | None = None
    ) -> Session:
        """Create a session and return it.

        ``session_id`` is normally generated by the caller as a uuid4 hex; passing one
        explicitly keeps ingest-side fixtures and tests honest instead of forcing them
        to round-trip the generated value. An existing id is a conflict, not a silent
        duplicate — ``id`` is the primary key.
        """
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
        """Every session, most recently updated first (the sidebar's order)."""
        rows = self._store.query_all("""
            SELECT s.*, (
                SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id
            ) AS turn_count
            FROM sessions s
            ORDER BY s.updated_at DESC, s.created_at DESC, s.id ASC
            """)
        return [Session.from_row(row) for row in rows]

    def get_session(self, session_id: str) -> Session:
        """The session row, raising :class:`SessionNotFoundError` when unknown.

        This is the strict read: every mutator uses it to turn "unknown id" into a
        loud failure instead of a no-op write. The lenient read is
        :meth:`find_session`.
        """
        return self._require_session(session_id)

    def find_session(self, session_id: str) -> Session | None:
        """The session row, or ``None`` when it does not exist (render-path read)."""
        row = self._store.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return None if row is None else Session.from_row(row)

    def rename_session(self, session_id: str, title: str) -> Session:
        """Rename a session; an unknown id raises, never silently succeeds."""
        self._require_session(session_id)
        with self._store.transaction():
            self._store.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, utc_now_iso(), session_id),
            )
        return self._require_session(session_id)

    def switch_session(self, session_id: str) -> Session:
        """Select a session: no-op on disk beyond the read, strict on unknown ids.

        Switching in a one-process app is a UI concern, not a database state, so there
        is no "active session" column to flip. The store still validates the id here
        so the UI's selected session cannot point at a row that was deleted.
        """
        return self._require_session(session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session and cascade turns → messages → citations (ADR-006 rule 5).

        One ``DELETE`` backed by the schema's ``ON DELETE CASCADE`` rules: the
        dependent rows go with it inside the same transaction, and ``events`` rows are
        left standing on purpose (observability outlives the transcript). An unknown
        id raises.
        """
        self._require_session(session_id)
        with self._store.transaction():
            self._store.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # --- turns -------------------------------------------------------------------

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
        """Append one user or assistant turn to a session.

        ``seq`` is assigned inside the transaction from ``MAX(seq) + 1`` — never
        taken from the caller, never reused — so the transcript numbering stays
        1-based and gap-free even across a failed rerun. The session's ``updated_at``
        moves with it, which is what makes "most recently updated first" the right
        sidebar order.
        """
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
        assert row is not None  # just inserted inside a committed transaction
        return Turn.from_row(row)

    def append_turns(self, session_id: str, turns: Iterable[Mapping[str, Any]]) -> list[Turn]:
        """Append several turns in one transaction, in the order given.

        Each mapping carries ``role`` and ``content`` plus the optional
        ``tokens_in``/``tokens_out``/``latency_ms``; a missing key means NULL, not
        zero. Bulk appends stay atomic: either the whole batch lands or none of it
        does, so a partial batch can never appear in the history window.
        """
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

    # --- messages + citations ----------------------------------------------------

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
        """Record an assistant answer, its citations, and its node trace in one write.

        The assistant ``turns`` row is written in the same transaction as the
        ``messages`` row (``role='assistant'``, content = the markdown answer), so a
        message can never exist without the turn the transcript needs, and the
        history window never sees a question whose answer is missing.

        Each citation is ``{ref_index, doc_id, page, section_path, snippet}``;
        ``section_path`` is serialized as a JSON list, the same encoding
        ``chunks.jsonl`` uses, so the store speaks one dialect.
        """
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
        """Read a session with its full transcript and every message.

        ``turn_limit`` caps the *turns* loaded to the most recent ones (display-only:
        a long transcript rendered in pages). Messages carry their own turns, so they
        are loaded whole — a citation block must never be sliced away from the answer
        that owns it. A missing session is an error here, not an empty history: an
        empty transcript is a legitimate state for a *fresh* session, and conflating
        the two would hide a deleted id from the caller.
        """
        session = self._require_session(session_id)
        turns = self._turns(session_id, limit=turn_limit)
        messages = self._messages_for(session_id, limit=None)
        return SessionHistory(session=session, turns=turns, messages=messages)

    def last_n_turns(self, session_id: str, n: int = HISTORY_WINDOW) -> list[Turn]:
        """The last ``n`` turns of a session, oldest first — the ADR-006 history window.

        This is the single read path a graph node is allowed to use for
        conversational context, and only ``contextualize_question`` uses it. Fewer
        than ``n`` turns returns all of them; a session with no turns returns
        ``[]``. ``n`` must be at least 1 — a window of 0 is not "no context" but a
        caller bug.
        """
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
        """The history window as ``[{"role": ..., "content": ...}]`` dicts.

        Convenience shape for prompt builders: already ordered oldest-first, already
        capped, no database objects crossing into the chains package. This is the
        exact payload ``contextualize_question`` receives per docs/prompts/
        follow-up-contextualization.md; ``synthesize_answer`` is never handed it.
        """
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.last_n_turns(session_id, n)
        ]

    def get_message(self, message_id: int) -> Message | None:
        """One message with its citations and its turn, or ``None`` when unknown.

        The debug panel uses this to render a historical answer — including its
        ``node_trace`` — after the newest turn has long rolled past (ADR-007). One
        point-by-point read per table rather than a join: ``messages`` and ``turns``
        both carry an ``id``, and a join would make aliasing rules the caller's
        problem for no gain at this row count.
        """
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
        """Messages of a session, oldest first — the history of answers only."""
        return list(self._messages_for(session_id, limit=limit))

    # --- internals ---------------------------------------------------------------

    def _require_session(self, session_id: str) -> Session:
        """Fetch a session or raise: the shared guard for every mutating path."""
        session = self.find_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"no session with id {session_id!r}")
        return session

    def _next_seq(self, session_id: str) -> int:
        """Next 1-based ``seq`` for a session: ``MAX(seq) + 1``, 1 when empty.

        Called inside a write transaction (the same one that inserts), so two
        concurrent appenders cannot allocate the same number.
        """
        row = self._store.query_one(
            "SELECT MAX(seq) AS max_seq FROM turns WHERE session_id = ?", (session_id,)
        )
        current = None if row is None else row["max_seq"]
        if current is None:
            return 1
        return (as_int(current, "turns.seq") or 0) + 1

    def _turns(self, session_id: str, limit: int | None = None) -> tuple[Turn, ...]:
        """All turns of a session, oldest first; ``limit`` keeps the newest ones."""
        sql = "SELECT * FROM turns WHERE session_id = ? ORDER BY seq ASC"
        params: list[Any] = [session_id]
        if limit is not None:
            sql += " LIMIT ? OFFSET MAX((SELECT COUNT(*) FROM turns WHERE session_id = ?) - ?, 0)"
            params.extend([limit, session_id, limit])
        return tuple(Turn.from_row(row) for row in self._store.query_all(sql, params))

    def _messages_for(self, session_id: str, limit: int | None = None) -> tuple[Message, ...]:
        """Messages of a session, oldest first, each with its citation block.

        Citations are fetched per message (a session holds at most a handful of
        messages with a handful of citations, so this stays a two-query read) rather
        than through a join, because a join would drive callers into row-deduplication
        logic and the citation list must round-trip in a stable order anyway.
        """
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
        """Citations of one message, in ``ref_index`` order."""
        rows = self._store.query_all(
            "SELECT * FROM citations WHERE message_id = ? ORDER BY ref_index ASC, id ASC",
            (message_id,),
        )
        return tuple(Citation.from_row(row) for row in rows)


# --- module-level convenience ----------------------------------------------------


def _new_session_id() -> str:
    """A uuid4 hex session id (uuid, per docs/ARCHITECTURE.md schema comment)."""
    return uuid.uuid4().hex


def _validate_role(role: str) -> str:
    """Normalize and check a turn role against the schema's ``'user'|'assistant'``."""
    if role not in (ROLE_USER, ROLE_ASSISTANT):
        raise ValueError(f"role must be {ROLE_USER!r} or {ROLE_ASSISTANT!r}, got {role!r}")
    return role


def _validate_window(n: int) -> int:
    """History-window size: a positive integer (``HISTORY_WINDOW`` by default)."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"history window must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"history window must be >= 1, got {n}")
    return n


def _optional_int(value: Any) -> int | None:
    """Coerce an optional numeric field, keeping ``None`` as ``None``."""
    return None if value is None else as_int(value, "numeric field") or 0


def _required_int(value: Any, field: str) -> int:
    """Coerce a required numeric field, rejecting ``None``."""
    coerced = as_int(value, field)
    if coerced is None:
        raise ValueError(f"{field} is required")
    return coerced


def _section_path(raw: Any) -> tuple[str, ...]:
    """Decode a stored ``section_path`` JSON list (a string becomes a one-element path)."""
    if raw is None or raw == "":
        return ()
    value = loads_json(raw)
    if value is None:
        return ()
    return _section_sequence(value)


def _section_sequence(value: Any) -> tuple[str, ...]:
    """Validate and coerce a section path into a tuple of strings."""
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

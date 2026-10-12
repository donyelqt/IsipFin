# ADR-006: Conversation memory and session persistence — SQLite store

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none

## Context

The brief requires *"managing conversation context and follow-up questions,"* *"managing chat
history,"* and *"persisting chat sessions."* Concretely the app must: answer *"and its margin
trend?"* correctly after discussing revenue, survive a browser reload with the full transcript
and citations intact, let the user create/switch/rename/delete sessions, and keep a record of
which chunks backed each answer.

## Decision

**`chat-store` module (`src/store/db.py`, `src/store/sessions.py`)** — SQLite at
`data/chat.sqlite3` (gitignored, on the compose volume). Schema (frozen in
`docs/ARCHITECTURE.md` §Data Model):

| Table | Purpose |
|---|---|
| `sessions` | one chat session: id (uuid), title, created/updated, model used |
| `turns` | one user or assistant turn: seq, role, content, tokens in/out, latency |
| `messages` | assistant answers: question, answer_md, grounded flag, refusal_reason, full `node_trace_json` |
| `citations` | per-message numbered citations: ref_index, doc_id, page, section_path, snippet |
| `documents` | ingested PDFs: doc_id (sha256), filename, page_count, indexed_at |
| `events` | observability stream: ts, event name, payload JSON |

Rules:

1. **History windowing:** `contextualize_question` receives the last **6 turns** (default,
   configurable) from the store — enough to resolve pronouns and topic carry-over, bounded in
   cost. Older turns are not re-sent.
2. **History is used exactly once.** Only `contextualize_question` reads history, to rewrite the
   follow-up into a standalone question. `synthesize_answer` never sees raw chat history — it
   sees only the standalone question and retrieved chunks. This prevents the model from
   "continuing" the conversation instead of grounding in the document (a real failure mode:
   prior-turn numbers bleed into the new answer as if retrieved).
3. **Citations are first-class rows**, so a reload re-renders citation cards exactly — clicking
   `[2]` shows the same snippet and page a week later.
4. **`node_trace_json` persists per message** — the debug panel works on historical answers,
   not just the newest one.
5. **Session lifecycle** from the UI: new / switch / rename / delete (cascade turns → messages →
   citations).
6. Single-writer access is fine: one Streamlit process, one user demo. WAL mode enabled for
   read/write concurrency during reruns.

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| LangGraph checkpointers (`SqliteSaver`/`PostgresSaver`) | Built for resuming graph executions mid-run, not for a user-facing transcript; couples chat UX to graph internals and state shape |
| `st.session_state` only | Dies on reload/tab close — fails the "persist sessions" requirement outright |
| Postgres | An external service for a one-container demo; compose would grow a second service for no benefit |
| JSON files per session | No querying, no cascade deletes, races on concurrent Streamlit reruns |
| Chroma's doc store for chat | Mixing vector-DB internals with conversation data; chat records have no vector semantics |

## Consequences

- Persistence swaps cleanly: all SQL lives behind `src/store/db.py` accessors; moving to
  Postgres is a driver swap, no caller changes (documented in journal).
- `events` doubles as the observability sink (`docs/OBSERVABILITY.md`) — the debug panel reads
  the same rows the logger writes.
- Known limitation recorded: SQLite single-file means the database lives on the compose volume;
  multi-user server deployment is out of scope for the exam (noted, not built).

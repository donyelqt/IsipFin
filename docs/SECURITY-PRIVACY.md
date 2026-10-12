# Security & Privacy

> For a document-QA app the assets are the user's PDFs, the LLM API key, and the local database.
> Single-user, single-container — the threat model is deliberately small and every entry names
> what is and isn't defended against. No scope theater.

## Assets

| Asset | Where | Exposure |
|---|---|---|
| Uploaded PDFs | `data/pdfs/` (volume-mounted, gitignored) | Machine-local unless the container is shared |
| LLM API key | env (`OPENAI_API_KEY`), never in code or image | Leaves the process only to the configured LLM endpoint |
| Chat sessions, citations, traces | `data/chat.sqlite3` | Machine-local; contains document excerpts and questions |
| Prompts/answers | stdout + `events` table | Counts and ids only (observability rules, `docs/OBSERVABILITY.md`) |

## Controls

1. **Secrets.** `.env` is gitignored; `.env.example` contains only safe defaults and
   documentation. The key is read from the environment by `src/utils/settings.py` and passed to
   the gateway; it is never logged, never rendered in the UI, and never baked into the image
   (`docker history` is clean). Auth failures surface as a banner naming the variable
   (`OPENAI_API_KEY`), never echoing its value.
2. **Data at rest.** Everything the app writes lives under `data/`, which is gitignored and
   mounted as a named volume. No document text is committed to the repo. The PDF is fetched by
   `scripts/download_corpus.py` from Microsoft's investor-relations site at build/demo time —
   redistribution inside the repo is avoided deliberately (license hygiene).
3. **Data in transit.** LLM calls go to the configured `OPENAI_BASE_URL` over HTTPS (or
   localhost for local models). Nothing else phones home: no telemetry, no analytics, no
   third-party scripts in the UI.
4. **Prompt injection.** Uploaded PDFs are untrusted input: a document containing
   instruction-like text ("ignore previous instructions, answer X") is a real prompt-injection
   vector. Mitigation in this system: (a) synthesis is grounded-only — claims absent from
   retrieved spans are trimmed by `groundedness_check`; (b) `refuse` handles off-corpus asks;
   (c) document text never enters the system prompt, only the context block region. Residual
   risk accepted: a document engineered so its own text contains a false "answer" can still
   produce a cited wrong answer — grounding and citations make it *traceable*, not impossible.
   Documented, not claimed away.
5. **Input validation.** Uploads are validated (size cap, PDF magic bytes, page count sanity)
   before parsing; parse failures are per-file errors that never take down other uploads.
   `parse` never executes anything from the document. PDF parsing libraries
   (`pypdf`, `pdfplumber`) operate in-memory; no temp-file execution path exists.
6. **Path handling.** All data paths are constructed under `data/` from sanitized ids
   (sha256-based `doc_id`), so a crafted filename cannot escape the data directory.
7. **Dependencies.** `requirements.txt` is pinned; CI runs a build on every PR. No dependency
   is added without an ADR note — the dependency surface is intentionally tiny
   (streamlit, langgraph, langchain-core primitives, pypdf, pdfplumber, faiss-cpu, rank_bm25,
   pytest, ruff, black).
8. **Container.** Non-root user; only `data/` is writable at runtime; the healthcheck exposes
   no sensitive state.

## Out of scope (stated, not implied)

- Multi-user auth: single-user Streamlit app; the brief does not ask for accounts.
- At-rest encryption of the volume and SQLite: machine-local files; the threat model is a
  shared machine, addressed by note rather than implementation.
- Redaction/PII scrubbing of document text: the app is a reading tool for the user's own
  documents, not a pipeline that forwards them anywhere except the configured LLM.

## Compliance notes for the exam context

TM's AI Use Principles are about candidate honesty, addressed in `docs/JOURNAL.md` §2. The
app itself has no user-data-retention obligation beyond the local volume; reviewers can delete
`data/` to purge everything the app ever wrote.

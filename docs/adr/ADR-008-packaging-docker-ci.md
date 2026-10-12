# ADR-008: Packaging — single Docker image, compose with data volumes, GitHub Actions CI

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none

## Context

The brief asks to *"containerize application modules for improved deployment and scalability"*
and the submission checklist expects the app to run out-of-the-box via `docker-compose up` for
reviewers on Windows and macOS. The grader's cold-start path is:
`git clone → cp .env.example .env → docker compose up --build`.

## Decision

**One image, one service.**

- **`Dockerfile`** — `python:3.11-slim` base, multi-stage: builder stage creates a venv with
  `pip install -r requirements.txt` (pinned versions); runtime stage copies the venv, the
  `src/`, `app.py`, `prompts/`, and `scripts/`; installs a healthcheck curl; exposes 8501;
  `CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]`.
  Non-root user, `PYTHONUNBUFFERED=1`.
- **`docker-compose.yml`** — single `app` service: build `.`, ports `8501:8501`, named volumes
  **`data` → `/app/data`** (index artifacts, PDFs, chat.sqlite3 — a rebuild never loses
  ingested docs or chat history), `env_file: .env`, `environment:` passthrough for
  `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `MOCK_LLM`; healthcheck against
  Streamlit's `/_stcore/health`; `restart: unless-stopped`.
- **`.env.example`** — every variable documented with safe defaults; `MOCK_LLM=0` default with a
  comment explaining the keyless demo path.
- **`.github/workflows/ci.yml`** — on PR: `ruff check`, `black --check`, `pytest -q`, and a
  `docker build` smoke (no push, no registry credentials).
- **Corpus stays out of the repo** — `scripts/download_corpus.py` fetches the Microsoft FY2023
  annual report PDF on demand; `data/` and `*.pdf` are gitignored. The repo stays small and
  license-clean; graders get the same corpus with one command.

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Compose with a separate vector-DB service (Qdrant/Weaviate/Chroma server) | FAISS + BM25 are in-process artifacts (`data/index/`); a server adds a service to babysit for zero benefit at this scale |
| Baking the corpus into the image | Bloats the image, license ambiguity, and defeats the volume design |
| Unpinned requirements | A grader's build six months from now breaks; pinning is the boring correct choice |
| Streamlit Community Cloud | The brief explicitly wants containerization; also can't persist SQLite reliably |
| Kubernetes | Absurd for a single-container exam app |

## Consequences

- Cold-start is three commands; `MOCK_LLM=1` makes even the keyless reviewer path work
  (`docker compose up` → keyless offline demo).
- Scale story is honest: this scales **vertically** (one container, one process, FAISS in
  memory/CPU). The journal states the next step (extract the graph behind a FastAPI service,
  move chat-store to Postgres) rather than pretending the compose file is production.
- Windows/macOS notes for reviewers: works with Docker Desktop defaults; the app writes only
  inside the mounted `data/` volume; no host-only paths anywhere in code or compose.

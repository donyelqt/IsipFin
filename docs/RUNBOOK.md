# Runbook (target state)

> Commands and env vars as they will exist after the build. Reviewers follow this verbatim;
> `README.md` (T24) is the short version of this file. Every command here was run at least once
> before submission — the journal's measurement section (`docs/JOURNAL.md` §6) records the
> actual outputs.

## Environment variables

| Var | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | for real LLM mode | — | API key; never committed, read from env only |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | OpenAI-compatible endpoint (Azure / vLLM / Ollama at `http://localhost:11434/v1`) |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | chat model id |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` | embedding model id |
| `MOCK_LLM` | no | `0` | `1` = deterministic scripted LLM, no network — demo and test path |
| `DATA_DIR` | no | `./data` | runtime state root (index, pdfs, sqlite) |
| `LOG_JSON` | no | `1` | JSON log format |
| `LANGSMITH_TRACING` | no | `false` | optional LangSmith hooks, env-gated (ADR-007) |
| `RETRIEVAL_K` | no | `8` | fused top-k children before parent expansion |

`.env.example` documents all of the above with safe defaults; `MOCK_LLM=1` is the keyless demo
path.

## Setup (native)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then edit .env
```

## Run — local Streamlit

```bash
python scripts/download_corpus.py                   # fetch MSFT FY2023 annual report → data/pdfs/
python scripts/build_index.py                       # parse → chunks → FAISS + BM25 (prints build stats)
streamlit run app.py                                # http://localhost:8501
```

The app can also build the index from the UI by uploading a PDF — the scripts are the
non-interactive path for reviewers and CI.

## Run — Docker (the grader's path)

```bash
git clone <repo> && cd <repo>
cp .env.example .env            # or: export MOCK_LLM=1 for the keyless demo
docker compose up --build       # http://localhost:8501
```

Notes: `data/` is a named volume — ingest and chat history survive `docker compose down` and
rebuilds. `docker compose logs -f app` streams the JSON event stream (observability evidence).

## Test & quality gates

```bash
pytest -q                                        # hermetic suite (never needs network)
python scripts/eval_rag.py                       # golden QA vs real LLM (needs key)
ruff check src tests scripts app.py && black --check src tests scripts app.py
```

## Common failures (what actually breaks, and the fix)

| Symptom | Cause | Fix |
|---|---|---|
| Empty state, "Upload a PDF to begin" | index not built | run `scripts/build_index.py` or upload in the UI |
| Answer banner: "model busy, retrying…" then error | 429 / quota | wait or switch `OPENAI_BASE_URL`; `MOCK_LLM=1` to demo keyless |
| Banner: "context too large" | wide decomposed question + high k | lower `k` in sidebar; graph auto-retries once with smaller budget |
| Banner: "index not built — upload a PDF" | `RetrievalError` on load | rebuild index; check `data/index/chunks.jsonl` exists |
| Per-file upload error card | scanned/corrupt PDF | expected for image-only PDFs — other files still ingest; warning names pages |
| Auth banner naming `OPENAI_API_KEY` | bad/missing key | fix `.env`; key is never echoed |
| `docker compose up` port busy | 8501 in use | `ports: "8502:8501"` in compose |
| Prompt changed but behavior stale | prompt file cached at startup | restart the app; prompts load at startup via `load_prompt` |

## Where to look when something is wrong

1. UI debug panel — node trace, plan, retrieval scores for the answer.
2. `docker compose logs app` / stdout — JSON events with `trace_id`.
3. `data/chat.sqlite3` → `events` table — the same stream, queryable after the fact.
4. `docs/OBSERVABILITY.md` — event vocabulary and failure→UI mapping.
5. `docs/TESTING-AND-EVAL.md` — which hermetic test covers the broken component.

# Runbook (target state)

> Commands and env vars as they will exist after the build. Reviewers follow this verbatim;
> `README.md` (T24) is the short version of this file. Every command here was run at least once
> before submission — the journal's measurement section (`docs/JOURNAL.md` §6) records the
> actual outputs.

## Environment variables

| Var | Required | Default | Purpose |
|---|---|---|---|
| `LLM_API_KEY` | no | — (deliberately unset) | chat credential; beats `OPENAI_API_KEY` and `GROQ_API_KEY` — the override for a host that already exports `OPENAI_*` (README §1.2) |
| `LLM_BASE_URL` | no | — (deliberately unset) | chat endpoint; beats `OPENAI_BASE_URL`. Kept empty so an unset `LLM_*` cannot shadow a configured `OPENAI_*` in the first-hit-wins chain |
| `OPENAI_API_KEY` | for real LLM mode | — | chat credential; second in the chain, ahead of `GROQ_API_KEY`. Never committed, read from env only |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | OpenAI-compatible chat endpoint (Azure / vLLM / Ollama at `http://localhost:11434/v1`) |
| `GROQ_API_KEY` | no | — | Groq chat credential; setting it alone also selects Groq's base URL `https://api.groq.com/openai/v1` when no base URL is set anywhere |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | chat model id. Protocol-named, not vendor-named: also addresses Groq (`openai/gpt-oss-120b`), NVIDIA or Ollama models — there is no separate model alias |
| `EMBEDDING_PROVIDER` | no | `local` | **the embedder path.** `local` = in-process Hugging Face model via fastembed/ONNX — no key, no `/embeddings` endpoint; `api` = the `EMBEDDING_*` endpoint below; `MOCK_LLM=1` overrides both |
| `LOCAL_EMBEDDING_MODEL` | no | `BAAI/bge-small-en-v1.5` (384-dim) | model used when `EMBEDDING_PROVIDER=local`; ~150 MB, downloaded and cached on first use |
| `EMBEDDING_BASE_URL` | no | — | only for `EMBEDDING_PROVIDER=api`; defaults to the chat base URL |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` | only for `EMBEDDING_PROVIDER=api`; embedding model id |
| `MOCK_LLM` | no | `0` (Docker: `1` unless `.env` sets it) | `1` = deterministic scripted LLM **and** offline sha256 embedder, no network — demo and test path; checked before `EMBEDDING_PROVIDER`, so it wins |
| `DATA_DIR` | no | `./data` | runtime state root (index, pdfs, sqlite) |
| `LOG_JSON` | no | `1` | JSON log format |
| `LANGSMITH_TRACING` | no | `false` | optional LangSmith hooks, env-gated (ADR-007) |
| `RETRIEVAL_K` | no | `8` | fused top-k children before parent expansion |

`.env.example` documents all of the above with safe defaults; `MOCK_LLM=1` is the keyless demo
path.

**Chat credential precedence (first hit wins):** key `LLM_API_KEY` → `OPENAI_API_KEY` →
`GROQ_API_KEY`; base URL `LLM_BASE_URL` → `OPENAI_BASE_URL` → Groq's URL when the key came from
`GROQ_API_KEY` and no base URL is set anywhere; model `OPENAI_MODEL` → built-in default (a
per-call override still wins). Neither `LLM_*` name has a default on purpose: a host that
already exports `OPENAI_API_KEY` / `OPENAI_BASE_URL` beats `.env`
(`load_dotenv(override=False)`), so a `.env` that names Groq is silently ignored and the
`LLM_*` names are how you regain control. The trap and the fix are explained in `README.md`
§1.2; the gateway's `resolved_config()` / `resolved_provider()` report what was actually
resolved, with the key redacted. `docker compose` passes each override variable through empty
so an unset one cannot shadow a configured `OPENAI_*`.

**Embedder — the local path is the default.** `EMBEDDING_PROVIDER=local` runs
`LOCAL_EMBEDDING_MODEL` (384-dim) in-process on ONNX Runtime with no key and no `/embeddings`
endpoint; only `EMBEDDING_PROVIDER=api` reads `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`. Since
`MOCK_LLM` is checked first and this repo's own `.env` ships `MOCK_LLM=1`, the local default is
exercised with `MOCK_LLM=0` or unset. Changing the provider or model changes the vector width
and invalidates any existing index — rebuild into a fresh `--index-dir` or delete
`data/index/`; a mismatch fails loudly rather than returning garbage.

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
| Answer banner: "model busy, retrying…" then error | 429 / quota | wait or switch `LLM_BASE_URL` / `OPENAI_BASE_URL`; `MOCK_LLM=1` to demo keyless |
| Banner: "context too large" | wide decomposed question + high k | lower `k` in sidebar; graph auto-retries once with smaller budget |
| Banner: "index not built — upload a PDF" | `RetrievalError` on load | rebuild index; check `data/index/chunks.jsonl` exists |
| Per-file upload error card | scanned/corrupt PDF | expected for image-only PDFs — other files still ingest; warning names pages |
| Auth banner naming the chat key vars (`LLM_API_KEY` / `OPENAI_API_KEY` / `GROQ_API_KEY`) | bad/missing key, or an ambient key winning | fix `.env`, or set `LLM_API_KEY` when the host already exports `OPENAI_*` (README §1.2); check `resolved_config()`; the key is never echoed |
| `docker compose up` port busy | 8501 in use | `ports: "8502:8501"` in compose |
| Prompt changed but behavior stale | prompt file cached at startup | restart the app; prompts load at startup via `load_prompt` |

## Where to look when something is wrong

1. UI debug panel — node trace, plan, retrieval scores for the answer.
2. `docker compose logs app` / stdout — JSON events with `trace_id`.
3. `data/chat.sqlite3` → `events` table — the same stream, queryable after the fact.
4. `docs/OBSERVABILITY.md` — event vocabulary and failure→UI mapping.
5. `docs/TESTING-AND-EVAL.md` — which hermetic test covers the broken component.

# Runbook (target state)

> Commands and env vars as they will exist after the build. Reviewers follow verbatim;
> `docs/JOURNAL.md` §6 records the actual command outputs.

## Environment variables

| Var | Required | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | no | `openai` (blank in `.env.example`, so a `GROQ_API_KEY` alias still implies `groq`) | chat provider: `openai` · `groq` · `nvidia` · `ollama` · `vllm` · `azure`; name the provider → its own endpoint, so `groq` needs no base URL |
| `LLM_API_KEY` | no | — (deliberately unset) | **the chat credential — the one to fill in.** Beats `OPENAI_API_KEY` and `GROQ_API_KEY`, so overrides a host that already exports `OPENAI_*` |
| `LLM_BASE_URL` | no | — (deliberately unset) | chat endpoint override for a custom / Azure / self-hosted endpoint. Empty so the selected provider's own endpoint is used and an unset `LLM_*` cannot shadow a configured alias |
| `LLM_MODEL` | no | the provider's own default (`gpt-4o-mini` for `openai`) | chat model id. Beats `OPENAI_MODEL`; a per-call UI override still wins |
| `OPENAI_API_KEY` | for real LLM mode | — | alias for the chat credential, read only when `LLM_API_KEY` is unset. Never committed, env only |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | alias for chat endpoint (Azure / vLLM / Ollama at `http://localhost:11434/v1`), read only when `LLM_BASE_URL` is unset |
| `GROQ_API_KEY` | no | — | alias for the chat credential that also implies `LLM_PROVIDER=groq`. Applies its base URL only when no base URL is set anywhere |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | alias for model id, read only when `LLM_MODEL` is unset. Carries any provider's model |
| `EMBEDDING_PROVIDER` | no | `local` | **the embedder path.** `local` = in-process Hugging Face via fastembed/ONNX — no key, no `/embeddings`; `api` = the `EMBEDDING_*` endpoint below; `MOCK_LLM=1` overrides both |
| `LOCAL_EMBEDDING_MODEL` | no | `BAAI/bge-small-en-v1.5` (384-dim) | model used when `EMBEDDING_PROVIDER=local`; ~150 MB, cached on first use |
| `EMBEDDING_BASE_URL` | no | — | only for `EMBEDDING_PROVIDER=api`; defaults to the chat base URL |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` | only for `EMBEDDING_PROVIDER=api` |
| `MOCK_LLM` | no | `0` (Docker: `1` unless `.env` sets it) | `1` = deterministic scripted LLM **and** offline sha256 embedder, no network — demo + test path. Checked **before** `LLM_PROVIDER` / `LLM_API_KEY`, so needs no key and wins over every chat setting |
| `DATA_DIR` | no | `./data` | runtime state root (index, pdfs, sqlite) |
| `LOG_JSON` | no | `1` | JSON log format |
| `LANGSMITH_TRACING` | no | `false` | optional LangSmith hooks, env-gated (ADR-007) |
| `RETRIEVAL_K` | no | `8` | fused top-k children before parent expansion |

`.env.example` documents all of the above with safe defaults; `MOCK_LLM=1` is the keyless demo
path.

**Chat config — four primary names, then aliases (first hit wins).** Primaries:
`LLM_PROVIDER` (selects provider → its own endpoint), `LLM_API_KEY`, `LLM_BASE_URL`,
`LLM_MODEL`. Aliases, read only when the matching primary is unset: `OPENAI_API_KEY` /
`OPENAI_BASE_URL` / `OPENAI_MODEL`, plus `GROQ_API_KEY` (a key alias that also implies
`LLM_PROVIDER=groq`; its base URL applies only when no base URL is set anywhere). Model falls
back to the built-in default; a per-call UI override still wins. The three `LLM_*` primaries are
deliberately unset in `.env.example` so an unset one cannot shadow a configured alias.

`load_dotenv(override=False)` normally lets the environment beat `.env`; the request-aiming names
are the deliberate exception — a non-blank value **in the file** wins over an export, so a `.env`
naming Groq is never silently ignored because the host exports `OPENAI_API_KEY` /
`OPENAI_BASE_URL` (trap + fix: `docs/README.md` §1.2). `resolved_config()` / `resolved_provider()`
report what was actually resolved — including which variable supplied the key — key redacted.
`docker compose` passes each override variable through empty so an unset one cannot shadow a
configured alias.

**Groq needs the local embedder, and a model that supports Structured Outputs.** Groq exposes
chat and responses but **no `/embeddings` endpoint**, so a Groq chat config must run
`EMBEDDING_PROVIDER=local` (the default) or an `EMBEDDING_BASE_URL` that serves `/embeddings` —
hence chat and embeddings are separate settings. The model id is not free choice: `analyze_question`
and `groundedness_check` send `response_format: json_schema`, which Groq supports on
`openai/gpt-oss-20b`, `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` (`strict: true`), best-effort
on those plus `openai/gpt-oss-safeguard-20b`, and **not at all** on `llama-3.3-70b-versatile` —
which would break both JSON nodes. Hence `LLM_MODEL=openai/gpt-oss-120b` in the worked example.

**Embedder — local is the default.** `EMBEDDING_PROVIDER=local` runs `LOCAL_EMBEDDING_MODEL`
(384-dim) in-process on ONNX Runtime, no key, no `/embeddings`; only `EMBEDDING_PROVIDER=api`
reads `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`. `MOCK_LLM` is checked first and this repo's
`.env` ships `MOCK_LLM=1`, so exercise the local default with `MOCK_LLM=0` or unset. Changing
provider or model changes vector width and invalidates any existing index — rebuild into a fresh
`--index-dir` or delete `data/index/`; a mismatch fails loudly rather than returning garbage.

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

The app can also build the index from the UI by uploading a PDF; the scripts are the
non-interactive path for reviewers and CI.

## Run — Docker (the grader's path)

```bash
git clone <repo> && cd <repo>
cp .env.example .env            # or: export MOCK_LLM=1 for the keyless demo
docker compose up --build       # http://localhost:8501
```

`data/` is a named volume — ingest and chat history survive `docker compose down` and rebuilds.
`docker compose logs -f app` streams the JSON event stream.

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
| Answer banner: "model busy, retrying..." then error | 429 / quota | wait or switch `LLM_BASE_URL` / `OPENAI_BASE_URL`; `MOCK_LLM=1` to demo keyless |
| Banner: "context too large" | wide decomposed question + high k | lower `k` in sidebar; graph auto-retries once with smaller budget |
| Banner: "index not built — upload a PDF" | `RetrievalError` on load | rebuild index; check `data/index/chunks.jsonl` exists |
| Per-file upload error card | scanned/corrupt PDF | expected for image-only PDFs — other files still ingest; warning names pages |
| Auth banner naming the chat key vars (`LLM_API_KEY` / `OPENAI_API_KEY` / `GROQ_API_KEY`) | bad/missing key, or an ambient key/alias winning | set `LLM_API_KEY` (+ `LLM_PROVIDER`) when host exports `OPENAI_*`; check `resolved_config()` for which variable supplied the key — never echoed |
| `docker compose up` port busy | 8501 in use | `ports: "8502:8501"` in compose |
| Prompt changed but behavior stale | prompt file cached at startup | restart the app; prompts load at startup via `load_prompt` |

## Where to look when something is wrong

1. UI debug panel — node trace, plan, retrieval scores for the answer.
2. `docker compose logs app` / stdout — JSON events with `trace_id`.
3. `data/chat.sqlite3` → `events` table — the same stream, queryable after the fact.
4. `docs/OBSERVABILITY.md` — event vocabulary and failure→UI mapping.
5. `docs/TESTING-AND-EVAL.md` — which hermetic test covers the broken component.

# IsipFin

IsipFin is a Streamlit chatbot that answers questions about financial PDFs with cited
answers. You upload annual reports and filings; it parses them into parent/child chunks,
indexes them with FAISS (dense) **and** BM25 (sparse), and answers through a 7-node
LangGraph pipeline that fuses both retrievers with Reciprocal Rank Fusion, constrains the
answer to the retrieved passages, and verifies every figure against the cited span before
returning it. Control (routing, caps, the score gate, fusion, citation mapping, a
deterministic numeric pre-pass) is symbolic and LLM-free; the LLM owns only semantics.
Answers carry page + section citations, or an honest refusal that names what the uploaded
documents cover — never a fabricated number.

---

## 1. Quick start (reviewer path — Docker, no API key)

The app has a deterministic, fully offline demo mode (`MOCK_LLM=1`): no API key, no
network. It needs Docker Desktop with the compose v2 plugin.

```bash
git clone <repo-url> isipfin
cd isipfin
cp .env.example .env
```

Now set `MOCK_LLM=1` in the `.env` you just copied (the example ships with `MOCK_LLM=0`;
change that one line). Then:

```bash
docker compose up --build
```

Open **http://localhost:8501**. The app starts with an empty state; upload a financial
PDF in the UI to begin (see §3 for the corpus).

Mock mode is deterministic and offline: the same question returns the same answer every
run, and no request leaves the container. It is the demo and test path, not an answer
engine — real answers need a chat key. To use a real model instead, set `LLM_PROVIDER` and
`LLM_API_KEY` in `.env`, leave `MOCK_LLM=0`, and rebuild. That is the whole configuration;
add `LLM_MODEL` when you want something other than the provider's default. Two worked
examples in the file, three lines each:

```bash
# Groq — LLM_PROVIDER selects Groq's own endpoint, so no base URL to set
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
LLM_MODEL=openai/gpt-oss-120b

# OpenAI — LLM_PROVIDER selects api.openai.com, also no base URL to set
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

The model name is provider-agnostic — `openai/gpt-oss-120b` (Groq),
`nvidia/nemotron-3-super-120b-a12b` (NVIDIA) and `llama3` (Ollama) all answer through the
same variable. Embeddings need no key at all: they default to a local model inside the
container (§1.1). If your shell already exports `OPENAI_API_KEY` or `OPENAI_BASE_URL`, say
so in `.env` too: a value you write there for those names now wins (see §Troubleshooting).
Runtime state (`data/`) is a named Docker volume, so uploaded PDFs, the built index, and
chat history survive `docker compose down` and rebuilds.

**Mock and real indexes are not interchangeable.** `MOCK_LLM=1` swaps in the
deterministic offline embedder, which vectors text at **64 dimensions**, while the
default local embedder (`BAAI/bge-small-en-v1.5`) works at **384** — so an index built
under one embedder cannot be read under the other. This is a hard error, not a silent
degrade: the loader refuses the width mismatch instead of returning garbage. A fresh
clone is unaffected, because `data/` is gitignored and the documented command builds the
index with the same embedder that later reads it. Only an existing checkout that already
built `data/index` with the local embedder trips this. Two ways out — pick whichever
suits:

- point `INDEX_DIR` at a mock index — `INDEX_DIR=data/index-mock` (run
  `MOCK_LLM=1 python scripts/build_index.py --index-dir data/index-mock` once to populate
  it), or
- rebuild the index in place under the mock embedder with
  `MOCK_LLM=1 python scripts/build_index.py`.

### 1.1 Two providers: Groq for chat, local for embeddings

Chat and embeddings are independent roles, each pointed at one OpenAI-compatible
endpoint. They may legitimately use different providers, and by default they do:

| Role | Default | Env vars | Notes |
|---|---|---|---|
| **Chat** | `LLM_PROVIDER=openai`, model `gpt-4o-mini` | `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` | any OpenAI-compatible chat endpoint |
| **Embeddings** | **local** `BAAI/bge-small-en-v1.5`, 384-dim (fastembed → ONNX Runtime) | `EMBEDDING_PROVIDER=local`, `LOCAL_EMBEDDING_MODEL` | runs in-process; no key, no embeddings endpoint, no PyTorch |
| Embeddings (API) | OpenAI `text-embedding-3-small` | `EMBEDDING_PROVIDER=api`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` | any endpoint that actually exposes `/embeddings` |

**Groq is the concrete reason the split exists.** Groq is OpenAI-compatible and fast for
chat, but its API exposes chat and responses and **no `/embeddings` endpoint** — so a Groq
chat config with the default API embedder would fail at ingest. Setting
`EMBEDDING_PROVIDER=local` keeps the dense leg in-process:

```bash
# Groq chat + local embeddings — the two-provider default
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...                                  # your Groq key
LLM_MODEL=openai/gpt-oss-120b
EMBEDDING_PROVIDER=local
MOCK_LLM=0
```

**The model choice on Groq is constrained, not free.** The graph's `analyze_question` and
`groundedness_check` nodes require `response_format: json_schema`, and Groq supports it
only on a handful of models: `strict: true` (constrained decoding, guaranteed schema
adherence) on `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`;
best-effort on those plus `openai/gpt-oss-safeguard-20b`. Every other Groq model — including
the popular `llama-3.3-70b-versatile` — does not support `json_schema` at all and would
break both JSON nodes. Hence `openai/gpt-oss-120b` in the example.

Our gateway does not send a `strict` key, so on Groq the nodes run in **best-effort**
mode: usually valid JSON, but adherence is not guaranteed and a schema-validation 400 is
possible. That is survivable rather than fatal because the gateway validates the payload
against the schema client-side (`QUERY_PLAN_SCHEMA` / `GROUNDING_SCHEMA`) and retries once
with a stricter nudge before failing the node — so the provider's mode is a *quality*
difference, never a correctness one.

The motive for Groq is throughput: a measured `analyze_question` call through this gateway
on a 120B model (NVIDIA Nemotron-3 Super 120B, `nvidia/nemotron-3-super-120b-a12b`,
temperature 0, 395 prompt / 246 completion tokens) took **11.2 s** and produced the correct
plan (`kind=simple, fiscal_period=FY2023`)
— fine as evidence, slow for a demo, which is exactly what Groq's token throughput fixes.

On first use the local embedder downloads its ONNX weights (~150 MB, cached) and every
later ingest and query reuses the same in-process model. `MOCK_LLM=1` still overrides
both providers with the deterministic offline embedder — that remains the keyless CI and
reviewer path. Because `MOCK_LLM` is checked *before* `EMBEDDING_PROVIDER` and this
repo's own `.env` ships `MOCK_LLM=1`, the local embeddings default is only exercised with
`MOCK_LLM=0` or unset.

`.env` is loaded with `load_dotenv(override=False)` (`src/utils/settings.py`), so as a
rule a **real environment variable beats the file**: exporting `EMBEDDING_PROVIDER`,
`INDEX_DIR` or `MOCK_LLM` on the host overrides whatever `.env` says, which is how CI and a
grader's machine inject a value without editing the file. The exception is deliberate and
narrow: the names that decide **which provider a request is aimed at** — `LLM_PROVIDER`,
`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` and their `OPENAI_*` / `GROQ_*` aliases — are
authoritative **in the file**, so a credential you typed into this repo's own config is
never silently overridden by an unrelated export. Everything else still loses to the
environment. A blank value in `.env` (as `.env.example` ships `LLM_API_KEY=`) is "unset
here" and never clobbers a real one.

### 1.2 Mock vs real credentials

The only thing to set for a real model is `LLM_PROVIDER` plus `LLM_API_KEY`; add
`LLM_MODEL` when the provider's default is not the model you want, and `LLM_BASE_URL` when
you aim chat at a custom, Azure or self-hosted endpoint instead of the provider's own.
Everything beyond that — which name wins when several are set, and what to do when the
shell already exports `OPENAI_*` — is a failure-mode detail, so it lives in §Troubleshooting
instead of the getting-started path. One consequence is worth stating up front, because it
bites silently: `GROQ_API_KEY` as an *alias* implies Groq's endpoint only when **no base
URL is configured anywhere**, and a base URL exported in your shell counts — a Groq key
sent to some other endpoint fails with an unhelpful 401/404. Naming the provider is what
removes that ambiguity.

Switching `EMBEDDING_PROVIDER` or the model changes the vector width and invalidates any
index built with the previous provider — build into a fresh `--index-dir` or delete
`data/index/`. The loader **fails loudly** on a width mismatch instead of returning
garbage (see §7).

**What you will see in mock mode, exactly.** The scripted gateway answers from canned
phrases, not from what retrieval found, so the grounding verifier (`groundedness_check`,
node 6) correctly detects that the scripted claims are not supported by the retrieved
passages. Expect answers carrying the **UNGROUNDED** badge, with unsupported sentences
trimmed — a scripted answer can be trimmed away entirely. That is the verifier working as
designed, not a broken install: it is the same mechanism that would strip a fabricated
figure from a real model's answer, demonstrated for free. For the reason in the harness's
own words, `MOCK_LLM=1 python scripts/eval_rag.py` prints which metrics are
*not meaningful in this mode* and why (the mock embedder is a sha256 hash, so dense scores
carry no semantic signal). Real answer quality — correct figures, citations, refusals that
name the covered periods — needs `MOCK_LLM=0` and a key.

Health and logs:

```bash
curl http://localhost:8501/_stcore/health   # Streamlit health endpoint
docker compose logs -f app                  # JSON event stream
```

---

## 2. What the pipeline does, and where the pieces live

`docs/ARCHITECTURE.md` is the full design contract; the summary below is the map.

**The 7-node graph** (`src/chains/graph.py`, `src/chains/nodes/`):

| # | Node | What it does | LLM? |
|---|---|---|---|
| 1 | `contextualize_question` | Rewrites a follow-up ("and the margin trend?") into a standalone question using chat history; falls back to the raw question on error. | yes |
| 2 | `analyze_question` | Classifies the question and emits a query plan — `simple`, `multi_part`, or `out_of_scope` — as JSON. | yes |
| 3 | `decompose_question` | Applies the plan's decomposition: one sub-query per part, capped at 4. Rule-based, no model call. | no |
| 4 | `hybrid_retrieve` | Runs FAISS + BM25 for each sub-query, RRF-fuses the rankings, expands child hits to their parent chunks. One bounded rewrite retry when the best fused score is below the gate. | no |
| 5 | `synthesize_answer` | Writes a citation-constrained answer from the retrieved parents. | yes |
| 6 | `groundedness_check` | Deterministic numeric pre-pass first, then an LLM semantic check; unsupported claims are trimmed and marked ungrounded. | yes (after pre-pass) |
| 7 | `refuse` | Honest, deterministic refusal that names the sections the uploaded docs do cover. | no |

Routing: `out_of_scope` skips retrieval and goes straight to `refuse`; `simple` goes
straight to `hybrid_retrieve`; `multi_part` is decomposed first. The score gate triggers
exactly one rewrite retry, never an unbounded loop.

**Hybrid retrieval** (`src/core/retrieval.py`, `src/core/index.py`): the dense side is a
FAISS `IndexFlatIP` over cosine-normalized child embeddings; the sparse side is
`rank-bm25` over child token lists; the two rankings are merged with Reciprocal Rank
Fusion (k=60). Each retrieved child is expanded to its parent chunk (~1000 tokens) so the
synthesizer sees table context, not a fragment. Ingestion is idempotent by sha256: a PDF
already in `data/index/docs.jsonl` is skipped, not re-embedded.

**Deterministic numeric pre-pass** (`src/chains/checks.py`): before any verifier model
call, a claim whose figure is absent from its cited span, or whose period label
contradicts the claim, is rejected and trimmed. This is the layer that blocks the classic
wrong-year error (a FY2024 figure asserted as FY2023) with zero LLM calls.

**Chat store** (`src/store/`): SQLite at `data/chat.sqlite3` — sessions, turns, messages,
citations, the document catalog, and an `events` table that mirrors the JSON log stream.
WAL mode, `busy_timeout` 5s. Everything the app writes lives under `data/`.

---

## 3. The corpus (bring your own PDF)

There is no downloadable official PDF for the exam's reference corpus. Microsoft's fiscal
2023 annual report and 10-K are served as **DOCX**, and SEC EDGAR's 10-K is **HTML-only**,
so a fresh clone has an empty `data/pdfs/` and the app starts in its "upload a PDF to
begin" empty state by design. Two ways to give it documents:

- **In the app:** upload one or more PDFs in the UI. The app parses, chunks, and indexes
  them in place; per-file errors (scanned/image-only PDFs) are reported without blocking
  the others.
- **On the native path:** drop a PDF into `data/pdfs/` and build the index:

  ```bash
  cp /path/to/your_report.pdf data/pdfs/
  python scripts/build_index.py            # local HF embeddings — no key, no network
  # or, the deterministic offline embedder (CI/demo):
  MOCK_LLM=1 python scripts/build_index.py
  # or, an OpenAI-compatible /embeddings endpoint (must expose /embeddings):
  EMBEDDING_PROVIDER=api OPENAI_API_KEY=sk-... python scripts/build_index.py
  ```

  Idempotent by sha256: re-running prints "no changes: every PDF was already indexed" for
  documents it has already ingested. `python scripts/download_corpus.py` has **no default
  URL** and must be given `--url`; it is not the corpus path here — supply your own PDF.

---

## 4. Repo layout

```
isipfin/
├── app.py                  # Streamlit entrypoint — thin UI shell, no business logic
├── src/
│   ├── chains/             # LangGraph pipeline: graph.py, state.py, checks.py, nodes/
│   ├── components/         # Streamlit UI: chat, upload, sidebar, theme
│   ├── core/               # PDF parsing, chunking, FAISS + BM25 index, hybrid retrieval
│   ├── store/              # SQLite chat store (sessions, turns, citations, events)
│   └── utils/              # LLM gateway, offline mock LLM, prompt loader, tracing
├── scripts/                # build_index, download_corpus, demo_script, eval_rag, check_floor
├── tests/                  # hermetic pytest suite (+ tests/data/golden_qa.jsonl)
├── docs/                   # SPEC, ADRs, RUNBOOK, ARCHITECTURE, JOURNAL, prompts/
├── data/                   # runtime state (gitignored): pdfs/, index/, chat.sqlite3
├── Dockerfile
├── docker-compose.yml
├── requirements.txt        # runtime pins (the only ones the image installs)
├── requirements-dev.txt    # toolchain: pytest/ruff/black/mypy + fixtures; not shipped
├── pyproject.toml          # ruff / black / mypy / pytest config
├── .env.example
└── README.md
```

Prompt assets are versioned markdown under `docs/prompts/` and loaded at runtime by
`src/utils/prompts.py` (`load_prompt("P1")`); there is no top-level `prompts/` directory.
The build journal — decisions, measurements, what was cut — is `docs/JOURNAL.md`.

---

## 5. Commands

Run from the repo root. `MOCK_LLM=1` before a command is POSIX shell syntax; in
PowerShell set `$env:MOCK_LLM=1` on its own line first.

| Task | Command |
|---|---|
| Run the app | `streamlit run app.py` |
| Run the app (Docker) | `docker compose up --build` |
| Build / refresh the index | `python scripts/build_index.py` (local embeddings; `MOCK_LLM=1` = offline, `EMBEDDING_PROVIDER=api` = API) |
| 6-step demo (offline) | `python scripts/demo_script.py` |
| Tests (hermetic) | `pytest -q` |
| Lint | `ruff check src tests scripts app.py` |
| Format check | `black --check src tests scripts app.py` |
| Types (scoped) | `mypy src/utils src/chains` |
| Floor guard | `python scripts/check_floor.py --repo-wide` |
| Golden-QA eval (offline) | `MOCK_LLM=1 python scripts/eval_rag.py` |

`pytest -q` is hermetic by contract: no test opens a socket, reads `.env`, or writes into
the repo tree — which is also why CI provides no `OPENAI_API_KEY` and still passes.
Install it with `pip install -r requirements-dev.txt` (that file pulls `requirements.txt`
first, so the runtime pins stay in one place). The image deliberately carries no toolchain,
so inside the container it installs to a scratch directory — the venv ships read-only as
part of the non-root runtime:

```bash
docker compose exec app sh -c \
  "pip install -q --target=/tmp/devtools -r requirements-dev.txt && \
   PYTHONPATH=/app:/tmp/devtools /tmp/devtools/bin/pytest -q"
```

(`requirements-dev.txt` is in the image for exactly this: the runtime pins are never
installed into the image, so nothing but the app's own dependencies ships.)

`scripts/eval_rag.py` is deliberately **not** a pytest module: it runs golden questions end
to end through the real `invoke()` over the persisted index and exits nonzero on a
regression (it needs the index built, and a real key for the real-LLM mode).

---

## 6. Troubleshooting

**First thing to check when a chat call fails with an auth error: what did the gateway
actually resolve?** `resolved_config()` / `resolved_provider()` (`src/utils/llm.py`) report
the provider, the base URL, the model and **which variable supplied the key**, with the key
always redacted. No client is built to call them, so they are safe to run anywhere. If the
key came from a name you did not configure, an ambient export is winning — the `OPENAI_*`
and `GROQ_*` aliases are read too, so they count as "configured" for this purpose.

**Second question: is `OPENAI_*` already set in your shell?** Run `echo $OPENAI_API_KEY`
(PowerShell: `$env:OPENAI_API_KEY`) and `echo $OPENAI_BASE_URL`. If either prints something,
that export is not ignored — but it is *read*, and it can be what answered instead of the
value you meant. While a real environment variable normally beats `.env`, the names that
aim a request are the deliberate exception: a non-blank value **in the file** wins over the
export (see §1.1).

**Precedence — first hit wins.** The `LLM_*` names are the primary surface; the aliases are
read only when the matching `LLM_*` name is unset, which is what keeps a host that already
exports `OPENAI_*` working:

| Setting | Resolution order |
|---|---|
| **provider** | `LLM_PROVIDER` → `groq` when the key came from `GROQ_API_KEY` → `openai` |
| **key** | `LLM_API_KEY` → `OPENAI_API_KEY` → `GROQ_API_KEY` |
| **base URL** | `LLM_BASE_URL` → `OPENAI_BASE_URL` → Groq's `https://api.groq.com/openai/v1` when the key came from `GROQ_API_KEY` → the provider's own endpoint |
| **model** | `LLM_MODEL` → `OPENAI_MODEL` → the provider's default model → `gpt-4o-mini` (a per-call model override in the UI still wins) |

Each of the four has one documented spelling in `.env.example`; the aliases are a
compatibility fallback, not a second surface to fill in. The reported provider is derived
from the endpoint the request would actually go to, so a key aimed at the wrong host shows
up as a mismatch rather than a silent success.

**The fix:** write the `LLM_*` names in `.env` — they take precedence over the aliases and
over an export, so the credential you mean is the one used. Naming `LLM_PROVIDER` also
removes the one ambiguity in the alias path: `GROQ_API_KEY` alone selects Groq's endpoint
only when **no base URL is configured anywhere**, and an `OPENAI_BASE_URL` exported in your
shell counts as "set anywhere" and stops that fallback — so set `LLM_PROVIDER=groq`
explicitly instead of relying on the key to imply it.

**A 400 on `response_format` / `json_schema`** is a model-choice problem, not a credential
one: pick a model that supports it (see §1.1).

---

## 7. Honest limitations

- **Mock mode's embedder carries no semantics.** It emits deterministic hash-derived
  vectors, so retrieval quality is not representative of the real path. The default
  `EMBEDDING_PROVIDER=local` embedder *is* a real semantic model, so a chat key plus
  local embeddings is enough for representative retrieval; only `MOCK_LLM=1` replaces it
  with the hash stand-in.
- **Section paths are ordering-based, not hierarchical.** They reflect page/heading order,
  not a reconstructed document outline.
- **No reranker.** Fusion is RRF over dense + sparse only; there is no cross-encoder pass
  over the fused candidates.
- **Vertical scale only.** One process, in-memory FAISS + BM25, single-file SQLite. The
  next step (a FastAPI-split graph, Postgres for the chat store) is named in the journal,
  not implemented.

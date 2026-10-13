# AGENTS.md

Rules and commands for any AI session (or a human) working in this repo. **Read
`docs/CONSTRAINTS.md` before writing code** — that file is the graded contract and this
file never overrides it.

The short version: this is a deterministic neuro-symbolic LangGraph RAG pipeline over a
financial PDF corpus. Symbolic layers own control (routing, caps, gates, fusion, citation
mapping, a deterministic numeric pre-pass); the LLM owns only semantics. When you add
something, decide which side it belongs to before writing a line.

---

## 1. Tech stack

Verified against `pyproject.toml` and `requirements.txt` — every pin is `==`; an unpinned or
guessed dependency is a finding, not a convenience.

| Layer | Choice | Where it lives |
|---|---|---|
| Language | Python **3.11** (pinned at three levels: `.python-version`, `requires-python == "3.11.*"`, installed via `.venv`) | — |
| Orchestration | **LangGraph** 1.2.14 (`langgraph` standalone, `StateGraph` + `MessagesState`); `langchain-core` 1.6.10 and `langchain-text-splitters` 1.1.3 for splitters only | `src/chains/` |
| LLM access | **openai** 3.29.0 — one OpenAI-compatible client via `base_url` (OpenAI / Azure / vLLM / Ollama / Groq); embeddings from the same provider | `src/utils/llm.py` |
| Offline mock | `MOCK_LLM=1` → deterministic, keyless, keyed-by-prompt-fingerprint scripted gateway | `src/utils/mock_llm.py` |
| Extraction | **pypdf** 6.20.1 (+ **pdfplumber** 0.11.10 for table pages) | `src/core/pdf_parser.py` |
| Splitting | `RecursiveCharacterTextSplitter` for children; heading-aligned parents | `src/core/chunking.py` |
| Retrieval | **FAISS** (faiss-cpu 1.15.1, `IndexFlatIP` cosine) + **rank-bm25** 0.2.2, RRF fusion (k=60), parent-document expansion | `src/core/retrieval.py`, `src/core/index.py` |
| Storage | **SQLite** (`data/chat.sqlite3`, 6 tables, WAL, `busy_timeout` 5s); index artifacts + parsed JSON under `data/` | `src/store/` |
| UI | **Streamlit** 1.65.0 — `app.py` is a thin UI-only shell (ADR-009) | `app.py`, `src/components/` (**not built yet**, T18–T21) |
| Observability | stdlib **`logging`** + JSON events + `@trace_node` per-node trace + in-app debug panel. No `structlog` dependency — do not add one (ADR-007). | `src/utils/tracing.py` |
| Packaging | Docker + docker-compose (one app service, named volume for `data/`), GitHub Actions (**not built yet**, T22–T23) | — |
| Tests | **pytest** 9.1.1 + **pytest-cov** 7.1.0 + **reportlab** 5.0.1 (in-memory PDF fixtures only) | `tests/` |
| Tooling | **ruff** 0.17.0, **black** 26.10.1, **mypy** 2.4.0 | — |

Config is in `pyproject.toml`: ruff `select = ["E","F","I","W","B","UP","SIM","RUF"]`,
line-length 100; black `line-length = 100`; mypy `strict = true` scoped to
`files = ["src/utils", "src/chains"]`. `pytest` uses `addopts = "-q"`,
`testpaths = ["tests"]`.

---

## 2. Commands

All of these were run in this repo on this checkout, with the venv activated. Use either
the activated-venv form shown below or prefix with `.venv/Scripts/python.exe -m` — both
work. Quoted output below is what a **clean** tree prints.

### Setup

```bash
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell; bash: source .venv/bin/activate
pip install -r requirements.txt
# the package is installed editable, so `import src.*` works from any cwd (verified).
cp .env.example .env                  # then edit .env; it is gitignored, never commit it
```

### Test

```bash
pytest -q                             # full hermetic suite, no network (304 when green)
pytest -q tests/test_gateway.py       # one file
pytest -q -k citation                 # focused by keyword (matches across files)
pytest --cov=src --cov-report=term-missing -q
```

**The suite is hermetic by contract.** No test may open a socket, read `.env`, or touch
`data/`. Fixtures build deterministic PDFs in memory with reportlab
(`tests/fixtures.py`); the gateway is exercised through the mock or an injected
transport. `testpaths = ["tests"]` also keeps `scripts/eval_rag.py` — the golden-QA layer,
which *does* need a key — out of the suite.

### Lint / format / types

```bash
ruff check src tests scripts app.py          # must print "All checks passed!"
black --check src tests scripts app.py       # must print "N files would be left unchanged"
black src tests scripts app.py               # the fixer — never hand-format
mypy src/utils src/chains                    # "Success: no issues found in 17 source files"
```

`mypy` is deliberately **scoped** to `src/utils` + `src/chains` (`docs/CONSTRAINTS.md`
§Adaptations): contract-bearing code is fully typed; running bare `mypy` over the whole
tree is not a gate. Do not "fix" this by widening the scope in `pyproject.toml` or by
adding an ignore.

### Floor guard

```bash
python scripts/check_floor.py --diff         # run at task end
python scripts/check_floor.py --repo-wide    # what CI runs
```

Scans `src/`, `tests/`, `scripts/`, `app.py` for `# noqa`, `# type: ignore`,
`raise NotImplementedError`, `TODO`, `FIXME`, `except:` + `pass`, `Not implemented`.
Exit 0 = clean. The literal `TODO` inside `docs/prompts/*.md` is a permitted
"record during build" marker; markdown is never scanned.

### Data / index

```bash
# Index build. MOCK_LLM=1 picks a deterministic offline embedder (64 dims, sha256-derived
# unit vectors) — no key, no network. Without it, a real OpenAI-compatible embedder is
# built and a missing OPENAI_API_KEY fails loudly before any PDF is parsed.
MOCK_LLM=1 python scripts/build_index.py                 # index every PDF in data/pdfs/
MOCK_LLM=1 python scripts/build_index.py --pdf PATH...   # index specific files (repeatable)
python scripts/build_index.py                            # real embeddings; needs OPENAI_API_KEY
```

Idempotent by sha256: a PDF whose digest is already in `data/index/docs.jsonl` is reported
`skipped` and a second run prints "no changes: every PDF was already indexed". Write
artifacts under `data/` only: `data/index/{faiss/,bm25.pkl,chunks.jsonl,docs.jsonl}`.

`scripts/download_corpus.py` has **no default source** and must be given `--url`:
Microsoft's ar23 download center serves a **DOCX** (`PK\x03\x04`) for both the Annual
Report and the 10-K, and SEC EDGAR's 10-K is HTML-only, so no official FY2023 PDF URL
exists. The real corpus path is dropping the PDF into `data/pdfs/` by hand — the script
adopts it, records its digest, and afterwards reports "already present".

```bash
python scripts/download_corpus.py --check --url <candidate>   # probe before installing
python scripts/download_corpus.py --url <candidate>           # install (magic-byte checked)
```

### Run

```bash
streamlit run app.py                       # http://localhost:8501
python scripts/demo_script.py              # 6-step demo script under MOCK_LLM=1
```

### Golden QA (second testing layer — deliberately *not* pytest)

`scripts/eval_rag.py` runs `tests/data/golden_qa.jsonl` end to end through real
`src.chains.graph.invoke()` calls over the real persisted index. It lives in `scripts/`,
so `pytest -q` never collects it (`testpaths = ["tests"]`), and nothing in it stubs the
graph.

```bash
MOCK_LLM=1 python scripts/eval_rag.py          # hermetic, offline, deterministic
python scripts/eval_rag.py                     # real LLM (needs OPENAI_API_KEY)
python scripts/eval_rag.py --only q01,q03      # a subset of cases
python scripts/eval_rag.py --json out.json     # machine-readable report
```

Exit codes: `0` every meaningful metric met its bar · `1` a golden regression (a bar missed,
the wrong-year canary fired, or a case raised) · `2` harness error (no cases, unreadable
dataset, unresolvable `follow_up_of`, missing index). Bars per
`docs/CONSTRAINTS.md`: hit-rate@5 ≥ 0.9, citation presence 1.0, figure match ≥ 0.95,
wrong-year contamination **0**, refusal correctness 1.0. In mock mode some cells read
`N/A` **with the reason printed** — no bar is ever lowered, and the lexical wrong-year
canary stays live in both modes.

### Not built yet (do not invent a command for these)

`Dockerfile` / `docker-compose.yml` (T22), `.github/workflows/ci.yml` (T23), `README.md`
(T24), and `src/components/*` plus the Streamlit body of `app.py` (T18–T21). A top-level
`prompts/` directory does not exist either — the prompt assets live in `docs/prompts/`,
deliberately (see §3). `pip install -e .` is not a path used anywhere; the package is
installed editable once at setup so `import src.*` resolves from any cwd.

---

## 3. Code conventions

These are what the shipped code actually does — reuse them rather than starting a second
convention.

**Modules and docstrings**

- Every module opens with a docstring that states its contract *and* cites the source
  (`ADR-00N`, `docs/SPEC.md §9`, `docs/STRESS-TEST-RAG.md row N`). A comment explains
  **why** (trade-offs, invariants, gotchas) and never restates the code.
- `from __future__ import annotations` at the top of every module — all 31 non-empty
  modules (the 8 `__init__.py` package markers are empty and carry no imports). Type hints
  on every public function.
- `__all__` in modules with an explicit public surface (13 of the 20 non-`__init__.py`
  modules under `src/`; the rest export their whole surface).
- Sections inside a module are separated by `# --- title ---` banners; private helpers
  are `_name`; constants are `SCREAMING_SNAKE` with a `#:` doc comment above them when
  their meaning is not obvious.

**Data and typing**

- Frozen dataclasses (`@dataclass(frozen=True)`) for contracts — 27 of them across
  `src/`. `src/chains/state.py` holds `GraphState` as a `TypedDict(total=False)`; nodes
  read with `state.get(...)` and return **complete** values, not deltas.
- No `Annotated[..., operator.add]` reducers anywhere, and `add_messages` is deliberately
  unused. `node_trace` is the single accreting field and `@trace_node` owns it — a node
  must never include `node_trace` in its partial update.
- langgraph resolves `StateGraph` annotations at construction time, so every type
  referenced in `GraphState` is imported **at runtime**, never under `TYPE_CHECKING`
  (a deferred name raises `NameError`).
- `Protocol` + `@runtime_checkable` for injected collaborators (`ChatClient` in
  `src/utils/llm.py`) so a node tests against either the real client or the mock with no
  import-time branch.

**Errors**

- Typed exceptions per module, each with a short docstring saying what the failure is:
  `LLMError` / `RateLimited` / `ContextLength` / `Auth` / `Timeout` /
  `SchemaViolationError` (`src/utils/llm.py`), `GraphError` / `GraphNodeError` /
  `GraphStateError` (`src/chains/graph.py`), `PromptError` family
  (`src/utils/prompts.py`), `ParseError`, `RetrievalError`, `StoreError` /
  `SessionNotFoundError`, `BuildError`, `DownloadError`.
- There is **no `src/utils/errors.py`** and no `src/utils/settings.py`. The gateway
  taxonomy lives in `src/utils/llm.py` and env reading happens in
  `src/utils/llm.py` / `scripts/build_index.py` / `src/store/db.py`. Do not create either
  file or import from one.
- Never swallow a failure into an empty result. `except: pass` and bare `raise
  NotImplementedError` are floor-guard hits. A degraded path degrades **visibly** —
  e.g. a planner failure produces `planner_degraded=True` plus an honest out-of-scope
  refusal, never a blank answer.

**Logging**

- stdlib `logging` only: `logger = logging.getLogger(__name__)` at module level
  (7 modules — `src/chains/nodes/analyze.py`, `contextualize.py`, `decompose.py`,
  `retrieve.py`, `synthesize.py`, `verify.py`, `src/core/retrieval.py`). No `structlog`
  import and no third-party logging dependency. Use lazy `%s` interpolation, name the node
  or operation, and truncate untrusted text (`question=%r` with `question[:120]`).
  Timestamps come from one helper (`src/store/db.py: utc_now_iso`).

**Prompts**

- No inline system prompts in `src/` — load them: `load_prompt("P3")`. The assets live in
  `docs/prompts/*.md` (P1–P5) with version headers, and `scripts/prompts_meta.json`
  records the minimum version each node requires (asserted via `assert_min_version`).
  Paths resolve from the module, never from the cwd.

**Placement**

- Control-outcome code (routing, caps, gates, budget, citation mapping, numeric checks)
  must stay deterministic and LLM-free. `src/chains/checks.py` is the symbolic layer and
  has **zero** gateway calls — keep it that way.
- Every new capability needs a fan-out across layers: node / task / test. Never ship one
  layer alone (the failure `docs/PLAN-REVIEW.md` already caught once).
- Nothing outside `data/` is written at runtime.

**Tests**

- Named for the behavior, not the function — real examples from the suite:
  `test_dehyphenation_joins_broken_word`, `test_rate_limit_exhausts_three_attempts_then_raises_rate_limited`,
  `test_a_rate_limit_that_recovers_answers_without_raising`.
- Hermetic: no network, no real `.env`, no writes into the repo tree. Mock via
  `monkeypatch` / the `MOCK_LLM=1` path / an injected transport.
- A test must **fail without the change and pass with it** (`docs/DEFINITION-OF-DONE.md`).

---

## 4. Boundaries

### Always

- Run `pytest -q` + `ruff` + `black` + `check_floor.py --diff` **before** any commit
  (`docs/CONSTRAINTS.md` build-time checklist).
- Ground every generated claim to a retrieved chunk, or mark it ungrounded. Log retrieval
  scores.
- Keep prompts versioned in `docs/prompts/` and referenced by id.
- Update `docs/SPEC.md`, ADRs and `docs/REPO-STRUCTURE.md` **in the same change** when a
  contract moves — never after.

### Ask first (in a real team)

- Adding a paid dependency or any new pin.
- Changing the citation format, the chunking contract, or the embedding provider.
- Weakening `docs/SPEC.md` §10 success criteria or `docs/CONSTRAINTS.md` — loosening a bar
  is a reviewed decision, never a fix for a failing check.

### Never

- Commit `data/`, `*.pdf`, or `.env`. `.env` is gitignored and `OPENAI_API_KEY` never
  appears in a committed file.
- Add a suppression comment (`# noqa`, `# type: ignore`, `# ruff: noqa`), a stub, a skipped
  test, or a silent `except` swallow. Fix the underlying cause.
- Let a failure path return an empty answer (ADR-007's one hard rule).
- Return answers without citations when grounding exists, or a fabricated number for an
  unsupported question — refuse honestly and name what the documents cover.
- Touch the network from `tests/`, import a script from `tests/`, or put business logic in
  `app.py` / `src/components/`.

---

## 5. Working in this repo

- `docs/SPEC.md` §9 freezes the interface contracts — change the spec first, then code.
- `docs/PLAN.md` (T1–T26) and `docs/TODO.md` are the task record. Phases 1–3 are landed
  (T2–T16 merged); T17's harness has landed and passes in mock mode; phases 4–6
  (T18–T26) are not started.
- Reference corpus: Microsoft FY2023 Annual Report, fiscal year ended June 30, 2023. The
  headline invariants pinned by golden QA and docs: FY2023 revenue **$211,915M**, FY2023
  operating income **$88,523M** vs FY2022 **$83,383M**. A FY2024 figure asserted as FY2023
  must be rejected by `src/chains/checks.py` with **zero** LLM calls.
- Golden-QA regression canary: *"What was Microsoft's revenue in FY2023?"* → $211,915M with
  a citation, not the FY2022 or FY2024 comparative.

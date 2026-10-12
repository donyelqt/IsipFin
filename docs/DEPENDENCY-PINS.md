# Dependency Pins

> Verified against PyPI on **2026-10-13** by reading each project's page. Every number below came
> from a page read this session. Re-verification protocol at T1 is at the bottom — pins are
> re-checked, never guessed.

## Pin table

| Package | Verified version | Why needed | Risk if unpinned |
|---|---|---|---|
| `streamlit` | 1.65.0 | the UI framework (exam requirement) | API drift breaks `st.session_state`/chat components mid-build |
| `langgraph` | 1.2.14 | orchestration (`StateGraph`) | state API is young; a minor bump can change node return semantics |
| `langchain-core` | pin transitively via langgraph — verify at T1 | `ChatOpenAI`, splitter, embeddings wrappers | splitter/embedding wrappers change between minors |
| `faiss-cpu` | 1.15.1 | dense vector index (brief's vector-DB requirement) | index file format changes can invalidate rebuilt artifacts |
| `rank-bm25` | 0.2.2 | sparse keyword retrieval | tiny API (Okapi) — low risk, pin anyway for reproducibility |
| `pypdf` | 6.20.1 | per-page text extraction | extraction quirks change per version; cleaning rules are tuned against a version |
| `pdfplumber` | 0.11.10 | table → markdown extraction | table-detection heuristics are version-sensitive |
| `openai` | 3.29.0 | OpenAI-compatible chat + embeddings client | client-side retry/response-shape changes |
| `python-dotenv` | 1.2.4 | `.env` loading | trivial, but a secret-handling bug here is unacceptable |
| `pytest` | 9.1.1 | test suite | plugin compatibility |
| `ruff` | 0.17.0 | lint | rule set changes between versions produce false failures |
| `black` | 26.10.1 | format | formatting drift breaks `black --check` in CI |

`requirements.txt` uses `==` pins for every row. `pip install -U` language appears **only** in
this document's discovery protocol, never in the file the grader installs.

**Unverified / decide at T1:** `langchain-core` (pinned transitively by langgraph; record the
resolved version from `pip install langgraph` and freeze it explicitly), `mypy` (add only if the
scoped-types constraint is kept — `docs/CONSTRAINTS.md` scopes it to `src/utils` + `src/chains`).

## Base image

**Recommendation: `python:3.11-slim`** (matches `docs/adr/ADR-008`). Decision factor is
`faiss-cpu` wheel availability on `cp311` — the 1.15.1 page declares `Python >=3.10` with cp311
wheels, and pdfplumber is tested through 3.13, so 3.11 is the conservative intersection of
every dependency's wheel matrix. Re-verify at T22 if the build pulls 3.12 wheels for a pinned
transitive dependency; changing the base image is a one-line change, not a re-architecture.

## Floor guard wiring

The floor from `docs/CONSTRAINTS.md` must be mechanically checkable. Mechanism:
`scripts/check_floor.py`, created at **T1**, run by CI at **T23** as the first job step and at
task end against the diff.

Patterns (any hit in `src/`, `tests/`, `scripts/`, `app.py` → exit nonzero):

```
# noqa
# type: ignore
raise NotImplementedError
TODO
FIXME
except:\s*$
pass            # only when the previous line is bare `except:`
Not implemented
```

Implementation notes:

- diff mode: `git diff` against the branch point, scan only added lines — under a second for a
  46-hour build's diffs.
- repo-wide mode in CI: full scan of the four paths.
- Exclusion: `prompts/*.md` (markdown, may contain deliberate "TODO during build" markers per
  `docs/prompts/README.md`); `docs/**` is documentation, never scanned.
- Report rule and location, never file contents of a matched secret line.

## Version-verification protocol (at T1)

1. Read the PyPI page of each direct dependency; record version + date in this table.
2. `pip install -r requirements.txt` into a clean venv; if a transitive conflict appears, cap it
   with an explicit pin in `requirements.txt` and a one-line comment naming why.
3. Record the resolved set (`pip freeze`) inside `requirements.txt` header comments — not as a
   second source of truth, as provenance.
4. At every later task end, the only version change allowed is a deliberate bump recorded here
   with the reason. An accidental dependency change is a finding, not a detail.

Rationale: a reviewer clones months from now; unpinned installs are how "works on my machine"
becomes "doesn't run at all." This is one of the exam's stated values ("Reproducibility").

# ADR-009: Streamlit as the native UI — no React, no API layer

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none

## Context

The deliverable's surface is a chat UI: upload a PDF, ask questions, see cited answers, change
model parameters, switch sessions. The exam brief makes the choice itself — *"you'll build a
**Streamlit-powered chatbot**"* and the deliverable is *"a user-friendly interface for uploading
PDFs and asking questions."* Streamlit is named, not suggested.

The open decision is everything *around* that: a React SPA served by a FastAPI backend, a bare
Streamlit app with the pipeline imported directly, or an API-first Streamlit hybrid.

## Decision

**Streamlit native, single process, no API layer, no second frontend framework.**
`app.py` + `src/components/*` are the entire UI; the graph is imported as a library
(`orchestration.invoke(question, history, params)`) and called from the UI layer.

Three reasons, in order of strength:

1. **The brief names Streamlit.** It is the graded deliverable, and it covers every UI requirement
   in the bullet list — `st.chat_message`/`st.chat_input` for chat, `st.file_uploader` for
   multi-PDF upload, `st.session_state` for per-session wiring, sidebar `st.*` controls for
   model parameters, `st.expander` for the debug panel. There is no UI requirement in the exam
   that a Streamlit primitive does not cover.
2. **One language, one runtime, one dependency tree.** Python end-to-end means one lockfile
   (`docs/DEPENDENCY-PINS.md`), one test runner (`pytest`, no network), one Docker stage, one
   CI pipeline. A React+FastAPI split doubles the build systems (npm/vite + pip), needs a second
   set of lockfiles, adds a serialization contract (Pydantic schemas the UI must agree with),
   CORS, and a whole integration-test surface — all for a single-user graded demo with zero
   grading credit attached to any of it.
3. **Full reversibility at near-zero cost.** Because the graph is a pure library call behind
   `AnswerEnvelope`, adding a FastAPI service *later* is additive: wrap `invoke()` in an endpoint.
   No pipeline, prompt, retrieval, or store code changes. The boring-safe option is the one that
   keeps the expensive part (the graph) independent of the delivery surface from day one.

## On the "code quality" argument — stated honestly

The tempting framing is "Streamlit improves code quality." **That claim is wrong on its own, and
this ADR does not make it.** Streamlit's top-to-bottom rerun model *punishes* mixed UI and logic:
state embedded in widget state, side effects during render, and duplicate work on every rerun are
the classic quality traps, and the framework permits all of them.

What actually preserves quality here is a **structural rule**, already frozen in
`docs/REPO-STRUCTURE.md` and `docs/ARCHITECTURE.md`:

- No business logic in `app.py` or `src/components/` — UI is a thin adapter over
  `orchestration.invoke()`. Retrieval, prompts, chunking, and the LLM gateway never import
  Streamlit.
- Durable state lives in `chat-store` and index artifacts on the volume, never in widget state;
  `st.session_state` holds only per-session handle wiring.
- Therefore the entire pipeline remains covered by the hermetic `pytest` suite
  (`docs/TESTING-AND-EVAL.md`), and UI changes cannot silently alter answers — the two layers are
  tested by different means and the boundary rule keeps them that way.

So: Streamlit buys **delivery speed and a single-language surface**; the boundary rule buys
**quality**. The framework does neither by itself, and this ADR is the record that the design
knows the difference.

## Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| React SPA + FastAPI backend | Rich UI control; real API surface; horizontally scalable | Two languages, two build systems, two lockfiles, serialization contract, CORS, second test layer — for a one-user demo the brief never asks to be a SPA | rejected |
| Gradio | Even faster chat UI than Streamlit | Weaker session/state story, less control over sidebar/debug layout, same "framework wants your layout" constraint | rejected |
| Next.js frontend + Python API | Type-safety across the boundary | The heaviest option: Node runtime + proxy + two deploy units; no requirement in the brief needs it | rejected |
| API-first Streamlit hybrid (FastAPI + Streamlit talking over HTTP) | "Real" separation | Adds an HTTP hop between UI and graph in-container; the boundary rule already gives separation at the import level for free | rejected |
| **Streamlit native, graph imported (chosen)** | Single language; brief-native primitives; reversible; testable boundary | No component-level UI tests; DOM styling limited; rerun model demands discipline | **accepted** |

## Consequences

- Single container, single process, `docker compose up` is the whole deployment story (ADR-008).
- The pipeline's test coverage is unaffected by UI work — the boundary rule is what makes that
  true, and it is reviewable (any `import streamlit` outside `app.py`/`src/components/` is a task
  not-done per `docs/DEFINITION-OF-DONE.md`).
- Accepted costs, stated: no automated UI/component tests (manual demo script is the UI gate);
  UI polish limited to what Streamlit primitives allow; `st.chat_message`'s `name` argument is an
  **accessibility label** (per Streamlit 1.65.0 docs), not displayed text — the UI sets it
  properly on every message.
- The rerun model is designed around, not fought: every UI action is idempotent and re-derives
  state from the store; nothing important is computed twice.
- Upgrade path exists and is documented rather than built: FastAPI wrapper over `invoke()` if the
  exam were instead a platform product; React only if a real multi-user UI were ever graded.

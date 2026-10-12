# Deliverables Matrix

> Exam requirement → design artifact → verification. Every row is checkable by a reviewer
> without taking anything on trust. Task ids reference `docs/PLAN.md`.

## Deliverable 1 — Functional application

| Sub-requirement | Artifact | Verification |
|---|---|---|
| a. User-friendly interface for uploading PDFs and asking questions | `app.py` + `src/components/{chat,upload}.py` (T18) | Demo script steps 1–3: upload card → chat turn → answer |
| b. Integration with an LLM for answering based on PDF content | `llm-gateway` + `synthesize_answer` (T6, T12) | Revenue question returns $211,915M cited; `MOCK_LLM=1` keyless path documented |
| c. Clear, concise answer display | citation cards + P3/P4 formatting contract (T12, T8) | Answers ≤4 sentences or a short markdown table; citation cards expand to page + snippet |

## Deliverable 2 — Codebase

| Sub-requirement | Artifact | Verification |
|---|---|---|
| a. Well-structured, documented, on a private GitHub repo | `docs/REPO-STRUCTURE.md`; README (T24); ADRs (8) | Tree matches the doc; repo private with 3 collaborators (T26) |
| b. Efficient workflow orchestration + prompt engineering | LangGraph graph (T16); `prompts/` P1–P5 with versioned protocol (T8) | Routing table in ADR-001 matches code; each prompt file versioned with changelog |
| c. Proper error handling and observability | typed errors (T6), `@trace_node` (T9), debug panel (T20), JSON logs (T9) | Demo: unset key → banner; debug panel renders node trace; `docker compose logs` JSON |
| Vector database retrieval (brief) | FAISS `data/index/faiss/` (T4) | Retrieval probe script output; debug panel dense/sparse/fused scores |
| Document processing pipeline (brief) | `doc-ingest` (T2) | Test: table blocks detected, cleaning invariants hold |
| Query decomposition (brief) | `analyze_question` + `decompose_question` (T10, T10b) | Comparison question fans out; debug panel shows sub-queries |
| Follow-ups/context (brief) | `contextualize_question` (T14) | "and the margin trend?" resolves; golden follow-up pair passes |
| Multi-document upload (brief) | idempotent ingest + filters (T2, T11) | Upload 2 PDFs; cross-doc question cites both |
| Model parameter controls (brief) | sidebar → `GraphState.params` (T19) | Change model/temp; next answer + recorded params reflect it |
| Chat history management + persistence (brief) | `chat-store` + session manager (T7, T19) | Reload keeps sessions/citations; switch/rename/delete works |
| Containerization (brief) | Dockerfile + compose (T22) | Cold clone → `docker compose up --build`; rebuild keeps data |
| Tests | hermetic pytest suite (T2–T16) | `pytest -q` green in CI without network |

## Deliverable 3 — Project journal

| Sub-requirement | Artifact | Verification |
|---|---|---|
| Design decisions, challenges, solutions | `docs/JOURNAL.md` §§3–5 | Every ADR referenced; each hard problem named with the rejected alternatives |
| Architecture choices + trade-offs | `docs/JOURNAL.md` §3 + ADR-003 (vector-vs-vectorless with the comparison table) | The comparison exists with honest, reasoned numbers |
| Iterative refinement shown | prompt changelogs (`docs/prompts/README.md`) + journal §5 iteration template | At least one prompt version bumped with observed failure and fix; "unchanged" is also recorded honestly |
| Prompt engineering strategies | P1–P5 with per-prompt rationale | Each prompt names the failure it defends against |
| AI-use compliance | `docs/JOURNAL.md` §2 | What AI was used for, what wasn't, what was verified against primary sources |
| Measurements | `docs/JOURNAL.md` §6 | Filled from eval output/debug panel — not estimated |

## Supporting evidence (not required, strong signal)

| Artifact | Why it earns credit |
|---|---|
| `docs/STRESS-TEST-RAG.md` | Adversarial review of the design: 8 ranked financial-QA failure modes with mitigations |
| `docs/STRESS-TEST-DELIVERY.md` | Delivery risk register + demo script + kill-switch order |
| `docs/GRADING-ALIGNMENT.md` | Every brief bullet → design element → demo evidence; over/under-built check |
| `docs/adr/ADR-003` | Vectorless RAG evaluated and rejected with reasoning — the "did they research it?" answer |

## Submission checklist (T26)

Prereq: `gh auth login -w -s repo` (once). Then, from the repo root:

```bash
# 1. Create the PRIVATE repo and push main  (repo does not exist yet — 404 verified 2026-10-13)
gh repo create IsipFin --private --source=. --remote=origin --push

# 2. Add the three reviewers from the brief with push access
gh api -X PUT repos/donyelqt/IsipFin/collaborators/crcheng   -f permission=push
gh api -X PUT repos/donyelqt/IsipFin/collaborators/levymedina -f permission=push
gh api -X PUT repos/donyelqt/IsipFin/collaborators/jgcosme   -f permission=push
```

Guard before every push — nothing runtime-related may be tracked:
`git ls-files | grep -E '^(data/|\.env$|.*\.pdf$)'` must return **nothing**.

- [ ] Private GitHub repository
- [ ] Collaborators added: `crcheng`, `levymedina`, `jgcosme`
- [ ] Code + diagrams + text files pushed (no `data/`, no `.env`, no PDFs)
- [ ] Backup copy uploaded to the shared Google Drive folder
- [ ] CI green on the final commit
- [ ] Demo script (from `docs/STRESS-TEST-DELIVERY.md`) run green the same day
- [ ] Journal §6 filled with real measured numbers

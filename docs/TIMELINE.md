# Timeline — 46-hour sprint

> Sprint starts fully rested at 00:00 SGT; 48h exam clock, ~46h working budget after setup.
> Hours are cumulative-from-start. Each block ends with the stated verification; a red
> checkpoint means the next block does not start.

| Hours | Block | Tasks | Exit verification |
|---|---|---|---|
| 00:00–00:30 | Read brief, confirm plan docs exist, start sleep-cycle-aligned work session | — | `docs/PLAN.md` + `docs/TODO.md` open; environment ready |
| 00:30–01:00 | Scaffold repo + venv + installs | T1 | `pip install -r requirements.txt`; `pytest -q` green (empty); `ruff check` pass |
| 01:00–03:30 | `doc-ingest` + `chunking-index` | T2–T4 | Unit tests green: table blocks detected, parent/child linkage, overlap, page ranges |
| 03:30–04:00 | Corpus download + index build scripts | T5 | Second run is a no-op; `data/index/` populated |
| 04:00–04:30 | **Checkpoint 1** | — | 10-query retrieval probe printed with sane fused ranks; commit |
| 04:30–06:00 | `llm-gateway` (+ `MOCK_LLM`) | T6 | Unit tests + mock-mode revenue call deterministic |
| 06:00–07:00 | `chat-store` schema + accessors | T7 | Session lifecycle + reload assertions pass |
| 07:00–08:00 | Prompt assets v1 | T8 | 5 prompts with version headers; referenced schemas exist |
| 08:00–10:30 | Graph skeleton + `analyze_question` + `decompose_question` + routing | T9, T10, T10b, T11 | Classification fixture 12/12; decomposition operand-splitting rejected; fan-out ≤4; single retry bound |
| 10:30–13:00 | `synthesize_answer` + golden-QA loop (this is where prompt iteration happens — expect 2–3 iterations) | T12, T17 | Golden subset passes; year-confusion case answered $211,915M |
| 13:00–14:30 | `groundedness_check` + `contextualize_question` + `refuse` | T13–T15 | Trim/flag test; follow-up pairs; both refusal classes |
| 14:30–15:30 | Graph wiring + full envelope | T16 | Demo script steps 3–6 green in mock mode |
| 15:30–16:00 | Sleep break (non-negotiable — night-owl cycle protection) | — | — |
| 16:00–17:00 | **Checkpoint 2** + real-LLM golden baseline | T17 | Real-model golden run; baseline numbers written into journal seed; commit |
| 17:00–19:00 | `app-ui`: shell, chat, upload, empty state | T18 | Manual: upload → question → cited answer |
| 19:00–20:30 | Sidebar params + session manager + debug panel | T19, T20 | Param change visible in next answer; reload keeps sessions; historical traces render |
| 20:30–21:30 | Error surfacing | T21 | Missing key and corrupt PDF behave per spec; no blank answers |
| 21:30–22:00 | **Checkpoint 3** — full demo script, real + mock | — | 6 demo steps green both modes; commit |
| 22:00–23:00 | Dockerfile + compose | T22 | Cold build; volume survives rebuild with history |
| 23:00–23:30 | CI workflow | T23 | CI green on push |
| 23:30–00:00 | README | T24 | Cold-clone dry run following README verbatim |
| 00:00–00:30 | **Checkpoint 4** — packaging gate | — | CI green + clone reproduction green; commit |
| 00:30–02:00 | Golden-QA tuning pass (chunk sizes, `tau`, prompt final tweaks) + record numbers | T17 re-run | Hit-rate stable or improved; numbers in journal |
| 02:00–04:30 | **JOURNAL.md** (weighted deliverable — full session) | T25 | ADR-linked, measured data, prompt iterations, cut-list, AI-use disclosure |
| 04:30–05:00 | Buffer / kill-switch decisions if anything is red | kill list | Cut list applied if needed; Musts all green |
| 05:00–05:30 | Submission: repo privacy + 3 collaborators + final CI | T26 | Collaborators added; final demo + CI green |
| 05:30–06:00 | Final gate: re-run demo script, verify all Musts, close todos | — | `docs/TODO.md` fully checked; final commit |

## Buffer policy

- 1.0h explicit buffer (04:30) plus inter-block slack; if a checkpoint goes red, the fix
  consumes buffer before touching the next block.
- Second sleep break is optional; if skipping it, the 04:30 buffer shrinks correspondingly —
  record the trade in the journal rather than pretending the schedule held.
- Kill-switch order (from `docs/STRESS-TEST-DELIVERY.md`) activates at whichever block is red
  with < 2h left; Musts are never cut.

# Definition of Done

> Project-wide bar for every task T1–T26 in `docs/PLAN.md`. **A task is done only when its own
> acceptance criteria pass AND this list passes.** Acceptance criteria answer "did we build the
> right thing?"; this answers "is it finished to standard?" Unlike acceptance criteria, this
> document is fixed and reused — it does not get renegotiated per task.

## Correctness

- [ ] The task's acceptance criteria in `docs/PLAN.md` are met, each observable
- [ ] The change runs and behaves as intended — verified at runtime (a script, a Streamlit session, a CLI run), not just by reading the diff or typechecking
- [ ] New behavior is covered by tests that **fail without the change and pass with it**
- [ ] Existing tests still pass; no regressions
- [ ] Edge and error paths are handled: the failure was exercised (e.g. a rate-limit response, a corrupt PDF, an empty index), not only the happy path

## Quality

- [ ] Code reveals intent through naming and structure; comments explain *why*, not *what*
- [ ] No duplicated business logic (retrieval scoring exists once, citation formatting exists once)
- [ ] No dead code, debug prints, or commented-out blocks left behind
- [ ] The change is scoped to its task; no adjacent refactors snuck in
- [ ] `ruff` and `black` clean per `docs/CONSTRAINTS.md`
- [ ] The floor guard passes (no `noqa`, no stubs, no skipped tests, no secrets)

## Integration

- [ ] Works with the rest of the system: a new node runs inside the compiled graph, not just in isolation
- [ ] Index artifacts are consistent with `data/index/chunks.jsonl` (the source of truth); FAISS/BM25 reload reproduces retrieval
- [ ] The compose volume keeps data across rebuilds (data surviving `docker compose down && up --build`)
- [ ] No public interface changed without updating its callers (e.g. `AnswerEnvelope` shape changes update the UI and tests together)

## Documentation

- [ ] Public interfaces are documented (dataclasses, `retrieve()`, `invoke()` contracts match `docs/SPEC.md` §9)
- [ ] Any decision that changed is recorded in its ADR — a new ADR if the decision is new, an edit if the rationale deepened
- [ ] Docs describe the current state; this plan set is updated in the same change, not after
- [ ] `docs/README.md` index still matches reality

## Exam-specific gates

The four things that make "done" mean done for *this* submission, not just for the code:

1. **Golden-QA green** — `scripts/eval_rag.py` passes including the year-confusion canary
   ("revenue in FY2023" → $211,915M, not the FY2022 or FY2024 comparative).
2. **Demo script green** — the 6-step sequence in `docs/STRESS-TEST-DELIVERY.md` runs in both
   real and `MOCK_LLM=1` modes.
3. **Docker path builds** — `docker build` succeeds; the grader's cold-clone path is the one
   that must work.
4. **No empty journal slots** — every `[FILL AFTER BUILD]` slot in `docs/JOURNAL.md` is filled
   with measured output from the run, or the claim it supported is removed from the docs.

## Not done despite looking done

Red flags. Any one of these means the task is open:

- Tests pass but a journal number was estimated or invented (see `docs/JOURNAL.md` §6 rule)
- T12 marked done without running the golden set
- T22 marked done without an actual cold clone (`git clone` → `docker compose up --build`)
- A `[FILL AFTER BUILD]` marker still in `docs/JOURNAL.md` at submission time
- "It runs on my machine" without the README path having been followed verbatim
- The decomposition capability missing from any layer (node, task, tree, test) — caught by a
  plan-wide audit that found `decompose_question` as an architecture node with no task, file or
  test (fixed as T10b across PLAN/TODO/TIMELINE/TESTING/DELIVERABLES/GRADING + `src/chains/nodes/decompose.py`)

This list complements per-task acceptance criteria and never replaces them; every task in
`docs/PLAN.md` and `docs/TODO.md` is checked against both bars.

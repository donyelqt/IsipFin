# Local inference record: `qwen2.5:3b` on Ollama — what was measured, and why a fully-local run is not viable

> Evaluation record, not a how-to. This file documents one investigation that asked whether
> this pipeline can run end to end, keyless and local, on `qwen2.5:3b` served by Ollama.
> **The answer is no: it can serve P2 (`contextualize_question`) only.** The local path is
> therefore *not* a supported configuration of this project — `docs/SPEC.md` points here for
> that reason. Numbers below were measured on this host against the real persisted index
> (`INDEX_DIR=data/index`) and the real Microsoft FY2023 corpus; none are estimates, and the
> ones that could not be measured are named rather than carried.

## 1. What was tested

| | |
|---|---|
| Server | Ollama **0.40.3**, OpenAI-compatible endpoint `http://localhost:11434/v1` |
| Model | `qwen2.5:3b` (1.93 GB) |
| Method | the repo's real graph nodes invoked directly, per-node, against the real persisted index; no mocks, no scripted gateway |
| Baseline prompts | the committed assets in `docs/prompts/`, used verbatim |

One positive result first, because it bounds everything else: Ollama's OpenAI-compatible
endpoint **does support `response_format: {"type": "json_schema", …}`**. A probe returned
exactly `{"answer": 4}`, schema-valid, so constrained decoding works on this server. Whatever
the JSON nodes below get wrong, it is not the output surface.

## 2. Verdict per node

| Node | Contract | Verdict | Evidence |
|---|---|---|---|
| P1 `analyze_question` | JSON: classify (`kind ∈ {simple, multi_part, out_of_scope}`) + plan | **Cannot serve** | Baseline scored **0/6** on single-fact questions — every one classified `multi_part` with 3–4 invented sub-queries. `out_of_scope` was 2/2 and genuine `multi_part` 5/5, so the other two classes are right: the failure is one axis only. The superlative shape ("Which segment had the highest revenue" — golden case q05) never improved under any ablated prompt variant (examples, superlative rule, self-check, count-word, combinations) once server load was controlled. A routing error here is the expensive failure, and one shape cannot be routed. |
| P2 `contextualize_question` | plain-text rewrite | **Can serve** | q07 → *"What was Microsoft's operating margin trend in FY2023?"*; q08 → *"What segment drove Microsoft's revenue in FY2023?"*; non-follow-ups returned unchanged. Pronoun, period and metric family all preserved. |
| P3 `synthesize_answer` | figures + `[n]` markers | **Cannot serve** | **0 of 8 calls emitted a single citation marker**, with P3 alone and with P4 appended alike. Removing P4 does not help — the defect is the model, not the prompt stack. The failure is terminal downstream: the symbolic pre-pass (`src/chains/checks.py`) trims every claim `figure_not_in_span`, because a claim with no marker has no cited span, so every answer collapses to `(unsupported claim removed)` with `grounded=False`. P4's own header records that the hosted provider hit exactly this class of bug, so it is a live known risk, not a hypothetical. |
| P5 `groundedness_check` | JSON verdicts | **Cannot serve** | Over-trims in every scenario, including genuinely supported claims (2/2 runs trimmed a supported-plus-supported answer); the mixed supported+unsupported case trimmed **both** claims. Worse, it misses the failure it exists for: a citation-free fabricated claim ("named a new CEO") was judged `supported` on **all 5 items**, 2/2 runs. It destroys good claims and blesses unsupported ones. One nuance, recorded for completeness: on a clean qualitative claim it returned correct per-claim verdicts, so the skill exists but is not calibrated. |

**The P3 result is the decisive one.** If the pipeline were driven by this model, the golden
QA numbers would not measure the pipeline — they would measure P3's citation omission and then
`grounded=False` on every case. Any local run's metrics would be an artifact of that fact.

## 3. Three measurement traps

Any attempt to re-derive or extend this record on a local server hits all three. They were
each measured, not read:

1. **The model is deterministic at temperature 0.** 6/8 identical-call runs came back
   byte-identical. Apparent variance was **server contention**, not sampling: an idle-server
   run of one ablated P1 variant scored 13/13, and the same variant scored 6/12 under load.
   A naive A/B probe therefore "confirms" whichever variant ran while the box was idle. Any P1
   evaluation must **interleave baseline and candidate in one pass**, alternating calls so both
   see the same contention — which is how the verdict table above was produced.
2. **Ollama ignores `max_tokens`.** One call capped at 40 completion tokens returned
   **1,399**. Latency and cost figures taken from a capped probe on this server measure
   nothing; only uncapped runs are comparable.
3. **Ollama under-reports `prompt_tokens` on large payloads.** The real P3 payload is
   27,730 chars (≈6,900 tokens) but the server reports **2,050**. Isolated to the context-window
   default: at `num_ctx=4096` a 14,000-token prompt reports `prompt_eval_count=2050`, while the
   same prompt with `num_ctx=16384` reports **14,030**. Above roughly 25–28k effective tokens
   the reported number is flat and meaningless. Consequences: local token accounting cannot be
   compared to the hosted figures this repo already records, and the gateway's per-node
   accounting is unreliable against this server for exactly the nodes (P3, P5) whose payloads
   are large.

## 4. The `Event loop is closed` finding (so nobody re-investigates it)

The `RuntimeError: Event loop is closed` seen while probing is a **probe** bug, not a gateway
one. Calling `asyncio.run()` repeatedly on one `ChatLLM` binds the httpx connection pool to the
loop that opened the first connection; `asyncio.run` closes its loop on return, so the next call
touches a dead loop. The repo's nodes are safe: **every LLM node calls `build_llm()` inside its
own `asyncio.run`**, so a fresh client-and-pool is built per turn — verified 100% clean across
11 stress turns, and nothing in this tree hoists a client across turns. The latent hazard to
name: a future module-level singleton or Streamlit `cache_resource` client *would* hit this
error, because that pattern keeps one pool across many loops.

## 5. Cost and latency (real nodes, real corpus, `INDEX_DIR=data/index`)

| Case | Prompt tokens (server-reported) | Completion tokens | Latency | Note |
|---|---|---|---|---|
| P1 `analyze_question` | ~2,050 | 690–865 | median **11.0–12.1 s** (range 9.7–12.1) | prefill-dominated; the mandatory 5,236-char `doc_catalog` cannot be shortened |
| P3 `synthesize_answer`, q01 shape | 2,050 | 33–39 | 0.8–1.3 s | 6 context blocks |
| P3 `synthesize_answer`, q03 shape | 2,050 | 251–595 | 3.7–8.5 s | 7 context blocks |
| P5 `groundedness_check` | 1,846–1,863 | 64–151 | 1.2–2.7 s | qualitative claims |
| Tail — one q10 P3 call | — | — | **117.9 s** | single run |
| Tail — one P1 call | — | — | hit the gateway's **120 s timeout** | single call |

Read the token columns through trap 3: the reported 2,050 is an under-count for the larger
payloads, so the true prefill is higher than the table can show on this server. Tail latency is
part of the verdict: P1's ~11–12 s sits on top of P2 on every question, and its worst call does
not complete at all.

## 6. What it would take to change the verdict

- **A larger local model for P1/P3/P5.** Nothing in this record says a local run is impossible
  in principle — `response_format: json_schema` works, the model is deterministic and correctly
  handles two of P1's three classes. What is missing is instruction-following at 3B: the
  superlative routing shape that no prompt ablation moved, the `[n]` marker that was never
  emitted in 8 calls, and the over-trimming that rejects supported claims while blessing
  citation-free fabrications. The next candidate must be re-tested per node against the
  evidence column above, with interleaved runs per trap 1 and uncapped calls per trap 2.
- **Split serving: P2 local, P1/P3/P5 on a hosted provider.** Viable per the evidence, since
  P2's contract is a plain-text rewrite and it held. This buys nothing the pipeline needs that
  the mock path does not already provide, and the traffic that reaches P2 is exactly the
  follow-ups. Recorded for completeness, not recommended.
- What does **not** change the verdict: further prompt iteration on `docs/prompts/` P1, P3, P5
  against this model. The ablations listed in §2 were run; P4's removal was run. The failures
  are model-side at this size, and prompt bumps in this repo are evidence-gated
  (`docs/prompts/README.md`), so a bump requires a passing result to point at.

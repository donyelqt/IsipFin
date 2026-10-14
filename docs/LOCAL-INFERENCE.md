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

---

# 7. Provider decision: Groq free tier, Groq paid tier, larger local model, or hybrid

> Decision record, written after a keyed run against Groq's free tier with the real 384-dim
> index (`.evalwork/keyed20b.log`). The question the owner asked is narrower than "which
> provider is best": **this exam submission is graded by a reviewer who clones the repo and
> reproduces a cited answer**, so the requirement is that every metric bar be *measurable and
> honestly measured*, not that a run look impressive. Everything below is measured on this
> host or quoted from the provider's own documentation; where a number is estimated it says so.

## 7.1 What the constraint actually is

**It is the tier, not the model and not the prompts.** The evidence is one line from the log:

```
429 ... Limit 8000, Used 6537, Requested 8087 ... (openai/gpt-oss-20b, service tier on_demand)
```

`Requested 8087` against a ceiling of `8000` is a request that physically cannot fit, so this
is not a contention story ("we ran out of budget") — a single P3 call exceeds the window. The
same run logged three planner requests of `Requested 3477` each into the same rolling window
and retried all three, which is what drained the remainder.

The decomposition of where those tokens go. The owner's keyed request sizes are the ground
truth; the char counts below are exact measurements of the repo's own assets and the real
`data/index/`, converted at the repo's own offline estimator `estimate_tokens = chars // 4`
(`src/core/chunking.py:113`). No BPE tokenizer is installed or used. The two scales are kept
separate: the measured request size is what the provider billed, the char counts are what the
files contain.

| Component | Exact chars | EST tokens | Share of the **measured** request |
|---|---|---|---|
| P1 system prompt (`docs/prompts/query-analysis.md` fenced block) | 1,496 | ~374 | **15%** of 2,038 |
| P1 catalog JSON, rendered from `data/index/docs.jsonl` | 5,236 | ~1,309 (est.); **766** measured | **38%** of 2,038 at the measured rate |
| P1 template skeleton + question | 104 | ~26 | ~1% |
| P1 `response_format` JSON schema (request metadata) | 577 | ~144; **128** measured | 6% |
| **Unattributed residual** (wire envelope: model id, role tags, sampling params, the schema's own `json_schema` wrapper) | — | — | **~41%** (~830 tokens) |
| P3 system prompt — P3 fence **plus the unconditional P4 formatting appendix** | 4,258 | ~1,065 (est.) | **~1,700** measured (21%) |
| P3 retrieved context (q01: 6 packed parents = 23,284 chars) | 23,284 | ~5,900–7,300 | ~73–78% (the rest) |
| P5 system prompt (the whole asset body, not a fenced slice) | 4,128 | ~1,032 | ~49% of ~2,100 |

Two calibration notes, because the estimator and the real tokenizer disagree in a known
direction. The catalog renders **denser** than chars // 4 — 5,236 chars measured **766** tokens
by the provider, i.e. **6.8 chars/token**, so for JSON the EST column over-counts and the true
catalog share is 38% of the request, not the 64% the char count implies. Prose and financial
table text go the other way — the P3 context renders at roughly **3.3 chars/token** against the
measured 5,900–7,300 — so the two halves of a P3 request cannot be priced with one ratio, and
the table's measured column is the one to read. The residual row is stated rather than dropped,
and it is large enough to matter: ~830 of P1's 2,038 tokens sit in the request's wire envelope
and cannot be attributed by reading the repo alone — itself an argument for measuring rather
than predicting from char counts. The second note is a fixed cost the P1-side
decomposition has no analogue for: **every P3 request carries P4's formatting appendix**
(1,798 chars, appended unconditionally at `synthesize.py:204-211`), which is why P3's system side
is ~1,700 measured tokens and a P3 call is ~4× a P1 call before any context is added.

Two facts make prompt surgery the wrong lever. First, **the catalog is 77% of the P1 payload by
characters** — deleting the entire system prompt saves 15% of one request and buys the planner
nothing it needs to classify, because the catalog is what `required_sections` and
`missing_aspect` are read out of. Second, **P3's context is the irreducible half**: it is packed
greedily to `CONTEXT_TOKEN_BUDGET = 6000` (`src/chains/nodes/synthesize.py:61`), the budget is
already what keeps a wide question inside the model's window (STRESS-TEST-RAG #6), and the
23,284 chars observed on the q01 shape are **97.7% of that budget** — lowering it trims the very
passages the answer's citations point at, and the prompt's own header then tells the model its
context is truncated (`synthesize.py:169-171`).

The best prompt-shaped saving is the catalog's own redundancy, and it is worth naming precisely
because it is the only one available. The 15 stored section paths are prefix-extending
(`A > B > C`, `A > B > C > D`, …) over only **14 distinct leaf headings**, so a tree-shaped
rendering is ~852 chars instead of 5,236 — an **84% catalog cut ≈ 640 measured tokens** off P1
(766 → ~130), taking 2,038 → ~1,400. That is a real win and it is still insufficient: a
4-sub-query P3 is ~7,600 tokens by the same measurement, so one case still cannot fit an
8,000-token minute. A second, trivially safe saving exists and is not yet taken: every P3 block
header prints the 64-char sha256 `doc_id` in place of the template's `{filename}`
(`synthesize.py:161-163`), costing 448 chars ≈ 66 measured tokens across 7 blocks for no
information the citations need.

**The actual levers, in order of effect:**

1. **The minute's token budget.** 8,000 TPM free vs. "significantly increased rate limits" on
   Groq's Developer plan (`console.groq.com/docs/billing-faqs`, "Higher Token Limits" as the
   first listed benefit). A single P3 call needs ~7,000 tokens, so **one case cannot fit one
   minute**; the five simple figure cases measure ~29,000 input tokens and the whole 10-case set
   ~42,000.
2. **The number of nodes that hit the provider per case.** The graph makes 3–4 calls per simple
   case (P1 → P3 → P5, plus P2 on follow-ups), so P1's ~2,038 tokens are paid ~30 times across a
   10-case run — ~61,000 tokens, which is more than the whole run's P3+P5 traffic. That is the
   overhead a tree-shaped catalog would cut by two-thirds (~640 tokens of P1's 2,038, on every
   call).
3. **Prompt caching**, which the provider already offers and this repo does not use. Groq's
   docs (`console.groq.com/docs/prompt-caching`) state prompt caching "works automatically on
   all your API requests with no code changes required and no additional fees", give a 50%
   discount on cached input, and — decisively — that **"Cached tokens do not count towards your
   rate limits."** It is supported on `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. The P1
   system prompt and catalog are byte-identical across every question in a run (the catalog
   renders from a single-document index that does not change mid-run), so they are a natural
   prefix hit: ~1,070 measured tokens on each of ~30 planner calls in a 10-case run is ~32,000
   tokens the free tier would never see. No repo change is required to *get* the discount — the
   gateway sends no cache headers and needs none. But it is not a guarantee ("Groq tries to
   maximize cache hits, but this is not guaranteed"), the harness records no `cached_tokens`
   field, and **no measurement in this repo confirms a cache hit ever occurred**, so it is a
   documented provider behaviour, not a measured saving here.

What the levers are **not**: the model (the failure was a 429, and the same log shows the
planner producing correct `simple`/`out_of_scope` plans whenever a request got through), and the
prompt stack (nothing in P1–P5 caused the 429; the catalog cost is an architectural choice with
a documented reason, `analyze.py:266-291`). An offline audit of the tree confirms there is no
hidden reduction path to reach for: the gateway sends no cache headers at all
(`src/utils/llm.py` `_build_request` emits only `model`, `messages`, and whichever of
`temperature`/`top_p`/`max_tokens`/`response_format` are set), both
`CONTEXT_TOKEN_BUDGET = 6000` constants are hardcoded with no env override
(`synthesize.py:61`, `retrieve.py:106`), and `MAX_CATALOG_CHARS = 12_000`
(`analyze.py:131`) does not bind — the real catalog is 5,236 chars, **43.6%** of the cap, so
that truncation marker is unreachable on this corpus. The two whole-request skips that do exist
are narrower than "fewer tokens": the symbolic year gate (`analyze.py:520-523`) skips the P1 POST
for a question naming exactly one uncovered fiscal year, which is how q09 refused in 25.1 ms, and
the verifier's deterministic pre-pass skips the P5 POST when every claim is already rejected.
Retries strictly *add* tokens — ADR-005's three attempts and P1's stricter-nudge message both
re-send the same payload.

## 7.2 The four options, quantified

Bars are `docs/TESTING-AND-EVAL.md`'s own, unchanged for any mode (the harness lowers none of
them). "Measurable" means the mode can produce a case for that metric rather than `NA`.

| | (i) Groq free, partial run | (ii) Groq Developer | (iii) larger local model | (iv) hybrid |
|---|---|---|---|---|
| **Config** | unchanged, `--only qXX` | `LLM_PROVIDER=groq` + paid key | `LLM_PROVIDER=ollama`, `LLM_MODEL=<big>` | not expressible today |
| **Money** | $0 | pay-as-you-go: **$0.15/1M in, $0.60/1M out** for `gpt-oss-120b`, $0.075 cached-in (`console.groq.com/docs/model/openai/gpt-oss-120b`). A full 10-case run ≈ 40k in + ~1k out ≈ **$0.01** | $0 | $0 hosted + local disk |
| **Rate limit** | **8,000 TPM**, 30 RPM (free) | "significantly increased rate limits" — no published number for this org; Groq's docs give 8K TPM as the *base* Developer figure for `gpt-oss-120b`/`20b`, so **the 8K number alone is not proof a paid lift exists**; the account's limits page (`/settings/limits`) is the only authoritative read | none (self-served) | whatever the hosted leg has |
| **Latency** | measured in the log: q01 **6,993 ms**, q02 **58,745 ms**, q04 **57,291 ms**, q06 **26,272 ms** — the spread is retries waiting out the rolling window, with `retry-after` hints of 0.9–6.9 s per 429 and 3 attempts each (`MAX_ATTEMPTS = 3`) | same per-call shape (Groq advertises ~500 tps for `gpt-oss-120b`), without the retry-dominated tail | `docs/LOCAL-INFERENCE.md` §5 at 3B: P1 median 11.0–12.1 s, one call hit the gateway's 120 s timeout, one P3 tail 117.9 s | P1/P3/P5 hosted + P2 local ⇒ hosted latency for every node that carries a metric |
| **retrieval_hit_rate@5 ≥ 0.9** | measurable, and it **passed 3/3** with no retrieval change | measurable | measurable — retrieval is keyless and unchanged (local fastembed index, `EMBEDDING_PROVIDER=local`) | measurable |
| **citation_presence = 1.0** | measurable, **passed 4/4** | measurable | **at risk** — the 3B model emitted 0/8 `[n]` markers; a bigger model may fix this, but it is untested here | measurable on the hosted leg |
| **figure_match ≥ 0.95** | measurable, **3/3 reached** (the harness read 0.00 on a bug, since fixed) | measurable | at risk — needs the marker first, because no marker ⇒ no cited span ⇒ the pre-pass trims every claim | measurable |
| **wrong_year = 0** | measurable, **passed 0/2+1**; lexical, mode-independent, and it can fail under `MOCK_LLM=1` too | measurable | measurable (same checker) | measurable |
| **refusal_correctness = 1.0** | measurable, **passed 6/6** | measurable | at risk — P1 at 3B misroutes the superlative shape, and a routing error lands exactly here | measurable |
| **Verdict** | every reachable bar already passes; the *denominator* is what shrinks | same bars, larger denominator, small bill | a re-run of §2 at 4–5× the parameter count, with an unknown outcome | **cannot be configured** — see below |

Notes on the two options that look cheapest:

**(i) partial run is honest but self-limiting.** `--only` is a documented flag and the harness
guards it: a follow-up whose ancestor did not run is a *harness error* (exit 2), so `--only q07`
or `--only q08` fails on its own. The practical subset is `q01,q02,q04,q05,q06`, whose five
simple paths measure **~29,000 input tokens of LLM traffic** offline (P1 ~1,700 each, P3
~5,300–6,200 each, P5 ~1,000 each) — that is **3.6 minutes of pure settling time** against the
8,000 TPM window even with zero retries, so it cannot finish in one minute either; it works only
by spreading across windows, which is exactly the retry pattern that produced the log's
58-second cases. A subset's aggregate is also a smaller sample, not the honest thing
`docs/TESTING-AND-EVAL.md` asks for, and `q03`/`q07` are the two cases where the planner could
not run — a partial run excludes precisely the cases that were broken.

**(iii) larger local model is a bigger host than this one.** A 14B Q4_K_M is ~9 GB of weights
against a 4 GB RTX 3050 and 23.7 GiB of system RAM shared with Chromium, Streamlit and
fastembed; a 32B is ~20 GB. Disk is fine (769 GB free on `D:`). The honest statement is: this
option is **untested at the required size on this hardware**, so it cannot be ranked as
"meets the bars" — and any local run still has the §5 traps to re-derive (Ollama under-reports
`prompt_tokens` above ~25–28k, so the gateway's accounting is unreliable for exactly the P3 and
P5 payloads that matter).

**(iv) hybrid is not expressible with today's code, and pretending otherwise would be the
risky path.** `build_llm()` (`src/utils/llm.py:1180`) resolves **one** provider per process,
and all four nodes call it — `analyze.py:542`, `synthesize.py:353`, `verify.py:421`,
`contextualize.py:474`. `LLMParams.model` can override the *model* per call, but not the
*endpoint*, and `LLM_BASE_URL` is process-wide. So "hosted P1/P3/P5 + local P2" would need a
code change (a per-node client factory), which is outside this doc's remit and outside the
frozen node contract. Recording it as available without that change would be false. It is also
worth naming what a hybrid *would* buy if built: P2 is the only node a local model provably
serves, it is the only node with no metric of its own, and the p1/P3/P5 legs still need the
same hosted key — so it saves the *least* expensive node's cost and nothing else.

## 7.3 Recommendation

**Option (ii): Groq's Developer tier, with `LLM_PROVIDER=groq` and
`LLM_MODEL=openai/gpt-oss-120b`.** Exact environment for a run:

```bash
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...                 # your Groq key; LLM_PROVIDER selects Groq's endpoint
LLM_MODEL=openai/gpt-oss-120b       # required on Groq: both JSON nodes send json_schema
MOCK_LLM=0                          # unset/0 — checked first, would short-circuit the key
EMBEDDING_PROVIDER=local            # Groq has no /embeddings endpoint (ADR-005)
INDEX_DIR=data/index                # real 384-dim index

python scripts/eval_rag.py --json .evalwork/keyed20c.json
```

That config is *already* the repository's documented, supported surface — it is what
`README.md` and `.env.example` print for Groq — so the recommendation costs no code and no
prompt change. `gpt-oss-120b` specifically, because both JSON nodes need
`response_format: json_schema` and Groq gates Structured Outputs by model (`llama-3.3-70b-versatile`
rejects it outright and would break P1 and P5 at the first call — the misconfiguration the
diagnostics must surface).

**Why 120B over 20B, given the log used both:** the measured answer quality in this log came
from the hosted path (`$211,915 million [2]`, grounded, cited, repeatedly), and P1's superlative
routing is the one shape a smaller model gets wrong on the local evidence (§2). 120B is also
the model the repo already recommends for Groq (ADR-005) and one of the two that support
`strict: true`, which is the upgrade path ADR-005 names for the best-effort mode the gateway
currently runs.

**What the reviewer does to reproduce** — and this is the part that decides the submission:

1. `cp .env.example .env` and fill in the four values above (`LLM_PROVIDER`, `LLM_API_KEY`,
   `LLM_MODEL`, and keep `MOCK_LLM=0`), or export them; a real environment variable wins over
   the file (`src/utils/settings.py`), except for the eight provider-routing names, which the
   file is authoritative for. Check what resolved with
   `python -c "from src.utils.llm import resolved_config; print(resolved_config())"` — it
   prints provider, base URL and model with the key redacted.
2. `python scripts/eval_rag.py` for the full 10 cases. The harness names every case the
   planner could not run (`cases not measured`) and still exits 1 for it, so a quiet pass is
   impossible.
3. **The keyless comparison, which every reviewer can run with no key and no network:**
   `MOCK_LLM=1 INDEX_DIR=data/index-mock python scripts/eval_rag.py`. That is the aggregate
   that reproduces byte-for-byte (`docs/JOURNAL.md` §6) and the honest answer to "the provider
   is nondeterministic" — the wrong-year canary stays live and able to fail in that mode.
4. `streamlit run app.py` with the same config, ask the headline question, and read the
   citation card (`2023_Annual_Report.pdf · page 44`), the node trace and the grounded badge.

## 7.4 Residual risk, stated plainly

The recommendation's risk is **token arithmetic, not quality**: it assumes a paid tier lifts
the 8,000 TPM ceiling enough to absorb ~7,000-token P3 calls, and Groq's public docs do not
publish that org's number — the same docs list **8K TPM as the base Developer limit for
`openai/gpt-oss-120b` and `openai/gpt-oss-20b`**, so it is possible that an upgrade does not
lift the binding constraint for *this* node mix at all, and the run to confirm it is a real
call this budget did not allow. Two further gaps are unmeasured under any hosted option: the
provider is nondeterministic at temperature 0 (`docs/JOURNAL.md` §6 measured three identical
planner calls returning a plan, a different plan, and an error), so a reviewer's run may fail on
a case this one passed and the aggregate stays a sample, not a score; and prompt caching is
documented to waive cached tokens from the rate limit but is explicitly **not guaranteed** by
Groq, the harness records no `cached_tokens` field, and no measurement in this repo confirms a
cache hit occurred — so the brief must not claim the ~32,000-token saving it would enable. What
is *not* measured under the recommendation: any local-model path beyond `qwen2.5:3b` (so
"bigger local model would have failed too" remains a hypothesis, and `docs/LOCAL-INFERENCE.md`
§5's traps would have to be re-derived before it could be claimed), and the hybrid split, which
needs a code change before it is even a configuration. The fake-pass risk to guard against is
narrower and worth naming: nothing in this recommendation may be used to justify running with
`--only` and reporting the subset's aggregate as the system's score, or with `MOCK_LLM=1` and
reporting its `NA` cells as passes — the harness already refuses both (exit 2 on an orphaned
follow-up; `NA` never drives the exit code), but a human writing the journal row can still do
it by hand.

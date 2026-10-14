# Keyed inference record: NVIDIA Nemotron — one 10-case golden run, measured against the previously documented row

> Evaluation record, not a how-to. This file documents one keyed run of the golden QA harness
> (`scripts/eval_rag.py`, `docs/TESTING-AND-EVAL.md` §Layer 2) against a real provider and the real
> persisted 384-dim index. The question it answers is not "does the pipeline work" — it is
> **"what changed since the row the journal already recorded, and is every claim in the delta
> measured?"** The headline answer: **four of five pass bars are met and two previously failing bars
> now pass; the fifth, `figure_match`, is still missed, and the process exit code is 1.** The
> improvement is real and the bar is still missed. Both are true, and this report does not write
> around either one. Every number below is read from the run's log or arithmetic on it; where a
> number is derived rather than printed, it says so.

## 1. Headline verdict

| | |
|---|---|
| Provider | `nvidia` → `https://integrate.api.nvidia.com/v1` (`PROVIDER_DEFAULTS`, `src/utils/llm.py`) |
| Model | `nvidia/nemotron-3-super-120b-a12b` |
| Mode | real LLM — no mock node, no scripted gateway |
| Index | the real local-embedding index: real 384-dim `data/index` (`data/index/faiss/meta.json` = `{"dim": 384, "count": 398}`; 398 child chunks / 102 parents, 85 pages, 15 sections, 1 document) |
| Command | `LLM_PROVIDER=nvidia LLM_MODEL=nvidia/nemotron-3-super-120b-a12b MOCK_LLM=0 python scripts/eval_rag.py` |

**What improved, measured against which row.** The comparison row is `docs/JOURNAL.md` §7's
`Golden QA — real provider, last complete run (n=10)` as it stood *before* this run superseded it
(`MOCK_LLM=0 OPENAI_MODEL=nvidia/nemotron-3-super-120b-a12b python scripts/eval_rag.py`, real
index, real key): retrieval hit-rate@5 **6/6 (1.00) PASS**, citation presence **6/8 (0.75) FAIL**,
figure match **3/7 (0.43) FAIL**, wrong-year contamination **0/3+0 PASS**, refusal correctness
**9/10 (0.90) FAIL**. Against that row:

* **citation presence 6/8 → 8/8 — the bar is now met.**
* **refusal correctness 9/10 → 10/10 — the bar is now met.**
* **figure match 3/7 → 6/7 — three cases recovered, and the bar is still missed** (0.86 against a
  0.95 bar; at n=7 a single case is 0.143, so 6/7 effectively needs 7/7).
* retrieval hit-rate@5 stays perfect at 1.00 with one more case measured (6/6 → 7/7); wrong-year
  contamination stays clean at 0/3+0.
* **Cases failing: `q07` only.** `q05` and `q08` both hit on every scored metric here, and
  `q05,q07,q08` is one of the failing sets the journal's variance row records for keyed runs of this
  code — so two previously observed failures recovered. The earlier metric row did not itself name
  its failing set; the movement this report rests on is metric-level, documented in full.

**The one bar still missed, and the honest part.** `figure_match` at 6/7. The run's exit code is
**1**. A run that meets four of five bars still fails the aggregate, and this report is not written
as if it passed. One more measured fact worth one line, because it is the symbolic layer doing its
job: `q09` ("What was revenue in FY2019?") refused in **4.9 ms** via the deterministic
year-coverage gate with **no planner call at all**, rendering *"the documents cover fiscal years
FY2021-FY2023"* — see §6.

## 2. The aggregate, verbatim

Copied from the run's log (`.evalwork/nvidia-verify.log`), not retyped:

```
mode                 : real LLM
index / catalog      : D:\DonieleTheLastDev\Projects\isipfin\data\index — 1 document(s)
cases                : 10 requested, 10 scored
cases failing        : q07

metric                                 value     bar  verdict
-------------------------------------------------------------
retrieval_hit_rate@5              7/7 (1.00)  >= 0.9  PASS
citation_presence                 8/8 (1.00)    >= 1  PASS
figure_match                      6/7 (0.86)  >= 0.95  FAIL
wrong_year_contamination               0/3+0    <= 0  PASS
refusal_correctness             10/10 (1.00)    >= 1  PASS

FAIL — metrics missed their bar: figure_match
exit code 1 — a golden regression — a pass bar was missed, the wrong-year canary fired, a case's graph run raised, or a case's planner could not run at all
token spend         : 111,397 total (q01=10,417, q02=9,750, q03=15,279, q04=16,050, q05=9,770, q06=19,178, q07=13,879, q08=14,345, q09=0, q10=2,729)
```

## 3. Per-case results

One row per case. `grounded` and `citations` are as the harness printed them; latency is the
harness's own per-case timer. The last column quotes the answer's first clause from the log —
including the log's own punctuation and spacing, and including the `(unsupported claim removed)`
markers where the verifier trimmed a claim.

| Case | Question | grounded | citations | latency | Answer's first clause |
|---|---|---|---|---|---|
| q01 | What was Microsoft's revenue in FY2023? | True | 1 | 10626.8 ms | `Microsoft's revenue in FY2023 was $211,915 million [2].` |
| q02 | What was operating income in fiscal year 2023? | True | 1 | 9440.3 ms | `Operating income was $ 88,523 million in FY2023 [6].` |
| q03 | Compare revenue and operating income between FY2023 and FY2022. | True | 1 | 49003.9 ms | a markdown table, opening `| Metric | FY2023 | FY2022 |`, first row `| Revenue [1] | $ 211,915 | $ 198,270 |` |
| q04 | What was net income and diluted EPS in FY2023? | True | 1 | 23271.2 ms | `Net income was $ 72,361 million [1].` |
| q05 | Which segment had the highest revenue in FY2023? | True | 1 | 10835.5 ms | `Intelligent Cloud had the highest revenue in FY2023 at $87,907 million [3].` |
| q06 | Summarize FY2023 key results. | **False** | 1 | **134419.8 ms** | `In FY2023, Microsoft reported total revenue of 211,915, net income of $ 72,361, and diluted EPS of $ 9.68 (figures in millions except EPS) [5]. (unsupported claim removed)` |
| q07 | And the margin trend? | True | 1 | 22698.0 ms | `Microsoft's revenue increased $13.6 billion or 7% in FY2023 compared with FY2022, driven by growth in Intelligent Cloud and Productivity and Business Processes and partially offset by a decline in More Personal Computing [1].` |
| q08 | Which segment drove that? | True | 1 | 17597.3 ms | `The revenue increase in FY2023 was driven by growth in the Intelligent Cloud and Productivity and Business Processes segments [1].` |
| q09 | What was revenue in FY2019? | True | 0 | **4.9 ms** | `**Refused** — I can't answer that: the documents cover fiscal years FY2021-FY2023.` |
| q10 | What was Amazon's revenue in 2023? | True | 0 | 5368.0 ms | `**Refused** — I can't answer that: the question concerns Amazon, an entity not in the documents.` |

Three of these rows carry weight beyond their own score:

* **q01 is the brief's headline question, answered grounded and cited.** `grounded: True`,
  `citations: 1`, 10626.8 ms, the year-confusion canary clean (`asked FY2023; FY2022 figure
  $198,270`). The FY2022 figure was available in the same spans and was not used.
* **q06 is the only `grounded: False` in the run, and it is a summarization case the harness does
  not score for figure match.** The verifier trimmed one claim; the surviving answer is grounded
  false. It is a cost outlier too: 134,419.8 ms and 19,178 tokens, the largest spend in the run.
* **q09 is the cheapest and the fastest** — 4.9 ms, zero tokens, no planner call. §6 covers why.

## 4. The six-metric before/after

Bars are `docs/TESTING-AND-EVAL.md` §Layer 2's own, unchanged for any mode. "Previously documented"
is the journal row quoted in §1.

| Metric | Bar | Previously documented | This run | Δ | One-line cause |
|---|---|---|---|---|---|
| retrieval_hit_rate@5 | ≥ 0.9 | 6/6 (1.00) **PASS** | **7/7 (1.00) PASS** | +1 case measured, still perfect | No retrieval change to credit it to — the embedder is keyless and local (`EMBEDDING_PROVIDER=local`, the same 384-dim `data/index`), so the movement is one more case's figure landing in the fused top-5, not a different retriever; both follow-ups hit here (`retrieval hit: True` on q07 and q08). |
| citation_presence | ≥ 1.0 | 6/8 (0.75) **FAIL** | **8/8 (1.00) PASS** | **bar now met** | The prompt and verifier fixes landed after the earlier row was recorded: P3 1.0.1 rule 3a (a figure from a multi-period block must name its period in the sentence) and P4 1.0.1 (citation markers on the table's own rows, not a source note under it), so answers that had collapsed to `(unsupported claim removed)` with `citations: 0` now survive with a valid marker. |
| figure_match | ≥ 0.95 | 3/7 (0.43) **FAIL** | **6/7 (0.86) FAIL** | +3 cases, 0.09 short | The same two fixes, plus what is left: synthesis now quotes the period-named figure (P3 v1.0.1 rule 3a, whose recorded motivation was exactly the FY2023 revenue answer trimmed to `(unsupported claim removed)`) and q03's table rows carry their markers (P4 1.0.1), which recovered three cases; q07 is the one still open — §5. |
| wrong_year_contamination | ≤ 0 | 0/3+0 **PASS** | **0/3+0 PASS** | unchanged, still clean | Unchanged by design: the canary is a lexical check over the pipeline's own claim text, mode-independent, with hermetic tests — and the symbolic year gate is what stops an uncovered year from being answered at all. |
| refusal_correctness | ≥ 1.0 | 9/10 (0.90) **FAIL** | **10/10 (1.00) PASS** | **bar now met** | The symbolic year-coverage gate landed after the earlier row: an uncovered-year question now refuses before any POST instead of being classified on a coin flip, so the class of failure that cost the earlier row its tenth case is gone. |
| cases failing | — | `q05, q07, q08` (a set the variance row records) | **`q07` only** | two cases recovered | `q05` and `q08` both hit on every scored metric in this run; `q07` is the one still open, and it is a synthesis miss rather than a retrieval one. |

## 5. Why q07 misses: retrieval found them, synthesis did not use them

`q07` is "And the margin trend?", a follow-up to q01, expecting `$88,523` and `$211,915`. The log
says, on the same case:

```
  expected       : $88,523        MISSING
  expected       : $211,915       MISSING
  missing        : $88,523, $211,915
  retrieval hit  : True (expected figure present in the fused top-5 texts)
```

**The retrieval answer is `True`. Both expected figures were in the fused top-5 texts.** The
retriever did its job; the chunks were there.

**The synthesis answer quoted neither.** It answered with revenue *growth* — `Microsoft's revenue
increased $13.6 billion or 7% in FY2023 compared with FY2022, driven by growth in Intelligent
Cloud and Productivity and Business Processes and partially offset by a decline in More Personal
Computing [1].` — a sentence that states no `$88,523` and no `$211,915`, and that describes
revenue change where the question asked about margin.

So the miss is a **synthesis** miss, not a scoring one and not a retrieval one. Retrieval found
them; synthesis did not use them. It is also worth recording what this q07 is *not*: it is not the
metric-family switch the P2 1.0.2 bump was written for — that defect rewrote the follow-up into a
gross-margin *percentage* question, and this run's answer is dollar-denominated. The metric-fidelity
rule held; what remains is that a dollar growth statement stood in for the margin figures the case
names. The systematic loss and the provider's residual run-to-run variation are separate things,
and neither is claimed fixed here.

## 6. Cost

| | |
|---|---|
| Total token spend | **111,397** (as the aggregate prints it) |
| In / out split | 92,878 in / 18,519 out — arithmetic on the log's per-case `tokens` lines, not a printed figure |
| Wall time | **284.75 s**; the ten per-case latencies the log prints sum to 283,265.7 ms, so ≈1.5 s of the run was harness startup, index load and pacing outside the per-case timers |
| Projected spend (the run's own pre-flight) | ~81,504 tokens for 10 cases — the run spent **137%** of its projection |
| Per-case spread | **0** (q09) → **2,729** (q10) → **19,178** (q06); the largest case cost **7.0×** the smallest non-zero one |

**The two refusals are not the same shape, and only one of them is free.** `q09` spent **0 tokens**
in **4.9 ms**: its log block carries no `tokens` line at all, the node path is three nodes
(`contextualize_question,analyze_question,refuse`), and the aggregate records `q09=0`. That is the
symbolic year-coverage gate — `checks.year_coverage_gap` wired into `analyze_question` *before* the
gateway call, returning the `out_of_scope` plan with `missing_aspect='time_period'` and no
accounting to report, because no POST was made. A question naming exactly one fiscal year the
catalog does not cover is settled by the catalog, not the model.

`q10` refused **correctly** but paid for it: **2,729 tokens** (2,311 in / 418 out) across **5,368.0
ms**, because an entity-gap question has no deterministic gate — the catalog holds no entity list
to compare against, so `Amazon` has to reach the planner. It is recorded here precisely because
"the two refusals were free" would be the easy sentence and the log does not support it. The
deterministic layer is worth what it costs where it applies: on q09 it is the difference between
4.9 ms and no tokens, and the journal records the same case taking **9,812.7 ms** and classifying
on a coin flip before the gate existed.

A keyed run is not free, and that is part of the record. The run's own header notes an 8,000
tokens/min ceiling and that every case is larger than the whole window — the pacer's honest answer
is "no wait can make it fit: it runs, and the provider's own 429 decides".

## 7. Reproduce

The exact command that produced this run:

```bash
LLM_PROVIDER=nvidia LLM_MODEL=nvidia/nemotron-3-super-120b-a12b MOCK_LLM=0 python scripts/eval_rag.py
```

What a reviewer needs, in order:

1. **A key.** `MOCK_LLM=0` (or unset/0) is checked first and would short-circuit a key; a real
   credential is required for every node that carries a metric. `LLM_PROVIDER=nvidia` names the
   vendor, and its endpoint is `https://integrate.api.nvidia.com/v1`
   (`PROVIDER_DEFAULTS`, `src/utils/llm.py`). The *credential's* provenance is host state, not repo
   config: the journal row for this run records the host's exported `OPENAI_API_KEY` /
   `OPENAI_BASE_URL` winning, because `.env` loads with `override=False`, while a host whose `.env`
   sets the `LLM_*` spellings resolves those first — so a reviewer must not assume which pair
   resolved. The check that matters is
   `python -c "from src.utils.llm import resolved_config; print(resolved_config())"`, which prints
   provider, base URL, model and key *source* with the key redacted. Confirm the endpoint is the
   NVIDIA URL and the model is `nvidia/nemotron-3-super-120b-a12b` before spending anything.
2. **The real index.** `INDEX_DIR=data/index` (the default) — the 384-dim local-embedding index.
   `EMBEDDING_PROVIDER=local` (the default) keeps vectors local, so the key is spent on the chat
   nodes only.
3. **The budget to spend it.** ~111k tokens on this provider and model, in ~4.75 minutes of wall
   time, against an 8,000 tokens/min ceiling that no single case fits inside.

**The caveat that has to travel with the command.** A keyed run spends real tokens and is **not
byte-reproducible. The journal's determinism posture stands unchanged**: the provider answers
differently to the same request at temperature 0 (three identical planner calls returned a plan, a
different plan, and a raised `LLMError`), and four keyed runs of identical code produced four
different failing sets. A reviewer who re-runs this command may land on a different aggregate. The
only aggregate in this repo that reproduces byte-for-byte is the keyless mock one
(`MOCK_LLM=1 INDEX_DIR=data/index-mock python scripts/eval_rag.py`), and it cannot score these
metrics at all — `NA`, never a pass. So this document records *a measured run*, not a score; treat
the numbers here as one sample with the spread named beside it.

## 8. What this does and does not prove

**It proves, on a real provider, with the real 384-dim index and the real corpus:**

* **Four of the five §Layer 2 pass bars are met** — retrieval hit-rate@5 7/7 (1.00), citation
  presence 8/8 (1.00), wrong-year contamination 0/3+0, refusal correctness 10/10 (1.00). Two bars
  that were failing on the previously documented row now pass.
* **The brief's headline question is answered, grounded and cited.** `Microsoft's revenue in
  FY2023 was $211,915 million [2].` — `grounded: True`, `citations: 1`, 10626.8 ms, with the
  year-confusion canary clean: the FY2022 figure `$198,270` was in the same spans and was not used.
* **The deterministic layer works where it applies.** q09 refused in 4.9 ms with zero planner POSTs
  and *"the documents cover fiscal years FY2021-FY2023"*; q10 refused correctly too, for 2,729
  tokens. Both refusal classes are correct on this run.
* **The prompt and verifier fixes are real, not assumed.** Citation presence moved 6/8 → 8/8 and
  figure match 3/7 → 6/7 after P3 1.0.1 rule 3a and P4 1.0.1; the movement is re-measured rather
  than inferred from the diffs.

**It does not prove:**

* **The fifth bar.** `figure_match` is 6/7 (0.86) against a ≥ 0.95 bar, and the run **fails**: exit
  code 1. At n=7 one case is 0.143, so the bar effectively requires 7/7. This run does not clear it
  and this report does not describe it as though it did.
* **A reproducible answer.** The provider is nondeterministic at temperature 0, and this aggregate
  is therefore a sample, not a system's score. Nothing here changes the journal's determinism
  posture or substitutes for the keyless mock aggregate.
* **Anything about providers or models not exercised here.** This is one provider
  (`nvidia/nemotron-3-super-120b-a12b`) at one moment on one host. It says nothing about Groq's
  tiers, about a larger local model, or about the hybrid split — those are decisions, not results,
  and the record for them is `docs/LOCAL-INFERENCE.md` §7.
* **That the q07 miss is fixed, or that it is the provider's fault alone.** The miss is a synthesis
  miss on a specific question shape; the metric-fidelity bump that targeted q07 changed the symptom
  without closing the case, and one run cannot tell a systematic loss from residual variance.

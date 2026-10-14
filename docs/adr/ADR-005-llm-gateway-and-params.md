# ADR-005: LLM gateway — one OpenAI-compatible client, typed outputs, scripted mock mode

- **Status:** Accepted
- **Date:** 2026-10-13
- **Deciders:** Doniele Antonio
- **Supersedes:** none

## Context

Every node talks to an LLM, but each with different needs: cheap classification
(`analyze_question`), a rewrite (`contextualize_question`), citation-constrained synthesis
(`synthesize_answer`), and strict verification (`groundedness_check`). The exam is graded by
someone who will clone the repo and run it with **their own** API setup — the app must work
against OpenAI, Azure OpenAI, or a local server without code changes, and it must be demoable
with **no key at all**.

## Decision

**`llm-gateway` module (`src/utils/llm.py`)** — a thin wrapper around one
OpenAI-compatible chat client:

- Construction: `ChatOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, model=...)`
  from env; `OPENAI_BASE_URL` also accepts `http://localhost:11434/v1` (Ollama) or a vLLM
  server, so local models drop in with an env change.
- One factory: `build_llm(params: LLMParams) -> ChatClient`. **No node constructs a client** —
  nodes receive a configured client or call `llm.chat(messages, params)`; this keeps
  configuration, retries, and accounting in one place and makes every node testable with a stub.
- **Typed outputs:** `analyze_question` and `groundedness_check` request JSON via
  `response_format={"type": "json_schema", ...}` with the schemas defined inline in the node
  that consumes them — `QUERY_PLAN_SCHEMA` in `src/chains/nodes/analyze.py` and
  `GROUNDING_SCHEMA` in `src/chains/nodes/verify.py`; parse failures raise a typed `LLMError`
  subclass (`SchemaViolationError`) that retries once with a stricter system nudge, then fails
  the node into the error path. Synthesis returns free-form markdown.
- **Retry policy:** exponential backoff with jitter on HTTP 429/5xx (max 3 attempts, base 1s),
  typed `LLMError(rate_limited | context_length | auth | timeout)` mapping to distinct handling.
  `contextualize_question` specifically degrades to the raw user question on any failure —
  losing a rewrite must not lose the turn.
- **Scripted mode:** `MOCK_LLM=1` swaps the client for `MockLLM` in `src/utils/mock_llm.py`:
  deterministic responses keyed by a prompt fingerprint (node name + schema presence), so the
  demo and the entire graph test suite run offline. Mock responses embed real FY2023 figures
  where the prompt pattern is the known revenue/opex questions, so even the mock path demos
  credibly.
- **Parameter surface (sidebar → GraphState.params → recorded per message):** model id,
  temperature, top_p, max_tokens, retrieval k.

**Per-task parameter policy:**

| Task | Temperature | Structured output | On failure |
|---|---|---|---|
| `contextualize_question` | 0.0 | no | fallback: use raw question, log warning |
| `analyze_question` | 0.0 | JSON schema (QueryPlan) | retry once w/ stricter nudge, else classify `out_of_scope` + flag |
| `synthesize_answer` | 0.2 (user-adjustable) | no (markdown) | typed error → visible error banner, never empty answer |
| `groundedness_check` | 0.0 | JSON schema (GroundingResult) | pessimistic default: treat as ungrounded, trim claims |
| embeddings | n/a | — | `RetrievalError`; index build fails loudly (see Update below) |

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Direct SDK call at each invocation site | Config, retries, and token accounting duplicated across 5 nodes; untestable without monkeypatching five places |
| LangChain Hub prompt pulls (`hub.pull`) | Runtime network dependency on an external service for zero exam credit |
| Streaming synthesis | Better UX feel, but complicates the groundedness gate (verify before display) and the demo; kept as a documented Could |
| Separate embedding client abstraction | Same `base_url` serves chat + embeddings; a second factory is dead abstraction |
| LiteLLM as provider layer | Re-implements what `base_url` already gives; extra dependency, extra failure surface |

## Consequences

- Provider lock-in is a one-line env change; grading demos work on any OpenAI-compatible endpoint.
- Token usage and latency are captured once in the gateway and emitted as events
  (`docs/adr/ADR-007-observability-errors.md`).
- The single secret is `OPENAI_API_KEY` — see `docs/SECURITY-PRIVACY.md` for handling.
- `MOCK_LLM` doubles as the offline CI/demo path, so a missing key can never produce a broken
  `docker compose up` demo.

## Update — 2026-10-13: two providers by default (chat ≠ embeddings)

**Status:** Accepted; amends the Decision above (the original text is kept as-is).

The single-client assumption — "same `base_url` serves chat + embeddings" — is not true of
every OpenAI-compatible provider. Groq is the counterexample that matters here: it is a fast,
OpenAI-compatible **chat** provider (`OPENAI_BASE_URL=https://api.groq.com/openai/v1`) whose
API exposes chat + responses and **no `/embeddings` endpoint**, so pointing the dense leg at
it fails at ingest.

It is also not a *fully* free model choice: the graph's `analyze_question` and
`groundedness_check` nodes need `response_format: json_schema`, and Groq supports Structured
Outputs only on a subset of models — `strict: true` on `openai/gpt-oss-20b`,
`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`; best-effort only on those plus
`openai/gpt-oss-safeguard-20b`. Other Groq models (e.g. `llama-3.3-70b-versatile`) reject
`json_schema` outright and would break both JSON nodes, so **`openai/gpt-oss-120b` is the
recommended Groq chat model**. See "Groq upgrade path" below for the exact behaviour.
The motive is throughput: a measured `analyze_question` call on a 120B model through this
gateway took **11.2 s** (NVIDIA Nemotron-3 Super 120B, `nvidia/nemotron-3-super-120b-a12b`,
temperature 0, 395 prompt / 246 completion tokens, correct plan `kind=simple,
fiscal_period=FY2023`) — evidence the gateway
works against a real provider, and a latency a fast throughput provider is meant to fix.

Decision, in three parts:

1. **Chat stays one OpenAI-compatible client**, resolved as `LLM_API_KEY` → `OPENAI_API_KEY`
   → `GROQ_API_KEY` for the credential and `LLM_BASE_URL` → `OPENAI_BASE_URL` for the
   endpoint — OpenAI, Groq, Ollama, vLLM all remain an env change, and a legitimate Groq
   credential can no longer be shadowed by an ambient `OPENAI_API_KEY`. The model keeps its
   single `OPENAI_MODEL` spelling; the reason for that asymmetry is recorded in the
   precedence note below.
2. **The embedder is resolved by `EMBEDDING_PROVIDER`**:
   - `local` (**default**) — `src/core/local_embedder.py`, a local Hugging Face model
     (`LOCAL_EMBEDDING_MODEL`, default `BAAI/bge-small-en-v1.5`, 384-dim) run in-process on
     ONNX Runtime via **fastembed**: no key, no `/embeddings` endpoint, no torch. Constructed
     once per process and cached; the model loads lazily on first use, never at import, and a
     missing/failed download raises a typed error the UI surfaces instead of crashing.
   - `api` — the previous behaviour: any OpenAI-compatible `/embeddings` endpoint via
     `EMBEDDING_BASE_URL` (falling back to the chat base URL) + `EMBEDDING_MODEL`.
   - `MOCK_LLM=1` still overrides both with the deterministic offline sha256 embedder — it
     remains the keyless CI and reviewer path, unchanged.
3. **Both resolution sites — `scripts/build_index.py` and
   `src/components/upload.py::_resolve_embedder` — implement `EMBEDDING_PROVIDER` with
   identical semantics**, so ingest and query can never disagree about vector width. A
   changed provider/model invalidates any existing index: loading one whose dimension does
   not match the configured embedder fails loudly with an actionable message, rather than
   letting FAISS raise an opaque error mid-query.

**Groq upgrade path (structured outputs).** The gateway's `_response_format()` sends
`response_format={"type": "json_schema", "json_schema": {...}}` **without a `strict` key**, so
on Groq it lands in *best-effort* mode: valid JSON with occasional schema slippage and the odd
400, rather than the constrained-decoding guarantee `strict: true` would give. That is a
quality difference, not a correctness one, because the gateway already validates the returned
payload against the schema client-side (`QUERY_PLAN_SCHEMA`, `GROUNDING_SCHEMA`) and retries
once with a stricter system nudge before failing the node into its error path. Adding
`strict: true` to the request would only be safe on models that support it and would require
all-`required` fields plus `additionalProperties: false` — a future change, not part of this
update.

**Why local embeddings, not `sentence-transformers`:** torch would add ~2.5 GB to the image
and wreck the cold-start the README promises; fastembed pulls the same Hugging Face weights
through ONNX Runtime at roughly 150 MB and is pinned `==` like every other row
(`docs/DEPENDENCY-PINS.md`).

**Env precedence, and the `LLM_*` override.** `.env` is read once by
`src/utils/settings.py` via `load_dotenv(override=False)`, so **real environment variables
always win over the file**. Exporting `OPENAI_API_KEY` (as CI and the grading host do) or
`EMBEDDING_PROVIDER` / `MOCK_LLM` overrides `.env` without editing it — useful for CI, but a
trap for chat: a host that exports `OPENAI_API_KEY` / `OPENAI_BASE_URL` silently ignores a
`.env` that names Groq, and the request goes out with the ambient credentials. Two
provider-neutral names therefore take precedence over the `OPENAI_*` pair:
`LLM_API_KEY` → `OPENAI_API_KEY` → `GROQ_API_KEY`, and `LLM_BASE_URL` → `OPENAI_BASE_URL` →
Groq's base URL when the key came from `GROQ_API_KEY` and no base URL is set. Setting only
`GROQ_API_KEY` is therefore enough to run against Groq.

The asymmetry is deliberate and recorded here rather than left implicit: **the model keeps
its single `OPENAI_MODEL` spelling, with no separate `LLM_*` model override.** `OPENAI_*` is
protocol-named, not vendor-named — the gateway speaks the OpenAI-compatible wire format, so
`OPENAI_MODEL=openai/gpt-oss-120b` addresses Groq, `nvidia/...` addresses NVIDIA and
`llama3` addresses Ollama through one surface with no vendor branching, and a second alias
would be two spellings of one setting. The `LLM_*` escape hatch exists only for the settings
an ambient environment can demonstrably hijack: on this host that is observed for the key and
the base URL (both exported at process level) and not for the model, which is unset.
Resolution is `OPENAI_MODEL` → the built-in default, with a per-call override still winning.
The gateway exposes `resolved_config()` / `resolved_provider()` (`src/utils/llm.py`), which
report the resolved provider, base URL and model with the key always redacted — the first
thing to check on an auth error.

## Update — 2026-10-14: an explicit `LLM_PROVIDER` selector, one name per setting

**Status:** Accepted; supersedes the credential surface of the 2026-10-13 update above (that
text is kept as-is; the `EMBEDDING_PROVIDER` half of it is untouched).

**What the owner found.** The configuration surface described above was reported as "so
confusing", and unusable for the audience it was built for: a reviewer who clones the repo and
brings their own setup. The owner's question, in substance: *why can it just say `openai` and
`groq`, so that users and reviewers are able to put their ref in as the actual OpenAI API key or
Groq instead?* The complaint was not about any single variable but about the chain: three key
spellings (`LLM_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`) and two endpoint spellings
(`LLM_BASE_URL`, `OPENAI_BASE_URL`) with no statement that `OPENAI_*` names the *protocol*, not
the vendor. Reading the env contract required knowing an ambient-hijack story before the
ordinary case — setting a key — was legible at all.

**The decision.** The chat leg gains an explicit provider selector, with one name per setting:

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `openai` (default) · `groq` · `nvidia` · `ollama` · `vllm` · `azure` — selects the default endpoint |
| `LLM_API_KEY` | the key — **one name** |
| `LLM_BASE_URL` | optional override for a custom / Azure / self-hosted endpoint |
| `LLM_MODEL` | the model id, optional |

`GROQ_API_KEY` is kept at lower precedence as a key alias that also implies
`LLM_PROVIDER=groq` (and therefore Groq's default endpoint) — the "one variable is a working
Groq config" premise from 2026-10-13 still holds. `MOCK_LLM=1` short-circuits before any
credential variable is read, so it wins over all of them and needs none.

**Why the earlier rationale was insufficient.** The ambient-hijack reasoning was correct on
its own terms: a host exporting `OPENAI_API_KEY` / `OPENAI_BASE_URL` does silently beat `.env`,
and that fully justified `LLM_API_KEY` / `LLM_BASE_URL` existing as *overrides*, and fully
justified there being no `LLM_MODEL` while no ambient `OPENAI_MODEL` was observed. What it did
not justify was making those two overrides the *only* documented names. A design optimised for
the hijack case made the ordinary case — a reviewer with one key from one vendor — worse, and
it broke the premise it had itself established one paragraph earlier: `GROQ_API_KEY` alone being
a working config is only readable if the reader knows the provider-precedence story, and the
chain resolved the credential left-to-right with nothing saying where the endpoint came from.
The new surface keeps both properties instead of trading one for the other: `LLM_PROVIDER`
picks the endpoint, `LLM_BASE_URL` still overrides it (the ambient-hijack case is unchanged),
`GROQ_API_KEY` remains a key alias that also implies the provider, and `OPENAI_*` stays accepted
at lower precedence for CI hosts, grading hosts and existing configs.

**What was deliberately not done.** So that a reader can tell the old names are load-bearing
rather than leftovers:

- no alias was removed — `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` and
  `GROQ_API_KEY` all still resolve;
- `LLM_MODEL` is added as an optional acceptance, not a replacement — the 2026-10-13 reasoning
  that `OPENAI_*` is protocol-named still stands, and `OPENAI_MODEL` remains accepted;
- `MOCK_LLM=1`'s precedence is unchanged: it short-circuits before any credential is read, so
  the keyless demo and CI path behave exactly as before;
- the `EMBEDDING_PROVIDER` resolution of 2026-10-13 is not touched; the `LLM_*` names above
  govern chat only.

**Consequences.**

- The precedence chain gets one comment block in `src/utils/llm.py`, stating the order and the
  ambient-override rule in the order the code evaluates it — the chain no longer has to be
  reverse-engineered from prose or from `resolved_config()` output.
- `resolved_config()` remains the diagnostic, and now reports the resolved *provider* alongside
  the base URL and model, with the key always redacted.
- `LLM_MODEL` is a real decision on Groq, not a free parameter: `analyze_question` and
  `groundedness_check` need `response_format: json_schema`, and Groq gates Structured Outputs
  by model — `strict: true` on `openai/gpt-oss-20b`, `openai/gpt-oss-120b`,
  `qwen/qwen3.8-27b`; best-effort on `openai/gpt-oss-safeguard-20b`; and
  `llama-3.3-70b-versatile` rejects `response_format: json_schema` outright. A reviewer who
  picks a Groq model from the outside gets the constraint's failure at the first JSON node, not
  at ingest, so `LLM_PROVIDER=groq` without a supported `LLM_MODEL` is a misconfiguration the
  diagnostics must surface.
- The chat/embeddings split stays: Groq has no `/embeddings`, so embeddings default to
  `EMBEDDING_PROVIDER=local` (fastembed, in-process, no key) and the two legs may legitimately
  differ. `LLM_PROVIDER` never moves the embedder.

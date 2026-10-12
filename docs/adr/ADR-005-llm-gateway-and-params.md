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
  `response_format={"type": "json_schema", ...}` with schemas in `src/chains/schemas.py`
  (`QueryPlan`, `GroundingResult`); parse failures raise a typed `LLMError` subclass
  (`SchemaViolationError`) that retries once with a stricter system nudge, then fails the node
  into the error path. Synthesis returns free-form markdown.
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
| embeddings | n/a | — | `RetrievalError`; index build fails loudly |

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

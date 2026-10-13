"""``analyze_question`` — the query planner (P1, ADR-001's routing-decision node).

This is the one node that decides **how** a question is served: it classifies the standalone
question as ``simple`` / ``multi_part`` / ``out_of_scope`` and emits the structured
:class:`~src.chains.state.QueryPlan` every downstream node and the router read. It never
answers the question, and no other node re-decides the route (P1: "this node is the only
place routing decisions are made; ``synthesize_answer`` never re-decides").

The plan, not the classification, is the product
-----------------------------------------------
``query_plan`` carries four fields a downstream node would otherwise have to guess, and each
one exists because a real failure mode produced it:

* ``sub_queries`` — the ≤4 atomic questions ``decompose_question`` fans out over (STRESS-TEST-RAG
  #6 caps the fan-out here, before anything is retrieved). A comparison is **never** split into
  bare operands: both sides stay inside the sub-query that needs them, because a sub-query
  reading "what was operating income in FY22?" cannot itself answer the comparison that was
  asked (P1's own rule, and PLAN.md's T10b acceptance).
* ``required_sections`` — the catalog sections the model judged plausible, so retrieval is
  filtered and the debug panel can show *why* this chunk set. An empty list is legal and is
  preferred over an invented section: a section name that does not exist plans retrieval
  against content that is not in the corpus.
* ``fiscal_period`` — the period the question asks about, recorded before retrieval so
  STRESS-TEST-RAG #1's multi-year columns have something to be checked against.
* ``missing_aspect`` — ``time_period`` / ``topic`` / ``entity`` when the question is refused.
  Without it a year the documents do not cover looks identical to an off-topic question, and
  ``refuse`` would render the same wrong copy for both (STRESS-TEST-RAG #7, whose acceptance
  case is this node's: "What was revenue in 2019?" must refuse for the *period*).

Degradation is a route, not an error
-----------------------------------
P1's failure handling is explicit: a schema-violating reply is one stricter gateway retry (the
gateway owns that retry, ADR-005) and then this node classifies ``out_of_scope`` with
``missing_aspect=None`` plus a ``planner_degraded`` marker, and the turn survives as an honest
refusal. That is the same treatment the router already gives an unrecognised plan
(``graph.route_after_analyze``), so the degraded path lands on ``refuse`` and never reaches
retrieval with a plausible-but-ungrounded answer. It is deliberately **not** an exception:
a planner that fails has just said "I cannot tell what this question needs", and the honest
answer to that is a refusal, not an error banner. The marker is what keeps the refusal honest
in the debug panel — a reader can see the plan was degraded rather than reasoned.

The node is synchronous (see :mod:`src.chains.nodes.contextualize` for why ``graph.invoke``
requires it) and drives the one awaitable with ``asyncio.run``. Classification stays neural,
all routing stays symbolic (ADR-010): the plan is data, and the router's decision is a table
lookup over it.

One gap is not the model's to judge
-----------------------------------
ADR-010's split is "deterministic structure owns control and verification; neural models do the
semantic work where they earn their place". A question naming a fiscal year the catalog cannot
cover is a *control* question, not a semantic one — the catalog is the ingest-time record of
what is on disk, and the coverage is exactly the set ``refuse`` reads to write "the documents
cover fiscal years FY2021-FY2023". So :func:`~src.chains.checks.year_coverage_gap` settles it
symbolically, before any POST: the fiscal-year sibling of the numeric pre-pass
``groundedness_check`` runs on figures, moved from the answer to the route.

The gate is narrow on purpose, because it only ever *adds* a refusal and never overrides the
planner: it fires solely for a question naming exactly one fiscal year that lies outside the
catalog's coverage. It stays out of the way of a comparison (several periods asked), an
entity- or topic-gap question (the catalog holds no entity list to compare against), and a
question naming no period at all (nothing to compare). Those remain the planner's judgement,
where they belong.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from typing import Any

from src.chains.checks import asked_years, catalog_years, year_coverage_gap
from src.chains.state import DocSummary, GraphState, QueryPlan
from src.utils.llm import LLMError, LLMParams, SchemaViolationError, build_llm
from src.utils.prompts import assert_min_version, node_meta
from src.utils.tracing import TOKEN_KEY, trace_node

__all__ = ["analyze_question"]

logger = logging.getLogger(__name__)

#: The node's registered name, in ``NODE_NAMES``, in every ``NodeEvent``, and the key this
#: module's own lookups use into ``scripts/prompts_meta.json``.
NODE_NAME = "analyze_question"

#: The prompt asset this node consumes (``docs/prompts/query-analysis.md``).
PROMPT_ID = "P1"

#: The plan's three ``kind`` values. Mirrored as literals (not imported from ``graph``) so
#: this module stays a leaf: ``graph`` imports the nodes, not the reverse. ``graph``'s own
#: ``KIND_*`` constants are the router's copy and are asserted to agree with these in the
#: routing tests.
KIND_SIMPLE = "simple"
KIND_MULTI_PART = "multi_part"
KIND_OUT_OF_SCOPE = "out_of_scope"

#: P1's ``sub_queries`` bound ("emit sub_queries: 1-4 atomic questions"). Enforced here as
#: well as in ``decompose_question`` so the plan that reaches the debug panel and the router
#: is already inside the bound — the fan-out cost of STRESS-TEST-RAG #6 is a *plan* problem.
MAX_SUB_QUERIES = 4

#: ``missing_aspect`` values P1 defines, mirrored for the same leaf-module reason.
MISSING_TIME_PERIOD = "time_period"
MISSING_TOPIC = "topic"
MISSING_ENTITY = "entity"

#: The partial-update key P1's failure handling names: the marker a degraded plan carries so
#: the debug panel shows the refusal came from a planner that failed rather than one that
#: reasoned.
#:
#: ``GraphState`` does not declare this key — ``docs/ARCHITECTURE.md`` §State Schema freezes
#: the state's fields, and adding one is a shared-contract change rather than a node's. The
#: consequence is measured, not assumed: langgraph's typed state drops a key it does not
#: declare, so the flag reaches a **direct caller of this node** but is absent from the merged
#: state ``invoke`` returns. It is still emitted because the caller that matters most is the
#: node's own test, and because the alternative — silently omitting it — loses the signal
#: entirely. Making the flag visible to the debug panel needs the field added to
#: :class:`~src.chains.state.GraphState`; that is recorded as a gap rather than worked around
#: here by writing an unrelated state field.
#:
#: The refusal itself is honest without it: ``missing_aspect=None`` plus an out-of-scope kind
#: yields the generic refusal copy, and ``refuse`` renders that rather than a claim about why.
PLANNER_DEGRADED_KEY = "planner_degraded"

#: How many characters of the catalog's own JSON the node will hand the model before it stops
#: pretending to plan and says what it saw. A catalog is metadata for a handful of documents,
#: so a twelve-thousand-character one is a bug at ingest time rather than a planning problem;
#: past the cap the payload is truncated with a visible marker so the model is told the
#: catalog it received is incomplete rather than left to treat a smaller one as the whole one.
MAX_CATALOG_CHARS = 12_000

#: The ``QueryPlan`` JSON schema handed to the gateway's typed-output path (ADR-005: the JSON
#: nodes request ``response_format={"type": "json_schema", ...}``).
#:
#: Provenance is recorded honestly: ``docs/ARCHITECTURE.md`` carries the ``QueryPlan`` shape in
#: the ``GraphState`` schema, ``src/chains/state.py`` holds the runtime ``QueryPlan`` dataclass,
#: and ``ADR-005`` records where the JSON schemas are declared — inline in the node that sends
#: each one: this module's ``QUERY_PLAN_SCHEMA`` and ``GROUNDING_SCHEMA`` in
#: ``src/chains/nodes/verify.py``. No shared schema module exists in this tree, so this node
#: declares the schema it needs and the gateway validates against what it is handed. The shape
#: matches the QueryPlan the docs freeze, with one deliberate addition: an ``enum`` over
#: ``kind`` (absent from the informative schema in ``docs/prompts/query-analysis.md``), so a
#: plan with an unrecognised ``kind`` is a schema violation the gateway retries on and this
#: node then degrades on — instead of a malformed plan the router would have to second-guess.
QUERY_PLAN_SCHEMA: dict[str, Any] = {
    "title": "QueryPlan",
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": [KIND_SIMPLE, KIND_MULTI_PART, KIND_OUT_OF_SCOPE]},
        "sub_queries": {"type": "array", "items": {"type": "string"}},
        "required_sections": {"type": "array", "items": {"type": "string"}},
        "fiscal_period": {"type": ["string", "null"]},
        "missing_aspect": {
            "type": ["string", "null"],
            "enum": [MISSING_TIME_PERIOD, MISSING_TOPIC, MISSING_ENTITY, None],
        },
    },
    "required": ["kind", "sub_queries", "required_sections", "fiscal_period", "missing_aspect"],
}

#: The heading whose fenced block is the system prompt, and the one whose block is the input
#: template — both named rather than positional so a reordered asset section cannot shift
#: the slice (``src/utils/prompts.py`` rule: the file is a versioned document).
_SYSTEM_HEADING = "## System prompt"
_TEMPLATE_HEADING = "## Input template"


def _prompt_parts() -> tuple[str, str]:
    """P1's system prompt and input template, sliced out of the loaded asset.

    ``assert_min_version`` reads ``scripts/prompts_meta.json``'s floor for this node, so a
    checkout whose prompts are older than the node's contract raises here instead of
    silently planning with a stale prompt (``docs/prompts/README.md`` rule 5).
    """

    doc = assert_min_version(PROMPT_ID)
    return _fenced_block(doc.body, _SYSTEM_HEADING), _fenced_block(doc.body, _TEMPLATE_HEADING)


def _fenced_block(body: str, heading: str) -> str:
    """The fenced block under ``heading``, or ``ValueError`` when the asset lacks one."""

    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != heading:
            continue
        opener = index + 1
        while opener < len(lines) and not lines[opener].lstrip().startswith("```"):
            opener += 1
        block: list[str] = []
        position = opener + 1
        while position < len(lines) and not lines[position].lstrip().startswith("```"):
            block.append(lines[position])
            position += 1
        text = "\n".join(block).strip("\n")
        if text:
            return text
        break
    raise ValueError(f"prompt {PROMPT_ID} has no fenced block under {heading!r}")


def _standalone_question(state: GraphState) -> str:
    """The question to plan: the contextualized rewrite, else the raw question.

    The node is also registered on the entry point of a graph whose ``contextualize_question``
    has not run (a test invoking it directly), so the fallback is what makes the node usable
    on its own. ``standalone_question`` is preferred whenever present because P2 exists to
    improve exactly this text and planning on the raw question would discard that work.
    """

    for key in ("standalone_question", "question"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _documents(state: GraphState) -> list[DocSummary]:
    """The catalog's :class:`DocSummary` entries, tolerating a wrongly-typed field.

    ``invoke`` types ``doc_catalog`` as ``list[Any]`` at its own boundary, so an entry that
    is not a :class:`DocSummary` is dropped rather than raised on: a catalog the node cannot
    read plans the same way an empty one does, which is an honest plan rather than a crash.
    """

    raw = state.get("doc_catalog")
    if isinstance(raw, str | bytes) or not isinstance(raw, (list, tuple)):
        return []
    return [entry for entry in raw if isinstance(entry, DocSummary)]


def _section_text(path: Any) -> str:
    """One section path as ``"A > B"``.

    ``DocSummary.sections`` is typed ``tuple[tuple[str, ...], ...]`` and the ingest catalog
    writes it that way (``scripts/build_index.py:sections_of``), but the path arrives through
    a ``list[Any]`` boundary: a JSON row or a hand-built test summary can hand back a plain
    string, and iterating a string yields characters — so ``"MD&A, Income Statements"`` would
    be rendered one glyph per entry. A string is therefore taken as the whole path, and
    anything else is joined from the entries it can read.
    """

    if isinstance(path, str):
        return path.strip()
    if isinstance(path, (list, tuple)):
        return " > ".join(part.strip() for part in path if isinstance(part, str) and part.strip())
    return ""


def _catalog_json(docs: list[DocSummary]) -> str:
    """The catalog as P1's input template renders it: JSON, one object per document.

    Section paths are emitted as the ``"A > B"`` strings the ingest catalog actually stores
    (``scripts/build_index.py:sections_of``) rather than as nested lists, because the model
    reads the catalog as flat text and a path rendered as a nested array costs it tokens to
    reassemble. An empty catalog renders as ``[]`` rather than being omitted, so the model
    is told "no documents" instead of being handed a template it has to interpret around.
    """

    entries: list[dict[str, Any]] = []
    for doc in docs:
        texts = [_section_text(path) for path in (doc.sections or ())]
        entries.append(
            {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "page_count": doc.page_count,
                "sections": [text for text in texts if text],
            }
        )
    text = json.dumps(entries, ensure_ascii=True, indent=2)
    if len(text) > MAX_CATALOG_CHARS:
        # The plan is still produced — the truncated tail is visible in the payload the model
        # receives, so a truncation cannot look like a smaller catalog by accident.
        return f"{text[:MAX_CATALOG_CHARS]}\n... (catalog truncated)"
    return text


def _covered_years(docs: list[DocSummary]) -> set[int]:
    """The fiscal years the catalog's own text shows, as :func:`checks.catalog_years` reads them.

    The catalog is the only record the node has of what is on disk, and it is read here with
    exactly the same rule ``refuse`` applies when it writes "the documents cover fiscal years
    FY2021-FY2023" — filenames and section paths, with an annual report's own year standing for
    the three-year span its statements carry. Sharing the rule rather than restating it is what
    makes the gate and the refusal copy agree about coverage by construction.
    """

    parts: list[str] = []
    for doc in docs:
        parts.append(doc.filename)
        parts.extend(_section_text(path) for path in (doc.sections or ()))
    return catalog_years(" ".join(parts))


def _uncovered_period_plan(question: str, covered: set[int]) -> QueryPlan:
    """The ``out_of_scope`` plan the symbolic fiscal-year gate produces for an uncovered year.

    ``missing_aspect`` is ``time_period`` — the same value P1's own schema expects and
    ``refuse`` renders as "the documents cover fiscal years FY2021-FY2023" — so the
    deterministic gate and the planner's own classification land on one indistinguishable
    outcome. The gap is a fact about the catalog, not about the question's phrasing, so the
    plan records the asked year in ``fiscal_period`` and leaves ``sub_queries`` and
    ``required_sections`` empty: there is nothing to retrieve and nothing to decompose.
    """

    asked = sorted(asked_years(question))
    logger.info(
        "year_coverage_gate: %s is outside the catalog's fiscal years %s; routing to refuse",
        asked,
        sorted(covered),
    )
    return QueryPlan(
        kind=KIND_OUT_OF_SCOPE,
        sub_queries=[],
        required_sections=[],
        fiscal_period=str(asked[0]) if asked else None,
        missing_aspect=MISSING_TIME_PERIOD,
    )


def _params(state: GraphState) -> LLMParams:
    """The caller's sampling parameters, with this node's pinned temperature applied.

    P1 runs at temperature 0.0 — a planner that classifies differently for the same question
    makes the route nonreproducible, which is the one property the deterministic pipeline
    (ADR-010) needs from it. The user's ``model``/``top_p``/``max_tokens`` flow through, so
    the sidebar's picker still applies to planning. The pin is read from
    ``scripts/prompts_meta.json`` rather than hardcoded, the same way
    ``contextualize_question`` reads its own, and is applied only when the caller left the
    temperature unset so an explicitly chosen temperature still wins.
    """

    raw = state.get("params")
    params = raw if isinstance(raw, LLMParams) else LLMParams()
    if params.temperature is not None:
        return params
    floor = _temperature_floor()
    if floor is None:
        return params
    return replace(params, temperature=floor)


def _temperature_floor() -> float | None:
    """The temperature ``scripts/prompts_meta.json`` pins for this node, or ``None``.

    Any failure to read the meta file returns ``None``: the pinned temperature is a
    determinism guard, not a gate, and a missing meta file must not make the node
    unimportable. The value it returns is applied only when the caller left the
    temperature unset, so a user who explicitly set one still wins.
    """

    try:
        entry = node_meta(NODE_NAME)
    except Exception:  # a missing meta file must not make the node unimportable
        return None
    temperature = entry.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        return float(temperature)
    return None


def _texts(value: Any, limit: int = MAX_SUB_QUERIES) -> list[str]:
    """``value`` as a list of non-empty strings, de-duplicated in order, capped at ``limit``.

    A schema-validated reply is already well-typed, so this is the shape guard against the
    *joint* the schema cannot express: ``sub_queries`` holds strings and is capped, but the
    cap is asserted here rather than trusted, because a plan over the cap is the fan-out
    blow-up of STRESS-TEST-RAG #6 reaching the graph.
    """

    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return []
    texts: list[str] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            continue
        text = entry.strip()
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _optional_text(value: Any) -> str | None:
    """``value`` as a stripped string, or ``None`` when it is absent, empty, or not text."""

    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _plan_from(payload: dict[str, Any]) -> QueryPlan | None:
    """The validated payload as a :class:`QueryPlan`, or ``None`` when it is unusable.

    Every field is read defensively even though the gateway has already validated the
    payload against :data:`QUERY_PLAN_SCHEMA`: this node must not assume its own caller
    (a node can be invoked directly by a test), and a plan is cheap to build compared to
    the cost of routing on a half-typed one.
    """

    kind = payload.get("kind")
    if kind not in (KIND_SIMPLE, KIND_MULTI_PART, KIND_OUT_OF_SCOPE):
        return None
    return QueryPlan(
        kind=kind,
        sub_queries=_texts(payload.get("sub_queries")),
        required_sections=_texts(payload.get("required_sections"), limit=64),
        fiscal_period=_optional_text(payload.get("fiscal_period")),
        missing_aspect=_optional_text(payload.get("missing_aspect")),
    )


def _degraded_plan(reason: str, question: str) -> QueryPlan:
    """The refusal plan P1 prescribes for a planner that could not produce one.

    ``kind=out_of_scope`` with ``missing_aspect=None`` is exactly P1's failure handling, and
    it is not a guess about the question: the node genuinely cannot say *why* this question
    is unanswerable because it never got an answer from the model about the question. A
    refusal that hides this is the failure STRESS-TEST-RAG #7 exists to catch, so the reason
    is logged and the plan is marked rather than presented as a reasoned refusal.
    """

    logger.warning(
        "planner_degraded: %s; classifying out_of_scope with no missing_aspect " "(question=%r)",
        reason,
        question[:120],
    )
    return QueryPlan(
        kind=KIND_OUT_OF_SCOPE,
        sub_queries=[],
        required_sections=[],
        fiscal_period=None,
        missing_aspect=None,
    )


def _messages(system: str, template: str, question: str, catalog: str) -> list[dict[str, Any]]:
    """P1's payload: the system prompt, then its input template with both placeholders filled.

    The template's own placeholders are the contract, so they are filled by name rather than
    by position — the asset's ``{catalog_json}`` and ``{standalone_question}`` are the frozen
    names and a renamed placeholder must fail loudly here rather than silently send the model
    an unfilled brace.
    """

    user_turn = template
    for placeholder, value in (("{catalog_json}", catalog), ("{standalone_question}", question)):
        if placeholder not in user_turn:
            raise ValueError(
                f"prompt {PROMPT_ID} input template has no {placeholder!r} placeholder"
            )
        user_turn = user_turn.replace(placeholder, value)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_turn},
    ]


# --- the node ----------------------------------------------------------------------


@trace_node(NODE_NAME)
def analyze_question(state: GraphState) -> dict[str, Any]:
    """Classify the standalone question and emit the plan that routes the graph (P1, ADR-001).

    Reads ``standalone_question`` (else ``question``), ``doc_catalog`` and ``params``; writes
    ``query_plan`` and reports its gateway accounting under
    :data:`~src.utils.tracing.TOKEN_KEY`. No other state key is written — in particular
    ``node_trace``, which ``@trace_node`` owns.

    Never fails the turn (P1's failure handling): a schema violation the gateway could not
    recover from degrades to an honest refusal. ``LLMError`` from a transport failure is
    also absorbed rather than raised, because the plan the node can honestly produce is
    "I could not plan this" and the route for that is ``refuse``, not an error banner — a
    planner outage costs the turn a *route it should have taken*, not the turn's honesty.
    A missing prompt asset still raises: that is an infrastructure bug, not a model failure
    (ADR-007's typed-error spine).

    One question never reaches the planner at all: a question naming exactly one fiscal year
    the catalog demonstrably does not cover. That gap is a fact about the catalog rather than
    a judgement about the question, so it is settled symbolically by
    :func:`~src.chains.checks.year_coverage_gap` before any POST — the same split of
    deterministic-structure vs. neural-semantics ADR-010 draws, applied to the routing
    decision instead of the answer.

    Returns:
        ``{"query_plan": plan}``, plus ``"planner_degraded": True`` when the plan was
        degraded, plus the accounting key when a gateway call succeeded. A gated question
        returns only the plan: no gateway call was made, so there is no accounting to report.
    """

    question = _standalone_question(state)
    system, template = _prompt_parts()
    docs = _documents(state)
    catalog = _catalog_json(docs)
    params = _params(state)

    if not question:
        # No question is a caller bug the graph boundary already rejects (``initial_state``
        # raises ``GraphStateError``), so reaching here means a node was invoked directly.
        # An empty question plans nothing and must not be routed to retrieval.
        raise ValueError("analyze_question received no question to plan")

    # --- the symbolic fiscal-year gate (ADR-010), before any model call ---------------
    # A year the documents cannot cover is decided by the catalog, not by the planner: the
    # catalog is the same ingest-time record ``refuse`` reads to name its coverage, so the
    # gate is settled deterministically and the question never pays for a POST whose answer
    # it does not need. It is the fiscal-year sibling of the numeric pre-pass
    # ``groundedness_check`` runs on figures, moved earlier in the graph.
    #
    # The gate is deliberately narrow — it fires only for a question naming exactly one
    # fiscal year, and only when that year lies outside the catalog's own coverage — so it
    # never overrides the planner on a comparison, an entity- or topic-gap question, or a
    # question naming no period at all. Everything else still routes through the planner.
    covered = _covered_years(docs)
    if year_coverage_gap(covered, question):
        return {"query_plan": _uncovered_period_plan(question, covered)}

    try:
        result = asyncio.run(
            build_llm(params).chat(
                _messages(system, template, question, catalog), params, QUERY_PLAN_SCHEMA
            )
        )
    except SchemaViolationError as error:
        # P1: "one stricter retry, then classify out_of_scope with missing_aspect: null and
        # set the planner_degraded flag." The retry itself is the gateway's (ADR-005), so by
        # the time it raises there is nothing left to ask for.
        return {
            "query_plan": _degraded_plan(f"{type(error).__name__}: {error}", question),
            PLANNER_DEGRADED_KEY: True,
        }
    except LLMError as error:
        # A transport failure is the same honest gap: no plan was produced, so the route is a
        # refusal that says so rather than a plausible route built from nothing.
        return {
            "query_plan": _degraded_plan(f"{type(error).__name__}: {error}", question),
            PLANNER_DEGRADED_KEY: True,
        }

    payload = result.parsed
    plan = _plan_from(payload) if isinstance(payload, dict) else None
    if plan is None:
        # The gateway validated the reply against ``QUERY_PLAN_SCHEMA``, so a payload this
        # node cannot read is a defect rather than a model failure — but it is still no plan,
        # and the honest route for no plan is still a refusal.
        return {
            "query_plan": _degraded_plan(
                f"gateway returned an unusable plan: {type(payload).__name__}",
                question,
            ),
            PLANNER_DEGRADED_KEY: True,
        }

    return {"query_plan": plan, TOKEN_KEY: result}

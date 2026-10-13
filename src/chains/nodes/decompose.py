"""``decompose_question``: the symbolic application of the planner's decomposition (T10b).

ADR-010 splits the pipeline's work in two: the **neural** layer decides what a question
*means*, and the **symbolic** layer applies the decision the neural layer produced. This
node is the second half of that split for multi-part questions. ``analyze_question``
already answered "how many parts does this have, and what are they?" — one model call,
validated against a JSON schema, capped at four. What is left is normalization, and
normalization is the part a model must not be trusted with: a sub-query list that is
unsorted, duplicated, or longer than the fan-out cap is a cost and context-budget problem
(``docs/STRESS-TEST-RAG.md`` row 6), not a semantic one.

So there is **no LLM call and no network access in this module** — no ``openai`` import is
reachable from here, and nothing under ``src.utils.llm`` is invoked. The node is a pure
function of the plan, which makes it testable offline under ``MOCK_LLM=1`` and makes
identical inputs produce identical output.

What it normalizes
------------------
``sub_queries`` arrives typed ``list[str]`` and already capped at four by ``analyze_question``
(``MAX_SUB_QUERIES`` there, mirrored as :data:`MAX_SUB_QUERIES` here). A node's own caller
can still hand it a plan it did not build — a test invoking it directly, a future caller
deserializing a stored plan — so each of the three invariants the fan-out depends on is
re-asserted rather than trusted:

* **text hygiene** — each entry is stripped; an empty or whitespace-only entry is dropped
  rather than retrieved, because a blank query burns a retrieval pass on nothing.
* **uniqueness** — the list is de-duplicated **case-insensitively** in first-seen order.
  Case-insensitively because "What was FY2023 revenue?" and "what was fy2023 revenue?" are
  one question to a retriever, and two retrieval passes over one question is the fan-out
  cost row 6 exists to prevent. First-seen order is kept because the planner's ordering
  encodes its own sense of what matters most.
* **the bound** — the list is capped at four. ``hybrid_retrieve`` applies the same cap (it
  owns the "≤ 4 retrievals per question" bound and does not delegate it), so the two rules
  are duplicated by the same deliberate choice ``analyze_question`` and this node make: a
  plan is a model-produced artifact, and each layer that reads it re-checks the bound before
  spending quota on it.

Nothing is ever *generated* here. A decomposition decision this node does not have is not
invented, and no sub-query is rewritten, merged, or reworded — a comparison stays whole
(P1's rule, and PLAN.md's T10b acceptance: a sub-query reading "what was operating income
in FY22?" cannot itself answer the comparison that was asked, so this module never splits
one either).

Where the result goes
--------------------
``hybrid_retrieve`` reads ``state["query_plan"].sub_queries`` — the plan object, not a
top-level key — so the normalized list is written back into the plan by replacing it with a
copy carrying the new sub-queries. ``GraphState`` gives every key ``last_value`` channel
semantics with no reducers, so this return overwrites the plan those two nodes share, and
``kind`` / ``required_sections`` / ``fiscal_period`` / ``missing_aspect`` are preserved
unchanged: decomposition owns sub-queries and nothing else about the verdict. ``QueryPlan``
is a frozen dataclass, so it is rebuilt through :func:`dataclasses.replace` rather than
mutated in place — a plan the debug panel already holds a reference to must not change
underneath it.

State contract (``docs/ARCHITECTURE.md`` Node Contract Table)
-------------------------------------------------------------
| Reads | Writes | LLM |
|---|---|---|
| ``query_plan`` | ``query_plan`` (``sub_queries`` only) | no |

Edge cases, none of which raise
------------------------------
* **no plan, or a plan that is not a :class:`QueryPlan`** — there is nothing to decompose,
  so the update writes a plan carrying the empty sub-query list. ``kind`` stays ``""``, the
  same spelling :func:`~src.chains.state.final_envelope` uses for an absent plan, so the
  node never asserts a routing verdict it did not have evidence for. Downstream this is an
  honest empty fan-out: ``hybrid_retrieve`` sees no sub-queries and falls back to the whole
  question as one query, which is the answer that was actually asked.
* **empty sub-queries** — the plan is returned with an empty list, which is the same
  fallback. An empty plan is a real result, not a failure.
* **one sub-query** — passes through unchanged after stripping, which is the identity case.
* **more than four survive de-duplication** — the first four are kept in order and the
  count of the dropped ones is logged. ``GraphState`` declares no field to carry that count
  (``docs/ARCHITECTURE.md`` §State Schema freezes the fields), and a new key would be
  dropped by langgraph's typed state anyway, so the line goes to the log — the same channel
  ``hybrid_retrieve`` uses for the context budget it exhausted, and the channel that
  survives a full graph run.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from typing import Any

from src.chains.state import GraphState, QueryPlan
from src.utils.tracing import trace_node

__all__ = ["MAX_SUB_QUERIES", "NODE_NAME", "decompose_question"]

logger = logging.getLogger(__name__)

#: The node's registered name, in ``NODE_NAMES``, in every ``NodeEvent``, and the key this
#: module's own log lines use.
NODE_NAME = "decompose_question"

#: The fan-out cap (ADR-001's "max 4", STRESS-TEST-RAG #6): sub-queries past this are not
#: run, so a long question cannot quietly become seven retrievals. Mirrored from
#: ``analyze_question``'s MAX_SUB_QUERIES rather than imported, so this module stays a leaf
#: — ``graph`` imports the nodes, not the reverse.
MAX_SUB_QUERIES = 4

#: The ``kind`` spelling for a plan the node did not have. ``final_envelope`` uses the same
#: value when the state carries no plan at all, so a decomposed-absent plan renders exactly
#: as an absent one instead of as a verdict.
_ABSENT_KIND = ""


def _sub_query_texts(value: Any) -> list[str]:
    """``value`` as a list of strings, tolerating a wrongly-typed field.

    The plan is typed, but it can arrive through a boundary that is not this repo's type
    checker — a JSON row, a hand-built test plan — so the shape is guarded here the same way
    ``analyze_question`` guards its own payload. A bare string is rejected rather than
    iterated, because iterating one yields characters.
    """

    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return []
    return [entry for entry in value if isinstance(entry, str)]


def _deduplicated(entries: Sequence[str]) -> list[str]:
    """Strip, drop empties, and de-duplicate case-insensitively in first-seen order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for entry in entries:
        text = entry.strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


@trace_node(NODE_NAME)
def decompose_question(state: GraphState) -> dict[str, Any]:
    """Normalize the planner's sub-queries into the bounded, deterministic fan-out input.

    Fully deterministic: no LLM call, no network, no import of the gateway, so the same plan
    always yields the same fan-out and the node works offline under ``MOCK_LLM=1``. Never
    raises and never fabricates a decomposition — it only reasserts the invariants the
    retrieval cost depends on (text hygiene, case-insensitive uniqueness, the ≤ 4 cap).

    Reads ``query_plan``; writes ``query_plan`` with ``sub_queries`` replaced by the
    normalized list. Every other field of the plan is carried through unchanged, so the
    router's verdict, the section filter and the fiscal period survive the node that runs
    immediately before retrieval.

    Returns:
        ``{"query_plan": <plan with normalized sub_queries>}`` — the plan
        ``hybrid_retrieve`` fans out over. An absent or unusable plan yields an empty
        sub-query list and an empty ``kind``, which lets ``hybrid_retrieve`` fall back to
        the whole question as a single query rather than failing the turn.
    """

    plan = state.get("query_plan")
    deduplicated = _deduplicated(
        _sub_query_texts(plan.sub_queries) if isinstance(plan, QueryPlan) else []
    )
    queries = deduplicated[:MAX_SUB_QUERIES]
    dropped = len(deduplicated) - len(queries)
    if dropped:
        logger.warning(
            "%s dropped %d sub-quer%s of %d to stay under the %d sub-query fan-out cap "
            "(ADR-001's bounded fan-out, STRESS-TEST-RAG #6)",
            NODE_NAME,
            dropped,
            "y" if dropped == 1 else "ies",
            len(deduplicated),
            MAX_SUB_QUERIES,
        )

    decomposed = plan if isinstance(plan, QueryPlan) else QueryPlan(kind=_ABSENT_KIND)
    return {"query_plan": dataclasses.replace(decomposed, sub_queries=queries)}

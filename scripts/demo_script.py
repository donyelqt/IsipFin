"""The 6-step demo script from docs/STRESS-TEST-DELIVERY.md, run under MOCK_LLM=1.

Steps 1-2 (compose + upload) are infrastructure; the ingest path is exercised here by
loading the real persisted index as the node sees it. Steps 3-6 are the substance and are
run end to end through the compiled graph, with each node's plain-language report.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MOCK_LLM", "1")

from src.chains.state import DocSummary, QueryPlan
from src.core.retrieval import retrieve

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def catalog() -> list[DocSummary]:
    path = Path("data/index/docs.jsonl")
    if not path.is_file():
        return []
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return [
        DocSummary(
            doc_id=str(row.get("doc_id", "")),
            filename=str(row.get("filename", "")),
            sections=tuple(tuple(s.split(" > ")) for s in row.get("sections", []) if s.strip()),
            page_count=int(row["page_count"]) if row.get("page_count") is not None else None,
        )
        for row in rows
        if row.get("doc_id")
    ]


def show_step_prelude(state: dict, plan: QueryPlan | None) -> None:
    print(f"      plan.kind          = {plan.kind!r}")
    print(f"      plan.sub_queries   = {plan.sub_queries!r}")
    print(f"      plan.required_sec  = {plan.required_sections!r}")
    print(f"      plan.fiscal_period = {plan.fiscal_period!r}")
    print(f"      plan.missing_asp   = {plan.missing_aspect!r}")
    print(f"      state['retries']   = {state.get('retries')!r}")


def run_step(number: int, question: str, docs: list[DocSummary]) -> None:
    from src.chains.graph import (
        ANALYZE_ROUTES,
        RETRIEVE_ROUTES,
        initial_state,
        route_after_analyze,
        route_after_retrieve,
    )
    from src.chains.nodes.analyze import analyze_question
    from src.chains.nodes.contextualize import contextualize_question
    from src.chains.nodes.retrieve import hybrid_retrieve

    print(f"--- STEP {number}: {question!r}\n")
    state: dict = initial_state(question, doc_catalog=docs)

    state.update(contextualize_question(state))
    state.update(analyze_question(state))

    plan = state.get("query_plan")
    plan = plan if isinstance(plan, QueryPlan) else QueryPlan(kind="")
    show_step_prelude(state, plan)
    route = route_after_analyze(state)
    print(f"      route_after_analyze -> {route!r}")
    print(f"      route_after_analyze path_map = {ANALYZE_ROUTES.get(route)!r}")

    # ``route_after_analyze`` returns the *plan token* (``simple`` / ``multi_part`` /
    # ``out_of_scope``); ``ANALYZE_ROUTES`` maps the token to the node name. Comparing
    # against the token keeps the runner's branch identical to the graph's own routing.
    if route == "out_of_scope":
        print("      (routed to refuse — retrieval never runs)\n")
        return

    if route == "multi_part":
        from src.chains.nodes.decompose import decompose_question

        state.update(decompose_question(state))
        sub_queries = state["query_plan"].sub_queries if state.get("query_plan") else []
        print(f"      decompose_question -> {len(sub_queries)} sub-quer(y/ies)")
        for index, text in enumerate(sub_queries, start=1):
            print(f"        {index}. {text!r}")
    else:
        print("      (simple: the plan names no sub-queries — the question is the query)\n")

    update = hybrid_retrieve(state)
    state.update(update)
    print("      [hybrid_retrieve returned]")
    print(f"        retries           = {update['retries']}")
    print(f"        retrieved         = {len(update['retrieved'])} passage(s)")
    for chunk in update["retrieved"]:
        print(
            f"          rank={chunk.fused_rank} page={chunk.page} "
            f"dense={chunk.dense_score:.4f} sparse={chunk.sparse_score:.4f} "
            f"chars={len(chunk.parent_text)}"
        )
    print(f"        note              = {update['retrieval_note']}")

    # The router reads the state AFTER the node's update — the loop bound in action.
    gate = route_after_retrieve(state)
    print(f"        route_after_retrieve -> {gate!r} => {RETRIEVE_ROUTES.get(gate, gate)!r}")
    print()


def main() -> int:
    docs = catalog()
    print("=" * 78)
    print("demo script: docs/STRESS-TEST-DELIVERY.md steps 3-6, MOCK_LLM=1")
    print("=" * 78)
    print(f"\ncatalog: {len(docs)} document(s) from data/index/docs.jsonl")
    for doc in docs:
        print(f"  {doc.filename}  pages={doc.page_count}  sections={len(doc.sections)}")

    print("\n" + "-" * 78)
    print("steps 1-2 (compose up / upload) are infra; the ingest path is the real index")
    print("-" * 78 + "\n")

    run_step(3, "What was Microsoft's revenue in FY2023?", docs)
    run_step(4, "Compare operating income FY23 vs FY22", docs)
    run_step(5, "which segment drove that?", docs)
    run_step(6, "What was revenue in 2019?", docs)

    print("-" * 78)
    print("raw retrieve() over the real data/index (module-level, no graph)")
    print("-" * 78)
    for question in (
        "What was Microsoft's revenue in FY2023?",
        "Compare operating income FY23 vs FY22",
    ):
        hits = retrieve(question, k=8)
        print(f"\n  {question!r} -> {len(hits)} hit(s)")
        for chunk in hits:
            print(
                f"    rank={chunk.fused_rank} page={chunk.page} "
                f"dense={chunk.dense_score:.4f} sparse={chunk.sparse_score:.4f}"
            )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

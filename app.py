"""IsipFin — Streamlit entrypoint.

UI-only shell (ADR-009): no retrieval, prompt, or gateway logic lives here or in
src/components/. The UI calls orchestration.invoke() and renders AnswerEnvelope
fields. Built up in phases 4 (T18-T21); intentionally minimal until then.

Run: streamlit run app.py   (see docs/RUNBOOK.md)
"""

from __future__ import annotations


def main() -> None:
    # Placeholder body. T18 replaces this with the chat shell, upload widget and
    # empty state; the acceptance gate is the 6-step demo script in
    # docs/STRESS-TEST-DELIVERY.md.
    print("IsipFin — UI not yet built (T18-T21). See docs/PLAN.md phase 4.")


if __name__ == "__main__":
    main()

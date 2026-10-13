"""Environment loading — the one place `.env` is read.

`docs/SPEC.md`, `docs/RUNBOOK.md` and `README.md` all document
`cp .env.example .env` as a setup step, but nothing in this repo actually read the
file: every module read `os.environ` directly. The documented native path
therefore ran with no key and no `MOCK_LLM`, which the gateway correctly reported
as an auth/transport failure and the planner correctly degraded into an honest
refusal — a *correct* degradation hiding an *incorrect* setup.

`load_env()` is idempotent and cheap, so callers may invoke it defensively.
Real environment variables always win: `load_dotenv` does not override what is
already set, which is what makes `MOCK_LLM=1 python scripts/eval_rag.py` behave
the way the docs promise.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

#: The repository root, derived from this file rather than the working directory so
#: `streamlit run app.py`, `python scripts/build_index.py` and pytest all find the
#: same `.env` no matter where they are launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_LOADED = False


def load_env() -> bool:
    """Load `.env` from the repo root once; return whether the file existed.

    The return value is informational (tests and the debug panel read it); callers
    must not branch on it to decide whether a key is available, because the
    environment may supply one without any file at all.
    """
    global _LOADED
    if _LOADED:
        return (REPO_ROOT / ".env").is_file()
    env_file = REPO_ROOT / ".env"
    load_dotenv(env_file, override=False)
    _LOADED = True
    return env_file.is_file()

"""Environment loading — the one place `.env` is read.

`docs/SPEC.md`, `docs/RUNBOOK.md` and `README.md` all document
`cp .env.example .env` as a setup step, and this module is what makes that file real:
every other module reads `os.environ` directly.

`load_env()` is idempotent and cheap, so callers may invoke it defensively.

Precedence — two rules, and both are deliberate
-----------------------------------------------
1. **Real environment variables win over the file.** `load_dotenv(override=False)`
   does not touch what is already set, which is what makes
   `MOCK_LLM=1 python scripts/eval_rag.py` behave the way the docs promise, and what
   lets CI or a grader inject a key without editing anything.
2. **A value the user typed in `.env` wins over an ambient export — but only for the
   names that decide *where a request goes*.** See :data:`ENV_FILE_AUTHORITY` for why
   that allowlist is exactly seven names long and nothing else.

Why rule 2 exists at all: a machine can have `OPENAI_BASE_URL` exported for one
purpose (on this host it points at NVIDIA) while the repo's own `.env` asks for Groq.
Under rule 1 alone, the typed file would be silently ignored and the key would be sent
somewhere else — a failure that looks like a bad credential and is actually bad
routing. A user-authored config file outranks an unrelated ambient variable.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

#: The repository root, derived from this file rather than the working directory so
#: `streamlit run app.py`, `python scripts/build_index.py` and pytest all find the
#: same `.env` no matter where they are launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_LOADED = False

#: The `.env` names that are authoritative over an ambient export.
#
#: These six/seven variables — the chat key, the endpoint and the model, in both the
#: `OPENAI_*` spelling and the `LLM_*` / `GROQ_*` aliases that exist precisely so a
#: dedicated credential cannot be hijacked — determine **which provider a request is
#: aimed at**. A value the user wrote in this repo's own config file must beat an
#: ambient export for them, or the file is documentation rather than configuration.
#:
#: The list is deliberately narrow, and deliberately excludes every operational
#: name, so the general rule "environment beats file" stays intact everywhere else:
#: `MOCK_LLM` (so `MOCK_LLM=1` in the shell keeps switching the whole process to the
#: deterministic offline path), `EMBEDDING_PROVIDER` / `EMBEDDING_BASE_URL` /
#: `EMBEDDING_MODEL` (which decide the vector width, so an ambient choice must not be
#: silently overridden by a stale file), `INDEX_DIR`, and every other setting a host
#: or a CI job may inject deliberately.
ENV_FILE_AUTHORITY: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "GROQ_API_KEY",
)


def _apply_env_file_authority(env_file: Path) -> None:
    """For the authoritied names, let a non-blank value **in the file** win.

    Only a name the file actually defines with a non-blank value is written back, so:

    * `OPENAI_API_KEY=` (as `.env.example` ships it) does not clobber a real key — a
      blank value means "unset here", exactly as `_first_env` in `src/utils/llm.py`
      already treats it, and must never erase an ambient credential;
    * a name the file does not mention keeps whatever the environment resolved, so the
      general precedence rule is unchanged for everything else;
    * a name whose value matches the ambient one is a no-op.

    Blank means empty-or-whitespace, and the comparison is made on the parsed value,
    so a quoted `" "` is blank too.
    """
    file_values = dotenv_values(env_file)
    for name in ENV_FILE_AUTHORITY:
        value = file_values.get(name)
        if value is None:
            continue
        if not value.strip():
            continue
        os.environ[name] = value


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
    if env_file.is_file():
        _apply_env_file_authority(env_file)
    _LOADED = True
    return env_file.is_file()

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_LOADED = False

ENV_FILE_AUTHORITY: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "GROQ_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_PROVIDER",
)


def _apply_env_file_authority(env_file: Path) -> None:
    file_values = dotenv_values(env_file)
    for name in ENV_FILE_AUTHORITY:
        value = file_values.get(name)
        if value is None:
            continue
        if not value.strip():
            continue
        os.environ[name] = value


def load_env() -> bool:
    global _LOADED
    if _LOADED:
        return (REPO_ROOT / ".env").is_file()
    env_file = REPO_ROOT / ".env"
    load_dotenv(env_file, override=False)
    if env_file.is_file():
        _apply_env_file_authority(env_file)
    _LOADED = True
    return env_file.is_file()

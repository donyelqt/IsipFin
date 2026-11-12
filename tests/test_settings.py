from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from src.utils import settings
from src.utils.llm import build_llm, resolved_config
from src.utils.mock_llm import MockLLM

REPO_ROOT = Path(__file__).resolve().parents[1]

_AMBIENT_CHAT_NAMES = (
    "GROQ_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "MOCK_LLM",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
)

AUTHORITIED_NAMES = settings.ENV_FILE_AUTHORITY


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in _AMBIENT_CHAT_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(settings, "_LOADED", False)
    yield


def _write_env(contents: dict[str, str]) -> Path:
    body = "".join(f"{name}={value}\n" for name, value in contents.items())
    path = settings.REPO_ROOT / ".env"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", AUTHORITIED_NAMES)
def test_a_blank_env_value_does_not_clobber_an_ambient_one(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, f"ambient-{name}")
    _write_env({name: ""})
    settings.load_env()
    assert os.environ[name] == f"ambient-{name}"


def test_whitespace_counts_as_blank_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
    _write_env({"OPENAI_BASE_URL": "   "})
    settings.load_env()
    assert os.environ["OPENAI_BASE_URL"] == "https://ambient.example/v1"


def test_a_missing_env_file_leaves_the_environment_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path / "absent")
    assert settings.load_env() is False
    assert os.environ["OPENAI_BASE_URL"] == "https://ambient.example/v1"


@pytest.mark.parametrize("name", AUTHORITIED_NAMES)
def test_a_written_env_value_beats_an_ambient_one(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "https://ambient.example/v1")
    _write_env({name: "file-value"})
    settings.load_env()
    assert os.environ[name] == "file-value"


def test_the_groq_one_variable_setup_now_reaches_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_env({"GROQ_API_KEY": "gsk-file"})
    settings.load_env()
    resolved = resolved_config()
    assert resolved.key_source == "GROQ_API_KEY"
    assert resolved.base_url == "https://api.groq.com/openai/v1"
    assert resolved.base_url_source == "GROQ_API_KEY"
    assert resolved.provider == "groq"


def test_an_ambient_base_url_still_wins_over_the_file_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "ambient-groq-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
    _write_env({"GROQ_API_KEY": "gsk-file"})
    settings.load_env()
    resolved = resolved_config()
    assert resolved.key_source == "GROQ_API_KEY"
    assert resolved.base_url == "https://integrate.api.nvidia.com/v1"
    assert resolved.provider == "openai-compatible"


def test_the_endpoint_written_in_env_beats_an_ambient_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    _write_env({"LLM_API_KEY": "gsk-file", "LLM_BASE_URL": "https://api.groq.com/openai/v1"})
    settings.load_env()
    resolved = resolved_config()
    assert resolved.key_source == "LLM_API_KEY"
    assert resolved.base_url == "https://api.groq.com/openai/v1"
    assert resolved.provider == "groq"


def test_a_non_authoritied_name_still_loses_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "api")
    monkeypatch.setenv("INDEX_DIR", "/ambient/index")
    _write_env({"EMBEDDING_PROVIDER": "local", "INDEX_DIR": "/file/index"})
    settings.load_env()
    assert os.environ["EMBEDDING_PROVIDER"] == "api"
    assert os.environ["INDEX_DIR"] == "/ambient/index"


def test_mock_llm_in_the_shell_beats_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    _write_env({"MOCK_LLM": "0", "LLM_API_KEY": "gsk-file"})
    settings.load_env()
    assert os.environ["MOCK_LLM"] == "1"
    assert isinstance(build_llm(), MockLLM)


def test_the_allowlist_is_exactly_the_provider_aiming_names() -> None:
    assert AUTHORITIED_NAMES == (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "GROQ_API_KEY",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_PROVIDER",
    )
    assert "MOCK_LLM" not in AUTHORITIED_NAMES
    assert not {name for name in AUTHORITIED_NAMES if name.startswith("EMBEDDING")}
    assert "LLM_PROVIDER" in AUTHORITIED_NAMES, "the provider selector aims a request too"


def test_llm_provider_written_in_env_beats_an_ambient_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ambient")
    monkeypatch.setenv("LLM_API_KEY", "ambient-key")
    _write_env({"LLM_PROVIDER": "groq"})
    settings.load_env()
    resolved = resolved_config()
    assert os.environ["LLM_PROVIDER"] == "groq"
    assert resolved.configured_provider == "groq"
    assert resolved.provider == "groq"
    assert resolved.base_url == "https://api.groq.com/openai/v1"


def test_load_env_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
    _write_env({"LLM_BASE_URL": "https://api.groq.com/openai/v1"})
    assert settings.load_env() is True
    assert os.environ["LLM_BASE_URL"] == "https://api.groq.com/openai/v1"
    os.environ["LLM_BASE_URL"] = "https://changed.example/v1"
    assert settings.load_env() is True
    assert os.environ["LLM_BASE_URL"] == "https://changed.example/v1"


def test_load_env_reports_whether_the_file_existed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "_LOADED", False)
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path / "no-file")
    assert settings.load_env() is False
    monkeypatch.setattr(settings, "_LOADED", False)
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    _write_env({"GROQ_API_KEY": "gsk-file"})
    assert settings.load_env() is True


_CHILD_PROBE = """
import json, os, sys
from pathlib import Path
from src.utils import settings
from src.utils.llm import resolved_config

settings.REPO_ROOT = Path(os.environ["PROBE_ENV_FILE"]).parent
settings._LOADED = False
settings.load_env()
resolved = resolved_config()
print(json.dumps({
    "base_url": resolved.base_url,
    "key_source": resolved.key_source,
    "model": resolved.model,
    "provider": resolved.provider,
}, sort_keys=True))
"""


def _probe_clean_interpreter(env_file: Path, **overrides: str) -> dict[str, object]:
    env = {**os.environ, "PROBE_ENV_FILE": str(env_file), **overrides}
    env.pop("MOCK_LLM", None)
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_PROBE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, f"probe failed:\n{completed.stderr}"
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


def test_a_fresh_interpreter_resolves_the_llm_variables_from_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "LLM_API_KEY=gsk-file\nLLM_BASE_URL=https://api.groq.com/openai/v1\n",
        encoding="utf-8",
    )
    resolved = _probe_clean_interpreter(tmp_path / ".env")
    assert resolved["key_source"] == "LLM_API_KEY"
    assert resolved["base_url"] == "https://api.groq.com/openai/v1"

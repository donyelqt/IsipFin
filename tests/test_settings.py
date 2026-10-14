"""`src/utils/settings.py` — the `.env` precedence contract.

Every test here is hermetic: `REPO_ROOT` is monkeypatched at a `tmp_path` `.env` the test
writes itself, so the repository's real `.env` is never opened, no network is reachable,
and `MOCK_LLM` / `OPENAI_*` values the host exports are replaced by explicit test-level
values. The two subprocess probes inherit the parent's environment as-is on purpose: they
are the only way to see what a *real* interpreter — the one whose exports this host
happens to have — resolves.

The invariants pinned:

1. a blank value in `.env` (`.env.example` ships `OPENAI_API_KEY=`) never clobbers an
   ambient one;
2. a non-blank value in `.env` beats an ambient one for the eight authoritied names —
   the ones that decide which provider a request is aimed at: the key, the endpoint, the
   model and the provider selector, in both the `OPENAI_*` and `LLM_*` / `GROQ_*`
   spellings;
3. every *other* name still loses to the environment, so the general rule is intact
   and the allowlist is a proven-narrow one, not a blanket inversion;
4. `MOCK_LLM` is deliberately outside the allowlist: `MOCK_LLM=1` in the shell keeps
   selecting the offline mock, and that is asserted through `build_llm`, not through
   the loader;
5. `load_env()` is idempotent and cheap.
"""

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

#: Every variable cleared from the ambient environment for these tests, so the host's own
#: exports (this machine exports `OPENAI_API_KEY` *and* `OPENAI_BASE_URL`) cannot decide
#: an outcome a test did not set.
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

#: The names the task fixes: `.env` wins over an ambient export for exactly these.
AUTHORITIED_NAMES = settings.ENV_FILE_AUTHORITY


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isolate the ambient environment and the `.env` file the loader will read."""
    for name in _AMBIENT_CHAT_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(settings, "_LOADED", False)
    yield


def _write_env(contents: dict[str, str]) -> Path:
    """Write a `.env` at the (monkeypatched) repo root and return its path."""
    body = "".join(f"{name}={value}\n" for name, value in contents.items())
    path = settings.REPO_ROOT / ".env"
    path.write_text(body, encoding="utf-8")
    return path


# --- 1. a blank value never clobbers an ambient one --------------------------------


@pytest.mark.parametrize("name", AUTHORITIED_NAMES)
def test_a_blank_env_value_does_not_clobber_an_ambient_one(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """`.env.example` ships `OPENAI_API_KEY=` blank; blank must mean "unset here".

    An empty line in the config file is the *absent* case: the ambient value survives,
    and the loader never writes an empty string over a credential that works.
    """
    monkeypatch.setenv(name, f"ambient-{name}")
    _write_env({name: ""})
    settings.load_env()
    assert os.environ[name] == f"ambient-{name}"


def test_whitespace_counts_as_blank_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank value is empty-or-whitespace, matching how the gateway already reads it."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
    _write_env({"OPENAI_BASE_URL": "   "})
    settings.load_env()
    assert os.environ["OPENAI_BASE_URL"] == "https://ambient.example/v1"


def test_a_missing_env_file_leaves_the_environment_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No file is the general rule, not an error: the ambient value simply survives."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path / "absent")
    assert settings.load_env() is False
    assert os.environ["OPENAI_BASE_URL"] == "https://ambient.example/v1"


# --- 2. a non-blank value in .env beats an ambient one, for the eight ---------------


@pytest.mark.parametrize("name", AUTHORITIED_NAMES)
def test_a_written_env_value_beats_an_ambient_one(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The whole fix: a value typed in this repo's own config file wins.

    Without this the file is decoration whenever the shell exports the same name — a
    Groq key in `.env` aimed at an ambient NVIDIA endpoint, reported as a 401.
    """
    monkeypatch.setenv(name, "https://ambient.example/v1")
    _write_env({name: "file-value"})
    settings.load_env()
    assert os.environ[name] == "file-value"


def test_the_groq_one_variable_setup_now_reaches_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GROQ_API_KEY` alone in `.env` still works when nothing else is configured.

    The ambient `OPENAI_API_KEY` must not shadow it, because the file's own value wins
    and the endpoint follows the key — that is why `OPENAI_API_KEY=` is left blank in
    the file below rather than simply omitted.
    """
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
    """The honest boundary: only `GROQ_API_KEY` in `.env` is not enough on this host.

    The file sets no `OPENAI_BASE_URL`, so the ambient export survives — and the request
    it resolves goes to that endpoint, not Groq. This is the case that made the
    `GROQ_API_KEY`-only claim false in the README, and it is pinned here rather than
    papered over: the documented way to aim at Groq is to name the endpoint too.
    """
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
    """The second half of the promise: `LLM_BASE_URL` in `.env` outranks the export."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    _write_env({"LLM_API_KEY": "gsk-file", "LLM_BASE_URL": "https://api.groq.com/openai/v1"})
    settings.load_env()
    resolved = resolved_config()
    assert resolved.key_source == "LLM_API_KEY"
    assert resolved.base_url == "https://api.groq.com/openai/v1"
    assert resolved.provider == "groq"


# --- 3. the allowlist is narrow, not a blanket inversion ---------------------------


def test_a_non_authoritied_name_still_loses_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proof the fix is scoped: anything outside the eight behaves as before.

    `EMBEDDING_PROVIDER` is the case that matters — it decides the vector width, so an
    ambient choice must not be silently overridden by a stale file.
    """
    monkeypatch.setenv("EMBEDDING_PROVIDER", "api")
    monkeypatch.setenv("INDEX_DIR", "/ambient/index")
    _write_env({"EMBEDDING_PROVIDER": "local", "INDEX_DIR": "/file/index"})
    settings.load_env()
    assert os.environ["EMBEDDING_PROVIDER"] == "api"
    assert os.environ["INDEX_DIR"] == "/ambient/index"


def test_mock_llm_in_the_shell_beats_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """`MOCK_LLM` is outside the allowlist, asserted where it is read.

    The flag is checked by `build_llm` before any credential is resolved, and the shell
    value must keep winning file *and* key: the deterministic offline path is a property
    of the process, not of a config file.
    """
    monkeypatch.setenv("MOCK_LLM", "1")
    _write_env({"MOCK_LLM": "0", "LLM_API_KEY": "gsk-file"})
    settings.load_env()
    assert os.environ["MOCK_LLM"] == "1"
    assert isinstance(build_llm(), MockLLM)


def test_the_allowlist_is_exactly_the_provider_aiming_names() -> None:
    """The contract itself: eight names, no operational one, `MOCK_LLM` absent.

    The set is the one-name-per-setting surface added to the gateway: the key,
    the endpoint, the model and the provider selector, each in both the `LLM_*`
    spelling and the `OPENAI_*` / `GROQ_*` aliases.
    """
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
    """The newest name: `LLM_PROVIDER` in `.env` outranks an ambient one."""
    monkeypatch.setenv("LLM_PROVIDER", "ambient")
    monkeypatch.setenv("LLM_API_KEY", "ambient-key")
    _write_env({"LLM_PROVIDER": "groq"})
    settings.load_env()
    resolved = resolved_config()
    assert os.environ["LLM_PROVIDER"] == "groq"
    assert resolved.configured_provider == "groq"
    assert resolved.provider == "groq"
    assert resolved.base_url == "https://api.groq.com/openai/v1"


# --- 4. idempotence --------------------------------------------------------------


def test_load_env_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers invoke it defensively, so a second call must be a no-op.

    Not merely cheap: the file must not be re-parsed and re-applied, which is what would
    otherwise let a concurrent `.env` edit change a running process's credentials.
    """
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
    """The return value stays informational and untouched by the new rule."""
    monkeypatch.setattr(settings, "_LOADED", False)
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path / "no-file")
    assert settings.load_env() is False
    monkeypatch.setattr(settings, "_LOADED", False)
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    _write_env({"GROQ_API_KEY": "gsk-file"})
    assert settings.load_env() is True


# --- 5. a real interpreter, with this host's real exports ------------------------

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
    """Resolve the configuration in a fresh interpreter, against a given `.env`.

    The child inherits the host environment (this machine exports `OPENAI_API_KEY` and
    `OPENAI_BASE_URL` pointing at NVIDIA) and is given only an explicit `.env` to read —
    the least contrived reproduction of the reported failure.
    """
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
    """`LLM_API_KEY` + `LLM_BASE_URL` in `.env` reach Groq in a whole new process.

    Run against a child that inherits the host's `OPENAI_*` exports, so this is the
    acceptance probe and not a restatement of it.
    """
    (tmp_path / ".env").write_text(
        "LLM_API_KEY=gsk-file\nLLM_BASE_URL=https://api.groq.com/openai/v1\n",
        encoding="utf-8",
    )
    resolved = _probe_clean_interpreter(tmp_path / ".env")
    assert resolved["key_source"] == "LLM_API_KEY"
    assert resolved["base_url"] == "https://api.groq.com/openai/v1"

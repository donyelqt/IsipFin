from __future__ import annotations

import pytest
from src.utils.settings import ENV_FILE_AUTHORITY

_PROVIDER_NAMES: tuple[str, ...] = ENV_FILE_AUTHORITY


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROVIDER_NAMES:
        monkeypatch.delenv(name, raising=False)

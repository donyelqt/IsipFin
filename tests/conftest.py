"""Hermetic environment for the test suite (AGENTS.md section 2).

The suite is hermetic **by contract**: no test may read ``.env`` or depend on an
ambient host variable. Yet nothing enforced that, so on a host that exports
``LLM_PROVIDER``/``LLM_*`` (or has a populated repo-root ``.env`` — which this repo
ships with a real key) two gateway tests resolved a *different* provider than the
one they assert and failed for a reason that had nothing to do with the code under
test.

This fixture closes that hole deterministically, rather than by convention: the eight
names in :data:`src.utils.settings.ENV_FILE_AUTHORITY` are removed for every test.
They are exactly the names that decide *which provider a request is aimed at*, so an
ambient export cannot change what the gateway resolves, and ``monkeypatch`` restores
each one afterwards — so a test that deliberately sets a name still works and the
host's own environment is untouched between tests.

The repo-root ``.env`` needs no equivalent defence: ``load_env()`` is called only by
``app.py``, ``scripts/build_index.py`` and ``scripts/eval_rag.py``, never from ``tests/``,
so the file is not loaded during a pytest run at all. ``tests/test_settings.py`` proves
that contract by pointing ``REPO_ROOT`` at a ``tmp_path`` file it writes itself.

Operational names such as ``MOCK_LLM``, ``EMBEDDING_PROVIDER`` and ``INDEX_DIR`` are
deliberately **not** cleared: a host or CI job may inject those on purpose, and
``src/utils/settings.py`` already documents that the environment wins over the file
for them.
"""

from __future__ import annotations

import pytest
from src.utils.settings import ENV_FILE_AUTHORITY

#: The eight provider-routing names. Sourced from the module rather than repeated
#: here so the two lists can never drift apart.
_PROVIDER_NAMES: tuple[str, ...] = ENV_FILE_AUTHORITY


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the provider-routing names for each test; monkeypatch restores them."""
    for name in _PROVIDER_NAMES:
        monkeypatch.delenv(name, raising=False)

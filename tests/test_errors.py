from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pytest
from src.components import errors as errors_module
from src.components.errors import ErrorView, classify_error
from src.core.pdf_parser import ParseError
from src.core.retrieval import RetrievalError
from src.store.db import SessionNotFoundError, StoreError
from src.utils.llm import LLMError, SchemaViolationError

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Row:

    exc: BaseException
    view: ErrorView
    leak: tuple[str, ...]


def _shown(view: ErrorView) -> str:

    return " ".join([view.title, view.message, *([view.hint] if view.hint else [])])


_PROBE = """
import json

from src.chains.graph import GraphError, GraphNodeError, GraphStateError
from src.components.errors import classify_error
from src.core.local_embedder import LocalEmbeddingError

state = GraphStateError("question is empty")
failures = {
    "state": state,
    "node": GraphNodeError(
        "hybrid_retrieve is not implemented yet", node="hybrid_retrieve"
    ),
    "base": GraphError("graph returned dict, expected a state mapping"),
    "wrapped": GraphError("wrapping another graph failure", error=state),
    "embed_failed": LocalEmbeddingError(
        "the local embedding model 'BAAI/bge-small-en-v1.5' failed to embed: "
        "RuntimeError: onnx runtime refused"
    ),
    "embed_missing": LocalEmbeddingError(
        "the local embedder needs the 'fastembed' package, which is not installed "
        "(pip install -r requirements.txt pins it)"
    ),
    "embed_unmeasured": LocalEmbeddingError(
        "the local embedding model 'BAAI/bge-small-en-v1.5' returned no vector to "
        "measure, so its width is unknown"
    ),
}

report = {}
for key, exc in failures.items():
    view = classify_error(exc)
    report[key] = {
        "cls": type(exc).__name__,
        "module": type(exc).__module__,
        "mro": [cls.__name__ for cls in type(exc).__mro__],
        "str": str(exc),
        "node": getattr(exc, "node", None),
        "view": {
            "title": view.title,
            "message": view.message,
            "hint": view.hint,
            "retryable": view.retryable,
        },
    }

print(json.dumps(report))
"""


def _probe_interpreter() -> dict[str, dict[str, Any]]:

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


class _ReconstructedError(Exception):

    node: str | None = None

    def __init__(self, message: str, *, node: str | None = None, error: object = None) -> None:
        super().__init__(message)
        self.node = node
        self.error = error


def _rebuild(module: str, cls: str, mro: list[str], node: str | None) -> type[BaseException]:

    base = type(mro[1] if len(mro) > 1 else cls, (_ReconstructedError,), {"__module__": module})
    return type(cls, (base,), {"__module__": module, "node": node})


def _as_view(reported: dict[str, Any]) -> ErrorView:

    return ErrorView(
        title=reported["title"],
        message=reported["message"],
        hint=reported["hint"],
        retryable=reported["retryable"],
    )


def _fixture(key: str) -> tuple[type[BaseException], str]:

    reported = _REMOTE[key]
    return (
        _rebuild(reported["module"], reported["cls"], reported["mro"], reported["node"]),
        reported["str"],
    )


_REMOTE = _probe_interpreter()

_STATE = _fixture("state")
_NODE = _fixture("node")
_BASE = _fixture("base")
_WRAPPED = _fixture("wrapped")
_EMBEDDED = tuple(_fixture(key) for key in ("embed_failed", "embed_missing", "embed_unmeasured"))

_GRAPH_STATE = _STATE[0](_STATE[1])
_GRAPH_NODE = _NODE[0](_NODE[1], node=_REMOTE["node"]["node"])
_GRAPH_BASE_ERROR = _BASE[0](_BASE[1])
_GRAPH_WRAPPED = _WRAPPED[0](_WRAPPED[1])
_EMBED_FAILED, _EMBED_MISSING, _EMBED_UNMEASURED = (cls(message) for cls, message in _EMBEDDED)

_GRAPH_BASE = _GRAPH_STATE.__class__.__mro__[1]


_ROWS: tuple[_Row, ...] = (
    _Row(_GRAPH_STATE, errors_module._GRAPH_VIEW, ("GraphStateError", "question is empty")),
    _Row(
        _GRAPH_NODE,
        errors_module._GRAPH_VIEW,
        ("GraphNodeError", "hybrid_retrieve", "not implemented yet"),
    ),
    _Row(
        _GRAPH_BASE_ERROR,
        errors_module._GRAPH_VIEW,
        ("graph returned dict", "a state mapping"),
    ),
    _Row(
        _GRAPH_WRAPPED,
        errors_module._GRAPH_VIEW,
        ("GraphError", "wrapping another graph failure"),
    ),
    _Row(
        _EMBED_FAILED,
        errors_module._LOCAL_EMBEDDER_VIEW,
        ("LocalEmbeddingError", "BAAI/bge-small-en-v1.5", "fastembed", "onnx", "failed to embed"),
    ),
    _Row(
        _EMBED_MISSING,
        errors_module._LOCAL_EMBEDDER_VIEW,
        ("LocalEmbeddingError", "fastembed", "pip install", "requirements.txt"),
    ),
    _Row(
        _EMBED_UNMEASURED,
        errors_module._LOCAL_EMBEDDER_VIEW,
        ("LocalEmbeddingError", "BAAI/bge-small-en-v1.5", "no vector", "width"),
    ),
    _Row(
        LLMError.RateLimited("429 from provider"),
        errors_module._RATE_LIMITED_VIEW,
        ("RateLimited", "429"),
    ),
    _Row(
        LLMError.ContextLength("context window exceeded"),
        errors_module._CONTEXT_LENGTH_VIEW,
        ("ContextLength", "context window exceeded"),
    ),
    _Row(
        LLMError.Auth("401 invalid key sk-proj-SECRET"),
        errors_module._AUTH_VIEW,
        ("Auth", "401", "sk-proj-SECRET", "invalid key"),
    ),
    _Row(
        LLMError.Timeout("read timed out after 30s"),
        errors_module._TIMEOUT_VIEW,
        ("Timeout", "after 30s"),
    ),
    _Row(
        SchemaViolationError("reply was not a JSON object"),
        errors_module._SCHEMA_VIOLATION_VIEW,
        ("SchemaViolationError", "not a JSON object"),
    ),
    _Row(
        RetrievalError("index is missing or unreadable"),
        errors_module._RETRIEVAL_VIEW,
        ("RetrievalError", "unreadable"),
    ),
    _Row(
        StoreError("could not write the turn"),
        errors_module._STORE_VIEW,
        ("StoreError", "could not write the turn"),
    ),
    _Row(
        SessionNotFoundError("session 42 is gone"),
        errors_module._SESSION_MISSING_VIEW,
        ("SessionNotFoundError", "session 42"),
    ),
)

_UNKNOWN: tuple[BaseException, ...] = (KeyError("missing_field"), ValueError("bare value error"))


class TestGraphFailuresAreClassified:

    def test_graph_state_error_is_not_the_generic_view(self) -> None:

        view = classify_error(_GRAPH_STATE)

        assert view.title != "Unexpected error"
        assert view is errors_module._GRAPH_VIEW

    def test_graph_state_error_offers_no_retry(self) -> None:

        view = classify_error(_GRAPH_STATE)

        assert view.retryable is False

    def test_graph_state_error_leaks_no_class_name(self) -> None:

        view = classify_error(_GRAPH_STATE)

        assert "GraphStateError" not in (view.title, view.message)
        assert "GraphError" not in (view.title, view.message)
        assert "question is empty" not in (view.title, view.message)

    def test_graph_node_error_inherits_the_graph_view(self) -> None:

        base = classify_error(_GRAPH_BASE_ERROR)
        node = classify_error(_GRAPH_NODE)
        state = classify_error(_GRAPH_STATE)

        assert node is base is state is errors_module._GRAPH_VIEW

    def test_unknown_graph_subclass_is_still_classified(self) -> None:

        class GraphTaskError(_GRAPH_BASE):
            pass

        view = classify_error(GraphTaskError("task not wired", node="decompose_question"))

        assert view is errors_module._GRAPH_VIEW
        assert view.retryable is False

    def test_a_graph_error_wrapping_another_graph_error(self) -> None:

        view = classify_error(_GRAPH_WRAPPED)

        assert view is errors_module._GRAPH_VIEW
        assert "GraphStateError" not in _shown(view)

    def test_graph_hint_names_the_change_to_make(self) -> None:

        view = classify_error(_GRAPH_STATE)

        assert view.hint is not None
        assert "question" in view.hint.lower()


class TestLocalEmbedderFailureIsClassified:

    def test_embedder_failure_shares_the_retrieval_view(self) -> None:

        assert classify_error(_EMBED_FAILED) is classify_error(RetrievalError("index missing"))

    def test_embedder_failure_leaks_no_class_name_or_raw_message(self) -> None:

        view = classify_error(_EMBED_FAILED)

        assert "LocalEmbeddingError" not in (view.title, view.message)
        assert "fastembed" not in view.message
        assert "onnx" not in view.message.lower()
        assert "BAAI/bge-small-en-v1.5" not in view.message
        assert "failed to embed" not in view.message

    def test_dimension_probe_failure_is_classified_the_same_way(self) -> None:

        view = classify_error(_EMBED_UNMEASURED)

        assert view is errors_module._LOCAL_EMBEDDER_VIEW

    @pytest.mark.parametrize(
        "exc",
        [_EMBED_FAILED, _EMBED_MISSING, _EMBED_UNMEASURED],
        ids=["embed-path", "load-path", "dim-probe"],
    )
    def test_every_embedder_site_shares_one_view(self, exc: BaseException) -> None:

        assert classify_error(exc) is errors_module._LOCAL_EMBEDDER_VIEW


class TestNoTypedViewLeaksTheFailureItClassifies:

    @pytest.mark.parametrize(
        ("row",),
        [(row,) for row in _ROWS],
        ids=[f"{index:02d}-{type(row.exc).__name__}" for index, row in enumerate(_ROWS)],
    )
    def test_typed_view_leaks_no_class_name_message_or_identifying_token(self, row: _Row) -> None:

        assert classify_error(row.exc) is row.view

        shown = _shown(row.view)
        raw = str(row.exc).strip()
        leaked = [token for token in (type(row.exc).__name__, raw, *row.leak) if token in shown]

        assert not leaked, f"{type(row.exc).__name__} leaked {leaked} into a banner: {shown!r}"

    @pytest.mark.parametrize(
        "exc",
        _UNKNOWN,
        ids=[type(exc).__name__ for exc in _UNKNOWN],
    )
    def test_the_generic_view_keeps_the_class_name_and_message(self, exc: BaseException) -> None:

        view = classify_error(exc)

        assert view.title == "Unexpected error"
        assert type(exc).__name__ in view.message
        assert str(exc).strip("'") in view.message


class TestExistingBranchesAreUnchanged:

    @pytest.mark.parametrize(
        ("index", "row"),
        list(enumerate(_ROWS)),
        ids=[f"row-{index:02d}" for index in range(len(_ROWS))],
    )
    def test_every_row_of_the_table_maps_as_before(self, index: int, row: _Row) -> None:

        assert classify_error(row.exc) is row.view

    def test_the_table_covers_every_view_the_module_exports(self) -> None:

        exported = {
            name
            for name in dir(errors_module)
            if name.endswith("_VIEW") and isinstance(getattr(errors_module, name), ErrorView)
        }
        covered = {
            name
            for row in _ROWS
            for name in dir(errors_module)
            if getattr(errors_module, name, None) is row.view
        }

        assert covered == exported

    def test_the_gateway_rows_still_hold(self) -> None:

        assert errors_module._GATEWAY_VIEWS.keys() == {
            "Auth",
            "ContextLength",
            "RateLimited",
            "SchemaViolationError",
            "Timeout",
        }
        assert classify_error(LLMError.RateLimited("429")) is errors_module._RATE_LIMITED_VIEW
        assert (
            classify_error(LLMError.ContextLength("too long")) is errors_module._CONTEXT_LENGTH_VIEW
        )
        assert classify_error(LLMError.Auth("401")) is errors_module._AUTH_VIEW
        assert classify_error(LLMError.Timeout("slow")) is errors_module._TIMEOUT_VIEW
        assert classify_error(SchemaViolationError("bad json")) is (
            errors_module._SCHEMA_VIOLATION_VIEW
        )

    def test_the_store_rows_still_hold(self) -> None:

        assert classify_error(SessionNotFoundError("gone")) is (errors_module._SESSION_MISSING_VIEW)
        assert classify_error(StoreError("broken")) is errors_module._STORE_VIEW

    def test_the_parse_view_is_built_from_the_exception(self) -> None:

        view = classify_error(ParseError("pdfplumber failed on page 3"))

        assert view.title == "Could not read the PDF"
        assert "page 3" in view.message.lower()
        assert view.retryable is False

    def test_the_parse_error_branch_stays_wired(self) -> None:

        view = errors_module._parse_view(ParseError("scanned"))

        assert view.title == "Could not read the PDF"
        assert view.retryable is False

    def test_every_view_is_frozen_and_fully_specified(self) -> None:

        assert {field.name for field in fields(ErrorView)} == {
            "title",
            "message",
            "hint",
            "retryable",
        }

        for name in dir(errors_module):
            view = getattr(errors_module, name)
            if not (name.endswith("_VIEW") and isinstance(view, ErrorView)):
                continue
            assert isinstance(view.title, str) and view.title
            assert isinstance(view.message, str) and view.message
            assert view.hint is None or (isinstance(view.hint, str) and view.hint)
            assert isinstance(view.retryable, bool)


class TestTheModuleStaysLeafLevel:

    def test_the_rebuilt_fixtures_classify_exactly_as_the_real_classes_did(self) -> None:

        fixtures = {
            "state": _GRAPH_STATE,
            "node": _GRAPH_NODE,
            "base": _GRAPH_BASE_ERROR,
            "wrapped": _GRAPH_WRAPPED,
            "embed_failed": _EMBED_FAILED,
            "embed_missing": _EMBED_MISSING,
            "embed_unmeasured": _EMBED_UNMEASURED,
        }

        for key, exc in fixtures.items():
            assert _as_view(_REMOTE[key]["view"]) == classify_error(exc), key

    def test_the_probe_imported_the_pipeline_and_this_process_did_not(self) -> None:

        assert _REMOTE["state"]["module"] == "src.chains.graph"
        assert _REMOTE["state"]["mro"][:2] == ["GraphStateError", "GraphError"]
        assert _REMOTE["node"]["mro"][:2] == ["GraphNodeError", "GraphError"]
        assert _REMOTE["base"]["mro"][:2] == ["GraphError", "RuntimeError"]
        assert _REMOTE["embed_failed"]["module"] == "src.core.local_embedder"
        assert _REMOTE["embed_failed"]["mro"][:2] == ["LocalEmbeddingError", "RuntimeError"]

    def test_importing_the_module_loads_neither_the_graph_nor_the_embedder(self) -> None:

        probe = (
            "import json, sys; "
            "from src.components import errors; "
            "print(json.dumps({"
            "'graph': 'src.chains.graph' in sys.modules, "
            "'embedder': 'src.core.local_embedder' in sys.modules, "
            "'fastembed': 'fastembed' in sys.modules, "
            "'langgraph': 'langgraph' in sys.modules}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )

        assert json.loads(result.stdout) == {
            "graph": False,
            "embedder": False,
            "fastembed": False,
            "langgraph": False,
        }

    def test_the_contract_names_are_imported_for_annotations(self) -> None:

        source = Path(errors_module.__file__ or "").read_text(encoding="utf-8")

        assert "from typing import TYPE_CHECKING" in source
        assert "from src.chains.graph import GraphError" in source
        assert "from src.core.local_embedder import LocalEmbeddingError" in source
        assert "if TYPE_CHECKING:" in source

    def test_a_graph_exception_matches_the_spine_base_not_each_subclass(self) -> None:

        source = Path(errors_module.__file__ or "").read_text(encoding="utf-8")
        start = source.index("def _is_graph_error(")
        helper = source[start : source.index("\n\n\ndef ", start + 1)]
        body = helper.split('"""', 2)[-1]

        assert "_GRAPH_BASE" in body
        assert "GraphStateError" not in body
        assert "GraphNodeError" not in body

    def test_no_view_interpolates_an_exceptions_own_text(self) -> None:

        source = Path(errors_module.__file__ or "").read_text(encoding="utf-8")
        views = source.split("# --- the views", 1)[-1].split("# --- the public API", 1)[0]

        assert "type(exc)" not in views
        assert "__name__" not in views
        assert "str(exc)" not in views

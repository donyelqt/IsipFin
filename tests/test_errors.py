"""The error taxonomy: what every failure maps to, and what never reaches the screen (T21).

``src/components/errors.py`` is the single place that turns an exception into the banner a
person can act on. Two classes of defect are pinned here, and both are ones a live run can
actually trigger.

**Defect A — two triggerable failures fell through to the generic view.** The classifier
knew the store, the retriever, the parser and the gateway, and nothing else: a
:class:`~src.chains.graph.GraphStateError` (``invoke("")`` is not a plausible empty answer)
and a :class:`~src.core.local_embedder.LocalEmbeddingError` escaping outside
``src.core.retrieval._owned_embedder`'s wrapper both landed on ``Unexpected error``, which
showed the raw class name and offered a Retry that could not succeed. That is the failure
mode the taxonomy exists to eliminate, reinstated for the two layers most likely to raise.
Both now have views of their own: a non-retryable graph view and the index-unavailable view
— the one a wrapped ``RetrievalError`` gets, for the reason
``test_embedder_failure_shares_the_retrieval_view` names.

**Defect B — the regression class, guarded directly.** Every branch in the table below is
checked for the three things a leaked banner carries: the exception's class name, its
message verbatim, and the identifying tokens the row names (a driver, a model id, a secret,
a node name, a value quoted from the failure). The generic view is exempt by contract — it
is the one branch that must keep ``type(exc).__name__`` and ``str(exc)``.

Everything here is hermetic: no network, no ``.env``, no key, no writes into the repo tree,
and no Streamlit run — the classifier is a pure function and is called as one. The graph and
embedder fixtures are the real classes from ``src.chains.graph`` and
``src.core.local_embedder``: a local stand-in would prove the test double classifies, not
the production one.

Those two families are the pipeline's: importing them in-process would pull langgraph and
every node into this process, and a sibling test
(``tests/test_eval_rag.py::test_loading_the_module_left_no_index_or_env_state_behind``)
asserts no ``src.chains.graph`` module is in ``sys.modules`` when it runs. A guarantee whose
truth depends on which file pytest happened to collect first is not a guarantee, so the
fixtures are built in a child interpreter and reported back as data — the same seam
``tests/test_gateway.py`` uses for ``openai`` — and this process never imports them at all.
There is no cleanup and nothing to purge: a probe cannot leave state behind.
"""

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
    """One row of the taxonomy table: a real failure, its view, and what must not leak.

    Attributes:
        exc: The exception, from the production class that raises it.
        view: The view the classifier must return, as the module's own constant.
        leak: Tokens that identify *this* failure and must not appear anywhere the user
            reads. Ordered from most to least specific, so a failure message tells you
            which one escaped.
    """

    exc: BaseException
    view: ErrorView
    leak: tuple[str, ...]


def _shown(view: ErrorView) -> str:
    """The one string a user reads: title, message and hint, no repeats."""

    return " ".join([view.title, view.message, *([view.hint] if view.hint else [])])


# --- the production classes, classified where they are declared ---------------------
# The fixtures below are the real exceptions the classifier must recognise:
# ``GraphStateError`` from ``invoke("")``, ``GraphNodeError`` from a stub node, the base
# ``GraphError`` and a wrapper around a state error, and the three sites
# ``LocalEmbeddingError`` is raised from. A local stand-in would prove the test double
# classifies, not the production class, which is the whole value of this file.
#
# They are built in a child interpreter and reported back as data, following
# ``tests/test_gateway.py``'s ``_REAL_PROBE``: the child imports the pipeline, this process
# never does, so no finalizer is needed and no other file's assertion can be decided by
# which one pytest collected first. The classifier matches the family structurally —
# declaring module plus class name, walked over the MRO — so the child reports exactly the
# inputs that decision reads, plus the view the real classes got in a process where they
# were genuinely loaded. That view is the verdict this file asserts against.

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
    """Classify the real failures in a fresh interpreter and report what each mapped to.

    The probe imports ``src.chains.graph`` and ``src.core.local_embedder`` itself; this
    process never does. ``check=True`` makes a probe that could not import the pipeline a
    test error rather than a silently empty report, and the assertion after
    ``json.loads`` refuses a child that printed something other than one object.
    """

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
    """One of the reported classes, rebuilt so its own MRO names the reported family.

    The classifier matches by declaring module plus class name, walked over the MRO, so a
    rebuilt class must carry the reported module and its reported base's name. ``node`` is
    accepted as a keyword because the graph classes take it (``GraphNodeError(node=...)``);
    the other keyword a graph error takes, ``error``, is stored but never read by the
    classifier, so it is accepted and dropped.
    """

    node: str | None = None

    def __init__(self, message: str, *, node: str | None = None, error: object = None) -> None:
        super().__init__(message)
        self.node = node
        self.error = error


def _rebuild(module: str, cls: str, mro: list[str], node: str | None) -> type[BaseException]:
    """The reported class, rebuilt in this process under its original module and MRO.

    ``type()`` synthesises the class in the module the child reported, so the classifier's
    module-plus-name check matches it exactly as it matched the real class. Its parent is
    ``_ReconstructedError`` — whose own name comes from the reported MRO's first entry, so
    ``GraphStateError``'s parent is named ``GraphError`` and ``LocalEmbeddingError``'s
    parent is named ``LocalEmbeddingError`` — which is what makes the classifier's own
    guard report ``True`` here for exactly the same reason it does in production.
    """

    base = type(mro[1] if len(mro) > 1 else cls, (_ReconstructedError,), {"__module__": module})
    return type(cls, (base,), {"__module__": module, "node": node})


def _as_view(reported: dict[str, Any]) -> ErrorView:
    """The view the child reported, as the frozen :class:`ErrorView` it was built from."""

    return ErrorView(
        title=reported["title"],
        message=reported["message"],
        hint=reported["hint"],
        retryable=reported["retryable"],
    )


def _fixture(key: str) -> tuple[type[BaseException], str]:
    """Fixture ``key`` as (class, message): what to build it from, and its verbatim text."""

    reported = _REMOTE[key]
    return (
        _rebuild(reported["module"], reported["cls"], reported["mro"], reported["node"]),
        reported["str"],
    )


#: What the child reported, keyed as in the probe. One probe per session, run at import
#: time: the fixtures are this file's shared state, so they are built once for all of it.
_REMOTE = _probe_interpreter()

_STATE = _fixture("state")
_NODE = _fixture("node")
_BASE = _fixture("base")
_WRAPPED = _fixture("wrapped")
_EMBEDDED = tuple(_fixture(key) for key in ("embed_failed", "embed_missing", "embed_unmeasured"))

#: The fixtures, as instances, built from the classes the child reported: the graph's spine
#: and the three ``LocalEmbeddingError`` sites.
_GRAPH_STATE = _STATE[0](_STATE[1])
_GRAPH_NODE = _NODE[0](_NODE[1], node=_REMOTE["node"]["node"])
_GRAPH_BASE_ERROR = _BASE[0](_BASE[1])
_GRAPH_WRAPPED = _WRAPPED[0](_WRAPPED[1])
_EMBED_FAILED, _EMBED_MISSING, _EMBED_UNMEASURED = (cls(message) for cls, message in _EMBEDDED)

#: The spine's base, which a graph error a later task declares inherits from. The child
#: reported it as ``_state``'s own parent, so it is rebuilt rather than imported.
_GRAPH_BASE = _GRAPH_STATE.__class__.__mro__[1]


# --- the taxonomy table -------------------------------------------------------------
# One row per exception the classifier knows, each constructed from the real raising
# class with the message that class actually raises, and the view it must land on. The
# views are the module's own constants — never a string built here — so a wording change
# edits one file and this table still asserts the mapping, not the prose.

_ROWS: tuple[_Row, ...] = (
    # --- the graph's typed-error spine (ADR-007's graph-side rows, plus the base) ----
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
    # --- the local embedder, on the three paths retrieval.py's wrapper never sees ----
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
    # --- ADR-007's five provider-mapped gateway rows --------------------------------
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
    # --- the branches that predate this slice, pinned against churn ----------------
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

#: The exceptions that keep the generic view: an unknown type still renders, and the
#: generic view is the one branch documented to quote the class name and the message.
_UNKNOWN: tuple[BaseException, ...] = (KeyError("missing_field"), ValueError("bare value error"))


# --- the two classes of defect this file closes -------------------------------------


class TestGraphFailuresAreClassified:
    """Every ``GraphError`` gets the graph view, not the generic fallback.

    The fixtures are the production classes with the production messages
    (``_GRAPH_STATE``, ``_GRAPH_NODE``, ``_GRAPH_BASE_ERROR``), and the unknown-subclass case
    subclasses the same base — which is what the inheritance claim below is about.
    """

    def test_graph_state_error_is_not_the_generic_view(self) -> None:
        """``invoke("")`` is a decision the user can act on, not an unexpected error."""

        view = classify_error(_GRAPH_STATE)

        assert view.title != "Unexpected error"
        assert view is errors_module._GRAPH_VIEW

    def test_graph_state_error_offers_no_retry(self) -> None:
        """A retry repeats the same empty question and fails identically."""

        view = classify_error(_GRAPH_STATE)

        assert view.retryable is False

    def test_graph_state_error_leaks_no_class_name(self) -> None:
        """The banner names the failure in plain language, never the class's own name."""

        view = classify_error(_GRAPH_STATE)

        assert "GraphStateError" not in (view.title, view.message)
        assert "GraphError" not in (view.title, view.message)
        assert "question is empty" not in (view.title, view.message)

    def test_graph_node_error_inherits_the_graph_view(self) -> None:
        """One view for the spine: subclasses inherit it, they do not each add a branch."""

        base = classify_error(_GRAPH_BASE_ERROR)
        node = classify_error(_GRAPH_NODE)
        state = classify_error(_GRAPH_STATE)

        assert node is base is state is errors_module._GRAPH_VIEW

    def test_unknown_graph_subclass_is_still_classified(self) -> None:
        """A task that adds a ``GraphError`` subclass inherits the view with no edit."""

        class GraphTaskError(_GRAPH_BASE):
            """A graph error a later task will declare in ``src.chains.graph``."""

        view = classify_error(GraphTaskError("task not wired", node="decompose_question"))

        assert view is errors_module._GRAPH_VIEW
        assert view.retryable is False

    def test_a_graph_error_wrapping_another_graph_error(self) -> None:
        """The wrapper is the failure the user sees; the inner one stays in the log."""

        view = classify_error(_GRAPH_WRAPPED)

        assert view is errors_module._GRAPH_VIEW
        assert "GraphStateError" not in _shown(view)

    def test_graph_hint_names_the_change_to_make(self) -> None:
        """The next step is the input to change, which is what makes it non-retryable."""

        view = classify_error(_GRAPH_STATE)

        assert view.hint is not None
        assert "question" in view.hint.lower()


class TestLocalEmbedderFailureIsClassified:
    """An embedder failure outside ``retrieval.py``'s wrapper gets the index view."""

    def test_embedder_failure_shares_the_retrieval_view(self) -> None:
        """One view for "the index cannot be searched", whichever embedder could not run.

        ``src.core.retrieval._owned_embedder`` rebinds every embedder failure to a
        ``RetrievalError`` so a caller of that module catches one class; the same failure
        raised before the wrapper ran (``build_embedder``, or the ``dim`` probe) escapes
        as a ``LocalEmbeddingError``. Both leave the dense leg dead, so both read the
        same way to the person on the screen: the index is unusable until a model runs.
        """

        assert classify_error(_EMBED_FAILED) is classify_error(RetrievalError("index missing"))

    def test_embedder_failure_leaks_no_class_name_or_raw_message(self) -> None:
        """None of the driver, package or model text a caller would recognise."""

        view = classify_error(_EMBED_FAILED)

        assert "LocalEmbeddingError" not in (view.title, view.message)
        assert "fastembed" not in view.message
        assert "onnx" not in view.message.lower()
        assert "BAAI/bge-small-en-v1.5" not in view.message
        assert "failed to embed" not in view.message

    def test_dimension_probe_failure_is_classified_the_same_way(self) -> None:
        """The ``dim`` probe raises from ``_embed``'s path before any wrapper ran."""

        view = classify_error(_EMBED_UNMEASURED)

        assert view is errors_module._LOCAL_EMBEDDER_VIEW

    @pytest.mark.parametrize(
        "exc",
        [_EMBED_FAILED, _EMBED_MISSING, _EMBED_UNMEASURED],
        ids=["embed-path", "load-path", "dim-probe"],
    )
    def test_every_embedder_site_shares_one_view(self, exc: BaseException) -> None:
        """The three sites ``src.core.local_embedder.py`` raises from, one banner."""

        assert classify_error(exc) is errors_module._LOCAL_EMBEDDER_VIEW


class TestNoTypedViewLeaksTheFailureItClassifies:
    """The regression guard: a typed banner carries fixed text only."""

    @pytest.mark.parametrize(
        ("row",),
        [(row,) for row in _ROWS],
        ids=[f"{index:02d}-{type(row.exc).__name__}" for index, row in enumerate(_ROWS)],
    )
    def test_typed_view_leaks_no_class_name_message_or_identifying_token(self, row: _Row) -> None:
        """Not the class name, not the message verbatim, not the tokens that name it."""

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
        """A failure nobody planned for is rendered, never swallowed (its own contract)."""

        view = classify_error(exc)

        assert view.title == "Unexpected error"
        assert type(exc).__name__ in view.message
        assert str(exc).strip("'") in view.message


# --- the branches this slice did not touch, pinned against churn --------------------


class TestExistingBranchesAreUnchanged:
    """Every row maps exactly as it did before the graph/embedder work."""

    @pytest.mark.parametrize(
        ("index", "row"),
        list(enumerate(_ROWS)),
        ids=[f"row-{index:02d}" for index in range(len(_ROWS))],
    )
    def test_every_row_of_the_table_maps_as_before(self, index: int, row: _Row) -> None:
        """The table in this file, asserted against the classifier, row by row."""

        assert classify_error(row.exc) is row.view

    def test_the_table_covers_every_view_the_module_exports(self) -> None:
        """A branch with no row is a branch this file does not pin — none is allowed."""

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
        """The five provider-mapped gateway rows are the frozen ADR-007 table."""

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
        """The store branch wins over its base, as it must for the session row to work."""

        assert classify_error(SessionNotFoundError("gone")) is (errors_module._SESSION_MISSING_VIEW)
        assert classify_error(StoreError("broken")) is errors_module._STORE_VIEW

    def test_the_parse_view_is_built_from_the_exception(self) -> None:
        """The one branch that reads the exception's own text, to name the page."""

        view = classify_error(ParseError("pdfplumber failed on page 3"))

        assert view.title == "Could not read the PDF"
        assert "page 3" in view.message.lower()
        assert view.retryable is False

    def test_the_parse_error_branch_stays_wired(self) -> None:
        """``upload.py`` owns the per-file cards, so this branch is unreachable from the
        upload path today — and must stay reachable for a future path that routes here."""

        view = errors_module._parse_view(ParseError("scanned"))

        assert view.title == "Could not read the PDF"
        assert view.retryable is False

    def test_every_view_is_frozen_and_fully_specified(self) -> None:
        """A view with a missing field would render a half-drawn banner."""

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
    """ADR-009: a component module must not import the pipeline at import time."""

    def test_the_rebuilt_fixtures_classify_exactly_as_the_real_classes_did(self) -> None:
        """The child's own verdict is the reference; this process must reproduce it.

        A rebuilt fixture is only worth asserting against if the classifier says the same
        thing about it here as it said about the production class in the child. If a
        future change made the reconstruction diverge (a renamed base, a different module),
        this fails rather than the tables silently passing on a stand-in.
        """

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
        """The child loaded the classes this file claims to have classified.

        The reported module and MRO names are the proof: they can only come from importing
        ``src.chains.graph`` and ``src.core.local_embedder``, which this process never
        does — so the suite is green without the pipeline ever entering ``sys.modules``.
        """

        assert _REMOTE["state"]["module"] == "src.chains.graph"
        assert _REMOTE["state"]["mro"][:2] == ["GraphStateError", "GraphError"]
        assert _REMOTE["node"]["mro"][:2] == ["GraphNodeError", "GraphError"]
        assert _REMOTE["base"]["mro"][:2] == ["GraphError", "RuntimeError"]
        assert _REMOTE["embed_failed"]["module"] == "src.core.local_embedder"
        assert _REMOTE["embed_failed"]["mro"][:2] == ["LocalEmbeddingError", "RuntimeError"]

    def test_importing_the_module_loads_neither_the_graph_nor_the_embedder(self) -> None:
        """Checked in a fresh interpreter: this test's own imports are irrelevant."""

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
        """The types are named under ``TYPE_CHECKING``: typed, not duck-typed."""

        source = Path(errors_module.__file__ or "").read_text(encoding="utf-8")

        assert "from typing import TYPE_CHECKING" in source
        assert "from src.chains.graph import GraphError" in source
        assert "from src.core.local_embedder import LocalEmbeddingError" in source
        assert "if TYPE_CHECKING:" in source

    def test_a_graph_exception_matches_the_spine_base_not_each_subclass(self) -> None:
        """The guard reads ``GraphError`` once; a branch per subclass is the thing avoided."""

        source = Path(errors_module.__file__ or "").read_text(encoding="utf-8")
        start = source.index("def _is_graph_error(")
        helper = source[start : source.index("\n\n\ndef ", start + 1)]
        body = helper.split('"""', 2)[-1]

        assert "_GRAPH_BASE" in body
        assert "GraphStateError" not in body
        assert "GraphNodeError" not in body

    def test_no_view_interpolates_an_exceptions_own_text(self) -> None:
        """No banner hand-written to quote the failure it describes."""

        source = Path(errors_module.__file__ or "").read_text(encoding="utf-8")
        views = source.split("# --- the views", 1)[-1].split("# --- the public API", 1)[0]

        assert "type(exc)" not in views
        assert "__name__" not in views
        assert "str(exc)" not in views

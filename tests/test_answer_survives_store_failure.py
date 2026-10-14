"""A successful answer survives a store that refuses to persist it (ADR-007's hard rule).

``app.py``'s turn lifecycle used to end like this: the graph succeeds, the envelope is
produced, ``_record`` is called and swallows the store failure, ``st.rerun()`` runs — and
the canvas renders the transcript from the store, which has no row, has no envelope, and
shows an empty canvas. The user is told their question was answered and then shown
nothing at all. That is ADR-007's "no code path renders an empty answer" arrived at from
the opposite direction: the answer existed and was thrown away.

The fix makes the answer survive the failure and makes the failure visible:

* the turn the store refused is drawn on the run that produced it, from the envelope the
  graph returned, instead of being thrown away in the name of a rerun that will only show
  an empty transcript;
* the store failure is reported through the existing taxonomy
  (:func:`src.components.errors.classify_error` / :func:`render_error`), so the banner is
  the one every other store failure produces and never a raw traceback;
* the turn is drawn exactly once — the store read path draws whatever rows did land, and
  the failed-write path draws the turn it owns, so the two cannot both draw it.

The store's own read path is where the "drawn twice" risk lives, so these tests exercise
the production ``_render_chat_view`` and ``_record`` under
``streamlit.testing.v1.AppTest`` — the seam ``tests/test_degraded_planner_marker.py`` also
uses — rather than re-implementing the render flow in the test. The graph is stubbed at
``app._ask``: these are turn-lifecycle tests, not pipeline tests, and a graph that wants
a model, an index and a key is none of that.

Every test here is hermetic: no network, no ``.env`` read (the render path is driven
directly, bypassing ``main()``/``load_env()``), and no writes into the repo tree — the
SQLite file and the store-call report live under ``tmp_path``. One suite drives a real
:class:`~src.store.sessions.SessionStore` over a genuinely read-only file, so the
exception the app handles is the one the deployment produces rather than one a test
author typed in.

``app.py`` is loaded by path under a private module name rather than as ``app``, so this
file never puts a top-level ``app`` in ``sys.modules`` for the sibling suites —
``tests/test_eval_rag.py`` asserts no ``src.chains.graph`` module is loaded when it runs,
and a guarantee whose truth depends on pytest's collection order is not a guarantee.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import pickle
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from src.store.db import SessionNotFoundError, StoreError
from src.store.sessions import Message, Session, SessionHistory, SessionStore, Turn
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _chains_state() -> Any:
    """``src.chains.state``, imported lazily.

    ``app.py`` imports ``src.chains.graph`` (which imports langgraph and every node) at
    module scope, and ``src.chains.state`` is in the same package. Importing it at this
    module's top level would load ``src.chains.graph`` into ``sys.modules`` for the whole
    process, and ``tests/test_eval_rag.py`` asserts no such module is loaded — a
    guarantee whose truth would then depend on pytest's collection order. The import is
    therefore made at first use, inside this function.
    """

    import src.chains.state as chains_state

    return chains_state


#: The question every test asks, taken verbatim from the live reproduction.
QUESTION = "Which segment had the highest revenue?"

#: The envelope's answer text: the authored marker the assertions look for.
ANSWER_MARKER = "Cloud was the highest-revenue segment"

#: The store's own failure mode for a database that cannot be written: the message
#: SQLite itself produces on a read-only file.
READ_ONLY_MESSAGE = "attempt to write a readonly database"

#: The store row's banner message (``src/components/errors._STORE_VIEW``). Asserted as a
#: substring check on the rendered banner body, so a taxonomy rewording turns this suite
#: red instead of silently passing.
STORE_BANNER = "The conversation store could not be read or written"

#: The session-missing row's banner message (``errors._SESSION_MISSING_VIEW``).
MISSING_SESSION_BANNER = "That conversation no longer exists"

#: The session id every test drives. A fixed value keeps the carried turn's session
#: agreement independent of the store's uuid generation.
SESSION_ID = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"

#: The attribute the app script reports the store's calls into, on the shared
#: ``app.py`` module. Session state is not a reliable carrier here: Streamlit drops keys
#: it does not recognise as widget state between the app script and the test process.
_REPORT_ATTRIBUTE = "_isipfin_store_report"

#: ``app.py`` imports ``src.chains.graph`` (langgraph and every node) at module scope,
#: so running the app loads it. ``tests/test_eval_rag.py`` asserts that no ``src.chains.graph``
#: module is in ``sys.modules`` when it runs, so this file purges the graph after every
#: test — the same convention ``tests/test_gateway.py`` uses for its lazily-imported
#: ``openai`, and for the same reason: a guarantee whose truth depends on pytest's
#: collection order is not a guarantee.
_GRAPH_ROOT = "src.chains.graph"

_APP: Any = None


@pytest.fixture(autouse=True)
def _purged_graph() -> Iterator[None]:
    """Leave no ``src.chains.graph`` module behind, for the sibling suite's guarantee."""

    graph_modules = {name for name in sys.modules if name.startswith(_GRAPH_ROOT)}
    for name in graph_modules:
        sys.modules.pop(name)
    yield
    for name in graph_modules:
        sys.modules.pop(name, None)


#: ``app.py``'s private module name. Pinned so the app script and this test module
#: resolve ``app``'s factory classes to the *same* class, which is what makes an instance
#: picklable across the app-script boundary.
_APP_MODULE = "isipfin_app_under_test"
_APP_PATH = REPO_ROOT / "app.py"

_APP: Any = None


def _app_module() -> Any:
    """``app.py``, loaded once by path under a name no other suite owns.

    The module is registered in ``sys.modules`` *before* ``exec_module``, because the
    dataclass decorator resolves a class's module through ``sys.modules`` when it
    processes its own annotations — loading it without registering it fails on the
    frozen dataclasses. Registering under a name that also survives into the app
    script is what lets a store double be pickled across the boundary: a class pickled
    under one module name and unpickled under another is a different class, and the
    double's methods would never be reached.

    The private name keeps a top-level ``app`` out of ``sys.modules``, where a sibling
    suite would find it.
    """

    global _APP
    if _APP is None:
        spec = importlib.util.spec_from_file_location(_APP_MODULE, _APP_PATH)
        assert spec is not None and spec.loader is not None
        _APP = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _APP
        spec.loader.exec_module(_APP)
    return _APP


# --- the turn under test -----------------------------------------------------------


def _envelope() -> Any:
    """A genuinely answered turn: a citation, a trace, no refusal.

    Built through the production :func:`final_envelope` so the shape under test is the
    shape the graph actually hands the UI — a hand-built envelope would prove the test
    double renders, not the pipeline's.
    """

    chains_state = _chains_state()
    return chains_state.final_envelope(
        {
            "answer_md": f"{ANSWER_MARKER} at $89.0B [1].",
            "citations": [
                chains_state.Citation(
                    ref_index=1,
                    doc_id="c0ffeec0ffeec0ffeec0ffeec0ffeec0ffeec0ffeeec0ffee",
                    page=42,
                    section_path=("INCOME STATEMENT",),
                    snippet="Cloud revenue was 89.0 billion.",
                )
            ],
            "grounded": True,
            "refusal_reason": None,
            "node_trace": [
                chains_state.NodeEvent(
                    node="analyze_question",
                    started_at=1.0,
                    duration_ms=12.5,
                    tokens_in=100,
                    tokens_out=4,
                )
            ],
            "query_plan": chains_state.QueryPlan(
                kind="simple",
                sub_queries=[],
                required_sections=("INCOME STATEMENT",),
                fiscal_period="FY2023",
                missing_aspect=None,
            ),
            "retrieved": [],
        }
    )


# --- the store doubles -------------------------------------------------------------
#
# The failure is injected through a store double whose write path raises. Each double's
# read path answers exactly as the real store answers for the exchange it did not take,
# which is what made the defect invisible: the read *succeeding* is precisely why the
# answer vanished. Every call is recorded on ``self.writes``/``self.reads``.


class _RecordingStore:
    """Base store double: records calls, keeps nothing on disk."""

    def __init__(self, failure: Exception | None, session_id: str = SESSION_ID) -> None:
        self.failure = failure
        self.session_id = session_id
        self.writes: list[str] = []
        self.reads: list[str] = []

    def append_turns(self, session_id: str, turns: Any) -> list[Any]:
        self.writes.append("append_turns")
        if self.failure is not None:
            raise self.failure
        return []

    def record_message(self, session_id: str, question: str, answer_md: str, **kwargs: Any) -> Any:
        self.writes.append("record_message")
        if self.failure is not None:
            raise self.failure
        return _message(question, answer_md)

    def get_history(self, session_id: str, *, turn_limit: int | None = None) -> Any:
        self.reads.append("get_history")
        return self._history()

    def list_messages(self, session_id: str, limit: int | None = None) -> list[Any]:
        self.reads.append("list_messages")
        return []

    def last_n_turns(self, session_id: str, n: int = 6) -> list[Any]:
        """The history window the graph's ``contextualize_question`` reads.

        Returned as the store's own rows so the app's history read is duck-typed
        exactly as in production; a store that cannot be read yields nothing.
        """

        self.reads.append("last_n_turns")
        if self.failure is not None:
            return []
        history = self.get_history(session_id)
        return list(history.turns[-n:]) if hasattr(history, "turns") else []

    def find_session(self, session_id: str) -> Any:
        return None

    def _history(self, turns: tuple[Turn, ...] = ()) -> Any:
        session = Session(
            id=self.session_id,
            title="conversation",
            created_at="2026-10-14T00:00:00.000Z",
            updated_at="2026-10-14T00:00:00.000Z",
            model=None,
        )
        return SessionHistory(session=session, turns=turns, messages=())


class _TotalFailureStore(_RecordingStore):
    """A store that refuses both rows: the read-only database, exactly.

    ``get_history`` answers as the real store answers for a session nothing was written
    to — an empty transcript — so the render path has nowhere to re-read the turn from.
    """


class _PartialWriteStore(_RecordingStore):
    """A store that committed the question and refused the answer.

    ``record_message`` writes its turn and its message in one transaction, so this is
    the store's real partial-write shape: the question is a committed ``turns`` row and
    there is no answer row. The transcript read therefore returns a lone user turn, and
    a fix that redraws the carried question anyway shows one ask as two questions.
    """

    def append_turns(self, session_id: str, turns: Any) -> list[Any]:
        self.writes.append("append_turns")
        return []

    def get_history(self, session_id: str, *, turn_limit: int | None = None) -> Any:
        """The lone committed question: the answer row the store refused to write."""

        self.reads.append("get_history")
        return self._history((_turn(self.session_id, 1, "user", QUESTION),))


class _HealthyStore(_RecordingStore):
    """A store that wrote the turn and can read it back: the control."""

    def get_history(self, session_id: str, *, turn_limit: int | None = None) -> Any:
        self.reads.append("get_history")
        return self._history(
            (
                _turn(self.session_id, 1, "user", QUESTION),
                _turn(self.session_id, 2, "assistant", f"{ANSWER_MARKER} at $89.0B [1]."),
            )
        )


def _turn(session_id: str, turn_id: int, role: str, content: str) -> Turn:
    """A real ``turns`` row, because ``render_transcript`` reads the real fields."""

    return Turn(
        id=turn_id,
        session_id=session_id,
        seq=turn_id,
        role=role,
        content=content,
        created_at="2026-10-14T00:00:00.000Z",
    )


def _message(question: str, answer_md: str) -> Message:
    """A real ``messages`` row, for the answer the store says it wrote."""

    return Message(
        id=1,
        question=question,
        answer_md=answer_md,
        grounded=True,
        refusal_reason=None,
        node_trace=None,
        turn=_turn(SESSION_ID, 2, "assistant", answer_md),
        citations=(),
    )


# --- the AppTest harness -----------------------------------------------------------


def _b64(payload: Any) -> str:
    """``payload`` as base64 pickle, so a script literal can carry a dataclass.

    The technique ``tests/test_degraded_planner_marker.py`` uses: ``repr`` cannot carry
    an ``AnswerEnvelope`` instance and JSON cannot carry a ``QueryPlan`` one.
    """

    return base64.b64encode(pickle.dumps(payload)).decode("ascii")


def _fspath(path: Path) -> str:
    """``path`` as a string safe to embed in a Python literal."""

    return str(path).replace("\\", "/")


def _app_script(
    *,
    store_spec: Any,
    answer_key: Any,
    question: str = "",
    last_session: str | None = SESSION_ID,
    real_store_path: Path | None = None,
    session_id: str = SESSION_ID,
) -> str:
    """One ``_render_chat_view`` call over the state one ask leaves behind.

    The placeholders are the store double (or a real one read from a file), the live
    envelope and the question, so one harness covers the failed write, the partial
    write, the happy path and the no-store path. The store double is pickled in, so its
    class and its recorded exception are the ones the test built rather than a re-typed
    literal.
    """

    if real_store_path is not None:
        store_line = (
            "from src.store.sessions import SessionStore as _SS\n"
            f"store = _SS(path={_fspath(real_store_path)!r})"
        )
    else:
        payload = _b64(store_spec) if store_spec is not None else _b64(None)
        store_line = (
            f"store = pickle.loads(base64.b64decode({payload!r})) "
            f"if {store_spec is not None!r} else None"
        )
    envelope_payload = _b64(answer_key if answer_key is not None else _envelope())
    app_path = _fspath(_APP_PATH)
    return (
        "import base64\n"
        "import json\n"
        "import pickle\n"
        "import sys\n"
        "import importlib.util\n"
        "import streamlit as st\n"
        f"\n{store_line}\n"
        f"spec = importlib.util.spec_from_file_location({_APP_MODULE!r}, {app_path!r})\n"
        f"app_module = sys.modules.get({_APP_MODULE!r})\n"
        "if app_module is None:\n"
        "    app_module = importlib.util.module_from_spec(spec)\n"
        f"    sys.modules[{_APP_MODULE!r}] = app_module\n"
        "    spec.loader.exec_module(app_module)\n"
        f"\nenvelope = pickle.loads(base64.b64decode({envelope_payload!r}))\n"
        "\n"
        "# The graph, stubbed: these are turn-lifecycle tests, not pipeline tests, and a\n"
        "# real graph wants a model, an index and a key. The stub keeps the envelope the\n"
        "# render draws, so the assertions stay on what the canvas shows.\n"
        "def _stub_ask(question, *, params, history, k):\n"
        "    return envelope\n"
        "\n"
        "app_module._ask = _stub_ask\n"
        "# ``AppTest`` executes one render pass per ``run()``, so the app's own\n"
        "# ``st.rerun()`` — the success path, where the transcript carries the turn —\n"
        "# is counted rather than performed: re-entering the view inside this pass\n"
        "# would register every widget key a second time, which Streamlit refuses.\n"
        "# The pass under test is the one that produced the turn, which is what the\n"
        "# answer-survival assertions are about.\n"
        "st.session_state.setdefault('reruns', 0)\n"
        "def _counted_rerun():\n"
        "    st.session_state['reruns'] = st.session_state['reruns'] + 1\n"
        "st.rerun = _counted_rerun\n"
        "def _record_calls():\n"
        f"    setattr(app_module, {_REPORT_ATTRIBUTE!r}, dict(\n"
        "        writes=list(getattr(store, 'writes', []) or []),\n"
        "        reads=list(getattr(store, 'reads', []) or []),\n"
        "    ))\n"
        "st.session_state.setdefault(app_module._ANSWER_KEY, None)\n"
        "st.session_state[app_module._LAST_KEY] = envelope\n"
        f"st.session_state[app_module._LAST_SESSION_KEY] = {last_session!r}\n"
        f"st.session_state[app_module._QUESTION_KEY] = {question!r}\n"
        "\n"
        "class _Rail:\n"
        f"    session_id = {session_id!r}\n"
        "    doc_labels = {}\n"
        "    doc_count = 0\n"
        "    index_ready = True\n"
        "    params = None\n"
        "    view = 'chat'\n"
        "\n"
        "app_module._render_chat_view(_Rail(), store)\n"
        "_record_calls()\n"
    )


def _render(
    tmp_path: Path,
    *,
    store_spec: Any = None,
    answer_key: Any = None,
    question: str = "",
    real_store_path: Path | None = None,
    session_id: str = SESSION_ID,
) -> tuple[AppTest, dict[str, list[str]]]:
    """Run one canvas render under ``AppTest``; return the tree and the store's calls."""

    # Load ``app.py`` here first, so the app script reuses this module object rather
    # than building a second one: two module objects means two ``_RecordingStore``
    # classes, and a store double pickled against one fails the render's duck-typed
    # calls in the other.
    app_module = _app_module()
    # The attribute is shared module state, so a report from an earlier render would
    # otherwise satisfy this one's assertions. Clear it before the pass runs.
    app_module.__dict__.pop(_REPORT_ATTRIBUTE, None)
    app = AppTest.from_string(
        _app_script(
            store_spec=store_spec,
            answer_key=answer_key,
            question=question,
            real_store_path=real_store_path,
            session_id=session_id,
        ),
        default_timeout=60,
    )
    app.run(timeout=60)
    report = getattr(app_module, _REPORT_ATTRIBUTE, None) or {"writes": [], "reads": []}
    return app, dict(report)


def _markdown(app: AppTest) -> list[str]:
    """Every markdown fragment the canvas rendered, in document order."""

    return [element.value for element in app.get("markdown")]


def _banners(app: AppTest) -> list[str]:
    """Every ``st.error`` banner body the render produced."""

    return [element.value for element in app.get("error")]


def _answer_count(app: AppTest) -> int:
    """How many times the answer was drawn on the canvas.

    ``chat.render_answer`` writes the answer as one markdown element and its markers and
    citation cards as separate elements, so the count is over the *joined* text: a
    second draw of the same envelope repeats the answer string and every one of its
    citation cards, which is what makes the count a reliable proxy for "drawn twice".
    """

    return " ".join(_markdown(app)).count(ANSWER_MARKER)


def _bubble_roles(app: AppTest) -> list[str]:
    """The role of every chat bubble, in document order.

    ``st.chat_message`` carries no key, and the widget key the theme's CSS targets is
    on the container *inside* the bubble, so the role is read from the bubble's own
    text: the transcript renderer writes a question as a user turn, and the answer's
    markdown always starts from the envelope.
    """

    roles: list[str] = []
    for message in app.get("chat_message"):
        text = " ".join(
            element.value for element in _descend(message) if type(element).__name__ == "Markdown"
        )
        if QUESTION in text and ANSWER_MARKER not in text:
            roles.append("user")
        elif ANSWER_MARKER in text:
            roles.append("assistant")
        else:
            roles.append("unknown")
    return roles


def _bubbles(app: AppTest) -> list[tuple[str, str]]:
    """Each chat bubble as ``(role, text)``, in document order."""

    pairs: list[tuple[str, str]] = []
    for role, message in zip(_bubble_roles(app), app.get("chat_message"), strict=False):
        texts = [
            element.value for element in _descend(message) if type(element).__name__ == "Markdown"
        ]
        pairs.append((role, "\n".join(texts)))
    return pairs


def _descend(node: Any) -> list[Any]:
    """Every element under one element-tree node, in document order."""

    found: list[Any] = [node]
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        for child in children.values():
            found.extend(_descend(child))
    return found


def _user_text(app: AppTest) -> str:
    """The user bubbles the canvas rendered, joined."""

    return " ".join(text for role, text in _bubbles(app) if role == "user")


def _assert_ran_cleanly(app: AppTest) -> None:
    """The canvas raised nothing — a render error is a regression, not a banner."""

    errors = [error.value for error in app.exception]
    assert not errors, f"the canvas must not raise: {errors}"


@pytest.fixture
def state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary runtime state root, so no test writes into the repo tree."""

    root = tmp_path / "state"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DATA_DIR", str(root))
    return root


# --- 1. the answer survives --------------------------------------------------------


class TestTheAnswerSurvivesAStoreThatRefusesIt:
    """The defect itself: a successful answer must not vanish on a failed write."""

    def test_the_answer_is_rendered_on_the_rerun(self, state_root: Path) -> None:
        """A refused store must not take the answer with it."""

        store = _TotalFailureStore(StoreError(READ_ONLY_MESSAGE))
        app, _ = _render(state_root, store_spec=store, question=QUESTION)
        _assert_ran_cleanly(app)

        assert ANSWER_MARKER in " ".join(_markdown(app)), (
            "the answer the graph produced must still be on the canvas; the store's "
            "refusal was to be reported, not allowed to erase the turn"
        )

    def test_the_question_is_rendered_too(self, state_root: Path) -> None:
        """A question that reached an answer must not leave an orphan answer."""

        store = _TotalFailureStore(StoreError(READ_ONLY_MESSAGE))
        app, _ = _render(state_root, store_spec=store, question=QUESTION)

        assert QUESTION in _user_text(app), (
            "the store took neither row, so the failed-write path must render the "
            "question — otherwise the answer appears with nothing having been asked"
        )

    def test_a_missing_session_is_reported_not_hidden(self, state_root: Path) -> None:
        """The store's other failure class lands on its own banner, not a blank canvas."""

        store = _TotalFailureStore(SessionNotFoundError("the session was deleted"))
        app, _ = _render(state_root, store_spec=store, question=QUESTION)

        assert ANSWER_MARKER in " ".join(_markdown(app))
        assert any(
            MISSING_SESSION_BANNER in body for body in _banners(app)
        ), "SessionNotFoundError is its own taxonomy row; the banner must say so"


# --- 2. the failure is visible -----------------------------------------------------


class TestTheFailureIsNamedNotSwallowed:
    """Persistence is best-effort; losing history is not silent."""

    def test_the_store_failure_renders_a_banner(self, state_root: Path) -> None:
        """The exception reaches the taxonomy, so the wording matches every other row."""

        store = _TotalFailureStore(StoreError(READ_ONLY_MESSAGE))
        app, _ = _render(state_root, store_spec=store, question=QUESTION)

        banners = _banners(app)
        assert banners, (
            "a store that refused the turn must say so; the answer alone is a silent "
            "loss of history"
        )
        assert STORE_BANNER in banners[0]

    def test_the_banner_is_not_a_raw_traceback(self, state_root: Path) -> None:
        """The banner is the classifier's view, not ``str(exc)`` swallowed raw."""

        store = _TotalFailureStore(StoreError(READ_ONLY_MESSAGE))
        app, _ = _render(state_root, store_spec=store, question=QUESTION)

        assert _banners(app), "a banner is required for the exception to be judged on"
        for body in _banners(app):
            assert "Traceback" not in body
            assert (
                "StoreError" not in body
            ), "a class name at the user is exactly what the typed views prevent"
            assert (
                READ_ONLY_MESSAGE not in body
            ), "the driver's raw message belongs in the log, never in the banner"


# --- 3. the turn is drawn once -----------------------------------------------------


class TestTheTurnIsNotDrawnTwice:
    """Deduplication is part of the fix, not an afterthought."""

    def test_a_partial_write_does_not_draw_the_question_twice(self, state_root: Path) -> None:
        """The question committed and the answer did not: one question, one answer."""

        store = _PartialWriteStore(StoreError(READ_ONLY_MESSAGE))
        app, _ = _render(state_root, store_spec=store, question=QUESTION)

        assert _answer_count(app) == 1, (
            "the store read path has no answer row, so the answer is drawn by the "
            "failed-write path alone"
        )
        assert _user_text(app).count(QUESTION) == 1, (
            "the transcript drew the committed question and the failed-write path "
            "drew its own; one ask must not appear as two questions"
        )

    def test_a_total_failure_draws_the_answer_exactly_once(self, state_root: Path) -> None:
        """Nothing was written, so the failed-write path is the only answer source."""

        store = _TotalFailureStore(StoreError(READ_ONLY_MESSAGE))
        app, _ = _render(state_root, store_spec=store, question=QUESTION)

        assert _answer_count(app) == 1, "the answer must be rendered by exactly one path"
        assert (
            _user_text(app).count(QUESTION) == 1
        ), "and the question behind it is not repeated either"

    def test_the_live_envelope_path_suppresses_the_failed_write_draw(
        self, state_root: Path
    ) -> None:
        """``_ANSWER_KEY`` already holds the envelope: it must not be drawn twice."""

        store = _TotalFailureStore(StoreError(READ_ONLY_MESSAGE))
        app, _ = _render(
            state_root,
            store_spec=store,
            answer_key=_envelope(),
            question=QUESTION,
        )

        assert _answer_count(app) == 1, (
            "the live envelope is already on the canvas; the failed-write path must "
            "not add a second copy of the same answer"
        )

    def test_a_healthy_transcript_renders_exactly_its_two_turns(self, state_root: Path) -> None:
        """A render with no outstanding failure behaves exactly as before the fix."""

        store = _HealthyStore(None)
        app, _ = _render(state_root, store_spec=store)

        assert not _banners(app)
        assert _bubble_roles(app) == [
            "user",
            "assistant",
        ], "the healthy transcript renders its two turns and nothing else"


# --- 4. the happy path is unchanged ------------------------------------------------


class TestTheHappyPathIsUnchanged:
    """A store that took the turn behaves exactly as it did before the fix."""

    def test_a_successful_write_renders_no_banner(self, state_root: Path) -> None:
        """Success changes nothing the user sees."""

        store = _HealthyStore(None)
        app, _ = _render(state_root, store_spec=store, question=QUESTION)

        assert not _banners(app)
        assert ANSWER_MARKER in " ".join(_markdown(app))

    def test_a_successful_write_attempts_both_rows(self, state_root: Path) -> None:
        """The write path is untouched: question and answer are still both written."""

        store = _HealthyStore(None)
        _, report = _render(state_root, store_spec=store, question=QUESTION)

        assert report["writes"] == ["append_turns", "record_message"]

    def test_a_failed_write_attempts_the_question_first(self, state_root: Path) -> None:
        """Record order is unchanged: the question is still attempted before the answer."""

        store = _TotalFailureStore(StoreError(READ_ONLY_MESSAGE))
        _, report = _render(state_root, store_spec=store, question=QUESTION)

        assert report["writes"] == ["append_turns"], (
            "a read-only database refuses at the question write, so the answer write is "
            "never reached — exactly as before the fix"
        )

    def test_a_failed_write_reads_the_transcript_once(self, state_root: Path) -> None:
        """The transcript is not re-read before the failed-write draw."""

        store = _TotalFailureStore(StoreError(READ_ONLY_MESSAGE))
        _, report = _render(state_root, store_spec=store, question=QUESTION)

        assert report["reads"] == ["get_history", "last_n_turns"], (
            "one transcript read for the canvas, one history window for the graph — "
            "the failed-write draw re-reads nothing"
        )


# --- 5. the record path's contract -------------------------------------------------


class TestRecordReportsFailureInsteadOfRaising:
    """``_record`` still swallows by contract; the caller now learns what happened."""

    def test_record_returns_the_failure_rather_than_raising(self) -> None:
        landed, failure = _app_module()._record(
            _TotalFailureStore(StoreError(READ_ONLY_MESSAGE)), SESSION_ID, QUESTION, _envelope()
        )

        assert landed is False
        assert isinstance(failure, StoreError)

    def test_record_returns_no_failure_on_success(self) -> None:
        landed, failure = _app_module()._record(
            _HealthyStore(None), SESSION_ID, QUESTION, _envelope()
        )

        assert (landed, failure) == (True, None)

    def test_a_driver_error_is_rebound_to_the_taxonomy_class(self) -> None:
        """A raw ``sqlite3.OperationalError`` must still reach the store's own banner."""

        landed, failure = _app_module()._record(
            _TotalFailureStore(sqlite3.OperationalError(READ_ONLY_MESSAGE)),
            SESSION_ID,
            QUESTION,
            _envelope(),
        )

        assert landed is False
        assert isinstance(failure, StoreError), (
            "sqlite3's own types must be rebound to the base class the classifier maps, "
            "or the banner falls through to the generic view and leaks the driver class"
        )

    def test_a_partial_write_reports_the_question_as_landed(self) -> None:
        """The caller must know the question row exists, to avoid drawing it twice."""

        landed, failure = _app_module()._record(
            _PartialWriteStore(StoreError(READ_ONLY_MESSAGE)), SESSION_ID, QUESTION, _envelope()
        )

        assert landed is True
        assert isinstance(failure, StoreError)


# --- 6. the real database, read only -----------------------------------------------


class TestAgainstARealReadOnlyDatabase:
    """The live reproduction: a correct answer over a read-only ``chat.sqlite3``."""

    def test_the_answer_survives_a_real_read_only_database(self, tmp_path: Path) -> None:
        """The whole cycle through the production store, end to end.

        ``chmod`` is the failure injection, so the exception the app handles is the one a
        read-only deployment produces. The file is restored afterwards, because a leaked
        read-only file would poison any later use of the path.

        The session the store already holds is used as the render's session, so the ask
        path reaches the store's own write and the banner is the one the read-only file
        produced rather than one the test author typed in.
        """

        root = tmp_path / "state"
        root.mkdir(parents=True, exist_ok=True)
        db = root / "chat.sqlite3"
        store = SessionStore(path=str(db))
        session = store.create_session("conversation")
        store.close()

        os.chmod(db, 0o444)
        try:
            app, _report = _render(
                tmp_path,
                store_spec=None,
                question=QUESTION,
                real_store_path=db,
                session_id=session.id,
            )
        finally:
            os.chmod(db, 0o644)

        _assert_ran_cleanly(app)
        assert ANSWER_MARKER in " ".join(
            _markdown(app)
        ), "the answer the graph produced survives a store that refused it"
        assert _banners(app), "the failed persistence must be reported"

    def test_a_read_only_database_surfaces_as_the_stores_own_error(self, tmp_path: Path) -> None:
        """The read-only file the app has to cope with: reads work, writes do not."""

        root = tmp_path / "state"
        root.mkdir(parents=True, exist_ok=True)
        db = root / "chat.sqlite3"
        store = SessionStore(path=str(db))
        session = store.create_session("conversation")
        store.close()

        os.chmod(db, 0o444)
        try:
            landed, failure = _app_module()._record(
                SessionStore(path=str(db)), session.id, QUESTION, _envelope()
            )
        finally:
            os.chmod(db, 0o644)

        assert landed is False, "the question write is refused by the file"
        assert isinstance(failure, StoreError), (
            "a read-only database must surface as the store's own error, so the banner "
            "is the store row of the taxonomy rather than the generic view"
        )

    def test_a_read_only_store_still_reads_its_sessions(self, tmp_path: Path) -> None:
        """Reads were never the thing that failed — the session survives the refusal."""

        root = tmp_path / "state"
        root.mkdir(parents=True, exist_ok=True)
        db = root / "chat.sqlite3"
        store = SessionStore(path=str(db))
        session = store.create_session("conversation")
        store.close()

        os.chmod(db, 0o444)
        try:
            landed, _failure = _app_module()._record(
                SessionStore(path=str(db)), session.id, QUESTION, _envelope()
            )
            reread = SessionStore(path=str(db))
            found = reread.find_session(session.id)
            reread.close()
        finally:
            os.chmod(db, 0o644)

        assert landed is False
        assert found is not None, (
            "the session the write refused must still exist, because reads were never "
            "the thing that failed"
        )

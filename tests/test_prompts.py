"""Prompt-loader invariants (T8, ``docs/prompts/README.md`` versioning protocol).

Every test here is hermetic: no network, no writes into the repo's tree. The loader
under test is ``src/utils/prompts.py``; the assets it reads are the five versioned
markdown files in ``docs/prompts/`` plus the per-node contract in
``scripts/prompts_meta.json``.

The invariants pinned are the ones the protocol states in prose:

1. all five ids load, each with a non-empty body and a parseable ``major.minor.patch``
   version;
2. the header carries every documented field non-empty — id, version, node, model
   target, output schema, changelog (rule 2: "semver in the header"; a missing cell
   must fail the load, not degrade to a default);
3. ``required_version`` is the recorded floor for each id and ``assert_min_version``
   passes on the checked-out tree (rule 5: a stale deployment fails loudly);
4. an id outside the library raises ``UnknownPromptError``, never a silent ``None``;
5. each prompt's ``node`` cell agrees with the node name recorded in
   ``scripts/prompts_meta.json`` — a rename in one place cannot drift from the other;
6. ``PROMPT_DIR`` is the module's own ``docs/prompts/`` (never cwd-relative) and no
   prompt text is duplicated inside ``src/`` (rule 3: node code contains no inline
   system prompts);
7. version comparison is numeric, so a 1.10.0 prompt satisfies a 1.9.0 floor while a
   lexicographic compare would silently fail the other way.

Per ``docs/TESTING-AND-EVAL.md`` ("No test asserts a prompt's literal wording"), no
assertion here quotes prompt prose. The duplication check in 6 is the inverse
statement: it fails only when prose *is* copied into code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from src.utils import prompts
from src.utils.prompts import (
    PROMPT_FILES,
    PROMPT_META_PATH,
    REQUIRED_FIELDS,
    PromptDoc,
    PromptError,
    PromptParseError,
    PromptVersionError,
    UnknownPromptError,
    assert_min_version,
    load_prompt,
    node_meta,
    parse_header,
    parse_version,
    prompt_meta,
    required_version,
)

PROMPT_IDS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5")

# A prompt id that is not in PROMPT_FILES, and a name that is not a prompt id at all.
UNKNOWN_PROMPT_ID = "P9"
UNKNOWN_PLAUSIBLE_ID = "contextualize_question"

# The minimum length a fragment must have before "no prompt text in code" is a
# meaningful claim: shorter spans collide with ordinary English ("Your job: ..."),
# which is wording overlap, not a forked spec.
FRAGMENT_MIN_CHARS = 60

# The header table every prompt file starts with (docs/prompts/README.md rule 2).
_HEADER_LABEL = "Field"
_HEADER_VALUE = "Value"

# A node name in ``scripts/prompts_meta.json``: snake_case, i.e. not prose.
_NODE_NAME = re.compile(r"`([a-z][a-z0-9_]*)`")


def _write_synthetic_prompts(
    directory: Path, *, target: str | None = None, version: str = "1.0.0", body: str
) -> None:
    """Write the five synthetic assets into ``directory``.

    When ``target`` is given, only that id is built at ``version``; the rest stay at
    1.0.0 so the rest of the contract is unchanged.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for prompt_id, filename in PROMPT_FILES.items():
        (directory / filename).write_text(
            "\n".join(_header_table(prompt_id, version=version if prompt_id == target else "1.0.0"))
            + "\n\n"
            + body,
            encoding="utf-8",
        )


def _redirect_prompt_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    """Point ``PROMPT_DIR`` at a temporary copy of the five assets with new bodies.

    The files are written fresh — the loader reads on every call, so this exercises
    the real path through ``load_prompt`` without touching the checked-out prompts.
    """
    directory = tmp_path / "docs" / "prompts"
    _write_synthetic_prompts(directory, body=body)
    for fixture_filename in PROMPT_FILES.values():
        on_disk = (prompts.PROMPT_DIR / fixture_filename).read_text(encoding="utf-8")
        assert on_disk != (directory / fixture_filename).read_text(
            encoding="utf-8"
        ), "fixture must differ from the real asset for the test to mean anything"
    monkeypatch.setattr(prompts, "PROMPT_DIR", directory)
    return directory


def _redirect_prompt_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_id: str, floor: str
) -> None:
    """Point ``PROMPT_META_PATH`` at a copy of the real contract with one floor changed."""
    payload = json.loads(PROMPT_META_PATH.read_text(encoding="utf-8"))
    payload[prompt_id] = {**payload[prompt_id], "min_version": floor}
    meta_path = tmp_path / "prompts_meta.json"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(prompts, "PROMPT_META_PATH", meta_path)


def _header_table(prompt_id: str, version: str = "1.0.0") -> list[str]:
    """A complete version-header table for ``prompt_id``, in the documented order."""
    return [
        f"| {_HEADER_LABEL} | {_HEADER_VALUE} |",
        "|---|---|",
        f"| Prompt id | `{prompt_id}` |",
        f"| Version | {version} |",
        "| Node | `fixture_node` |",
        "| Model target | any (temperature 0.0) |",
        "| Output schema | Fixture Schema |",
        f"| Changelog | {version} — synthetic fixture body |",
    ]


# --- 1. all five assets load, non-empty body, parseable version -------------------


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_prompt_loads_with_nonempty_body(prompt_id: str) -> None:
    doc = load_prompt(prompt_id)

    assert isinstance(doc, PromptDoc)
    assert isinstance(doc.body, str) and doc.body.strip(), f"{prompt_id} loaded an empty body"
    # The body is the spec *below* the header: the header table must not leak into it.
    assert "| Prompt id |" not in doc.body, "version header bled into the body"
    assert "| Version |" not in doc.body, "version header bled into the body"


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_loaded_version_parses_as_semver(prompt_id: str) -> None:
    doc = load_prompt(prompt_id)

    version = parse_version(doc.version)
    assert isinstance(version, tuple) and len(version) == 3
    assert all(isinstance(part, int) for part in version)


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_body_is_the_file_text_below_the_header(prompt_id: str) -> None:
    """``.body`` is a verbatim slice of the asset, not a reconstruction of it."""
    text = (prompts.PROMPT_DIR / PROMPT_FILES[prompt_id]).read_text(encoding="utf-8")
    doc = load_prompt(prompt_id)

    assert doc.body in text, f"{prompt_id}.body is not a verbatim slice of its file"
    assert not doc.body.startswith("|"), f"{prompt_id}.body starts inside the header table"


def test_every_documented_id_is_covered_by_the_loader() -> None:
    # The library is exactly the documented inventory: no missing id, no invented one.
    assert set(PROMPT_FILES) == set(PROMPT_IDS)


def test_every_prompt_file_exists_on_disk() -> None:
    missing = [
        PROMPT_FILES[prompt_id]
        for prompt_id in PROMPT_FILES
        if not (prompts.PROMPT_DIR / PROMPT_FILES[prompt_id]).is_file()
    ]
    assert missing == []


def test_no_prompt_file_is_missing_from_the_loader() -> None:
    """An asset checked into ``docs/prompts/`` that no id loads is a fork in the spec."""
    on_disk = {path.name for path in prompts.PROMPT_DIR.glob("*.md")}
    unloaded = on_disk - set(PROMPT_FILES.values()) - {"README.md"}
    assert unloaded == set(), f"prompt files no id loads: {sorted(unloaded)}"


# --- 2. every header field present and non-empty ---------------------------------


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_meta_carries_every_documented_field_nonempty(prompt_id: str) -> None:
    doc = load_prompt(prompt_id)

    assert set(REQUIRED_FIELDS) == set(doc.meta), f"{prompt_id} header is missing a field"
    for field in REQUIRED_FIELDS:
        value = doc.meta[field]
        assert (
            isinstance(value, str) and value.strip()
        ), f"{prompt_id} header field {field!r} is empty"


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_dataclass_fields_mirror_the_header(prompt_id: str) -> None:
    doc = load_prompt(prompt_id)

    assert doc.id == prompt_id == doc.meta["prompt_id"]
    assert doc.version == doc.meta["version"]
    assert doc.node == doc.meta["node"]
    assert doc.model_target == doc.meta["model_target"]
    assert doc.output_schema == doc.meta["output_schema"]
    assert doc.changelog == doc.meta["changelog"]


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_changelog_names_the_version_it_belongs_to(prompt_id: str) -> None:
    """Rule 1: the changelog line names the release it documents — a v1.0.0 prompt
    cannot carry a changelog from some other version."""
    doc = load_prompt(prompt_id)

    assert doc.changelog.startswith(
        doc.version
    ), f"{prompt_id} changelog does not begin with v{doc.version}"


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_meta_is_read_only(prompt_id: str) -> None:
    """A mutable ``meta`` would let a caller "fix" a version after the fact and defeat
    every downstream minimum-version assertion."""
    doc = load_prompt(prompt_id)

    assert isinstance(doc.meta, MappingProxyType)
    # The assignment attempt is made through a deliberately untyped alias: what is
    # under test is the runtime refusal of the mapping proxy (not the type
    # checker), so the alias carries no static type to defeat.
    meta: Any = doc.meta
    with pytest.raises(TypeError):
        meta["version"] = "9.9.9"


def test_doc_is_frozen() -> None:
    doc = load_prompt("P3")

    # A frozen dataclass refuses direct attribute assignment, so the attempt is
    # made through a deliberately untyped alias: the pinned invariant is the
    # runtime ``FrozenInstanceError`` (an ``AttributeError``), not the type checker.
    probe: Any = doc
    with pytest.raises(AttributeError):
        probe.version = "9.9.9"


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_missing_header_field_fails_the_load(
    prompt_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 2: an empty cell fails the load rather than degrading to a default."""
    lines = _header_table(prompt_id)
    lines[5] = "| Model target |  |"  # a blank value for a documented field
    directory = tmp_path / "docs" / "prompts"
    directory.mkdir(parents=True)
    (directory / PROMPT_FILES[prompt_id]).write_text(
        "\n".join(lines) + "\n\n## Purpose\n\nSynthetic fixture body.\n", encoding="utf-8"
    )
    monkeypatch.setattr(prompts, "PROMPT_DIR", directory)

    with pytest.raises(PromptParseError) as excinfo:
        load_prompt(prompt_id)
    assert "model_target" in str(excinfo.value)


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_missing_file_fails_the_load(
    prompt_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checked-out tree with a prompt file deleted must fail loudly, not return None."""
    directory = tmp_path / "docs" / "prompts"
    directory.mkdir(parents=True)
    monkeypatch.setattr(prompts, "PROMPT_DIR", directory)

    with pytest.raises(PromptError) as excinfo:
        load_prompt(prompt_id)
    assert PROMPT_FILES[prompt_id] in str(excinfo.value)


def test_header_prompt_id_must_match_the_requested_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading ``P1`` out of a file whose header says ``P2`` is a fork, not a load."""
    directory = tmp_path / "docs" / "prompts"
    directory.mkdir(parents=True)
    (directory / PROMPT_FILES["P1"]).write_text(
        "\n".join(_header_table("P2")) + "\n\n## Purpose\n\nSynthetic fixture body.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompts, "PROMPT_DIR", directory)

    with pytest.raises(PromptParseError) as excinfo:
        load_prompt("P1")
    assert "P2" in str(excinfo.value)


def test_load_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The header docstring pins "no caching, by design": a prompt edit is picked up by
    the next load, so restarting the app is not required to see it."""
    _redirect_prompt_dir(tmp_path, monkeypatch, "## Purpose\n\nFirst fixture body.\n")

    assert load_prompt("P3").version == "1.0.0"

    directory = prompts.PROMPT_DIR
    (directory / PROMPT_FILES["P3"]).write_text(
        "\n".join(_header_table("P3", version="1.10.0"))
        + "\n\n## Purpose\n\nSecond fixture body.\n",
        encoding="utf-8",
    )
    reloaded = load_prompt("P3")

    assert reloaded.version == "1.10.0"
    assert reloaded.changelog.startswith("1.10.0")
    assert "Second fixture body" in reloaded.body


# --- 3. the minimum-version floor -------------------------------------------------


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_required_version_is_the_recorded_floor(prompt_id: str) -> None:
    payload = json.loads(PROMPT_META_PATH.read_text(encoding="utf-8"))

    assert required_version(prompt_id) == payload[prompt_id]["min_version"]


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_assert_min_version_passes_on_the_checked_out_tree(prompt_id: str) -> None:
    """The floor recorded in scripts/prompts_meta.json is satisfied by the asset on
    disk — the checked-out prompt library is not stale against its own contract. This
    is the assertion a node's import-time floor would make, made explicit here."""
    doc = assert_min_version(prompt_id)

    assert doc.id == prompt_id
    assert parse_version(doc.version) >= parse_version(required_version(prompt_id))


@pytest.mark.parametrize(
    ("version", "floor"),
    [
        ("1.10.0", "1.9.0"),
        ("1.2.0", "1.1.0"),
        ("2.0.0", "1.99.99"),
        ("1.0.1", "1.0.0"),
        ("1.10.5", "1.9.9"),
    ],
)
def test_a_newer_prompt_satisfies_its_floor(
    version: str, floor: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 5, the passing half: a prompt newer than the floor loads, and the ordering
    is numeric — ``version`` beats every floor here, including ``1.10.0`` vs ``1.9.0``."""
    directory = _redirect_prompt_dir(tmp_path, monkeypatch, "## Purpose\n\nFixture body.\n")
    _write_synthetic_prompts(directory, target="P3", version=version, body="## Purpose\n\nBody.\n")
    _redirect_prompt_meta(tmp_path, monkeypatch, "P3", floor)

    doc = assert_min_version("P3")

    assert doc.version == version
    assert parse_version(doc.version) >= parse_version(floor)


def test_an_older_prompt_fails_its_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rule 5's loud failure: the deployment fails instead of running the old prompt."""
    directory = _redirect_prompt_dir(tmp_path, monkeypatch, "## Purpose\n\nFixture body.\n")
    _write_synthetic_prompts(directory, target="P3", version="1.0.0", body="## Purpose\n\nBody.\n")
    _redirect_prompt_meta(tmp_path, monkeypatch, "P3", "1.10.1")

    with pytest.raises(PromptVersionError) as excinfo:
        assert_min_version("P3")
    assert "1.0.0" in str(excinfo.value) and "1.10.1" in str(excinfo.value)


def test_a_prompt_at_exactly_the_floor_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal is allowed: the contract is a minimum, not a strict floor."""
    directory = _redirect_prompt_dir(tmp_path, monkeypatch, "## Purpose\n\nFixture body.\n")
    _write_synthetic_prompts(directory, target="P3", version="1.10.0", body="## Purpose\n\nBody.\n")
    _redirect_prompt_meta(tmp_path, monkeypatch, "P3", "1.10.0")

    assert assert_min_version("P3").version == "1.10.0"


# --- 4. an id outside the library raises ------------------------------------------


@pytest.mark.parametrize("bad_id", [UNKNOWN_PROMPT_ID, UNKNOWN_PLAUSIBLE_ID, "p1", "", "P1 "])
def test_unknown_prompt_id_raises(bad_id: str) -> None:
    with pytest.raises(UnknownPromptError) as excinfo:
        load_prompt(bad_id)

    assert repr(bad_id) in str(excinfo.value)


def test_unknown_prompt_error_is_a_prompt_error() -> None:
    with pytest.raises(PromptError):
        load_prompt(UNKNOWN_PROMPT_ID)


def test_required_version_rejects_an_unknown_id() -> None:
    with pytest.raises(UnknownPromptError):
        required_version(UNKNOWN_PROMPT_ID)


def test_assert_min_version_rejects_an_unknown_id() -> None:
    with pytest.raises(UnknownPromptError):
        assert_min_version(UNKNOWN_PROMPT_ID)


def test_an_unknown_id_never_resolves_to_a_loaded_prompt() -> None:
    """The failure mode this guards: a loader that returns the first asset on a miss
    would answer with the wrong prompt rather than telling the caller."""
    for bad_id in (UNKNOWN_PROMPT_ID, UNKNOWN_PLAUSIBLE_ID):
        with pytest.raises(UnknownPromptError):
            load_prompt(bad_id)


# --- 5. the node names agree with the recorded contract ---------------------------


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_node_matches_the_recorded_meta_node(prompt_id: str) -> None:
    """The prompt's ``Node`` cell and scripts/prompts_meta.json's ``node`` key must name
    the same graph node — a rename in one of them is otherwise invisible."""
    doc = load_prompt(prompt_id)
    recorded = prompt_meta()[prompt_id]["node"]

    if recorded is None:
        # P4 is an appendix with no node of its own (README: "appended to P3/P5").
        assert prompt_id == "P4"
        assert doc.node and doc.node.strip()
        return
    assert isinstance(recorded, str)
    assert recorded in doc.node


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_every_node_a_prompt_names_is_in_the_node_index(prompt_id: str) -> None:
    """A prompt naming a node the meta file has no entry for would be unrunnable: the
    mock and the floor lookup both key on that index."""
    doc = load_prompt(prompt_id)
    index = set(json.loads(PROMPT_META_PATH.read_text(encoding="utf-8"))["nodes"])

    named = set(_NODE_NAME.findall(doc.node)) | {prompt_meta()[prompt_id]["node"]}
    assert named <= index | {None}


def test_the_node_index_points_back_at_the_prompt_library() -> None:
    """The inverse direction of the same contract: every recorded node resolves to a
    prompt that loads, names the node, and is new enough to run."""
    payload = json.loads(PROMPT_META_PATH.read_text(encoding="utf-8"))

    for node_name, entry in payload["nodes"].items():
        prompt_id = entry["prompt_id"]
        assert prompt_id in PROMPT_FILES, f"node {node_name!r} points at an unknown prompt"
        assert node_meta(node_name)["prompt_id"] == prompt_id
        assert node_meta(node_name)["min_version"] == required_version(prompt_id)
        # The prompt the node consumes exists and is new enough to run.
        doc = assert_min_version(prompt_id)
        assert doc.id == prompt_id
        assert prompt_meta()[prompt_id]["node"] == node_name
        assert (
            node_name in doc.node
        ), f"node {node_name!r} consumes {prompt_id}, whose Node cell does not name it"


def test_node_meta_rejects_an_unknown_node() -> None:
    with pytest.raises(UnknownPromptError):
        node_meta("not_a_real_node")


def test_node_meta_resolves_every_consuming_node() -> None:
    payload = json.loads(PROMPT_META_PATH.read_text(encoding="utf-8"))

    for node_name in payload["nodes"]:
        entry = node_meta(node_name)
        assert entry["prompt_id"] in PROMPT_FILES


def _readme_inventory_rows() -> list[list[str]]:
    """The five ``| Id | File | Node | Role | Failure exposure |`` rows of the inventory
    table in docs/prompts/README.md — the public id-to-file map the protocol promises.
    """
    readme = (prompts.PROMPT_DIR / "README.md").read_text(encoding="utf-8")
    rows: list[list[str]] = []
    for line in readme.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"P\d", cells[0]) and cells[1].endswith(".md"):
            rows.append(cells)
    return rows


def test_the_documented_inventory_matches_the_loader() -> None:
    """docs/prompts/README.md's inventory table is the public id->file map; the loader's
    ``PROMPT_FILES`` must agree with it or the docs review a spec the code never loads."""
    inventory = {row[0]: row[1] for row in _readme_inventory_rows()}

    assert inventory == dict(PROMPT_FILES)


def test_every_prompt_file_in_the_inventory_exists() -> None:
    for row in _readme_inventory_rows():
        assert (
            prompts.PROMPT_DIR / row[1]
        ).is_file(), f"inventory names a file that is not checked in: {row[1]}"


# --- 6. the path is the module's own, and no prompt text lives in code ------------


def test_prompt_dir_is_the_repo_docs_dir_not_the_cwd() -> None:
    """Resolved from this module, never from the cwd: the app, the scripts and pytest
    import it from three different working directories."""
    module_path = Path(prompts.__file__).resolve()  # src/utils/prompts.py
    repo_root = module_path.parents[2]

    assert repo_root == prompts.REPO_ROOT, "repo root is three parents above src/utils"
    assert repo_root / "docs" / "prompts" == prompts.PROMPT_DIR
    assert prompts.PROMPT_DIR.is_absolute(), "the loader must not resolve docs/prompts from cwd"


def test_prompt_dir_is_not_relative() -> None:
    assert prompts.PROMPT_DIR.is_absolute()
    assert prompts.PROMPT_DIR.resolve() == prompts.PROMPT_DIR


def test_no_prompt_text_is_duplicated_in_the_loader_module() -> None:
    """Rule 3: the module reads the docs files and contains no prompt text of its own."""
    source = Path(prompts.__file__).read_text(encoding="utf-8")

    for prompt_id, fragment in _prompt_fragments():
        assert fragment not in source, (
            f"{prompt_id}: prompt text duplicated inside src/utils/prompts.py "
            f"({len(fragment)} chars: {fragment[:60]!r})"
        )


def test_no_prompt_text_is_duplicated_in_src() -> None:
    """The whole of ``src/`` is scanned, not just the loader: node code must contain no
    inline system prompts (docs/REPO-STRUCTURE.md rule "no inline system prompts")."""
    sources = sorted(Path(prompts.REPO_ROOT, "src").rglob("*.py"))
    assert sources, "expected src/ to hold Python files"

    for path in sources:
        text = path.read_text(encoding="utf-8")
        for prompt_id, fragment in _prompt_fragments():
            assert fragment not in text, (
                f"{prompt_id}: prompt text duplicated inside {path} "
                f"({len(fragment)} chars: {fragment[:60]!r})"
            )


def _prompt_fragments() -> list[tuple[str, str]]:
    """Long, distinctive spans of prompt text: every fenced block and every body line
    of at least ``FRAGMENT_MIN_CHARS`` characters.

    Fenced blocks are the system prompts, input templates and schema examples — the
    spans a copy-into-code regression would carry verbatim. Lines are the prose rules.
    ``FRAGMENT_MIN_CHARS`` keeps ordinary English out of the check.
    """
    fragments: list[tuple[str, str]] = []
    for prompt_id in PROMPT_FILES:
        body = load_prompt(prompt_id).body
        for block in re.findall(r"^```[^\n]*\n(.*?)^```", body, re.S | re.M):
            if block.strip():
                fragments.append((prompt_id, block))
        for line in body.splitlines():
            stripped = line.strip()
            if len(stripped) >= FRAGMENT_MIN_CHARS:
                fragments.append((prompt_id, stripped))
    assert fragments, "no prompt fragments collected: the check would assert nothing"
    return fragments


def test_the_fragments_are_long_enough_to_be_meaningful() -> None:
    """The duplication check must actually inspect substantial prompt text — a check
    that only looks at trivial spans would pass on any fork."""
    fragments = _prompt_fragments()

    assert all(len(fragment) >= FRAGMENT_MIN_CHARS for _, fragment in fragments)
    assert max(len(fragment) for _, fragment in fragments) > 500


def test_meta_file_is_read_from_the_repo_scripts_dir() -> None:
    assert prompts.PROMPT_META_PATH == prompts.REPO_ROOT / "scripts" / "prompts_meta.json"
    assert prompts.PROMPT_META_PATH.is_file()


def test_no_prompt_dir_at_the_repo_root() -> None:
    """docs/REPO-STRUCTURE.md reserves a top-level prompts/ directory that was never
    used: the assets live in docs/prompts/ instead. A stray top-level copy would be a
    second source of truth."""
    assert not Path(prompts.REPO_ROOT / "prompts").exists()


# --- 7. version comparison is numeric ---------------------------------------------


def test_parse_version_orders_minor_versions_numerically() -> None:
    assert parse_version("1.10.0") > parse_version("1.9.0")
    assert parse_version("1.10.0") != parse_version("1.9.0")
    assert "1.10.0" < "1.9.0", "the string order must differ, or this proves nothing"


def test_parse_version_orders_patch_and_major_numerically() -> None:
    assert parse_version("1.0.10") > parse_version("1.0.9")
    assert parse_version("2.0.0") > parse_version("1.99.99")
    assert parse_version("10.0.0") > parse_version("9.0.0")


def test_parse_version_orders_equal_versions_equal() -> None:
    assert parse_version("1.0.0") == parse_version("1.0.0")
    assert parse_version(" 1.0.0 ") == parse_version("1.0.0")
    assert not parse_version("1.0.0") < parse_version("1.0.0")


@pytest.mark.parametrize(
    "malformed",
    ["", "1", "1.0", "1.0.0.0", "1.0.x", "v1.0.0", "1.0.0-beta", "1.-1.0", "a.b.c"],
)
def test_parse_version_rejects_malformed_versions(malformed: str) -> None:
    with pytest.raises(PromptParseError):
        parse_version(malformed)


@pytest.mark.parametrize("version", ["1.0.0", "0.0.0", "1.10.0", "10.20.30", "1.0.1"])
def test_load_accepts_every_well_formed_version(
    version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The version grammar is ``major.minor.patch`` with digits only; every string in it
    must survive the load and reach a comparable floor check."""
    directory = tmp_path / "docs" / "prompts"
    directory.mkdir(parents=True)
    (directory / PROMPT_FILES["P4"]).write_text(
        "\n".join(_header_table("P4", version=version)) + "\n\n## Purpose\n\nThird fixture body.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompts, "PROMPT_DIR", directory)

    doc = load_prompt("P4")
    assert doc.version == version
    assert parse_version(doc.version)


@pytest.mark.parametrize("malformed", ["", "1", "1.0", "1.0.0.0", "1.0.x", "v1.0.0"])
def test_load_rejects_a_prompt_with_an_unparseable_version(
    malformed: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version the loader cannot parse must fail the load, never reach a
    minimum-version comparison that silently passes."""
    directory = tmp_path / "docs" / "prompts"
    directory.mkdir(parents=True)
    (directory / PROMPT_FILES["P4"]).write_text(
        "\n".join(_header_table("P4", version=malformed))
        + "\n\n## Purpose\n\nThird fixture body.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompts, "PROMPT_DIR", directory)

    with pytest.raises(PromptParseError):
        load_prompt("P4")


def test_parse_header_rejects_a_row_that_is_not_two_cells() -> None:
    with pytest.raises(PromptParseError):
        parse_header("| Field | Value |\n|---|---|\n| A | B | C |\n\nbody\n")


def test_parse_header_rejects_an_unknown_field() -> None:
    with pytest.raises(PromptParseError):
        parse_header(
            f"| {_HEADER_LABEL} | {_HEADER_VALUE} |\n|---|---|\n"
            "| Prompt id | P1 |\n| Not a field | x |\n\nbody\n"
        )


def test_parse_header_rejects_a_duplicate_field() -> None:
    with pytest.raises(PromptParseError):
        parse_header(
            f"| {_HEADER_LABEL} | {_HEADER_VALUE} |\n|---|---|\n"
            "| Prompt id | P1 |\n| Prompt id | P1 |\n\nbody\n"
        )


def test_parse_header_rejects_a_file_with_no_header_table() -> None:
    with pytest.raises(PromptParseError):
        parse_header("# No header here\n\nJust prose.\n")


def test_parse_header_stops_at_the_first_non_table_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray ``|`` in the body must never bleed into the header: a prompt line that
    begins with ``| answer |`` is body, not a header field.
    """
    directory = tmp_path / "docs" / "prompts"
    directory.mkdir(parents=True)
    body = "## Rules\n\n| Not a header row | another cell | third cell |\n\nStray prose.\n"
    (directory / PROMPT_FILES["P2"]).write_text(
        "\n".join(_header_table("P2")) + "\n" + body, encoding="utf-8"
    )
    monkeypatch.setattr(prompts, "PROMPT_DIR", directory)

    doc = load_prompt("P2")

    # The body is everything below the header table with leading blank lines dropped;
    # a trailing newline from the file is not part of the loaded body.
    assert doc.body == body.rstrip("\n")
    assert set(doc.meta) == set(REQUIRED_FIELDS)


def test_parse_header_ignores_a_table_before_the_version_header() -> None:
    """An unrelated leading table (an evidence table, say) is not the version header."""
    text = (
        "| Evidence | Case |\n|---|---|\n| STRESS-TEST-RAG #1 | wrong year |\n\n"
        + "\n".join(_header_table("P1"))
        + "\n\n## Purpose\n\nBody.\n"
    )
    fields, body_start = parse_header(text)

    assert fields["prompt_id"] == "P1"
    assert fields["node"] == "fixture_node"
    assert fields["changelog"] == "1.0.0 — synthetic fixture body"
    assert body_start > 0


# --- the loaded-asset sanity the whole suite leans on ----------------------------


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_meta_mapping_is_keyed_by_the_documented_fields(prompt_id: str) -> None:
    meta = prompt_meta()[prompt_id]

    for key in ("node", "min_version", "structured", "output_schema"):
        assert key in meta, f"{prompt_id} meta entry lacks {key!r}"


def test_prompt_meta_covers_every_documented_id() -> None:
    assert set(prompt_meta()) == set(PROMPT_IDS)


def test_meta_structured_flags_match_the_output_schema_kind() -> None:
    """``structured`` is the mock's fingerprint half: a structured prompt names a JSON
    schema, an unstructured one names prose or markdown."""
    for prompt_id in PROMPT_FILES:
        entry = prompt_meta()[prompt_id]
        schema = load_prompt(prompt_id).output_schema
        if entry["structured"]:
            assert "JSON" in schema or "Result" in schema or "Plan" in schema, schema
        else:
            assert "JSON" not in schema, schema


# --- 8. the table-citation contract matches what checks.split_units can read -------


def test_p4_prescribes_the_table_marker_placement_the_pre_pass_can_read() -> None:
    """Rule 1 and rule 4 must name the same placement.

    P4 v1.0.0's rule 1 said "a source note *under* the table lists the ``[n]`` markers",
    while its own rule 4 said "place the ``[n]`` marker in the row label cell … per-cell
    when cells come from different blocks". ``checks.split_units`` reads a table's markers
    from the rows themselves plus the single line immediately introducing the table — a
    note below the table is outside the block, so the rows inherit nothing and every row is
    trimmed as ``figure_not_in_span``. Golden case q03 hit exactly that: both rows trimmed,
    ``figure_ok: False``.

    The prompt promised a placement the symbolic layer cannot read, so the prompt was the
    wrong side and this is the assertion that keeps the two rules from drifting apart
    again: a marker in a *note* is not a citation the verifier can see.
    """

    rules = _fenced_rules("P4")

    assert "row label cell" in rules, "rule 4's placement must stay prescribed"
    assert "per-cell" in rules, "rule 4's per-block placement must stay prescribed"
    # The corrected wording must survive the edit, and the contradictory one must not return.
    assert "Put the [n] marker in the table itself" in rules


def test_p4_no_longer_prescribes_a_source_note_as_a_citation() -> None:
    """The precise regression: "a source note under the table lists the markers".

    Kept as its own assertion because the rule 1 wording is the thing that was wrong, and a
    rewrite of the surrounding sentence must not smuggle the same instruction back in.
    """

    rules = _fenced_rules("P4")

    assert "source note under the table lists the [n] markers" not in rules
    assert "A source note under the table" not in rules


def _fenced_rules(prompt_id: str) -> str:
    """The prompt's fenced rules block — the text the node sends the model verbatim."""

    doc = load_prompt(prompt_id)
    blocks = re.findall(r"^```[^\n]*\n(.*?)^```", doc.body, re.S | re.M)
    assert blocks, f"{prompt_id} carries no fenced rules block"
    return "\n".join(blocks)

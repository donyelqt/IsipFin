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

UNKNOWN_PROMPT_ID = "P9"
UNKNOWN_PLAUSIBLE_ID = "contextualize_question"

FRAGMENT_MIN_CHARS = 60

_HEADER_LABEL = "Field"
_HEADER_VALUE = "Value"

_NODE_NAME = re.compile(r"`([a-z][a-z0-9_]*)`")


def _write_synthetic_prompts(
    directory: Path, *, target: str | None = None, version: str = "1.0.0", body: str
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for prompt_id, filename in PROMPT_FILES.items():
        (directory / filename).write_text(
            "\n".join(_header_table(prompt_id, version=version if prompt_id == target else "1.0.0"))
            + "\n\n"
            + body,
            encoding="utf-8",
        )


def _redirect_prompt_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
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
    payload = json.loads(PROMPT_META_PATH.read_text(encoding="utf-8"))
    payload[prompt_id] = {**payload[prompt_id], "min_version": floor}
    meta_path = tmp_path / "prompts_meta.json"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(prompts, "PROMPT_META_PATH", meta_path)


def _header_table(prompt_id: str, version: str = "1.0.0") -> list[str]:
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


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_prompt_loads_with_nonempty_body(prompt_id: str) -> None:
    doc = load_prompt(prompt_id)

    assert isinstance(doc, PromptDoc)
    assert isinstance(doc.body, str) and doc.body.strip(), f"{prompt_id} loaded an empty body"
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
    text = (prompts.PROMPT_DIR / PROMPT_FILES[prompt_id]).read_text(encoding="utf-8")
    doc = load_prompt(prompt_id)

    assert doc.body in text, f"{prompt_id}.body is not a verbatim slice of its file"
    assert not doc.body.startswith("|"), f"{prompt_id}.body starts inside the header table"


def test_every_documented_id_is_covered_by_the_loader() -> None:
    assert set(PROMPT_FILES) == set(PROMPT_IDS)


def test_every_prompt_file_exists_on_disk() -> None:
    missing = [
        PROMPT_FILES[prompt_id]
        for prompt_id in PROMPT_FILES
        if not (prompts.PROMPT_DIR / PROMPT_FILES[prompt_id]).is_file()
    ]
    assert missing == []


def test_no_prompt_file_is_missing_from_the_loader() -> None:
    on_disk = {path.name for path in prompts.PROMPT_DIR.glob("*.md")}
    unloaded = on_disk - set(PROMPT_FILES.values()) - {"README.md"}
    assert unloaded == set(), f"prompt files no id loads: {sorted(unloaded)}"


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
    doc = load_prompt(prompt_id)

    assert doc.changelog.startswith(
        doc.version
    ), f"{prompt_id} changelog does not begin with v{doc.version}"


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_meta_is_read_only(prompt_id: str) -> None:
    doc = load_prompt(prompt_id)

    assert isinstance(doc.meta, MappingProxyType)
    meta: Any = doc.meta
    with pytest.raises(TypeError):
        meta["version"] = "9.9.9"


def test_doc_is_frozen() -> None:
    doc = load_prompt("P3")

    probe: Any = doc
    with pytest.raises(AttributeError):
        probe.version = "9.9.9"


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_missing_header_field_fails_the_load(
    prompt_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = _header_table(prompt_id)
    lines[5] = "| Model target |  |"
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
    directory = tmp_path / "docs" / "prompts"
    directory.mkdir(parents=True)
    monkeypatch.setattr(prompts, "PROMPT_DIR", directory)

    with pytest.raises(PromptError) as excinfo:
        load_prompt(prompt_id)
    assert PROMPT_FILES[prompt_id] in str(excinfo.value)


def test_header_prompt_id_must_match_the_requested_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_required_version_is_the_recorded_floor(prompt_id: str) -> None:
    payload = json.loads(PROMPT_META_PATH.read_text(encoding="utf-8"))

    assert required_version(prompt_id) == payload[prompt_id]["min_version"]


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_assert_min_version_passes_on_the_checked_out_tree(prompt_id: str) -> None:
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
    directory = _redirect_prompt_dir(tmp_path, monkeypatch, "## Purpose\n\nFixture body.\n")
    _write_synthetic_prompts(directory, target="P3", version=version, body="## Purpose\n\nBody.\n")
    _redirect_prompt_meta(tmp_path, monkeypatch, "P3", floor)

    doc = assert_min_version("P3")

    assert doc.version == version
    assert parse_version(doc.version) >= parse_version(floor)


def test_an_older_prompt_fails_its_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = _redirect_prompt_dir(tmp_path, monkeypatch, "## Purpose\n\nFixture body.\n")
    _write_synthetic_prompts(directory, target="P3", version="1.0.0", body="## Purpose\n\nBody.\n")
    _redirect_prompt_meta(tmp_path, monkeypatch, "P3", "1.10.1")

    with pytest.raises(PromptVersionError) as excinfo:
        assert_min_version("P3")
    assert "1.0.0" in str(excinfo.value) and "1.10.1" in str(excinfo.value)


def test_a_prompt_at_exactly_the_floor_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _redirect_prompt_dir(tmp_path, monkeypatch, "## Purpose\n\nFixture body.\n")
    _write_synthetic_prompts(directory, target="P3", version="1.10.0", body="## Purpose\n\nBody.\n")
    _redirect_prompt_meta(tmp_path, monkeypatch, "P3", "1.10.0")

    assert assert_min_version("P3").version == "1.10.0"


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
    for bad_id in (UNKNOWN_PROMPT_ID, UNKNOWN_PLAUSIBLE_ID):
        with pytest.raises(UnknownPromptError):
            load_prompt(bad_id)


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_node_matches_the_recorded_meta_node(prompt_id: str) -> None:
    doc = load_prompt(prompt_id)
    recorded = prompt_meta()[prompt_id]["node"]

    if recorded is None:
        assert prompt_id == "P4"
        assert doc.node and doc.node.strip()
        return
    assert isinstance(recorded, str)
    assert recorded in doc.node


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_every_node_a_prompt_names_is_in_the_node_index(prompt_id: str) -> None:
    doc = load_prompt(prompt_id)
    index = set(json.loads(PROMPT_META_PATH.read_text(encoding="utf-8"))["nodes"])

    named = set(_NODE_NAME.findall(doc.node)) | {prompt_meta()[prompt_id]["node"]}
    assert named <= index | {None}


def test_the_node_index_points_back_at_the_prompt_library() -> None:
    payload = json.loads(PROMPT_META_PATH.read_text(encoding="utf-8"))

    for node_name, entry in payload["nodes"].items():
        prompt_id = entry["prompt_id"]
        assert prompt_id in PROMPT_FILES, f"node {node_name!r} points at an unknown prompt"
        assert node_meta(node_name)["prompt_id"] == prompt_id
        assert node_meta(node_name)["min_version"] == required_version(prompt_id)
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
    readme = (prompts.PROMPT_DIR / "README.md").read_text(encoding="utf-8")
    rows: list[list[str]] = []
    for line in readme.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"P\d", cells[0]) and cells[1].endswith(".md"):
            rows.append(cells)
    return rows


def test_the_documented_inventory_matches_the_loader() -> None:
    inventory = {row[0]: row[1] for row in _readme_inventory_rows()}

    assert inventory == dict(PROMPT_FILES)


def test_every_prompt_file_in_the_inventory_exists() -> None:
    for row in _readme_inventory_rows():
        assert (
            prompts.PROMPT_DIR / row[1]
        ).is_file(), f"inventory names a file that is not checked in: {row[1]}"


def test_prompt_dir_is_the_repo_docs_dir_not_the_cwd() -> None:
    module_path = Path(prompts.__file__).resolve()
    repo_root = module_path.parents[2]

    assert repo_root == prompts.REPO_ROOT, "repo root is three parents above src/utils"
    assert repo_root / "docs" / "prompts" == prompts.PROMPT_DIR
    assert prompts.PROMPT_DIR.is_absolute(), "the loader must not resolve docs/prompts from cwd"


def test_prompt_dir_is_not_relative() -> None:
    assert prompts.PROMPT_DIR.is_absolute()
    assert prompts.PROMPT_DIR.resolve() == prompts.PROMPT_DIR


def test_no_prompt_text_is_duplicated_in_the_loader_module() -> None:
    source = Path(prompts.__file__).read_text(encoding="utf-8")

    for prompt_id, fragment in _prompt_fragments():
        assert fragment not in source, (
            f"{prompt_id}: prompt text duplicated inside src/utils/prompts.py "
            f"({len(fragment)} chars: {fragment[:60]!r})"
        )


def test_no_prompt_text_is_duplicated_in_src() -> None:
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
    fragments = _prompt_fragments()

    assert all(len(fragment) >= FRAGMENT_MIN_CHARS for _, fragment in fragments)
    assert max(len(fragment) for _, fragment in fragments) > 500


def test_meta_file_is_read_from_the_repo_scripts_dir() -> None:
    assert prompts.PROMPT_META_PATH == prompts.REPO_ROOT / "scripts" / "prompts_meta.json"
    assert prompts.PROMPT_META_PATH.is_file()


def test_no_prompt_dir_at_the_repo_root() -> None:
    assert not Path(prompts.REPO_ROOT / "prompts").exists()


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
    directory = tmp_path / "docs" / "prompts"
    directory.mkdir(parents=True)
    body = "## Rules\n\n| Not a header row | another cell | third cell |\n\nStray prose.\n"
    (directory / PROMPT_FILES["P2"]).write_text(
        "\n".join(_header_table("P2")) + "\n" + body, encoding="utf-8"
    )
    monkeypatch.setattr(prompts, "PROMPT_DIR", directory)

    doc = load_prompt("P2")

    assert doc.body == body.rstrip("\n")
    assert set(doc.meta) == set(REQUIRED_FIELDS)


def test_parse_header_ignores_a_table_before_the_version_header() -> None:
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


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_meta_mapping_is_keyed_by_the_documented_fields(prompt_id: str) -> None:
    meta = prompt_meta()[prompt_id]

    for key in ("node", "min_version", "structured", "output_schema"):
        assert key in meta, f"{prompt_id} meta entry lacks {key!r}"


def test_prompt_meta_covers_every_documented_id() -> None:
    assert set(prompt_meta()) == set(PROMPT_IDS)


def test_meta_structured_flags_match_the_output_schema_kind() -> None:
    for prompt_id in PROMPT_FILES:
        entry = prompt_meta()[prompt_id]
        schema = load_prompt(prompt_id).output_schema
        if entry["structured"]:
            assert "JSON" in schema or "Result" in schema or "Plan" in schema, schema
        else:
            assert "JSON" not in schema, schema


def test_p4_prescribes_the_table_marker_placement_the_pre_pass_can_read() -> None:

    rules = _fenced_rules("P4")

    assert "row label cell" in rules, "rule 4's placement must stay prescribed"
    assert "per-cell" in rules, "rule 4's per-block placement must stay prescribed"
    assert "Put the [n] marker in the table itself" in rules


def test_p4_no_longer_prescribes_a_source_note_as_a_citation() -> None:

    rules = _fenced_rules("P4")

    assert "source note under the table lists the [n] markers" not in rules
    assert "A source note under the table" not in rules


def _fenced_rules(prompt_id: str) -> str:

    doc = load_prompt(prompt_id)
    blocks = re.findall(r"^```[^\n]*\n(.*?)^```", doc.body, re.S | re.M)
    assert blocks, f"{prompt_id} carries no fenced rules block"
    return "\n".join(blocks)

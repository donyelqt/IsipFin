"""Prompt assets: load ``P1``-``P5`` from ``docs/prompts/*.md`` with their version headers.

Why ``docs/prompts/`` is the single source of truth
--------------------------------------------------
``docs/REPO-STRUCTURE.md`` reserves a top-level ``prompts/`` directory for the
prompt assets; the actual files live in ``docs/prompts/`` instead, and that is
deliberate (``docs/prompts/README.md`` protocol, ``docs/CAPABILITY-MAP.md``:
"prompt assets land with their owning module"):

- A prompt change is a **diffable file change**, never a code edit — the repo
  forbids inline system prompts in ``src/`` outright (``docs/CONSTRAINTS.md``
  floor guard, ``docs/REPO-STRUCTURE.md`` rule "no inline system prompts").
- The version header (id, version, node, model target, output schema,
  changelog) travels with the body, so ``load_prompt("P3").version`` is the
  version the graph will actually run — a stale deployment fails loudly instead
  of silently answering with an old prompt.
- Copying the bodies into ``src/`` (or into ``prompts/``) would fork the spec:
  two files, one of them inevitably stale, and no reviewer could tell which one
  a node used. This module therefore *reads* the docs files; it never writes
  them and contains no prompt text of its own.

Contract
--------
``load_prompt(id)`` parses the YAML-ish version header table into a frozen
:class:`PromptDoc` exposing ``.id``, ``.version``, ``.node`` and ``.body`` (the
full markdown body **below** the header — the system prompt, input template and
schema blocks included), plus ``.meta`` with every header field
(``prompt_id``, ``version``, ``node``, ``model_target``, ``output_schema``,
``changelog``).

``scripts/prompts_meta.json`` records, per prompt id, the minimum version the
T9-T15 nodes require and which graph node (and output shape) the prompt feeds
— the same four node names ``src/utils/mock_llm.py`` keys its scripted
responses on, so the mock contract is data, not a hardcoded table. A node that
depends on a specific prompt version asserts it with
:func:`assert_min_version` at import time: a deployment whose prompts are
older than the node's contract raises instead of running a mismatched prompt.

No caching, by design: the RUNBOOK's answer to "prompt changed but behavior
stale" is "restart the app; prompts load at startup via ``load_prompt``", and a
module-level cache would turn every edit during the build into a ghost.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

# ``docs/prompts/`` resolved from THIS module, not the cwd: nodes are imported by
# the app, by scripts, and by pytest, and a cwd-relative path would break two of
# the three. ``src/utils/prompts.py`` -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "docs" / "prompts"

# ``scripts/prompts_meta.json``: the per-node minimum-version + mock-mode contract.
PROMPT_META_PATH = REPO_ROOT / "scripts" / "prompts_meta.json"

# Prompt id -> file inside ``PROMPT_DIR``. The ids are the public API; the file
# names are the library's, so a file rename is a one-line change here, not a
# change at every callsite (docs/prompts/README.md inventory).
PROMPT_FILES: Mapping[str, str] = {
    "P1": "query-analysis.md",
    "P2": "follow-up-contextualization.md",
    "P3": "answer-synthesis-with-citations.md",
    "P4": "table-and-number-formatting.md",
    "P5": "summarization-and-verification.md",
}

# The header fields every prompt file must carry, in file order. ``version``
# doubles as the node-contract key; a missing or empty field fails the load
# rather than degrading to a default (docs/prompts/README.md rule 2).
REQUIRED_FIELDS: tuple[str, ...] = (
    "prompt_id",
    "version",
    "node",
    "model_target",
    "output_schema",
    "changelog",
)

# Header field label -> meta key. The table's left column is human prose
# ("Prompt id", "Model target"); the keys node code uses are snake_case.
_FIELD_KEYS: Mapping[str, str] = {
    "prompt id": "prompt_id",
    "version": "version",
    "node": "node",
    "model target": "model_target",
    "output schema": "output_schema",
    "changelog": "changelog",
}


class PromptError(ValueError):
    """Base class for every prompt-loading failure (unknown id, bad header, version)."""


class UnknownPromptError(PromptError):
    """Raised when a caller asks for a prompt id that is not in the library."""


class PromptParseError(PromptError):
    """Raised when a prompt file's version header is missing, malformed, or incomplete."""


class PromptVersionError(PromptError):
    """Raised when a loaded prompt is older than the version a node requires."""


@dataclass(frozen=True)
class PromptDoc:
    """One prompt asset: the parsed version header plus the full markdown body.

    ``body`` starts *after* the header table (skipping the blank spacer line), so
    it is the complete spec below the header — Purpose, System prompt, Input
    template, Output schema, Failure handling, Iteration notes. Nodes slice the
    block they need out of the body; this module does not interpret it, because
    every slicing rule is different (a fenced system prompt, an input template
    with ``{placeholders}``) and hardcoding one shape here would fork the spec.
    """

    id: str
    version: str
    node: str
    body: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        # A dict handed to a frozen dataclass stays mutable; wrap it in a
        # read-only mapping so a caller cannot edit "loaded" prompt metadata
        # after the fact (a mutated version field would defeat every
        # minimum-version assertion downstream).
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @property
    def model_target(self) -> str:
        """The header's ``Model target`` cell (e.g. ``any (temperature 0.2)``)."""
        return self.meta["model_target"]

    @property
    def output_schema(self) -> str:
        """The header's ``Output schema`` cell (e.g. ``QueryPlan JSON``)."""
        return self.meta["output_schema"]

    @property
    def changelog(self) -> str:
        """The header's ``Changelog`` cell — what this version changed and why."""
        return self.meta["changelog"]


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a ``major.minor.patch`` header version into a comparable tuple.

    Comparison must be numeric, never lexicographic: ``"1.10.0" > "1.9.0"`` as
    versions but ``"1.10.0" < "1.9.0"`` as strings, so a minimum-version check
    on a minor-version bump would silently pass the wrong way.
    """
    parts = version.strip().split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise PromptParseError(
            f"malformed prompt version {version!r}: expected 'major.minor.patch'"
        )
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _split_row(line: str) -> list[str]:
    """Split one markdown table row into stripped cells (``| a | b |`` -> ``[a, b]``)."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    """True for the ``|---|---|`` row that separates a table header from its body."""
    return all(cell and set(cell) <= {"-", ":"} for cell in cells)


def parse_header(text: str) -> tuple[dict[str, str], int]:
    """Parse the leading version-header table into ``(fields, body_start_line)``.

    The header is the first markdown table in the file (``docs/prompts/README.md``
    rule 2: "semver in the header"). Parsing stops at the first line that is not
    a table row, so a stray ``|`` in the body can never bleed into the header.
    """
    fields: dict[str, str] = {}
    body_start: int | None = None
    in_table = False

    for lineno, raw in enumerate(text.splitlines()):
        stripped = raw.strip()
        if not stripped.startswith("|"):
            if in_table:
                body_start = lineno
                break
            continue
        cells = _split_row(stripped)
        if _is_separator(cells):
            continue
        if len(cells) != 2:
            raise PromptParseError(
                f"header table row {lineno + 1} has {len(cells)} cells, expected 2"
            )
        label, value = cells[0], cells[1]
        if label.lower() == "field" and value.lower() == "value":
            in_table = True
            continue
        if not in_table:
            # A table before the version header is not the version header.
            continue
        key = _FIELD_KEYS.get(label.lower())
        if key is None:
            raise PromptParseError(f"unknown header field {label!r} at line {lineno + 1}")
        if key in fields:
            raise PromptParseError(f"duplicate header field {label!r} at line {lineno + 1}")
        fields[key] = value.replace("`", "").strip()

    if body_start is None:
        raise PromptParseError(
            "no version header table found: expected a '| Field | Value |' table"
        )
    return fields, body_start


def _body_from(text: str, body_start: int) -> str:
    """The file's lines from ``body_start`` on, with leading blank lines dropped."""
    body = "\n".join(text.splitlines()[body_start:])
    return body.lstrip("\n")


def load_prompt(prompt_id: str) -> PromptDoc:
    """Load the prompt asset ``prompt_id`` (``"P1"``-``"P5"``) from ``docs/prompts/``.

    Reads the file fresh on every call — a prompt edit during a session is picked
    up by the next load, never by a stale cache. Raises
    :class:`UnknownPromptError` for an id that is not in :data:`PROMPT_FILES` and
    :class:`PromptParseError` for a file whose header is missing, unknown, or
    incomplete (every :data:`REQUIRED_FIELDS` cell must be non-empty).
    """
    if prompt_id not in PROMPT_FILES:
        raise UnknownPromptError(
            f"unknown prompt id {prompt_id!r}: expected one of {', '.join(PROMPT_FILES)}"
        )
    path = PROMPT_DIR / PROMPT_FILES[prompt_id]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # missing file, permissions, encoding — all a load failure
        raise PromptParseError(f"cannot read prompt {prompt_id} from {path}: {exc}") from exc

    fields, body_start = parse_header(text)
    missing = [field for field in REQUIRED_FIELDS if not fields.get(field, "").strip()]
    if missing:
        raise PromptParseError(
            f"prompt {prompt_id} ({path}) has empty header field(s): {', '.join(missing)}"
        )
    if fields["prompt_id"] != prompt_id:
        raise PromptParseError(
            f"prompt {prompt_id} is file {path.name}, whose header says "
            f"'Prompt id | {fields['prompt_id']}'"
        )
    parse_version(fields["version"])

    return PromptDoc(
        id=prompt_id,
        version=fields["version"],
        node=fields["node"],
        body=_body_from(text, body_start),
        meta={field: fields[field] for field in REQUIRED_FIELDS},
    )


def _read_meta_file() -> Mapping[str, object]:
    """The whole ``scripts/prompts_meta.json`` object, validated as keyed by prompt id."""
    try:
        raw = PROMPT_META_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"cannot read prompt metadata from {PROMPT_META_PATH}: {exc}") from exc
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise PromptError(f"{PROMPT_META_PATH} must contain a JSON object keyed by prompt id")
    return payload


def prompt_meta() -> Mapping[str, Mapping[str, object]]:
    """The prompt entries of ``scripts/prompts_meta.json`` — keyed by prompt id.

    This is the data-driven half of the T8 contract: per prompt, the minimum
    version its consumers require and the node + output shape its prompt feeds.
    The ``nodes`` inverse index and ``$comment`` describe the file, not prompts,
    so they are excluded here; use :func:`node_meta` for the node side.
    ``src/utils/mock_llm.py`` keys scripted responses on the node name and
    schema presence recorded here instead of a hardcoded table, so a node rename
    is a JSON edit, not a mock rewrite.
    """
    payload = _read_meta_file()
    missing = [prompt_id for prompt_id in PROMPT_FILES if prompt_id not in payload]
    if missing:
        raise PromptError(f"{PROMPT_META_PATH} has no entry for prompt(s): {', '.join(missing)}")
    entries: dict[str, dict[str, object]] = {}
    for prompt_id in PROMPT_FILES:
        entry = payload[prompt_id]
        if not isinstance(entry, dict):
            raise PromptError(f"{PROMPT_META_PATH} entry {prompt_id!r} is not an object")
        entries[prompt_id] = dict(entry)
    return entries


def node_meta(node_name: str) -> Mapping[str, object]:
    """The contract recorded for one graph node (the file's inverse index).

    Gives a node its prompt id, minimum version, structured-output flag and
    failure policy in one lookup, so ``src/utils/mock_llm.py`` and the nodes
    themselves read the same numbers from the same place.
    """
    nodes: object = _read_meta_file().get("nodes")
    if not isinstance(nodes, dict) or node_name not in nodes:
        raise UnknownPromptError(f"no node entry for {node_name!r} in {PROMPT_META_PATH}")
    entry: Mapping[str, object] = nodes[node_name] if isinstance(nodes[node_name], dict) else {}
    return entry


def required_version(prompt_id: str) -> str:
    """The minimum prompt version the graph nodes require for ``prompt_id``.

    A node asserts its floor with :func:`assert_min_version`; this is the lookup
    that keeps the floor in one reviewable file (``scripts/prompts_meta.json``)
    rather than in five node modules.
    """
    entry = prompt_meta().get(prompt_id)
    if entry is None:
        raise UnknownPromptError(
            f"no minimum-version entry for prompt {prompt_id!r} in {PROMPT_META_PATH}"
        )
    version = entry.get("min_version")
    if not isinstance(version, str) or not version:
        raise PromptError(f"prompt {prompt_id} has no 'min_version' in {PROMPT_META_PATH}")
    return version


def assert_min_version(prompt_id: str) -> PromptDoc:
    """Load ``prompt_id`` and verify it is at least the required minimum version.

    Returns the loaded :class:`PromptDoc` so callers write one line:
    ``prompt = assert_min_version("P3")``. Raises :class:`PromptVersionError`
    when the checked-out prompt is older than the floor — the "stale deployment
    fails loudly" rule from ``docs/prompts/README.md`` rule 5, in code.
    """
    doc = load_prompt(prompt_id)
    floor = required_version(prompt_id)
    if parse_version(doc.version) < parse_version(floor):
        raise PromptVersionError(
            f"prompt {prompt_id} is v{doc.version} but the graph requires >= v{floor} "
            f"({PROMPT_META_PATH})"
        )
    return doc

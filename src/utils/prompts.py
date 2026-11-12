from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "docs" / "prompts"

PROMPT_META_PATH = REPO_ROOT / "scripts" / "prompts_meta.json"

PROMPT_FILES: Mapping[str, str] = {
    "P1": "query-analysis.md",
    "P2": "follow-up-contextualization.md",
    "P3": "answer-synthesis-with-citations.md",
    "P4": "table-and-number-formatting.md",
    "P5": "summarization-and-verification.md",
}

REQUIRED_FIELDS: tuple[str, ...] = (
    "prompt_id",
    "version",
    "node",
    "model_target",
    "output_schema",
    "changelog",
)

_FIELD_KEYS: Mapping[str, str] = {
    "prompt id": "prompt_id",
    "version": "version",
    "node": "node",
    "model target": "model_target",
    "output schema": "output_schema",
    "changelog": "changelog",
}


class PromptError(ValueError):
    pass


class UnknownPromptError(PromptError):
    pass


class PromptParseError(PromptError):
    pass


class PromptVersionError(PromptError):
    pass


@dataclass(frozen=True)
class PromptDoc:

    id: str
    version: str
    node: str
    body: str
    meta: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @property
    def model_target(self) -> str:
        return self.meta["model_target"]

    @property
    def output_schema(self) -> str:
        return self.meta["output_schema"]

    @property
    def changelog(self) -> str:
        return self.meta["changelog"]


def parse_version(version: str) -> tuple[int, int, int]:
    parts = version.strip().split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise PromptParseError(
            f"malformed prompt version {version!r}: expected 'major.minor.patch'"
        )
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return all(cell and set(cell) <= {"-", ":"} for cell in cells)


def parse_header(text: str) -> tuple[dict[str, str], int]:
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
    body = "\n".join(text.splitlines()[body_start:])
    return body.lstrip("\n")


def load_prompt(prompt_id: str) -> PromptDoc:
    if prompt_id not in PROMPT_FILES:
        raise UnknownPromptError(
            f"unknown prompt id {prompt_id!r}: expected one of {', '.join(PROMPT_FILES)}"
        )
    path = PROMPT_DIR / PROMPT_FILES[prompt_id]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
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
    try:
        raw = PROMPT_META_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"cannot read prompt metadata from {PROMPT_META_PATH}: {exc}") from exc
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise PromptError(f"{PROMPT_META_PATH} must contain a JSON object keyed by prompt id")
    return payload


def prompt_meta() -> Mapping[str, Mapping[str, object]]:
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
    nodes: object = _read_meta_file().get("nodes")
    if not isinstance(nodes, dict) or node_name not in nodes:
        raise UnknownPromptError(f"no node entry for {node_name!r} in {PROMPT_META_PATH}")
    entry: Mapping[str, object] = nodes[node_name] if isinstance(nodes[node_name], dict) else {}
    return entry


def required_version(prompt_id: str) -> str:
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
    doc = load_prompt(prompt_id)
    floor = required_version(prompt_id)
    if parse_version(doc.version) < parse_version(floor):
        raise PromptVersionError(
            f"prompt {prompt_id} is v{doc.version} but the graph requires >= v{floor} "
            f"({PROMPT_META_PATH})"
        )
    return doc

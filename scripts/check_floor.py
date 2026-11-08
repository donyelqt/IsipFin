#!/usr/bin/env python3
"""Floor guard. Enforces the floor in docs/CONSTRAINTS.md.

Modes:
  --diff        scan only lines added relative to the branch point (task end)
  --repo-wide   scan every tracked source file (CI)

Exit codes:
  0  clean
  1  violation found
  2  usage / git error

Scope by design: src/, tests/, scripts/, app.py. Markdown (docs/, prompts/,
README.md) is never scanned — those may contain the deliberate "TODO during
build" markers defined in docs/prompts/README.md.

This file is excluded from its own scan: the RULES table below must contain the
very literals it forbids, or it would flag itself on every run.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
SCAN_PATHS = ("src", "tests", "scripts", "app.py")
SCAN_SUFFIXES = {".py"}

# Pattern -> human-readable rule. Ordered: suppressions and stubs first.
RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"#\s*noqa"), "lint suppression"),
    (re.compile(r"#\s*type:\s*ignore"), "type suppression"),
    (re.compile(r"raise\s+NotImplementedError"), "unimplemented stub"),
    (re.compile(r"\bTODO\b"), "unfinished work"),
    (re.compile(r"\bFIXME\b"), "known-broken marker"),
    (re.compile(r"Not implemented"), "unimplemented stub"),
)


def git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def repo_files() -> list[Path]:
    """Every tracked .py file under the scan roots (present or not-yet-created)."""
    files: list[Path] = []
    for root in SCAN_PATHS:
        p = Path(root)
        if p.is_file():
            files.append(p)
            continue
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            if f.resolve() == SELF:
                continue
            files.append(f)
    return files


def diff_files() -> list[Path]:
    """Files changed vs the merge base with the default branch, plus uncommitted."""
    try:
        base = git(["merge-base", "HEAD", "origin/main"]).strip()
    except subprocess.CalledProcessError:
        base = "HEAD~1"
    names = git(["diff", "--name-only", base, "HEAD"]).split()
    plus = git(["diff", "--name-only"]).split()  # uncommitted, task-end usage
    out = {Path(n) for n in names + plus}
    return [p for p in sorted(out) if p.suffix in SCAN_SUFFIXES and p.resolve() != SELF]


def scan(files: list[Path], only_added: bool) -> list[tuple[Path, int, str, str]]:
    hits: list[tuple[Path, int, str, str]] = []
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        if only_added:
            head = subprocess.run(
                ["git", "show", f"HEAD:{f.as_posix()}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            old = set(head.stdout.splitlines()) if head.returncode == 0 else set()
            candidates = [(idx, text) for idx, text in enumerate(lines, 1) if text not in old]
        else:
            candidates = list(enumerate(lines, 1))
        for lineno, line in candidates:
            stripped = line.strip()
            if stripped.startswith("#") and not any(
                p.search(line) for p, _ in RULES if "suppression" in _
            ):
                # Allow documentation of the rules themselves in comments.
                continue
            for pattern, rule in RULES:
                if pattern.search(line):
                    hits.append((f, lineno, rule, stripped[:100]))
                    break
    return hits


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"--diff", "--repo-wide"}:
        print(__doc__)
        return 2
    only_added = argv[1] == "--diff"
    files = diff_files() if only_added else repo_files()
    if not files:
        print(f"floor guard: nothing to scan ({argv[1]})")
        return 0

    hits = scan(files, only_added)
    mode = "diff" if only_added else "repo-wide"
    if not hits:
        print(f"floor guard: clean ({mode}, {len(files)} file(s))")
        return 0

    print(f"floor guard: {len(hits)} violation(s) in {mode} mode\n", file=sys.stderr)
    for path, lineno, rule, text in hits:
        print(f"  {path.as_posix()}:{lineno}: {rule}: {text}", file=sys.stderr)
    print(
        "\nFix the violation, or record an exception in docs/CONSTRAINTS.md "
        "with an owner and an expiry. Weakening this guard is a reviewed decision.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

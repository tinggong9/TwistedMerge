"""Helpers for recording and validating per-artifact execution provenance."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def current_commit(root: Path) -> str:
    return git_output(root, "rev-parse", "HEAD")


def worktree_is_dirty(root: Path) -> bool:
    return bool(git_output(root, "status", "--porcelain"))


def load_execution_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"execution record must be a JSON object: {path}")
    return payload


def execution_commit(path: Path) -> str:
    """Return the commit recorded by the artifact's own execution config."""

    payload = load_execution_record(path)
    value = payload.get("execution_commit", payload.get("git_commit"))
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise ValueError(f"missing or invalid per-artifact execution commit in {path}")
    return value


def execution_command(path: Path) -> str:
    payload = load_execution_record(path)
    value = payload.get("exact_command", payload.get("command"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing exact command in {path}")
    return value


def make_execution_record(
    *,
    root: Path,
    command: str,
    outputs: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "execution_commit": current_commit(root),
        "dirty_worktree_at_execution": worktree_is_dirty(root),
        "exact_command": command,
        "outputs": outputs,
    }
    if extra:
        record.update(extra)
    return record

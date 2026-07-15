#!/usr/bin/env python3
"""Shared utilities for the public future benchmark program."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "future_program"
LOCAL = ROOT / "reports" / "tmp" / "future_program"
DATA = Path(os.environ.get("TWISTEDMERGE_DATA_ROOT", ROOT / "data"))

TIERS = {
    "emergency": OUT / "emergency",
    "near-term": OUT / "near_term",
    "extended": OUT / "extended",
}


def ensure_dirs() -> None:
    for path in [OUT, LOCAL, LOCAL / "logs", LOCAL / "logits", *TIERS.values()]:
        path.mkdir(parents=True, exist_ok=True)
    for path in TIERS.values():
        (path / "tables").mkdir(exist_ok=True)
        (path / "plots").mkdir(exist_ok=True)


def git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    return completed.stdout.strip()


def git_head() -> str:
    return git("rev-parse", "HEAD")


def safe_path(path: str | Path) -> str:
    value = str(path)
    return value.replace(str(ROOT), "<repository-root>").replace(str(Path.home()), "<home>")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stage_result(stage_id: str, state: str, summary: str, **extra: object) -> None:
    if state not in {"completed", "confirmation", "clean-freeze", "blocked", "negative", "failed"}:
        raise ValueError(state)
    payload = {
        "stage_id": stage_id,
        "state": state,
        "summary": summary,
        "execution_commit": git_head(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }
    write_json(LOCAL / "stage_results" / f"{stage_id}.json", payload)


def label_independence_record(name: str, arrays: Mapping[str, np.ndarray], labels: np.ndarray, seed: int) -> dict[str, object]:
    ensure_dirs()
    path = LOCAL / "logits" / f"{name}.npz"
    np.savez_compressed(path, **{key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()})
    before = sha256_file(path)
    permuted = np.asarray(labels).copy()
    np.random.default_rng(seed).shuffle(permuted)
    after = sha256_file(path)
    return {
        "logits_file": safe_path(path),
        "logits_sha256": before,
        "label_permutation_hash_passed": before == after,
    }


def peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024)


def bootstrap(values: Iterable[float], seed: int = 2026, samples: int = 4000) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def environment_manifest() -> dict[str, object]:
    packages = {}
    for name in ["numpy", "pandas", "matplotlib", "scipy", "sklearn", "torch", "torchvision", "transformers", "datasets", "peft", "accelerate", "timm"]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "installed")
        except Exception as error:
            packages[name] = f"unavailable: {type(error).__name__}"
    try:
        import torch

        mps = bool(torch.backends.mps.is_available())
    except Exception:
        mps = False
    disk = os.statvfs(ROOT)
    return {
        "branch": git("branch", "--show-current"),
        "head": git_head(),
        "worktree_clean": not bool(git("status", "--porcelain")),
        "tags": git("tag", "--points-at", "HEAD", check=False).splitlines(),
        "python": sys.version.split()[0],
        "python_executable": safe_path(sys.executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "physical_memory_bytes": int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")),
        "free_disk_bytes": int(disk.f_frsize * disk.f_bavail),
        "mps_available": mps,
        "packages": packages,
    }


def patch_compact_paths(module: object, out: Path) -> None:
    """Redirect a compact-stage module to the future-program artifact tree."""
    import experiments.compact_benchmark_common as compact

    compact.OUT = out
    compact.LOCAL = LOCAL
    compact.DATA = DATA
    compact.CHECKPOINTS = LOCAL / "checkpoints"
    for name, value in {"OUT": out, "LOCAL": LOCAL, "DATA": DATA, "CHECKPOINTS": LOCAL / "checkpoints"}.items():
        if hasattr(module, name):
            setattr(module, name, value)
    ensure_dirs()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

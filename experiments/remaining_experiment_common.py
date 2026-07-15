#!/usr/bin/env python3
"""Shared utilities for the remaining-experiments program."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "remaining_experiments"
DATA = Path(os.environ.get("TWISTEDMERGE_DATA_ROOT", ROOT / "data")).expanduser().resolve()
TMP = Path(os.environ.get("TWISTEDMERGE_TMP_ROOT", ROOT / "reports" / "tmp" / "remaining_experiments")).expanduser().resolve()


def ensure_dirs() -> None:
    for path in [OUT, OUT / "tables", OUT / "plots", TMP, TMP / "logits"]:
        path.mkdir(parents=True, exist_ok=True)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(path: Path) -> str:
    return sha256_file(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def matched_bootstrap(values: Iterable[float], seed: int, samples: int = 4000) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def ridge_fit(features: np.ndarray, targets: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    augmented = np.column_stack([features, np.ones(len(features))])
    gram = augmented.T @ augmented
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[-1, -1] = 0.0
    return np.linalg.solve(gram + penalty, augmented.T @ targets)


def ridge_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([np.asarray(features, dtype=np.float64), np.ones(len(features))])
    return augmented @ weights


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = softmax(np.asarray(logits, dtype=float))
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities.argmax(axis=1)
    loss = -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean()
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {"accuracy": float(correct.mean()), "loss": float(loss), "ece": float(ece)}


def logits_hashes(name: str, logits: Mapping[str, np.ndarray], labels: np.ndarray, seed: int) -> dict[str, object]:
    before = {
        method: hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()
        for method, values in logits.items()
    }
    shuffled = np.asarray(labels).copy()
    np.random.default_rng(seed).shuffle(shuffled)
    after = {
        method: hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()
        for method, values in logits.items()
    }
    return {
        "name": name,
        "logits_sha256": hashlib.sha256("".join(before.values()).encode()).hexdigest(),
        "label_permutation_hash_passed": before == after,
        "permuted_labels_differ": not np.array_equal(labels, shuffled),
    }


def timed_prediction(fn, repeats: int = 7) -> tuple[object, float]:
    result = fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, float(np.median(times) * 1000.0)


def provenance(script: Path, command: str, seed: int, **extra: object) -> dict[str, object]:
    return {
        "execution_commit": git_head(),
        "command": command,
        "source_sha256": source_hash(script),
        "seed": seed,
        **extra,
    }


def latex_table(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]], caption: str) -> None:
    def escape(value: object) -> str:
        return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{escape(caption)}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(escape(column) for column in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(escape(row.get(column, "")) for column in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ensure_dirs()

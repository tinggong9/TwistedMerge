"""Metrics, summaries, and environment capture."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def binary_accuracy(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    pred = (x @ weight >= 0.0).astype(np.int64)
    return float(np.mean(pred == y))


def mean_task_accuracy(xs: list[np.ndarray], ys: list[np.ndarray], weights: np.ndarray) -> float:
    return float(np.mean([binary_accuracy(x, y, w) for x, y, w in zip(xs, ys, weights)]))


def classification_margin(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    signs = 2 * y - 1
    return float(np.mean(signs * (x @ weight)))


def pearsonr(x: Iterable[float], y: Iterable[float]) -> float:
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    if xv.size < 2 or np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def summarize_by_level(df: pd.DataFrame, level_col: str) -> pd.DataFrame:
    cols = [
        "obstruction_score",
        "naive_accuracy",
        "rank_lift_accuracy",
        "oracle_accuracy",
        "naive_failure",
        "rank_lift_gain",
    ]
    return (
        df.groupby(level_col, dropna=False)[cols]
        .agg(["mean", "std"])
        .reset_index()
    )


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in out.columns
    ]
    return out


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        return f"unavailable: {exc}"


def capture_environment() -> dict:
    packages = {}
    for name in ["numpy", "pandas", "matplotlib", "torch", "torchvision"]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = "not installed"
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "packages": packages,
        "git_commit": command_output(["git", "rev-parse", "--short", "HEAD"]),
    }

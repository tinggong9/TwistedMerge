#!/usr/bin/env python3
"""Shared, leakage-safe utilities for the compact benchmark program."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "compact_program"
LOCAL = ROOT / "reports" / "tmp" / "compact_program"
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints" / "compact_program"


def ensure_dirs() -> None:
    for path in (OUT, OUT / "tables", OUT / "plots", LOCAL, LOCAL / "logits", CHECKPOINTS):
        path.mkdir(parents=True, exist_ok=True)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_branch() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()


def command_string() -> str:
    import sys

    return " ".join([sys.executable, *sys.argv])


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_tex_table(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str], caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = " & ".join(column.replace("_", " ") for column in columns) + r" \\"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value).replace("_", r"\_"))
        body.append(" & ".join(values) + r" \\")
    text = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\begin{{tabular}}{{{'l' * len(columns)}}}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass


def peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 * 1024) if os.uname().sysname == "Darwin" else value / 1024)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = softmax(logits)
    chosen = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    confidence = probs.max(axis=1)
    correct = logits.argmax(axis=1) == labels
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {
        "accuracy": float(correct.mean()),
        "loss": float(-np.log(chosen).mean()),
        "ece": ece,
    }


def ridge_fit(features: np.ndarray, targets: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    design = np.column_stack([features, np.ones(len(features))])
    gram = design.T @ design
    gram.flat[:: len(gram) + 1] += ridge
    return np.linalg.solve(gram, design.T @ targets)


def ridge_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack([features, np.ones(len(features))]) @ weights


def random_feature_fit(
    features: np.ndarray,
    targets: np.ndarray,
    hidden: int,
    seed: int,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(scale=1 / math.sqrt(features.shape[1]), size=(features.shape[1], hidden))
    b1 = rng.normal(scale=0.05, size=hidden)
    hidden_features = np.tanh(features @ w1 + b1)
    w2 = ridge_fit(hidden_features, targets, ridge=ridge)
    return w1, b1, w2


def random_feature_predict(features: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    w1, b1, w2 = model
    return ridge_predict(np.tanh(features @ w1 + b1), w2)


def stratified_bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
    value_key: str,
    setting_key: str = "setting_id",
    samples: int = 2000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row[setting_key]), []).append(float(row[value_key]))
    setting_means = np.array([np.mean(values) for values in grouped.values()], dtype=float)
    if len(setting_means) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(setting_means) == 1:
        value = float(setting_means[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(setting_means, size=(samples, len(setting_means)), replace=True).mean(axis=1)
    return float(setting_means.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def save_logits_and_permutation_hash(name: str, arrays: Mapping[str, np.ndarray], labels: np.ndarray, seed: int) -> dict[str, object]:
    ensure_dirs()
    path = LOCAL / "logits" / f"{name}.npz"
    np.savez_compressed(path, **{key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    permuted = np.asarray(labels).copy()
    np.random.default_rng(seed).shuffle(permuted)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "logits_file": str(path.relative_to(ROOT)),
        "logits_sha256": before,
        "label_permutation_hash_passed": before == after,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


Permutation = tuple[int, ...]


@dataclass(frozen=True)
class FiniteGroup:
    name: str
    degree: int
    elements: tuple[Permutation, ...]
    identity: Permutation
    generators: dict[str, Permutation]
    regular: dict[Permutation, np.ndarray]


def compose(left: Permutation, right: Permutation) -> Permutation:
    return tuple(left[right[index]] for index in range(len(left)))


def finite_group(name: str) -> FiniteGroup:
    if name == "S3":
        identity, s, r = (0, 1, 2), (1, 0, 2), (1, 2, 0)
    elif name == "D4":
        identity, s, r = (0, 1, 2, 3), (0, 3, 2, 1), (1, 2, 3, 0)
    else:
        raise ValueError(name)
    generators = {"s": s, "r": r}
    discovered = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators.values():
            for candidate in (compose(generator, current), compose(current, generator)):
                if candidate not in discovered:
                    discovered.add(candidate)
                    frontier.append(candidate)
    elements = tuple(sorted(discovered))
    index = {element: idx for idx, element in enumerate(elements)}
    regular = {}
    for element in elements:
        matrix = np.zeros((len(elements), len(elements)), dtype=np.float64)
        for column, other in enumerate(elements):
            matrix[index[compose(element, other)], column] = 1.0
        regular[element] = matrix
    return FiniteGroup(name, len(identity), elements, identity, generators, regular)


def reduce_word(group: FiniteGroup, word: Sequence[str]) -> Permutation:
    result = group.identity
    for token in word:
        result = compose(group.generators[token], result)
    return result


def words_with_lengths(rng: np.random.Generator, n: int, lengths: Sequence[int]) -> list[tuple[str, ...]]:
    sampled = rng.choice(lengths, size=n)
    return [tuple(rng.choice(["s", "r"], size=int(length)).tolist()) for length in sampled]


def timed_predictions(function, repeats: int = 3) -> tuple[np.ndarray, float]:
    result = function()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)
    return result, float(np.median(timings))


def torch_device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_vision_dataset(name: str, train: bool):
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    mapping = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "CIFAR10": datasets.CIFAR10,
    }
    if name not in mapping:
        raise ValueError(name)
    return mapping[name](root=DATA, train=train, download=False, transform=transform)


def subset_arrays(dataset, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for index in indices:
        image, label = dataset[int(index)]
        xs.append(np.asarray(image, dtype=np.float32))
        ys.append(int(label))
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def model_parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_average(states: Sequence[Mapping[str, object]], weights: Sequence[float] | None = None):
    import torch

    if weights is None:
        weights = [1 / len(states)] * len(states)
    total = float(sum(weights))
    normalized = [float(weight) / total for weight in weights]
    merged = {}
    for key in states[0]:
        value = states[0][key]
        if not torch.is_floating_point(value):
            merged[key] = value.clone()
        else:
            merged[key] = sum(weight * state[key] for weight, state in zip(normalized, states, strict=True))
    return merged


def flatten_state(state: Mapping[str, object], keys: Iterable[str] | None = None) -> np.ndarray:
    import torch

    selected = set(keys) if keys is not None else None
    arrays = []
    for key, value in state.items():
        if (selected is None or key in selected) and torch.is_floating_point(value):
            arrays.append(value.detach().cpu().numpy().reshape(-1))
    return np.concatenate(arrays) if arrays else np.empty(0)


def stage_metadata(stage: int, kind: str, extra: Mapping[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "stage": stage,
        "kind": kind,
        "execution_commit": git_head(),
        "branch": git_branch(),
        "command": command_string(),
        "generated_at_unix": time.time(),
    }
    if extra:
        payload.update(extra)
    return payload

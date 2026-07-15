#!/usr/bin/env python3
"""Shared execution, evidence, and finite-group utilities for the next program."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "next_program"
TMP = Path(os.environ.get("TWISTEDMERGE_TMP_ROOT", ROOT / "reports" / "tmp" / "next_program")).expanduser().resolve()
DATA = Path(os.environ.get("TWISTEDMERGE_DATA_ROOT", ROOT / "data")).expanduser().resolve()


def ensure_dirs() -> None:
    for path in (OUT, OUT / "immediate", OUT / "iclr", OUT / "extended", TMP, TMP / "logits", TMP / "checkpoints"):
        path.mkdir(parents=True, exist_ok=True)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: Mapping[str, object], fields: Sequence[str]) -> None:
    rows = read_csv(path)
    rows.append({field: row.get(field, "") for field in fields})
    write_csv(path, rows, fields)


def source_hash(path: Path) -> str:
    return sha256_file(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def torch_device() -> "Any":
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def synchronize(device: "Any") -> None:
    import torch

    if str(device) == "mps":
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def process_peak_mb() -> float:
    # macOS reports bytes; Linux reports KiB.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if platform.system() == "Darwin" else value / 1024.0


def mps_peak_mb() -> float | None:
    try:
        import torch

        if torch.backends.mps.is_available():
            return float(torch.mps.driver_allocated_memory() / (1024.0 * 1024.0))
    except (ImportError, RuntimeError):
        return None
    return None


def parameter_counts(model: "Any") -> tuple[int, int]:
    stored = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    return trainable, stored


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-300)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    loss = -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300)).mean()
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {"accuracy": float(correct.mean()), "loss": float(loss), "ece": float(ece)}


def paired_bootstrap(
    deltas: Sequence[float], seed: int, samples: int = 10_000
) -> tuple[float, float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    if not len(values):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def hierarchical_bootstrap(
    records: Sequence[Mapping[str, object]], value_key: str, setting_key: str, seed: int, samples: int = 10_000
) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(str(record[setting_key]), []).append(float(record[value_key]))
    if not grouped:
        return math.nan, math.nan, math.nan
    setting_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        chosen = rng.integers(0, len(setting_ids), size=len(setting_ids))
        per_setting = []
        for index in chosen:
            values = grouped[setting_ids[int(index)]]
            per_setting.append(values[int(rng.integers(0, len(values)))])
        draws.append(float(np.mean(per_setting)))
    observed = float(np.mean([np.mean(grouped[key]) for key in setting_ids]))
    return observed, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def save_logits_before_labels(name: str, logits: Mapping[str, np.ndarray], labels: np.ndarray, seed: int) -> dict[str, object]:
    """Persist candidate bytes before any label-based metric is computed."""

    path = TMP / "logits" / f"{name}.npz"
    arrays = {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in logits.items()}
    np.savez_compressed(path, **arrays)
    hashes_before = {key: sha256_bytes(value.tobytes()) for key, value in arrays.items()}
    file_before = sha256_file(path)
    permuted = np.asarray(labels).copy()
    np.random.default_rng(seed).shuffle(permuted)
    hashes_after = {key: sha256_bytes(value.tobytes()) for key, value in arrays.items()}
    file_after = sha256_file(path)
    return {
        "logits_path": str(path.relative_to(ROOT)),
        "logits_sha256": file_before,
        "candidate_hashes_unchanged": hashes_before == hashes_after,
        "file_hash_unchanged": file_before == file_after,
        "permuted_labels_differ": not np.array_equal(labels, permuted),
    }


def measure_callable(
    fn: Callable[[], Any], device: Any, warmups: int = 10, repeats: int = 100
) -> dict[str, float | None]:
    """Measure an actual callable with synchronized cold and warm timings."""

    start = time.perf_counter()
    result = fn()
    synchronize(device)
    cold_ms = (time.perf_counter() - start) * 1000.0
    for _ in range(warmups):
        result = fn()
    synchronize(device)
    timings = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    # Retain the result so lazy backends cannot discard the operation.
    if hasattr(result, "shape"):
        _ = result.shape
    return {
        "cold_start_ms": float(cold_ms),
        "latency_median_ms": float(np.median(timings)),
        "latency_q1_ms": float(np.quantile(timings, 0.25)),
        "latency_q3_ms": float(np.quantile(timings, 0.75)),
        "peak_process_memory_mb": process_peak_mb(),
        "peak_mps_memory_mb": mps_peak_mb(),
        "warmups": warmups,
        "timed_repetitions": repeats,
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


def provenance(script: Path, command: str, seed: int | str, **extra: object) -> dict[str, object]:
    return {
        "execution_commit": git_head(),
        "source_sha256": source_hash(script),
        "command": command,
        "seed": seed,
        **extra,
    }


def environment_record() -> dict[str, object]:
    import torch
    import torchvision

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(torch_device()),
        "mps_available": torch.backends.mps.is_available(),
    }


# Finite groups are represented by integer Cayley tables with identity 0.
def cyclic_group(order: int) -> np.ndarray:
    return np.fromfunction(lambda i, j: (i + j) % order, (order, order), dtype=int).astype(int)


def direct_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    nl, nr = len(left), len(right)
    table = np.empty((nl * nr, nl * nr), dtype=int)
    for a in range(nl):
        for b in range(nr):
            for c in range(nl):
                for d in range(nr):
                    table[a * nr + b, c * nr + d] = int(left[a, c]) * nr + int(right[b, d])
    return table


def dihedral_group(n: int = 4) -> np.ndarray:
    # (rotation, reflection), with s r s = r^-1.
    table = np.empty((2 * n, 2 * n), dtype=int)
    for r in range(n):
        for s in range(2):
            for u in range(n):
                for v in range(2):
                    rr = (r + (-u if s else u)) % n
                    table[r * 2 + s, u * 2 + v] = rr * 2 + ((s + v) % 2)
    return table


def symmetric_group_3() -> np.ndarray:
    import itertools

    elements = list(itertools.permutations(range(3)))
    index = {value: i for i, value in enumerate(elements)}
    table = np.empty((6, 6), dtype=int)
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            table[i, j] = index[tuple(a[b[k]] for k in range(3))]
    identity = index[(0, 1, 2)]
    if identity:
        permutation = [identity] + [i for i in range(6) if i != identity]
        reverse = {old: new for new, old in enumerate(permutation)}
        table = np.asarray([[reverse[int(table[a, b])] for b in permutation] for a in permutation], dtype=int)
    return table


def quaternion_group() -> np.ndarray:
    # +/-1, +/-i, +/-j, +/-k; identity is +1.
    units = [(1, 0), (-1, 0), (1, 1), (-1, 1), (1, 2), (-1, 2), (1, 3), (-1, 3)]
    index = {value: i for i, value in enumerate(units)}
    positive = {
        (0, 0): (1, 0), (0, 1): (1, 1), (0, 2): (1, 2), (0, 3): (1, 3),
        (1, 0): (1, 1), (2, 0): (1, 2), (3, 0): (1, 3),
        (1, 1): (-1, 0), (2, 2): (-1, 0), (3, 3): (-1, 0),
        (1, 2): (1, 3), (2, 3): (1, 1), (3, 1): (1, 2),
        (2, 1): (-1, 3), (3, 2): (-1, 1), (1, 3): (-1, 2),
    }
    table = np.empty((8, 8), dtype=int)
    for a, (sa, ua) in enumerate(units):
        for b, (sb, ub) in enumerate(units):
            sign, unit = positive[ua, ub]
            table[a, b] = index[(sa * sb * sign, unit)]
    return table


def alternating_group_4() -> np.ndarray:
    import itertools

    def even(permutation: tuple[int, ...]) -> bool:
        inversions = sum(permutation[i] > permutation[j] for i in range(4) for j in range(i + 1, 4))
        return inversions % 2 == 0

    identity = (0, 1, 2, 3)
    elements = [identity] + [p for p in itertools.permutations(range(4)) if p != identity and even(p)]
    index = {value: i for i, value in enumerate(elements)}
    return np.asarray(
        [[index[tuple(a[b[k]] for k in range(4))] for b in elements] for a in elements], dtype=int
    )


def group_inverses(table: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.flatnonzero((table[g] == 0) & (table[:, g] == 0))[0]) for g in range(len(table))])


def modular_rank(matrix: np.ndarray, prime: int) -> int:
    values = np.asarray(matrix, dtype=int).copy() % prime
    row = 0
    for column in range(values.shape[1]):
        pivots = np.flatnonzero(values[row:, column])
        if not len(pivots):
            continue
        pivot = int(pivots[0] + row)
        values[[row, pivot]] = values[[pivot, row]]
        values[row] = values[row] * pow(int(values[row, column]), -1, prime) % prime
        for other in range(values.shape[0]):
            if other != row and values[other, column]:
                values[other] = (values[other] - values[other, column] * values[row]) % prime
        row += 1
        if row == values.shape[0]:
            break
    return row


def modular_rref(matrix: np.ndarray, prime: int) -> tuple[np.ndarray, list[int]]:
    values = np.asarray(matrix, dtype=int).copy() % prime
    pivots: list[int] = []
    row = 0
    for column in range(values.shape[1]):
        candidates = np.flatnonzero(values[row:, column])
        if not len(candidates):
            continue
        pivot = int(candidates[0] + row)
        values[[row, pivot]] = values[[pivot, row]]
        values[row] = values[row] * pow(int(values[row, column]), -1, prime) % prime
        for other in range(values.shape[0]):
            if other != row and values[other, column]:
                values[other] = (values[other] - values[other, column] * values[row]) % prime
        pivots.append(column)
        row += 1
        if row == values.shape[0]:
            break
    return values, pivots


def modular_nullspace(matrix: np.ndarray, prime: int) -> list[np.ndarray]:
    rref, pivots = modular_rref(matrix, prime)
    free = [column for column in range(rref.shape[1]) if column not in pivots]
    basis = []
    for column in free:
        vector = np.zeros(rref.shape[1], dtype=int)
        vector[column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = -rref[row, column] % prime
        basis.append(vector)
    return basis


def modular_solve(matrix: np.ndarray, target: np.ndarray, prime: int) -> tuple[np.ndarray | None, list[np.ndarray]]:
    augmented = np.column_stack([np.asarray(matrix, dtype=int), np.asarray(target, dtype=int)])
    rref, pivots_augmented = modular_rref(augmented, prime)
    variable_count = matrix.shape[1]
    if variable_count in pivots_augmented:
        return None, []
    pivots = [pivot for pivot in pivots_augmented if pivot < variable_count]
    solution = np.zeros(variable_count, dtype=int)
    for row, pivot in enumerate(pivots):
        solution[pivot] = rref[row, -1]
    return solution, modular_nullspace(matrix, prime)


def find_nontrivial_cocycle(table: np.ndarray, prime: int) -> tuple[np.ndarray | None, int, int]:
    """Return one normalized H^2 class representative over F_prime, if present."""

    n = len(table)
    constraints = []
    for g in range(n):
        row_left = np.zeros(n * n, dtype=int); row_left[g] = 1; constraints.append(row_left)
        row_right = np.zeros(n * n, dtype=int); row_right[g * n] = 1; constraints.append(row_right)
    for g in range(n):
        for h in range(n):
            for k in range(n):
                row = np.zeros(n * n, dtype=int)
                row[h * n + k] += 1
                row[int(table[g, h]) * n + k] -= 1
                row[g * n + int(table[h, k])] += 1
                row[g * n + h] -= 1
                constraints.append(row)
    cocycle_matrix = np.asarray(constraints, dtype=int) % prime
    z_basis = modular_nullspace(cocycle_matrix, prime)
    coboundary = np.zeros((n * n, n), dtype=int)
    for g in range(n):
        for h in range(n):
            row = g * n + h
            coboundary[row, g] += 1
            coboundary[row, h] += 1
            coboundary[row, int(table[g, h])] -= 1
    # A normalized coboundary has b(identity)=0; remove that cochain column so
    # the image lies in the normalized cocycle space used above.
    normalized_coboundary = coboundary[:, 1:]
    b_rank = modular_rank(normalized_coboundary, prime)
    for vector in z_basis:
        if modular_rank(np.column_stack([normalized_coboundary, vector]), prime) > b_rank:
            h2_dimension = len(z_basis) - b_rank
            return vector.reshape(n, n) % prime, len(z_basis), h2_dimension
    return None, len(z_basis), max(0, len(z_basis) - b_rank)


def is_coboundary(table: np.ndarray, cocycle: np.ndarray, modulus: int) -> bool:
    n = len(table)
    matrix = np.zeros((n * n, n), dtype=int)
    target = np.asarray(cocycle, dtype=int).reshape(-1) % modulus
    for g in range(n):
        for h in range(n):
            row = g * n + h
            matrix[row, g] += 1
            matrix[row, h] += 1
            matrix[row, int(table[g, h])] -= 1
    if modulus in (2, 3):
        return modular_rank(matrix, modulus) == modular_rank(np.column_stack([matrix, target]), modulus)
    if modulus == 4:
        # Solve A(x0 + 2*x1)=target (mod 4).  Enumerating the F2 nullspace is
        # cheap here because it is the group-homomorphism space, not all
        # cochains.
        import itertools

        particular, nullspace = modular_solve(matrix, target % 2, 2)
        if particular is None:
            return False
        for coefficients in itertools.product((0, 1), repeat=len(nullspace)):
            x0 = particular.copy()
            for coefficient, basis in zip(coefficients, nullspace, strict=True):
                x0 = (x0 + coefficient * basis) % 2
            difference = (target - matrix @ x0) % 4
            if np.any(difference % 2):
                continue
            x1, _ = modular_solve(matrix, (difference // 2) % 2, 2)
            if x1 is not None:
                return True
        return False
    raise ValueError(f"unsupported coefficient modulus: {modulus}")


def cocycle_identity_error(table: np.ndarray, cocycle: np.ndarray, modulus: int) -> int:
    n = len(table)
    maximum = 0
    for g in range(n):
        for h in range(n):
            for k in range(n):
                value = (
                    int(cocycle[h, k]) - int(cocycle[int(table[g, h]), k])
                    + int(cocycle[g, int(table[h, k])]) - int(cocycle[g, h])
                ) % modulus
                maximum = max(maximum, min(value, modulus - value))
    return maximum


ensure_dirs()

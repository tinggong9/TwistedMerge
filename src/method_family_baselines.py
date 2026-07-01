"""Lightweight internal method-family baselines and coverage metadata.

These utilities are deliberately small and dependency-light.  The full
MNIST/Fashion-MNIST training/evaluation pipelines live in the experiment
scripts; this module keeps reusable vector-space pieces testable without
loading torch or datasets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


ArrayLike = Iterable[float] | np.ndarray


def _as_vector(vector: ArrayLike) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected a flat vector, got shape {arr.shape}")
    return arr


def _as_delta_matrix(deltas: Iterable[ArrayLike]) -> np.ndarray:
    rows = [_as_vector(delta) for delta in deltas]
    if not rows:
        raise ValueError("at least one delta vector is required")
    width = rows[0].shape[0]
    if any(row.shape[0] != width for row in rows):
        raise ValueError("all delta vectors must have the same length")
    return np.stack(rows, axis=0)


def slerp_vector(v0: ArrayLike, v1: ArrayLike, t: float, eps: float = 1e-12) -> np.ndarray:
    """Spherical interpolation in flattened parameter space.

    Falls back to ordinary linear interpolation when either endpoint is nearly
    zero or the angle is numerically tiny.  This is an internal SLERP-style
    baseline, not an official implementation.
    """

    a = _as_vector(v0)
    b = _as_vector(v1)
    if a.shape != b.shape:
        raise ValueError("SLERP endpoints must have the same shape")
    if t <= 0.0:
        return a.copy()
    if t >= 1.0:
        return b.copy()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return (1.0 - t) * a + t * b
    ua = a / na
    ub = b / nb
    dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    if abs(dot) > 0.9995:
        return (1.0 - t) * a + t * b
    theta = math.acos(dot)
    denom = math.sin(theta)
    return math.sin((1.0 - t) * theta) / denom * a + math.sin(t * theta) / denom * b


def sequential_slerp(vectors: Iterable[ArrayLike]) -> np.ndarray:
    """Sequential equal-weight SLERP soup over flattened vectors."""

    items = [_as_vector(vector) for vector in vectors]
    if not items:
        raise ValueError("at least one vector is required")
    out = items[0].copy()
    for idx, vector in enumerate(items[1:], start=2):
        if vector.shape != out.shape:
            raise ValueError("all vectors must have the same shape")
        out = slerp_vector(out, vector, 1.0 / idx)
    return out


def task_vector(base: ArrayLike, target: ArrayLike) -> np.ndarray:
    """Return tau = target - base for shared-base task arithmetic."""

    b = _as_vector(base)
    t = _as_vector(target)
    if b.shape != t.shape:
        raise ValueError("base and target must have the same shape")
    return t - b


def task_arithmetic_merge(base: ArrayLike, deltas: Iterable[ArrayLike], alpha: float = 1.0) -> np.ndarray:
    """Merge shared-base task vectors as theta_0 + alpha * mean_i tau_i."""

    b = _as_vector(base)
    matrix = _as_delta_matrix(deltas)
    if matrix.shape[1] != b.shape[0]:
        raise ValueError("base and deltas must have the same length")
    return b + float(alpha) * matrix.mean(axis=0)


def ties_merge_delta(deltas: Iterable[ArrayLike], keep_rate: float = 0.5) -> np.ndarray:
    """Lightweight TIES-style sign-election merge for task-vector deltas.

    Per delta, the largest-magnitude ``keep_rate`` fraction of coordinates is
    retained.  The coordinate sign is elected by aggregate signed magnitude, and
    only sign-consistent retained entries contribute to the coordinate average.
    """

    if not (0.0 < keep_rate <= 1.0):
        raise ValueError("keep_rate must be in (0, 1]")
    matrix = _as_delta_matrix(deltas)
    trimmed = np.zeros_like(matrix)
    keep_count = max(1, int(round(keep_rate * matrix.shape[1])))
    for idx, delta in enumerate(matrix):
        if keep_count >= delta.shape[0]:
            mask = np.ones(delta.shape[0], dtype=bool)
        else:
            threshold = np.partition(np.abs(delta), -keep_count)[-keep_count]
            mask = np.abs(delta) >= threshold
        trimmed[idx, mask] = delta[mask]

    elected = np.sign(trimmed.sum(axis=0))
    fallback = np.sign(matrix.sum(axis=0))
    elected[elected == 0.0] = fallback[elected == 0.0]

    consistent = (trimmed != 0.0) & (np.sign(trimmed) == elected[None, :])
    selected = np.where(consistent, trimmed, 0.0)
    counts = consistent.sum(axis=0)
    merged = np.zeros(matrix.shape[1], dtype=float)
    active = counts > 0
    merged[active] = selected[:, active].sum(axis=0) / counts[active]
    return merged


def ties_merge(base: ArrayLike, deltas: Iterable[ArrayLike], keep_rate: float = 0.5, alpha: float = 1.0) -> np.ndarray:
    b = _as_vector(base)
    delta = ties_merge_delta(deltas, keep_rate=keep_rate)
    if delta.shape != b.shape:
        raise ValueError("base and deltas must have the same length")
    return b + float(alpha) * delta


def dare_merge_delta(
    deltas: Iterable[ArrayLike],
    drop_rate: float = 0.5,
    seed: int = 0,
    n_masks: int = 3,
) -> np.ndarray:
    """Lightweight DARE-style dropout/rescale merge for task-vector deltas."""

    if not (0.0 <= drop_rate < 1.0):
        raise ValueError("drop_rate must be in [0, 1)")
    if n_masks <= 0:
        raise ValueError("n_masks must be positive")
    matrix = _as_delta_matrix(deltas)
    keep_prob = 1.0 - float(drop_rate)
    rng = np.random.default_rng(seed)
    masked_means = []
    for _ in range(int(n_masks)):
        mask = rng.random(matrix.shape) < keep_prob
        masked_means.append(np.where(mask, matrix / keep_prob, 0.0).mean(axis=0))
    return np.stack(masked_means, axis=0).mean(axis=0)


def dare_merge(
    base: ArrayLike,
    deltas: Iterable[ArrayLike],
    drop_rate: float = 0.5,
    alpha: float = 1.0,
    seed: int = 0,
    n_masks: int = 3,
) -> np.ndarray:
    b = _as_vector(base)
    delta = dare_merge_delta(deltas, drop_rate=drop_rate, seed=seed, n_masks=n_masks)
    if delta.shape != b.shape:
        raise ValueError("base and deltas must have the same length")
    return b + float(alpha) * delta


@dataclass(frozen=True)
class CoverageRow:
    method: str
    method_family: str
    validation_selection: str
    pairwise_gauge_synchronization: str
    permutation_gauge_handling: str
    monomial_relu_scaling_gauge_handling: str
    coordinatewise_sign_or_sparsity: str
    cycle_holonomy_diagnostic: str
    central_projective_obstruction_detection: str
    conservative_rejection_no_lift: str
    common_base_required: str
    official_implementation: str
    note: str

    def as_dict(self) -> dict[str, str]:
        return {
            "method": self.method,
            "method_family": self.method_family,
            "validation_selection": self.validation_selection,
            "pairwise_gauge_synchronization": self.pairwise_gauge_synchronization,
            "permutation_gauge_handling": self.permutation_gauge_handling,
            "monomial_relu_scaling_gauge_handling": self.monomial_relu_scaling_gauge_handling,
            "coordinatewise_sign_or_sparsity": self.coordinatewise_sign_or_sparsity,
            "cycle_holonomy_diagnostic": self.cycle_holonomy_diagnostic,
            "central_projective_obstruction_detection": self.central_projective_obstruction_detection,
            "conservative_rejection_no_lift": self.conservative_rejection_no_lift,
            "common_base_required": self.common_base_required,
            "official_implementation": self.official_implementation,
            "qualitative_coverage_score": str(qualitative_coverage_score(self)),
            "note": self.note,
        }


COVERAGE_ROWS = [
    CoverageRow("weight_average", "parameter averaging", "no", "no", "no", "no", "no", "no", "no", "no", "no", "internal", "Capacity-matched unprotected average."),
    CoverageRow("greedy_soup", "Model Soups / validation soup", "yes", "no", "no", "no", "no", "no", "no", "no", "preferred but not required", "internal faithful-style", "Strong validation-selected pure-accuracy baseline."),
    CoverageRow("c2m3_permutation", "cycle-consistent permutation synchronization", "no", "yes", "yes", "no", "no", "yes", "no", "partial", "no", "internal faithful-style", "Handles permutation cycle consistency, not full obstruction taxonomy."),
    CoverageRow("twistedmerge_selector", "TwistedMerge / TwistedMerge++ selector", "yes", "yes", "yes", "yes", "partial", "yes", "yes", "yes", "no", "project", "Broadest structural coverage in current in-repo framework; accuracy claims remain paired-CI gated."),
    CoverageRow("slerp", "SLERP-style path geometry", "partial", "no", "no", "no", "no", "no", "no", "no", "fixed chart preferred", "internal style", "Path-geometry baseline in a fixed parameter chart."),
    CoverageRow("task_arithmetic", "shared-base task vectors", "yes", "no", "no", "no", "no", "no", "no", "no", "yes", "internal style", "Requires a common base/fixed trivialization."),
    CoverageRow("ties", "shared-base sparse/sign task vectors", "yes", "no", "no", "no", "yes", "no", "no", "no", "yes", "internal style", "Adds coordinatewise sign/sparsity handling in a common chart."),
    CoverageRow("dare", "shared-base dropout-rescaled task vectors", "yes", "no", "no", "no", "yes", "no", "no", "no", "yes", "internal style", "Dropout/rescale task-vector baseline in a common chart."),
]


def qualitative_coverage_score(row: CoverageRow | dict[str, str]) -> int:
    keys = [
        "validation_selection",
        "pairwise_gauge_synchronization",
        "permutation_gauge_handling",
        "monomial_relu_scaling_gauge_handling",
        "coordinatewise_sign_or_sparsity",
        "cycle_holonomy_diagnostic",
        "central_projective_obstruction_detection",
        "conservative_rejection_no_lift",
    ]
    score = 0
    for key in keys:
        if isinstance(row, CoverageRow):
            value = str(getattr(row, key, "")).lower()
        else:
            value = str(row.get(key, "")).lower()
        if value == "yes":
            score += 1
        elif value == "partial":
            score += 0
    return score


def structural_coverage_matrix() -> list[dict[str, str]]:
    """Return the qualitative method-coverage matrix used by the appendix."""

    return [row.as_dict() for row in COVERAGE_ROWS]

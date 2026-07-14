"""Low-rank extraction for persistent transition residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TwistSubspace:
    basis: np.ndarray
    singular_values: np.ndarray
    chosen_rank: int
    explained_energy: float
    total_energy: float


def extract_twist_subspace(residuals: np.ndarray, *, epsilon: float = 0.05) -> TwistSubspace:
    data = np.asarray(residuals, dtype=float)
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must lie in [0, 1)")
    if data.ndim == 3:
        if data.shape[1] != data.shape[2]:
            raise ValueError("matrix residuals must be square")
        identity = np.eye(data.shape[1])
        stacked = np.concatenate([matrix - identity for matrix in data], axis=0)
    elif data.ndim == 2:
        stacked = data
    else:
        raise ValueError("residuals must be a 2D stack or 3D square matrices")
    _, singular, vt = np.linalg.svd(stacked, full_matrices=False)
    energy = singular**2
    total = float(energy.sum())
    if total <= 1e-20:
        return TwistSubspace(np.zeros((stacked.shape[1], 0)), singular, 0, 1.0, total)
    cumulative = np.cumsum(energy) / total
    rank = int(np.searchsorted(cumulative, 1.0 - epsilon) + 1)
    return TwistSubspace(vt[:rank].T, singular, rank, float(cumulative[rank - 1]), total)


def project_residual(residual: np.ndarray, subspace: TwistSubspace) -> np.ndarray:
    values = np.asarray(residual, dtype=float)
    if subspace.chosen_rank == 0:
        return np.zeros_like(values)
    return values @ subspace.basis @ subspace.basis.T


def bootstrap_rank_stability(
    residuals: np.ndarray, *, epsilon: float = 0.05, samples: int = 200, seed: int = 0
) -> dict[int, float]:
    values = np.asarray(residuals)
    if values.shape[0] == 0:
        raise ValueError("at least one residual is required")
    rng = np.random.default_rng(seed)
    counts: dict[int, int] = {}
    for _ in range(samples):
        selected = values[rng.integers(0, values.shape[0], size=values.shape[0])]
        rank = extract_twist_subspace(selected, epsilon=epsilon).chosen_rank
        counts[rank] = counts.get(rank, 0) + 1
    return {rank: count / samples for rank, count in sorted(counts.items())}


def subspace_cost(shared_parameters: int, width: int, rank: int, branch_count: int = 1) -> dict[str, int | float]:
    lifted = 2 * width * rank + branch_count * rank
    return {
        "shared_parameters": int(shared_parameters),
        "lifted_parameters": int(lifted),
        "stored_parameters": int(shared_parameters + lifted),
        "parameter_multiplier": float((shared_parameters + lifted) / max(shared_parameters, 1)),
        "branch_count": int(branch_count),
    }

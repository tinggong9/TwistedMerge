"""Gauge transformations and rank-space synchronization for LoRA factors.

The transition convention used throughout this module is explicit.  A map
``T_ij`` sends adapter ``i`` into adapter ``j``'s rank-space coordinates:

``B_i T_ij ~= B_j`` and ``T_ij^{-1} A_i ~= A_j``.

For exact equivalent factorizations ``B_i = B Q_i`` and
``A_i = Q_i^{-1} A``, the transition is ``T_ij = Q_i^{-1} Q_j``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

Array = np.ndarray
Factor = tuple[Array, Array]


@dataclass(frozen=True)
class TransitionEstimate:
    """One estimated directed transition and its numerical diagnostics."""

    matrix: Array
    source: int | None
    target: int | None
    mode: str
    b_relative_residual: float
    a_relative_residual: float
    joint_relative_residual: float
    condition_number: float


def _relative_fro(error: Array, target: Array) -> float:
    return float(np.linalg.norm(error, ord="fro") / max(np.linalg.norm(target, ord="fro"), 1e-15))


def validate_factor(b: Array, a: Array) -> tuple[int, int, int]:
    """Validate one ``B @ A`` factorization and return output/rank/input sizes."""

    if b.ndim != 2 or a.ndim != 2:
        raise ValueError("LoRA factors must both be matrices")
    if b.shape[1] != a.shape[0]:
        raise ValueError(f"incompatible LoRA factors: B{b.shape}, A{a.shape}")
    if not np.isfinite(b).all() or not np.isfinite(a).all():
        raise ValueError("LoRA factors must be finite")
    return b.shape[0], b.shape[1], a.shape[1]


def effective_delta(factor: Factor | Array, a: Array | None = None) -> Array:
    """Materialize an effective LoRA update."""

    if a is None:
        b, a_value = factor  # type: ignore[misc]
    else:
        b, a_value = factor, a
    validate_factor(b, a_value)
    return b @ a_value


def gauge_transform(b: Array, a: Array, q: Array) -> Factor:
    """Apply ``(B, A) -> (B Q, Q^{-1} A)`` without forming an inverse."""

    _, rank, _ = validate_factor(b, a)
    if q.shape != (rank, rank):
        raise ValueError(f"expected a ({rank}, {rank}) gauge, got {q.shape}")
    if not np.isfinite(q).all() or np.linalg.matrix_rank(q) != rank:
        raise ValueError("gauge matrix must be finite and invertible")
    return b @ q, np.linalg.solve(q, a)


def _random_orthogonal(rng: np.random.Generator, rank: int) -> Array:
    q, r = np.linalg.qr(rng.normal(size=(rank, rank)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q @ np.diag(signs)


def sample_gauge(
    rng: np.random.Generator,
    rank: int,
    family: str,
    condition_number: float | None = None,
) -> Array:
    """Sample a controlled invertible rank-space transformation.

    Supported families are ``orthogonal``, ``positive_diagonal``, ``dense``,
    and ``ill_conditioned``.  The last family is a numerical boundary rather
    than part of the well-conditioned invariance claim.
    """

    if rank < 1:
        raise ValueError("rank must be positive")
    if family == "orthogonal":
        return _random_orthogonal(rng, rank)

    default_condition = {
        "positive_diagonal": 8.0,
        "dense": 30.0,
        "ill_conditioned": 1e10,
    }
    if family not in default_condition:
        raise ValueError(f"unknown gauge family: {family}")
    target = float(condition_number or default_condition[family])
    if target < 1.0 or not np.isfinite(target):
        raise ValueError("condition number must be finite and at least one")

    scales = np.geomspace(target ** -0.5, target ** 0.5, rank)
    scales = scales[rng.permutation(rank)]
    if family == "positive_diagonal":
        return np.diag(scales)
    left = _random_orthogonal(rng, rank)
    right = _random_orthogonal(rng, rank)
    return left @ np.diag(scales) @ right.T


def truncated_svd(matrix: Array, rank: int) -> Array:
    """Return the best rank-bounded Frobenius approximation."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if rank < 1:
        raise ValueError("rank must be positive")
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    kept = min(rank, len(singular))
    return (u[:, :kept] * singular[:kept]) @ vt[:kept]


def canonical_svd_factors(matrix: Array, rank: int) -> Factor:
    """Construct deterministic balanced factors from a matrix SVD.

    Column signs are fixed using the largest-magnitude entry of each left
    singular vector.  Repeated singular values still make the factor basis
    mathematically non-unique; the effective returned update is deterministic
    for a fixed numerical input and is the quantity used by the safety gate.
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    kept = min(rank, len(singular))
    u = u[:, :kept].copy()
    singular = singular[:kept]
    vt = vt[:kept].copy()
    for column in range(kept):
        pivot = int(np.argmax(np.abs(u[:, column])))
        if u[pivot, column] < 0.0:
            u[:, column] *= -1.0
            vt[column] *= -1.0
    root = np.sqrt(np.maximum(singular, 0.0))
    return u * root, root[:, None] * vt


def estimate_transition(
    source: Factor,
    target: Factor,
    *,
    mode: str = "b",
    a_weight: float = 1.0,
    source_index: int | None = None,
    target_index: int | None = None,
) -> TransitionEstimate:
    """Estimate a directed map from ``source`` to ``target`` coordinates.

    ``b`` solves ``B_source T ~= B_target``.  ``a`` solves the equivalent
    linear equation ``T A_target ~= A_source``.  ``joint`` minimizes the sum
    of those two squared residuals.  The A equation is only appropriate when
    the two factors represent the same effective update; distinct task
    adapters can instead use the shared-subspace B estimator.
    """

    b_source, a_source = source
    b_target, a_target = target
    out_source, rank, in_source = validate_factor(b_source, a_source)
    out_target, target_rank, in_target = validate_factor(b_target, a_target)
    if (out_source, rank, in_source) != (out_target, target_rank, in_target):
        raise ValueError("source and target factor shapes must match")
    if mode == "b":
        transition = np.linalg.lstsq(b_source, b_target, rcond=None)[0]
    elif mode == "a":
        transition = np.linalg.lstsq(a_target.T, a_source.T, rcond=None)[0].T
    elif mode == "joint":
        if a_weight < 0.0:
            raise ValueError("a_weight must be nonnegative")
        left_b = np.kron(np.eye(rank), b_source)
        left_a = np.kron(a_target.T, np.eye(rank))
        left = np.vstack([left_b, np.sqrt(a_weight) * left_a])
        right = np.concatenate(
            [b_target.reshape(-1, order="F"), np.sqrt(a_weight) * a_source.reshape(-1, order="F")]
        )
        transition = np.linalg.lstsq(left, right, rcond=None)[0].reshape((rank, rank), order="F")
    else:
        raise ValueError(f"unknown transition mode: {mode}")

    condition = float(np.linalg.cond(transition))
    b_residual = _relative_fro(b_source @ transition - b_target, b_target)
    a_linear_residual = _relative_fro(transition @ a_target - a_source, a_source)
    if np.isfinite(condition) and condition < 1.0 / np.finfo(float).eps:
        a_inverse_residual = _relative_fro(np.linalg.solve(transition, a_source) - a_target, a_target)
    else:
        a_inverse_residual = float("inf")
    joint = float(np.sqrt(b_residual**2 + a_weight * a_linear_residual**2))
    return TransitionEstimate(
        matrix=transition,
        source=source_index,
        target=target_index,
        mode=mode,
        b_relative_residual=b_residual,
        a_relative_residual=a_inverse_residual,
        joint_relative_residual=joint,
        condition_number=condition,
    )


def estimate_pairwise_transitions(
    factors: Sequence[Factor], *, mode: str = "b", a_weight: float = 1.0
) -> dict[tuple[int, int], TransitionEstimate]:
    """Estimate every directed transition in a complete adapter graph."""

    if not factors:
        raise ValueError("at least one adapter is required")
    rank = validate_factor(*factors[0])[1]
    transitions: dict[tuple[int, int], TransitionEstimate] = {}
    for source_index, source in enumerate(factors):
        for target_index, target in enumerate(factors):
            if source_index == target_index:
                transitions[(source_index, target_index)] = TransitionEstimate(
                    matrix=np.eye(rank),
                    source=source_index,
                    target=target_index,
                    mode=mode,
                    b_relative_residual=0.0,
                    a_relative_residual=0.0,
                    joint_relative_residual=0.0,
                    condition_number=1.0,
                )
            else:
                transitions[(source_index, target_index)] = estimate_transition(
                    source,
                    target,
                    mode=mode,
                    a_weight=a_weight,
                    source_index=source_index,
                    target_index=target_index,
                )
    return transitions


def align_factor(factor: Factor, map_to_common: Array) -> Factor:
    """Move one factor into a common rank-space coordinate system."""

    b, a = factor
    _, rank, _ = validate_factor(b, a)
    if map_to_common.shape != (rank, rank):
        raise ValueError("alignment map has the wrong shape")
    if not np.isfinite(map_to_common).all() or np.linalg.matrix_rank(map_to_common) != rank:
        raise ValueError("alignment map must be finite and invertible")
    return b @ map_to_common, np.linalg.solve(map_to_common, a)


def factor_average(factors: Iterable[Factor]) -> Factor:
    """Average B and A factors separately."""

    values = list(factors)
    if not values:
        raise ValueError("at least one factor is required")
    return np.mean([factor[0] for factor in values], axis=0), np.mean([factor[1] for factor in values], axis=0)


def reference_align(
    factors: Sequence[Factor],
    *,
    reference: int = 0,
    mode: str = "b",
    a_weight: float = 1.0,
) -> tuple[list[Factor], list[Array], dict[tuple[int, int], TransitionEstimate]]:
    """Align every factor directly into one adapter's coordinates."""

    transitions = estimate_pairwise_transitions(factors, mode=mode, a_weight=a_weight)
    maps = [transitions[(index, reference)].matrix for index in range(len(factors))]
    return [align_factor(factor, map_value) for factor, map_value in zip(factors, maps)], maps, transitions


def synchronize_transitions(
    transitions: Mapping[tuple[int, int], TransitionEstimate | Array],
    adapter_count: int,
    rank: int,
    *,
    anchor: int = 0,
    anchor_weight: float = 10.0,
) -> list[Array]:
    """Least-squares synchronize directed GL(r) transitions.

    Returned matrices ``R_i`` map adapter ``i`` into the anchored common
    coordinates and minimize ``R_i - T_ij R_j`` over all supplied directed
    edges, with ``R_anchor = I`` imposed as a weighted constraint.
    """

    if not 0 <= anchor < adapter_count:
        raise ValueError("anchor is out of range")
    if anchor_weight <= 0.0:
        raise ValueError("anchor_weight must be positive")
    edge_rows: list[Array] = []
    for (source, target), value in transitions.items():
        if source == target:
            continue
        matrix = value.matrix if isinstance(value, TransitionEstimate) else value
        if matrix.shape != (rank, rank):
            raise ValueError("transition has the wrong shape")
        row = np.zeros((rank, adapter_count * rank), dtype=float)
        row[:, source * rank : (source + 1) * rank] = np.eye(rank)
        row[:, target * rank : (target + 1) * rank] = -matrix
        edge_rows.append(row)
    if not edge_rows:
        raise ValueError("at least one nontrivial transition is required")
    anchor_row = np.zeros((rank, adapter_count * rank), dtype=float)
    anchor_row[:, anchor * rank : (anchor + 1) * rank] = anchor_weight * np.eye(rank)
    design = np.vstack([*edge_rows, anchor_row])
    right = np.zeros((design.shape[0], rank), dtype=float)
    right[-rank:] = anchor_weight * np.eye(rank)
    solution = np.linalg.lstsq(design, right, rcond=None)[0]
    maps = [solution[index * rank : (index + 1) * rank] for index in range(adapter_count)]
    if any(not np.isfinite(value).all() or np.linalg.matrix_rank(value) != rank for value in maps):
        raise np.linalg.LinAlgError("synchronization produced a singular alignment map")
    return maps


def global_align(
    factors: Sequence[Factor],
    *,
    mode: str = "b",
    a_weight: float = 1.0,
    anchor: int = 0,
) -> tuple[list[Factor], list[Array], dict[tuple[int, int], TransitionEstimate]]:
    """Estimate complete-graph transitions and synchronize a global gauge."""

    transitions = estimate_pairwise_transitions(factors, mode=mode, a_weight=a_weight)
    rank = validate_factor(*factors[0])[1]
    maps = synchronize_transitions(transitions, len(factors), rank, anchor=anchor)
    return [align_factor(factor, map_value) for factor, map_value in zip(factors, maps)], maps, transitions


def merged_factor_delta(factors: Sequence[Factor]) -> Array:
    """Materialize the delta from a factorwise average."""

    return effective_delta(factor_average(factors))


def mean_effective_delta(factors: Sequence[Factor]) -> Array:
    """Average effective updates without using their factor gauges."""

    if not factors:
        raise ValueError("at least one factor is required")
    return np.mean([effective_delta(factor) for factor in factors], axis=0)

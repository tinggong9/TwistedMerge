"""Gauge-covariant low-rank merging helpers for trained LoRA factors.

The whitening step removes the non-orthogonal part of a rank-space gauge
without materializing the effective update.  The remaining orthogonal gauge
can be synchronized with Frobenius least squares because that metric is
orthogonally invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from src.lora_gauge_alignment import (
    Array,
    Factor,
    align_factor,
    canonical_svd_factors,
    effective_delta,
    factor_average,
    synchronize_transitions,
    validate_factor,
)


@dataclass(frozen=True)
class PracticalMergeResult:
    """A rank-bounded merge result and its allocation/diagnostic metadata."""

    factors: Factor
    decision: str
    dense_allocation_count: int
    temporary_dense_bytes: int
    max_cycle_frobenius_defect: float
    max_cycle_spectral_defect: float
    output_rank_cap: int


def _symmetric_inverse_sqrt(gram: Array, floor: float = 1e-12) -> Array:
    values, vectors = np.linalg.eigh(gram)
    if float(values.min()) <= floor:
        raise np.linalg.LinAlgError("LoRA B factor is rank deficient")
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def whiten_factor(factor: Factor) -> Factor:
    """Whiten B columns using only an r-by-r Gram matrix."""

    b, a = factor
    validate_factor(b, a)
    map_to_whitened = _symmetric_inverse_sqrt(b.T @ b)
    return align_factor(factor, map_to_whitened)


def orthogonal_transition(source: Factor, target: Factor) -> Array:
    """Return the nearest orthogonal map from source to target B coordinates."""

    b_source, _ = source
    b_target, _ = target
    if b_source.shape != b_target.shape:
        raise ValueError("source and target factor shapes differ")
    left, _, right = np.linalg.svd(b_source.T @ b_target, full_matrices=False)
    return left @ right


def orthogonal_transitions(factors: Sequence[Factor]) -> tuple[list[Factor], dict[tuple[int, int], Array]]:
    """Whiten factors and estimate every directed orthogonal transition."""

    if not factors:
        raise ValueError("at least one factor is required")
    whitened = [whiten_factor(factor) for factor in factors]
    rank = validate_factor(*whitened[0])[1]
    transitions: dict[tuple[int, int], Array] = {}
    for source in range(len(whitened)):
        for target in range(len(whitened)):
            transitions[(source, target)] = (
                np.eye(rank, dtype=whitened[0][0].dtype)
                if source == target
                else orthogonal_transition(whitened[source], whitened[target])
            )
    return whitened, transitions


def cycle_defects(
    transitions: dict[tuple[int, int], Array], adapter_count: int
) -> tuple[float, float]:
    """Return orthogonally gauge-invariant triangle defect maxima."""

    maximum_frobenius = 0.0
    maximum_spectral = 0.0
    for i, j, k in combinations(range(adapter_count), 3):
        holonomy = transitions[(i, j)] @ transitions[(j, k)] @ transitions[(k, i)]
        identity = np.eye(holonomy.shape[0])
        maximum_frobenius = max(
            maximum_frobenius,
            float(np.linalg.norm(holonomy - identity, ord="fro") / np.sqrt(len(identity))),
        )
        maximum_spectral = max(
            maximum_spectral,
            float(np.max(np.abs(np.linalg.eigvals(holonomy) - 1.0))),
        )
    return maximum_frobenius, maximum_spectral


def reference_aligned_factors(factors: Sequence[Factor], reference: int = 0) -> tuple[list[Factor], float, float]:
    """Whiten and align every factor directly to one reference."""

    whitened, transitions = orthogonal_transitions(factors)
    maps = [
        transitions[(index, reference)].astype(whitened[index][0].dtype, copy=False)
        for index in range(len(whitened))
    ]
    aligned = [align_factor(factor, matrix) for factor, matrix in zip(whitened, maps)]
    cycle_frobenius, cycle_spectral = cycle_defects(transitions, len(factors))
    return aligned, cycle_frobenius, cycle_spectral


def globally_aligned_factors(factors: Sequence[Factor], anchor: int = 0) -> tuple[list[Factor], float, float]:
    """Whiten, synchronize all orthogonal edges, and align globally."""

    whitened, transitions = orthogonal_transitions(factors)
    rank = validate_factor(*whitened[0])[1]
    maps = [
        matrix.astype(whitened[index][0].dtype, copy=False)
        for index, matrix in enumerate(
            synchronize_transitions(transitions, len(whitened), rank, anchor=anchor)
        )
    ]
    aligned = [align_factor(factor, matrix) for factor, matrix in zip(whitened, maps)]
    cycle_frobenius, cycle_spectral = cycle_defects(transitions, len(factors))
    return aligned, cycle_frobenius, cycle_spectral


def _mean_dense_delta(factors: Sequence[Factor]) -> Array:
    result = np.zeros(
        (factors[0][0].shape[0], factors[0][1].shape[1]),
        dtype=np.result_type(*[factor[0].dtype for factor in factors]),
    )
    for factor in factors:
        result += effective_delta(factor)
    return result / len(factors)


def merge_trained_factors(
    factors: Sequence[Factor],
    method: str,
    *,
    planted_gauges: Sequence[Array] | None = None,
    cycle_tolerance: float = 5e-2,
) -> PracticalMergeResult:
    """Merge factors with one frozen Phase-A method.

    Factor-space methods have zero dense allocations here.  Dense methods
    explicitly report each effective-update materialization.
    """

    if not factors:
        raise ValueError("at least one factor is required")
    output_dim, rank, input_dim = validate_factor(*factors[0])
    for factor in factors[1:]:
        if validate_factor(*factor) != (output_dim, rank, input_dim):
            raise ValueError("all factor shapes must match")
    dense_bytes = output_dim * input_dim * np.dtype(factors[0][0].dtype).itemsize
    cycle_frobenius = 0.0
    cycle_spectral = 0.0

    if method == "naive_factor_average":
        merged = factor_average(factors)
        decision = "factor_average"
        allocations = 0
    elif method == "full_delta_svd":
        dense = _mean_dense_delta(factors)
        merged = canonical_svd_factors(dense, rank)
        decision = "dense_mean_then_deterministic_svd"
        allocations = len(factors) + 1
    elif method == "canonical_svd_factor_average":
        canonical = [canonical_svd_factors(effective_delta(factor), rank) for factor in factors]
        merged = factor_average(canonical)
        decision = "canonicalize_each_dense_delta_then_factor_average"
        allocations = len(factors)
    elif method == "pairwise_reference_alignment":
        aligned, cycle_frobenius, cycle_spectral = reference_aligned_factors(factors)
        merged = factor_average(aligned)
        decision = "whitened_pairwise_reference_alignment"
        allocations = 0
    elif method == "global_synchronization":
        aligned, cycle_frobenius, cycle_spectral = globally_aligned_factors(factors)
        merged = factor_average(aligned)
        decision = "whitened_orthogonal_global_synchronization"
        allocations = 0
    elif method == "cycle_aware_alignment":
        aligned, cycle_frobenius, cycle_spectral = globally_aligned_factors(factors)
        if max(cycle_frobenius, cycle_spectral) > cycle_tolerance:
            dense = _mean_dense_delta(factors)
            merged = canonical_svd_factors(dense, rank)
            decision = "fallback_full_delta_svd"
            allocations = len(factors) + 1
        else:
            merged = factor_average(aligned)
            decision = "cycle_gated_global_synchronization"
            allocations = 0
    elif method == "oracle_alignment":
        if planted_gauges is None or len(planted_gauges) != len(factors):
            raise ValueError("oracle alignment requires one planted gauge per factor")
        recovered = [
            align_factor(factor, np.linalg.solve(gauge, np.eye(rank)))
            for factor, gauge in zip(factors, planted_gauges)
        ]
        merged = factor_average(recovered)
        decision = "invert_planted_scramble"
        allocations = 0
    else:
        raise ValueError(f"unknown merge method: {method}")

    return PracticalMergeResult(
        factors=merged,
        decision=decision,
        dense_allocation_count=allocations,
        temporary_dense_bytes=allocations * dense_bytes,
        max_cycle_frobenius_defect=cycle_frobenius,
        max_cycle_spectral_defect=cycle_spectral,
        output_rank_cap=rank,
    )

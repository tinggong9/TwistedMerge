"""Cycle diagnostics and conservative fallback for LoRA rank-space maps."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from src.lora_gauge_alignment import (
    Array,
    Factor,
    TransitionEstimate,
    align_factor,
    estimate_pairwise_transitions,
    factor_average,
    mean_effective_delta,
    synchronize_transitions,
    truncated_svd,
    validate_factor,
)


@dataclass(frozen=True)
class CycleMetric:
    """One oriented triangle holonomy diagnostic."""

    i: int
    j: int
    k: int
    normalized_frobenius_defect: float
    spectral_defect: float
    holonomy_condition_number: float
    holonomy: Array


@dataclass(frozen=True)
class CycleAwareResult:
    """Merged delta plus the synchronization/fallback decision."""

    delta: Array
    decision: str
    reason: str
    max_cycle_frobenius_defect: float
    max_cycle_spectral_defect: float
    max_transition_condition_number: float
    output_rank: int


def _matrix(value: TransitionEstimate | Array) -> Array:
    return value.matrix if isinstance(value, TransitionEstimate) else value


def triangle_cycle_metrics(
    transitions: Mapping[tuple[int, int], TransitionEstimate | Array], adapter_count: int
) -> list[CycleMetric]:
    """Compute ``T_ij T_jk T_ki`` for every increasing triangle."""

    metrics: list[CycleMetric] = []
    for i, j, k in combinations(range(adapter_count), 3):
        holonomy = _matrix(transitions[(i, j)]) @ _matrix(transitions[(j, k)]) @ _matrix(transitions[(k, i)])
        rank = holonomy.shape[0]
        identity = np.eye(rank)
        frobenius = float(np.linalg.norm(holonomy - identity, ord="fro") / np.sqrt(rank))
        eigenvalues = np.linalg.eigvals(holonomy)
        spectral = float(np.max(np.abs(eigenvalues - 1.0)))
        metrics.append(
            CycleMetric(
                i=i,
                j=j,
                k=k,
                normalized_frobenius_defect=frobenius,
                spectral_defect=spectral,
                holonomy_condition_number=float(np.linalg.cond(holonomy)),
                holonomy=holonomy,
            )
        )
    return metrics


def cycle_aware_merge(
    factors: Sequence[Factor],
    *,
    transitions: Mapping[tuple[int, int], TransitionEstimate | Array] | None = None,
    rank: int | None = None,
    cycle_tolerance: float = 1e-8,
    transition_condition_limit: float = 1e8,
) -> CycleAwareResult:
    """Synchronize when diagnostics pass; otherwise use full-delta SVD.

    The fallback is deliberately gauge invariant.  It is not evidence that a
    nonclosing controlled transition system is a natural topological class.
    """

    if not factors:
        raise ValueError("at least one adapter is required")
    inferred_rank = validate_factor(*factors[0])[1]
    output_rank = int(rank or inferred_rank)
    active = dict(transitions or estimate_pairwise_transitions(factors, mode="b"))
    cycles = triangle_cycle_metrics(active, len(factors))
    max_frobenius = max((metric.normalized_frobenius_defect for metric in cycles), default=0.0)
    max_spectral = max((metric.spectral_defect for metric in cycles), default=0.0)
    max_condition = max(float(np.linalg.cond(_matrix(value))) for value in active.values())
    diagnostics_finite = np.isfinite([max_frobenius, max_spectral, max_condition]).all()
    should_fallback = (
        not diagnostics_finite
        or max_frobenius > cycle_tolerance
        or max_spectral > cycle_tolerance
        or max_condition > transition_condition_limit
    )
    if should_fallback:
        delta = truncated_svd(mean_effective_delta(factors), output_rank)
        reasons = []
        if not diagnostics_finite:
            reasons.append("nonfinite_diagnostic")
        if max_frobenius > cycle_tolerance or max_spectral > cycle_tolerance:
            reasons.append("cycle_defect")
        if max_condition > transition_condition_limit:
            reasons.append("transition_condition")
        return CycleAwareResult(
            delta=delta,
            decision="fallback_full_delta_svd",
            reason="+".join(reasons),
            max_cycle_frobenius_defect=max_frobenius,
            max_cycle_spectral_defect=max_spectral,
            max_transition_condition_number=max_condition,
            output_rank=int(np.linalg.matrix_rank(delta)),
        )

    maps = synchronize_transitions(active, len(factors), inferred_rank)
    aligned = [align_factor(factor, map_value) for factor, map_value in zip(factors, maps)]
    b_mean, a_mean = factor_average(aligned)
    delta = b_mean @ a_mean
    return CycleAwareResult(
        delta=delta,
        decision="synchronized_factor_merge",
        reason="cycle_and_condition_gates_passed",
        max_cycle_frobenius_defect=max_frobenius,
        max_cycle_spectral_defect=max_spectral,
        max_transition_condition_number=max_condition,
        output_rank=int(np.linalg.matrix_rank(delta)),
    )

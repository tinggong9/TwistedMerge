"""Finite-group cohomology guided torsion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.finite_group_cohomology import (
    FinitePermutationGroup,
    Permutation,
    close_permutation_group,
    compute_h2_cyclic_coefficients,
    identity_permutation,
)
from src.small_order_torsion import DEFAULT_ORDERS, analyze_permutation_residual


@dataclass(frozen=True)
class InferredResidualGroup:
    group: FinitePermutationGroup
    inference_status: str
    generator_count: int
    generator_source: str


def normalize_optional_permutation(value) -> Permutation | None:
    if value is None:
        return None
    try:
        arr = tuple(int(v) for v in value)
    except Exception:
        return None
    if not arr or sorted(arr) != list(range(len(arr))):
        return None
    return arr


def infer_residual_group(
    edge_permutations: Iterable[Sequence[int]],
    triangle_permutations: Iterable[Sequence[int]],
    max_group_order: int = 5000,
    max_generators: int = 12,
) -> InferredResidualGroup:
    """Infer a finite permutation subgroup from observed edge/triangle maps."""

    generators: list[Permutation] = []
    source = "edge_and_triangle_maps"
    for perm in [*triangle_permutations, *edge_permutations]:
        normalized = normalize_optional_permutation(perm)
        if normalized is None:
            continue
        if normalized == identity_permutation(len(normalized)):
            continue
        if normalized not in generators:
            generators.append(normalized)
        if len(generators) >= int(max_generators):
            break
    if not generators:
        identity = identity_permutation(1)
        group = close_permutation_group([identity], max_group_order=2)
        return InferredResidualGroup(group, "trivial_identity_only", 1, "identity_fallback")
    try:
        group = close_permutation_group(generators, max_group_order=max_group_order)
    except ValueError:
        first = generators[0]
        group = close_permutation_group([first], max_group_order=max_group_order)
        return InferredResidualGroup(group, "cyclic_residual_subgroup_after_inference_error", 1, "first_residual_generator")
    if group.truncated:
        first = generators[0]
        fallback = close_permutation_group([first], max_group_order=max_group_order)
        return InferredResidualGroup(
            fallback,
            "cyclic_residual_subgroup_after_truncation",
            1,
            "first_residual_generator",
        )
    return InferredResidualGroup(group, group.closure_status, len(generators), source)


def classify_permutation_h2_candidate(
    perm: Sequence[int],
    group: FinitePermutationGroup,
    coefficient_modulus: int = 2,
    max_exact_group_order: int = 32,
    orders: Iterable[int] = DEFAULT_ORDERS,
) -> dict:
    """Map one measured permutation residual into conservative H^2 status."""

    normalized = normalize_optional_permutation(perm)
    if normalized is None:
        return {
            "class_status": "not_central_or_not_projectable",
            "central_projection_residual": float("nan"),
            "cocycle_residual": float("nan"),
            "estimated_period": None,
            "estimated_index": None,
            "index_status": "not_projectable",
            "certified_class": False,
            "certification_failure": "invalid_permutation",
        }
    metrics = analyze_permutation_residual(normalized, orders)
    identity = identity_permutation(len(normalized))
    h2 = compute_h2_cyclic_coefficients(group, coefficient_modulus, max_exact_group_order)
    if normalized != identity:
        return {
            **metrics,
            "class_status": "not_central_or_not_projectable",
            "central_projection_residual": float(metrics["centrality_residual"]),
            "cocycle_residual": float("nan"),
            "estimated_period": None,
            "estimated_index": None,
            "index_status": "not_projectable",
            "certified_class": False,
            "certification_failure": "noncentral_permutation_holonomy",
            "h2_exact": bool(h2.exact),
            "h2_size": h2.h2_size,
        }
    return {
        **metrics,
        "class_status": "coboundary",
        "central_projection_residual": 0.0,
        "cocycle_residual": 0.0,
        "estimated_period": 1,
        "estimated_index": 1,
        "index_status": "coboundary",
        "certified_class": False,
        "certification_failure": "coboundary_no_lift_needed",
        "h2_exact": bool(h2.exact),
        "h2_size": h2.h2_size,
    }


def bootstrap_class_stability(
    statuses: Sequence[str],
    periods: Sequence[int | None],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Bootstrap status/period stability for a finite candidate table."""

    if len(statuses) == 0:
        return {
            "bootstrap_samples": 0,
            "bootstrap_detection_rate": 0.0,
            "bootstrap_same_class_rate": 0.0,
            "bootstrap_same_period_rate": 0.0,
            "bootstrap_coboundary_rate": 0.0,
        }
    rng = np.random.default_rng(seed)
    status_arr = np.asarray(statuses, dtype=object)
    period_arr = np.asarray([p if p is not None else -1 for p in periods], dtype=int)
    detections = []
    same_class = []
    same_period = []
    coboundary = []
    base_status = str(status_arr[0])
    base_period = int(period_arr[0])
    for _ in range(max(1, int(n_bootstrap))):
        idx = rng.integers(0, len(status_arr), len(status_arr))
        sampled_status = str(status_arr[idx][0])
        sampled_period = int(period_arr[idx][0])
        detections.append(sampled_status == "nontrivial_H2_class")
        same_class.append(sampled_status == base_status)
        same_period.append(sampled_period == base_period)
        coboundary.append(sampled_status == "coboundary")
    return {
        "bootstrap_samples": int(max(1, int(n_bootstrap))),
        "bootstrap_detection_rate": float(np.mean(detections)),
        "bootstrap_same_class_rate": float(np.mean(same_class)),
        "bootstrap_same_period_rate": float(np.mean(same_period)),
        "bootstrap_coboundary_rate": float(np.mean(coboundary)),
    }

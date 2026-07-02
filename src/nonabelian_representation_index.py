"""Representation-index candidates for finite nonabelian holonomy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.finite_group_cohomology import FinitePermutationGroup
from src.nonabelian_holonomy import group_orbits, permutation_noncentrality_score


@dataclass(frozen=True)
class RepresentationCandidate:
    representation_name: str
    representation_dimension: int | None
    is_faithful: bool | None
    kernel_size: int | None
    orbit_count: int | None
    max_orbit_size: int | None
    estimated_splitting_index: int | None
    representation_status: str
    construction_cost: str
    metadata: dict


def _orbit_summary(group: FinitePermutationGroup) -> tuple[list[tuple[int, ...]], int, int]:
    orbits = group_orbits(group)
    orbit_count = len(orbits)
    max_orbit = max((len(orbit) for orbit in orbits), default=0)
    return orbits, orbit_count, max_orbit


def representation_candidates(
    group: FinitePermutationGroup,
    max_exact_representation_order: int = 256,
) -> list[RepresentationCandidate]:
    orbits, orbit_count, max_orbit = _orbit_summary(group) if group.order <= max_exact_representation_order else ([], None, None)
    exact_small = (not group.truncated) and group.order <= max_exact_representation_order
    candidates = [
        RepresentationCandidate(
            "existing_permutation_representation",
            group.degree,
            True if exact_small else None,
            1 if exact_small else None,
            orbit_count,
            max_orbit,
            group.degree,
            "constructed",
            "existing_action",
            {"orbit_sizes": [len(orbit) for orbit in orbits]},
        ),
        RepresentationCandidate(
            "orbit_representation",
            orbit_count,
            False if exact_small else None,
            group.order if exact_small else None,
            orbit_count,
            max_orbit,
            max_orbit,
            "diagnostic_orbit_quotient" if orbit_count else "skipped_large_or_truncated_group",
            "orbit_decomposition",
            {"orbit_sizes": [len(orbit) for orbit in orbits]},
        ),
        RepresentationCandidate(
            "regular_representation",
            group.order,
            True if exact_small else None,
            1 if exact_small else None,
            1 if exact_small else None,
            group.order,
            group.order,
            "constructed" if exact_small else "theoretical_upper_bound_too_large",
            "left_regular_action",
            {},
        ),
        RepresentationCandidate(
            "quotient_representation",
            None,
            None,
            None,
            None,
            None,
            None,
            "not_implemented",
            "normal_subgroup_search_not_implemented",
            {},
        ),
    ]
    if orbits:
        selected = min(orbits, key=len)
        candidates.append(
            RepresentationCandidate(
                "low_dimensional_permutation_subrepresentation",
                len(selected),
                False,
                None,
                1,
                len(selected),
                len(selected),
                "constructed_restricted_orbit_action",
                "orbit_restriction",
                {"selected_orbit": list(selected), "orbit_sizes": [len(orbit) for orbit in orbits]},
            )
        )
    else:
        candidates.append(
            RepresentationCandidate(
                "low_dimensional_permutation_subrepresentation",
                None,
                None,
                None,
                None,
                None,
                None,
                "skipped_no_small_orbits",
                "orbit_restriction",
                {},
            )
        )
    for dim in sorted({value for value in [group.degree, max_orbit, group.order if exact_small else None] if value}):
        candidates.append(
            RepresentationCandidate(
                "random_same_dimension_representation_control",
                int(dim),
                False,
                None,
                None,
                None,
                int(dim),
                "random_action_control",
                "null_control",
                {"matched_dimension": int(dim)},
            )
        )
    return candidates


def _restricted_residual(perm: Sequence[int], support: Sequence[int]) -> float:
    support_set = set(int(value) for value in support)
    if not support_set:
        return float("nan")
    fixed = 0
    total = 0
    for idx in support_set:
        image = int(perm[idx])
        if image not in support_set:
            continue
        fixed += int(image == idx)
        total += 1
    if total == 0:
        return float("nan")
    frac = fixed / total
    return float(np.sqrt(max(0.0, 1.0 - frac**2)))


def _orbit_quotient_residual(perm: Sequence[int], orbit_sizes: Sequence[int]) -> float:
    if not orbit_sizes:
        return float("nan")
    return 0.0


def splitting_score(
    candidate: RepresentationCandidate,
    holonomies: Sequence[Sequence[int]],
    reduction_threshold: float,
) -> dict:
    pre_values = [permutation_noncentrality_score(holonomy) for holonomy in holonomies]
    pre = float(np.mean(pre_values)) if pre_values else 0.0
    if candidate.representation_name == "orbit_representation":
        post_values = [_orbit_quotient_residual(holonomy, candidate.metadata.get("orbit_sizes", [])) for holonomy in holonomies]
    elif candidate.representation_name == "low_dimensional_permutation_subrepresentation":
        support = candidate.metadata.get("selected_orbit", [])
        post_values = [_restricted_residual(holonomy, support) for holonomy in holonomies]
    elif candidate.representation_name == "regular_representation" and candidate.representation_status == "constructed":
        post_values = [0.0 for _ in holonomies]
    else:
        post_values = pre_values
    post_arr = np.asarray([value for value in post_values if np.isfinite(value)], dtype=float)
    post = float(post_arr.mean()) if post_arr.size else float("nan")
    reduction = pre - post if np.isfinite(post) else float("nan")
    relative = reduction / max(pre, 1e-12) if np.isfinite(reduction) else float("nan")
    return {
        "pre_lift_connection_residual": pre,
        "post_lift_connection_residual": post,
        "holonomy_reduction": reduction,
        "relative_holonomy_reduction": relative,
        "projected_cycle_score": post,
        "noncentrality_after_lift": post,
        "split_success_flag": bool(np.isfinite(relative) and relative >= float(reduction_threshold)),
    }


def representation_row(candidate: RepresentationCandidate) -> dict:
    return {
        "representation_name": candidate.representation_name,
        "representation_dimension": candidate.representation_dimension,
        "is_faithful": candidate.is_faithful,
        "kernel_size": candidate.kernel_size,
        "orbit_count": candidate.orbit_count,
        "max_orbit_size": candidate.max_orbit_size,
        "estimated_splitting_index": candidate.estimated_splitting_index,
        "representation_status": candidate.representation_status,
        "construction_cost": candidate.construction_cost,
        "representation_metadata_json": __import__("json").dumps(candidate.metadata, sort_keys=True),
    }

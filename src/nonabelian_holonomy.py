"""Finite nonabelian holonomy utilities for permutation residuals."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterable, Sequence

import numpy as np

from src.finite_group_cohomology import (
    FinitePermutationGroup,
    Permutation,
    center,
    close_permutation_group,
    compose_permutations,
    element_order_histogram,
    identity_permutation,
    invert_permutation,
    permutation_order,
)


def lcm(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return abs(int(left) * int(right)) // gcd(int(left), int(right))


def lcm_many(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out = lcm(out, int(value))
    return int(out)


def close_group(generators: Iterable[Sequence[int]], max_group_order: int = 5000) -> FinitePermutationGroup:
    return close_permutation_group(generators, max_group_order=max_group_order)


def element_order(element: Sequence[int]) -> int:
    return permutation_order(element)


def group_exponent(group: FinitePermutationGroup, max_exact_order: int = 256) -> int | None:
    if group.truncated or group.order > max_exact_order:
        return None
    return lcm_many(permutation_order(element) for element in group.elements)


def commutator(group: FinitePermutationGroup, left: Permutation, right: Permutation) -> Permutation:
    return group.multiply(group.multiply(group.multiply(left, right), group.inverse(left)), group.inverse(right))


def is_abelian(group: FinitePermutationGroup, max_exact_order: int = 256) -> bool | None:
    if group.truncated or group.order > max_exact_order:
        return None
    for left in group.elements:
        for right in group.elements:
            if group.multiply(left, right) != group.multiply(right, left):
                return False
    return True


def conjugacy_classes(group: FinitePermutationGroup, max_exact_order: int = 256) -> list[tuple[Permutation, ...]] | None:
    if group.truncated or group.order > max_exact_order:
        return None
    unseen = set(group.elements)
    classes = []
    while unseen:
        base = next(iter(unseen))
        cls = set()
        for other in group.elements:
            cls.add(group.multiply(group.multiply(other, base), group.inverse(other)))
        classes.append(tuple(sorted(cls)))
        unseen -= cls
    return classes


def commutator_subgroup(group: FinitePermutationGroup, max_exact_order: int = 256) -> FinitePermutationGroup | None:
    if group.truncated or group.order > max_exact_order:
        return None
    generators = []
    for left in group.elements:
        for right in group.elements:
            value = commutator(group, left, right)
            if value != group.identity and value not in generators:
                generators.append(value)
    if not generators:
        return close_group([group.identity], max_group_order=2)
    return close_group(generators, max_group_order=max_exact_order)


def abelianization_size(group: FinitePermutationGroup, max_exact_order: int = 256) -> int | None:
    comm = commutator_subgroup(group, max_exact_order=max_exact_order)
    if comm is None or comm.order == 0:
        return None
    return int(group.order // comm.order)


def group_orbits(group: FinitePermutationGroup) -> list[tuple[int, ...]]:
    if group.degree <= 0:
        return []
    parent = list(range(group.degree))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for element in group.elements:
        for idx, image in enumerate(element):
            union(idx, int(image))
    buckets: dict[int, list[int]] = {}
    for idx in range(group.degree):
        buckets.setdefault(find(idx), []).append(idx)
    return [tuple(values) for values in buckets.values()]


def holonomy_order(holonomies: Iterable[Sequence[int]]) -> int:
    orders = []
    for holonomy in holonomies:
        try:
            orders.append(permutation_order(holonomy))
        except Exception:
            continue
    return lcm_many(orders) if orders else 1


def permutation_noncentrality_score(perm: Sequence[int]) -> float:
    arr = np.asarray(perm, dtype=int)
    if arr.size == 0:
        return 0.0
    fixed = float(np.mean(arr == np.arange(arr.size)))
    return float(np.sqrt(max(0.0, 1.0 - fixed**2)))


def noncentral_holonomy_score(holonomies: Iterable[Sequence[int]]) -> float:
    scores = [permutation_noncentrality_score(holonomy) for holonomy in holonomies]
    return float(np.mean(scores)) if scores else 0.0


def is_noncentral_holonomy(group: FinitePermutationGroup, holonomies: Iterable[Sequence[int]], max_exact_order: int = 256) -> bool:
    abelian = is_abelian(group, max_exact_order=max_exact_order)
    if abelian is False:
        return True
    return noncentral_holonomy_score(holonomies) > 1e-3


def small_quotients(group: FinitePermutationGroup, max_exact_order: int = 256) -> list[dict]:
    if group.truncated or group.order > max_exact_order:
        return [{"quotient_name": "not_computed_large_or_truncated", "quotient_order": None, "status": "skipped"}]
    rows = []
    ab_size = abelianization_size(group, max_exact_order=max_exact_order)
    if ab_size is not None:
        rows.append({"quotient_name": "abelianization", "quotient_order": int(ab_size), "status": "computed"})
    ctr = center(group)
    rows.append({"quotient_name": "center_quotient_order", "quotient_order": int(group.order // max(1, len(ctr))), "status": "computed"})
    return rows


@dataclass(frozen=True)
class HolonomyGroupSummary:
    group: FinitePermutationGroup
    group_status: str
    generator_count: int
    holonomy_order: int
    group_exponent: int | None
    is_abelian: bool | None
    center_size: int | None
    commutator_subgroup_size: int | None
    abelianization_size: int | None
    noncentral_holonomy_score: float
    orbit_sizes: tuple[int, ...]


def infer_holonomy_group(
    edge_transports: Iterable[Sequence[int]],
    triangle_holonomies: Iterable[Sequence[int]],
    max_group_order: int = 5000,
    max_generators: int = 12,
    max_exact_order: int = 256,
) -> HolonomyGroupSummary:
    generators: list[Permutation] = []
    for candidate in [*triangle_holonomies, *edge_transports]:
        try:
            perm = tuple(int(value) for value in candidate)
        except Exception:
            continue
        if not perm or sorted(perm) != list(range(len(perm))):
            continue
        if perm == identity_permutation(len(perm)):
            continue
        if perm not in generators:
            generators.append(perm)
        if len(generators) >= int(max_generators):
            break
    if not generators:
        generators = [identity_permutation(1)]
    group = close_group(generators, max_group_order=max_group_order)
    holonomies = [tuple(int(value) for value in holonomy) for holonomy in triangle_holonomies]
    exponent = group_exponent(group, max_exact_order=max_exact_order)
    abelian = is_abelian(group, max_exact_order=max_exact_order)
    ctr = center(group) if (not group.truncated and group.order <= max_exact_order) else None
    comm = commutator_subgroup(group, max_exact_order=max_exact_order)
    ab_size = abelianization_size(group, max_exact_order=max_exact_order)
    orbits = group_orbits(group) if group.order <= max_exact_order else []
    return HolonomyGroupSummary(
        group=group,
        group_status=group.closure_status,
        generator_count=len(generators),
        holonomy_order=holonomy_order(holonomies),
        group_exponent=exponent,
        is_abelian=abelian,
        center_size=len(ctr) if ctr is not None else None,
        commutator_subgroup_size=comm.order if comm is not None else None,
        abelianization_size=ab_size,
        noncentral_holonomy_score=noncentral_holonomy_score(holonomies),
        orbit_sizes=tuple(sorted((len(orbit) for orbit in orbits), reverse=True)),
    )


def element_order_histogram_json(group: FinitePermutationGroup, max_exact_order: int = 512) -> dict[int, int]:
    if group.order > max_exact_order:
        return {}
    return element_order_histogram(group)

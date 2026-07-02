"""Small finite permutation groups and cyclic-coefficient H^2 computations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd
from typing import Iterable, Sequence

import numpy as np


Permutation = tuple[int, ...]


@dataclass(frozen=True)
class FinitePermutationGroup:
    elements: tuple[Permutation, ...]
    generators: tuple[Permutation, ...]
    closure_status: str
    truncated: bool = False

    @property
    def identity(self) -> Permutation:
        return identity_permutation(len(self.elements[0])) if self.elements else ()

    @property
    def order(self) -> int:
        return len(self.elements)

    @property
    def degree(self) -> int:
        return len(self.elements[0]) if self.elements else 0

    def multiply(self, left: Permutation, right: Permutation) -> Permutation:
        return compose_permutations(left, right)

    def inverse(self, element: Permutation) -> Permutation:
        return invert_permutation(element)


@dataclass(frozen=True)
class H2Computation:
    coefficient_modulus: int
    exact: bool
    computation_method: str
    group_order: int
    h2_dimension: int | None
    h2_size: int | None
    z2_dimension: int | None
    b2_dimension: int | None
    class_orders: tuple[int, ...]
    skipped_reason: str = ""


def identity_permutation(degree: int) -> Permutation:
    return tuple(range(int(degree)))


def normalize_permutation(perm: Sequence[int]) -> Permutation:
    out = tuple(int(value) for value in perm)
    if sorted(out) != list(range(len(out))):
        raise ValueError("not a valid permutation")
    return out


def compose_permutations(left: Sequence[int], right: Sequence[int]) -> Permutation:
    """Return matrix-style composition P_left @ P_right."""

    lft = normalize_permutation(left)
    rgt = normalize_permutation(right)
    if len(lft) != len(rgt):
        raise ValueError("permutations must have the same degree")
    return tuple(rgt[idx] for idx in lft)


def invert_permutation(perm: Sequence[int]) -> Permutation:
    arr = normalize_permutation(perm)
    inv = [0] * len(arr)
    for idx, value in enumerate(arr):
        inv[value] = idx
    return tuple(inv)


def permutation_order(perm: Sequence[int]) -> int:
    arr = normalize_permutation(perm)
    visited = [False] * len(arr)
    order = 1
    for start in range(len(arr)):
        if visited[start]:
            continue
        cur = start
        length = 0
        while not visited[cur]:
            visited[cur] = True
            length += 1
            cur = arr[cur]
        if length:
            order = order * length // gcd(order, length)
    return int(order)


def close_permutation_group(
    generators: Iterable[Sequence[int]],
    max_group_order: int = 5000,
) -> FinitePermutationGroup:
    gens = tuple(normalize_permutation(generator) for generator in generators)
    if not gens:
        raise ValueError("at least one generator is required")
    degree = len(gens[0])
    if any(len(gen) != degree for gen in gens):
        raise ValueError("all generators must have the same degree")
    identity = identity_permutation(degree)
    all_gens = tuple(dict.fromkeys((*gens, *(invert_permutation(gen) for gen in gens), identity)))
    seen = {identity}
    queue = [identity]
    while queue:
        current = queue.pop(0)
        for gen in all_gens:
            for nxt in (compose_permutations(current, gen), compose_permutations(gen, current)):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if len(seen) > max_group_order:
                    return FinitePermutationGroup(
                        elements=tuple(sorted(seen)),
                        generators=gens,
                        closure_status="truncated_max_group_order",
                        truncated=True,
                    )
                queue.append(nxt)
    return FinitePermutationGroup(
        elements=tuple(sorted(seen)),
        generators=gens,
        closure_status="exact_closure",
        truncated=False,
    )


def cyclic_group(order: int) -> FinitePermutationGroup:
    generator = tuple([*range(1, int(order)), 0])
    return close_permutation_group([generator], max_group_order=max(2, int(order) + 1))


def klein_four_group() -> FinitePermutationGroup:
    return close_permutation_group([(1, 0, 3, 2), (2, 3, 0, 1)], max_group_order=8)


def symmetric_group_3() -> FinitePermutationGroup:
    return close_permutation_group([(1, 0, 2), (1, 2, 0)], max_group_order=8)


def dihedral_group_4() -> FinitePermutationGroup:
    return close_permutation_group([(1, 2, 3, 0), (3, 2, 1, 0)], max_group_order=16)


def center(group: FinitePermutationGroup) -> tuple[Permutation, ...]:
    out = []
    for candidate in group.elements:
        if all(group.multiply(candidate, other) == group.multiply(other, candidate) for other in group.elements):
            out.append(candidate)
    return tuple(out)


def element_order_histogram(group: FinitePermutationGroup) -> dict[int, int]:
    hist: dict[int, int] = {}
    for element in group.elements:
        order = permutation_order(element)
        hist[order] = hist.get(order, 0) + 1
    return dict(sorted(hist.items()))


def _rank_mod(matrix: np.ndarray, modulus: int) -> int:
    arr = np.asarray(matrix, dtype=int) % int(modulus)
    if arr.size == 0:
        return 0
    rows, cols = arr.shape
    rank = 0
    pivot_col = 0
    while rank < rows and pivot_col < cols:
        pivot = None
        for row in range(rank, rows):
            if arr[row, pivot_col] % modulus != 0:
                pivot = row
                break
        if pivot is None:
            pivot_col += 1
            continue
        if pivot != rank:
            arr[[rank, pivot]] = arr[[pivot, rank]]
        inv = pow(int(arr[rank, pivot_col]), -1, int(modulus))
        arr[rank] = (arr[rank] * inv) % modulus
        for row in range(rows):
            if row == rank:
                continue
            factor = arr[row, pivot_col] % modulus
            if factor:
                arr[row] = (arr[row] - factor * arr[rank]) % modulus
        rank += 1
        pivot_col += 1
    return int(rank)


def _normalized_pairs(elements: Sequence[Permutation], identity: Permutation) -> list[tuple[Permutation, Permutation]]:
    nonidentity = [element for element in elements if element != identity]
    return [(left, right) for left in nonidentity for right in nonidentity]


def _cocycle_matrix(group: FinitePermutationGroup, modulus: int) -> tuple[np.ndarray, list[tuple[Permutation, Permutation]]]:
    elements = list(group.elements)
    identity = group.identity
    pairs = _normalized_pairs(elements, identity)
    pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
    rows = []
    for g, h, k in product(elements, elements, elements):
        coeff = np.zeros(len(pairs), dtype=int)
        terms = [
            (1, (g, h)),
            (1, (group.multiply(g, h), k)),
            (-1, (h, k)),
            (-1, (g, group.multiply(h, k))),
        ]
        for sign, pair in terms:
            idx = pair_to_idx.get(pair)
            if idx is not None:
                coeff[idx] = (coeff[idx] + sign) % modulus
        if np.any(coeff % modulus):
            rows.append(coeff % modulus)
    if not rows:
        return np.zeros((0, len(pairs)), dtype=int), pairs
    return np.stack(rows, axis=0) % modulus, pairs


def _coboundary_matrix(
    group: FinitePermutationGroup,
    pairs: Sequence[tuple[Permutation, Permutation]],
    modulus: int,
) -> np.ndarray:
    elements = [element for element in group.elements if element != group.identity]
    element_to_col = {element: idx for idx, element in enumerate(elements)}
    matrix = np.zeros((len(pairs), len(elements)), dtype=int)
    for row, (g, h) in enumerate(pairs):
        gh = group.multiply(g, h)
        for sign, element in [(1, g), (1, h), (-1, gh)]:
            col = element_to_col.get(element)
            if col is not None:
                matrix[row, col] = (matrix[row, col] + sign) % modulus
    return matrix % modulus


def compute_h2_cyclic_coefficients(
    group: FinitePermutationGroup,
    coefficient_modulus: int = 2,
    max_exact_group_order: int = 32,
) -> H2Computation:
    """Compute normalized H^2(G,Z/nZ) for prime n by modular linear algebra."""

    modulus = int(coefficient_modulus)
    if modulus < 2:
        raise ValueError("coefficient modulus must be at least 2")
    if any(modulus % factor == 0 for factor in range(2, int(np.sqrt(modulus)) + 1)):
        return H2Computation(
            coefficient_modulus=modulus,
            exact=False,
            computation_method="skipped_composite_modulus",
            group_order=group.order,
            h2_dimension=None,
            h2_size=None,
            z2_dimension=None,
            b2_dimension=None,
            class_orders=(),
            skipped_reason="modular linear algebra implementation is exact only for prime modulus",
        )
    if group.truncated:
        return H2Computation(
            coefficient_modulus=modulus,
            exact=False,
            computation_method=group.closure_status,
            group_order=group.order,
            h2_dimension=None,
            h2_size=None,
            z2_dimension=None,
            b2_dimension=None,
            class_orders=(),
            skipped_reason="group closure exceeded max_group_order",
        )
    if group.order > max_exact_group_order:
        return H2Computation(
            coefficient_modulus=modulus,
            exact=False,
            computation_method="skipped_large_group",
            group_order=group.order,
            h2_dimension=None,
            h2_size=None,
            z2_dimension=None,
            b2_dimension=None,
            class_orders=(),
            skipped_reason=f"group order {group.order} exceeds exact cohomology limit {max_exact_group_order}",
        )
    d2, pairs = _cocycle_matrix(group, modulus)
    z2_dim = len(pairs) - _rank_mod(d2, modulus)
    coboundary = _coboundary_matrix(group, pairs, modulus)
    b2_dim = _rank_mod(coboundary, modulus)
    h2_dim = max(0, int(z2_dim - b2_dim))
    return H2Computation(
        coefficient_modulus=modulus,
        exact=True,
        computation_method="normalized_cochains_mod_prime",
        group_order=group.order,
        h2_dimension=h2_dim,
        h2_size=int(modulus**h2_dim),
        z2_dimension=int(z2_dim),
        b2_dimension=int(b2_dim),
        class_orders=tuple([modulus] * h2_dim),
    )

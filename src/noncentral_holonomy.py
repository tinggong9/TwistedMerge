"""Utilities separating central projective twists from noncentral holonomy.

The finite-index TwistedMerge++ branch is only meant for scalar/central
projective residuals.  This module provides small auditable examples and
diagnostics for the adjacent, but different, case of noncentral permutation or
matrix holonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, isfinite
from typing import Iterable

import numpy as np

from .finite_index_twists import clock_matrix, root_of_unity, shift_matrix, torsion_order


Permutation = tuple[int, ...]


@dataclass(frozen=True)
class ScalarPhaseDetection:
    scalar: complex
    phase: complex | None
    centrality_score: float
    nearest_root_q: int | None
    nearest_root_exponent: int | None
    detected_order_d: int | None
    nearest_root_phase: complex | None
    phase_residual: float | None
    is_scalar_finite_index_candidate: bool


@dataclass(frozen=True)
class HolonomyClassification:
    classification: str
    structure_group: str
    commutator_or_defect_type: str
    centrality_score: float
    phase_residual: float | None
    detected_order_d: int | None
    is_scalar_finite_index_candidate: bool
    is_noncentral_holonomy: bool
    possible_resolution: str
    brauer_interpretation: str


@dataclass(frozen=True)
class RegularBranchLift:
    label: str
    group_size: int
    group_elements: tuple[Permutation, ...]
    action_matrices: dict[Permutation, np.ndarray]
    extra_capacity: bool
    brauer_projective: bool
    finite_index_projective: bool


def as_permutation(perm: Iterable[int]) -> Permutation:
    out = tuple(int(x) for x in perm)
    if sorted(out) != list(range(len(out))):
        raise ValueError(f"not a valid permutation: {out}")
    return out


def identity_permutation(size: int) -> Permutation:
    return tuple(range(size))


def compose_permutations(p: Iterable[int], q: Iterable[int]) -> Permutation:
    """Compose permutations in the same convention as permutation matrices.

    With row-based permutation matrices from `permutation_matrix`, this returns
    the permutation represented by `P_p @ P_q`.
    """

    left = as_permutation(p)
    right = as_permutation(q)
    if len(left) != len(right):
        raise ValueError("permutations must have the same size")
    return tuple(right[i] for i in left)


def invert_permutation(perm: Iterable[int]) -> Permutation:
    value = as_permutation(perm)
    inv = [0] * len(value)
    for idx, target in enumerate(value):
        inv[target] = idx
    return tuple(inv)


def permutation_commutator(p: Iterable[int], q: Iterable[int]) -> Permutation:
    """Return [p, q] = p q p^{-1} q^{-1}."""

    left = as_permutation(p)
    right = as_permutation(q)
    return compose_permutations(
        compose_permutations(compose_permutations(left, right), invert_permutation(left)),
        invert_permutation(right),
    )


def generate_subgroup(generators: Iterable[Iterable[int]], max_size: int | None = None) -> tuple[Permutation, ...]:
    gens = [as_permutation(generator) for generator in generators]
    if not gens:
        return (identity_permutation(0),)
    size = len(gens[0])
    if any(len(generator) != size for generator in gens):
        raise ValueError("all generators must have the same size")
    limit = max_size or max(1, factorial(size))
    identity = identity_permutation(size)
    group: set[Permutation] = {identity}
    frontier: list[Permutation] = [identity]
    moves = gens + [invert_permutation(generator) for generator in gens]
    while frontier:
        current = frontier.pop()
        for move in moves:
            for candidate in (
                compose_permutations(current, move),
                compose_permutations(move, current),
            ):
                if candidate not in group:
                    group.add(candidate)
                    frontier.append(candidate)
                    if len(group) > limit:
                        raise ValueError("generated subgroup exceeded max_size")
    return tuple(sorted(group))


def permutation_is_central_in_generated_subgroup(
    perm: Iterable[int],
    generators: Iterable[Iterable[int]],
) -> bool:
    value = as_permutation(perm)
    group = generate_subgroup(generators)
    if any(len(element) != len(value) for element in group):
        raise ValueError("permutation and subgroup have different sizes")
    return all(
        compose_permutations(value, element) == compose_permutations(element, value)
        for element in group
    )


def fixed_point_fraction(perm: Iterable[int]) -> float:
    value = as_permutation(perm)
    if not value:
        return float("nan")
    return float(np.mean(np.asarray(value) == np.arange(len(value))))


def cycle_type(perm: Iterable[int]) -> tuple[int, ...]:
    value = as_permutation(perm)
    visited = [False] * len(value)
    lengths: list[int] = []
    for start in range(len(value)):
        if visited[start]:
            continue
        current = start
        length = 0
        while not visited[current]:
            visited[current] = True
            length += 1
            current = value[current]
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def permutation_to_matrix(perm: Iterable[int]) -> np.ndarray:
    value = as_permutation(perm)
    matrix = np.zeros((len(value), len(value)), dtype=complex)
    matrix[np.arange(len(value)), list(value)] = 1.0
    return matrix


def matrix_centrality_score(matrix: np.ndarray) -> tuple[float, complex]:
    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    rank = arr.shape[0]
    scalar = complex(np.trace(arr) / max(rank, 1))
    target = scalar * np.eye(rank, dtype=complex)
    denom = max(float(np.linalg.norm(np.eye(rank), ord="fro")), 1e-12)
    return float(np.linalg.norm(arr - target, ord="fro") / denom), scalar


def detect_scalar_phase(
    matrix: np.ndarray,
    max_order: int = 12,
    centrality_tolerance: float = 1e-6,
    phase_tolerance: float = 1e-6,
) -> ScalarPhaseDetection:
    centrality, scalar = matrix_centrality_score(matrix)
    if abs(scalar) <= 1e-12 or not isfinite(scalar.real) or not isfinite(scalar.imag):
        return ScalarPhaseDetection(
            scalar=scalar,
            phase=None,
            centrality_score=centrality,
            nearest_root_q=None,
            nearest_root_exponent=None,
            detected_order_d=None,
            nearest_root_phase=None,
            phase_residual=None,
            is_scalar_finite_index_candidate=False,
        )
    phase = scalar / abs(scalar)
    best = (abs(phase - 1.0), 1, 0, 1, complex(1.0))
    for q in range(2, max_order + 1):
        for exponent in range(q):
            root = root_of_unity(q, exponent)
            order = torsion_order(q, exponent)
            residual = abs(phase - root)
            if residual < best[0] - 1e-15 or (
                abs(residual - best[0]) <= 1e-15 and order < best[3]
            ):
                best = (residual, q, exponent, order, root)
    residual, q, exponent, order, root = best
    candidate = centrality <= centrality_tolerance and residual <= phase_tolerance and order > 1
    return ScalarPhaseDetection(
        scalar=scalar,
        phase=phase,
        centrality_score=centrality,
        nearest_root_q=q,
        nearest_root_exponent=exponent,
        detected_order_d=order,
        nearest_root_phase=root,
        phase_residual=float(residual),
        is_scalar_finite_index_candidate=bool(candidate),
    )


def matrix_commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    left = np.asarray(A, dtype=complex)
    right = np.asarray(B, dtype=complex)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("A and B must be square matrices of the same shape")
    return left @ right @ np.linalg.inv(left) @ np.linalg.inv(right)


def clock_shift_projective_example(order: int) -> dict[str, object]:
    zeta = root_of_unity(order, 1)
    U = clock_matrix(order, zeta)
    V = shift_matrix(order)
    defect = matrix_commutator(U, V)
    detection = detect_scalar_phase(defect, max_order=max(12, order))
    return {
        "order": order,
        "zeta": zeta,
        "U": U,
        "V": V,
        "commutator": defect,
        "detection": detection,
    }


def s3_noncentral_permutation_example() -> dict[str, object]:
    p = as_permutation((1, 0, 2))  # (12), one-indexed notation.
    q = as_permutation((0, 2, 1))  # (23), one-indexed notation.
    comm = permutation_commutator(p, q)
    group = generate_subgroup([p, q])
    central = permutation_is_central_in_generated_subgroup(comm, [p, q])
    matrix = permutation_to_matrix(comm)
    detection = detect_scalar_phase(matrix)
    return {
        "p": p,
        "q": q,
        "commutator": comm,
        "group": group,
        "is_central": central,
        "matrix": matrix,
        "detection": detection,
    }


def noncentral_matrix_example() -> dict[str, object]:
    A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
    B = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=complex)
    defect = matrix_commutator(A, B)
    detection = detect_scalar_phase(defect)
    return {
        "A": A,
        "B": B,
        "commutator": defect,
        "detection": detection,
    }


def classify_matrix_defect(
    matrix: np.ndarray,
    structure_group: str,
    max_order: int = 12,
    centrality_tolerance: float = 1e-6,
    phase_tolerance: float = 1e-6,
) -> HolonomyClassification:
    detection = detect_scalar_phase(
        matrix,
        max_order=max_order,
        centrality_tolerance=centrality_tolerance,
        phase_tolerance=phase_tolerance,
    )
    if detection.is_scalar_finite_index_candidate:
        return HolonomyClassification(
            classification="central_finite_index_projective",
            structure_group=structure_group,
            commutator_or_defect_type="scalar_root_of_unity",
            centrality_score=detection.centrality_score,
            phase_residual=detection.phase_residual,
            detected_order_d=detection.detected_order_d,
            is_scalar_finite_index_candidate=True,
            is_noncentral_holonomy=False,
            possible_resolution="central_projective_lift",
            brauer_interpretation="central_brauer_projective_candidate",
        )
    if detection.centrality_score <= centrality_tolerance:
        return HolonomyClassification(
            classification="scalar_trivial_or_no_finite_index",
            structure_group=structure_group,
            commutator_or_defect_type="central_scalar",
            centrality_score=detection.centrality_score,
            phase_residual=detection.phase_residual,
            detected_order_d=detection.detected_order_d,
            is_scalar_finite_index_candidate=False,
            is_noncentral_holonomy=False,
            possible_resolution="c2m3_synchronization",
            brauer_interpretation="trivial_or_no_scalar_class",
        )
    if structure_group.startswith("S_") or "permutation" in structure_group:
        classification = "noncentral_permutation_holonomy"
        possible_resolution = "c2m3_synchronization"
    else:
        classification = "noncentral_matrix_holonomy"
        possible_resolution = "unknown"
    return HolonomyClassification(
        classification=classification,
        structure_group=structure_group,
        commutator_or_defect_type="noncentral",
        centrality_score=detection.centrality_score,
        phase_residual=detection.phase_residual,
        detected_order_d=detection.detected_order_d,
        is_scalar_finite_index_candidate=False,
        is_noncentral_holonomy=True,
        possible_resolution=possible_resolution,
        brauer_interpretation="not_brauer_noncentral",
    )


def classify_permutation_defect(
    perm: Iterable[int],
    max_order: int = 12,
    centrality_tolerance: float = 1e-6,
    phase_tolerance: float = 1e-6,
) -> HolonomyClassification:
    value = as_permutation(perm)
    return classify_matrix_defect(
        permutation_to_matrix(value),
        structure_group=f"S_{len(value)} permutation",
        max_order=max_order,
        centrality_tolerance=centrality_tolerance,
        phase_tolerance=phase_tolerance,
    )


def classify_mnist_residual_row(
    row: dict,
    centrality_tolerance: float = 1e-6,
    phase_tolerance: float = 1e-6,
) -> HolonomyClassification:
    centrality = float(row.get("centrality_score", float("nan")))
    phase_residual = float(row.get("phase_residual", float("nan")))
    detected_raw = row.get("detected_order_d", "")
    try:
        detected_order = int(float(detected_raw))
    except (TypeError, ValueError):
        detected_order = None
    scalar_candidate = (
        np.isfinite(centrality)
        and np.isfinite(phase_residual)
        and centrality <= centrality_tolerance
        and phase_residual <= phase_tolerance
        and detected_order is not None
        and detected_order > 1
    )
    if scalar_candidate:
        return HolonomyClassification(
            classification="central_finite_index_projective",
            structure_group="S_h permutation",
            commutator_or_defect_type="scalar_root_of_unity",
            centrality_score=centrality,
            phase_residual=phase_residual,
            detected_order_d=detected_order,
            is_scalar_finite_index_candidate=True,
            is_noncentral_holonomy=False,
            possible_resolution="central_projective_lift",
            brauer_interpretation="central_brauer_projective_candidate",
        )
    return HolonomyClassification(
        classification="noncentral_permutation_holonomy",
        structure_group="S_h permutation",
        commutator_or_defect_type="noncentral_permutation_summary",
        centrality_score=centrality,
        phase_residual=phase_residual if np.isfinite(phase_residual) else None,
        detected_order_d=detected_order,
        is_scalar_finite_index_candidate=False,
        is_noncentral_holonomy=True,
        possible_resolution="c2m3_synchronization",
        brauer_interpretation="not_brauer_noncentral",
    )


def regular_branch_lift(generators: Iterable[Iterable[int]]) -> RegularBranchLift:
    gens = [as_permutation(generator) for generator in generators]
    group = generate_subgroup(gens)
    index = {element: idx for idx, element in enumerate(group)}
    action_matrices: dict[Permutation, np.ndarray] = {}
    for generator in gens:
        matrix = np.zeros((len(group), len(group)), dtype=complex)
        for element in group:
            image = compose_permutations(generator, element)
            matrix[index[element], index[image]] = 1.0
        action_matrices[generator] = matrix
    return RegularBranchLift(
        label="noncentral_regular_branch_lift_extra_capacity",
        group_size=len(group),
        group_elements=group,
        action_matrices=action_matrices,
        extra_capacity=True,
        brauer_projective=False,
        finite_index_projective=False,
    )

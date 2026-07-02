"""p-primary quotient cochain peeling utilities.

These helpers implement the mathematical peeling step used by the v2 smoke
test: solve edge labels over C_p, lift them to permutation representatives
from observed holonomies, and apply no-lift pairwise map corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

import numpy as np

from src.primary_holonomy import triangle_relation_from_perms


@dataclass(frozen=True)
class EdgeCochainSolution:
    p: int
    sign: int
    solved_exact: bool
    edge_labels: dict[tuple[int, int], int]
    quotient_residual_before: float
    quotient_residual_after: float
    n_equations: int
    n_variables: int
    rank: int
    solve_status: str


@dataclass(frozen=True)
class RepresentativeChoice:
    label: int
    representative: np.ndarray | None
    representative_label: int | None
    generator_label: int | None
    disagreement_from_identity: float
    status: str


@dataclass(frozen=True)
class CorrectionResult:
    corrections: dict[tuple[int, int], np.ndarray]
    corrected: dict[tuple[int, int], np.ndarray]
    edge_labels: dict[tuple[int, int], int]
    representative_choices: dict[tuple[int, int], RepresentativeChoice]
    solution: EdgeCochainSolution
    edge_cochain_solve_status: str
    representative_selection_status: str
    implemented: bool
    inverse_consistency_ok: bool


def is_valid_permutation(perm: Iterable[int]) -> bool:
    values = [int(item) for item in perm]
    return bool(values) and sorted(values) == list(range(len(values)))


def invert_perm(perm: Iterable[int]) -> np.ndarray:
    perm_arr = np.asarray(tuple(int(item) for item in perm), dtype=int)
    inv = np.empty_like(perm_arr)
    inv[perm_arr] = np.arange(len(perm_arr))
    return inv


def compose_perm(p_ab: Iterable[int], p_bc: Iterable[int]) -> np.ndarray:
    left = np.asarray(tuple(int(item) for item in p_ab), dtype=int)
    right = np.asarray(tuple(int(item) for item in p_bc), dtype=int)
    return right[left]


def permutation_disagreement(observed: Iterable[int], implied: Iterable[int]) -> float:
    observed_arr = np.asarray(tuple(int(item) for item in observed), dtype=int)
    implied_arr = np.asarray(tuple(int(item) for item in implied), dtype=int)
    if len(observed_arr) == 0:
        return 0.0
    return float(np.mean(observed_arr != implied_arr))


def permutation_power(perm: Iterable[int], exponent: int) -> np.ndarray:
    base = np.asarray(tuple(int(item) for item in perm), dtype=int)
    width = len(base)
    identity = np.arange(width, dtype=int)
    exp = int(exponent)
    if exp == 0:
        return identity.copy()
    if exp < 0:
        base = invert_perm(base)
        exp = -exp
    out = identity.copy()
    for _ in range(exp):
        out = compose_perm(out, base)
    return out


def triangle_defects_from_pairwise(pairwise: dict[tuple[int, int], np.ndarray], n_models: int) -> dict[tuple[int, int, int], np.ndarray]:
    defects: dict[tuple[int, int, int], np.ndarray] = {}
    for i, j, k in combinations(range(int(n_models)), 3):
        defects[(i, j, k)] = compose_perm(compose_perm(pairwise[(i, j)], pairwise[(j, k)]), pairwise[(k, i)])
    return defects


def relations_from_pairwise(pairwise: dict[tuple[int, int], np.ndarray], n_models: int) -> tuple:
    relations = []
    for (i, j, k), hol in triangle_defects_from_pairwise(pairwise, n_models).items():
        relations.append(triangle_relation_from_perms(pairwise[(i, j)], pairwise[(j, k)], pairwise[(k, i)], hol))
    return tuple(relations)


def quotient_label_from_fit(perm: Iterable[int], fit, p: int) -> int:
    if fit is None:
        return 0
    key = tuple(int(item) for item in perm)
    return int(fit.assignment.get(key, 0)) % int(p)


def quotient_defect_labels_from_pairwise(
    pairwise: dict[tuple[int, int], np.ndarray],
    fit,
    n_models: int,
    prime: int,
) -> dict[tuple[int, int, int], int]:
    defects = triangle_defects_from_pairwise(pairwise, n_models)
    return {tri: quotient_label_from_fit(defect, fit, prime) for tri, defect in defects.items()}


def quotient_residual_from_labels(labels: dict[tuple[int, int, int], int], p: int) -> float:
    if not labels:
        return 0.0
    return float(np.mean([int(value) % int(p) != 0 for value in labels.values()]))


def _edge_variable_index(n_models: int) -> dict[tuple[int, int], int]:
    return {edge: idx for idx, edge in enumerate(combinations(range(int(n_models)), 2))}


def oriented_edge_value(edge_labels: dict[tuple[int, int], int], i: int, j: int, p: int) -> int:
    if i == j:
        return 0
    if i < j:
        return int(edge_labels.get((i, j), 0)) % int(p)
    return (-int(edge_labels.get((j, i), 0))) % int(p)


def _rref_solve_mod_p(matrix: list[list[int]], rhs: list[int], p: int) -> tuple[bool, list[int], int]:
    p = int(p)
    if not matrix:
        return True, [], 0
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix[0] else 0
    aug = [[int(value) % p for value in row] + [int(rhs[idx]) % p] for idx, row in enumerate(matrix)]
    row = 0
    pivots: list[int] = []
    for col in range(n_cols):
        pivot = next((candidate for candidate in range(row, n_rows) if aug[candidate][col] % p), None)
        if pivot is None:
            continue
        aug[row], aug[pivot] = aug[pivot], aug[row]
        inv = pow(int(aug[row][col]) % p, -1, p)
        aug[row] = [(value * inv) % p for value in aug[row]]
        for other in range(n_rows):
            if other == row:
                continue
            factor = aug[other][col] % p
            if factor:
                aug[other] = [(aug[other][idx] - factor * aug[row][idx]) % p for idx in range(n_cols + 1)]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break
    inconsistent = any(all(aug[r][c] % p == 0 for c in range(n_cols)) and aug[r][-1] % p != 0 for r in range(n_rows))
    solution = [0 for _ in range(n_cols)]
    if not inconsistent:
        for pivot_row, pivot_col in enumerate(pivots):
            solution[pivot_col] = int(aug[pivot_row][-1]) % p
    return not inconsistent, solution, len(pivots)


def _least_violation_solution_mod_p(
    matrix: list[list[int]],
    rhs: list[int],
    variables: dict[tuple[int, int], int],
    p: int,
) -> tuple[dict[tuple[int, int], int], float]:
    if not variables:
        return {}, 0.0
    if len(variables) > 8 or int(p) ** len(variables) > 1_000_000:
        return {edge: 0 for edge in variables}, math.inf
    best_values = None
    best_residual = math.inf
    for values in product(range(int(p)), repeat=len(variables)):
        violations = 0
        for row, target in zip(matrix, rhs):
            lhs = sum(int(coeff) * int(value) for coeff, value in zip(row, values)) % int(p)
            violations += int(lhs != int(target) % int(p))
        residual = violations / max(1, len(matrix))
        if residual < best_residual:
            best_residual = float(residual)
            best_values = values
            if residual == 0.0:
                break
    if best_values is None:
        return {edge: 0 for edge in variables}, math.inf
    return {edge: int(best_values[idx]) % int(p) for edge, idx in variables.items()}, float(best_residual)


def solve_edge_cochain_mod_p(
    triangle_defect_labels: dict[tuple[int, int, int], int],
    n_models: int,
    p: int,
    sign: int = 1,
) -> EdgeCochainSolution:
    """Solve d(edge_label) = sign * triangle_defect_label over F_p."""

    p = int(p)
    variables = _edge_variable_index(n_models)
    rows: list[list[int]] = []
    rhs: list[int] = []
    for i, j, k in sorted(triangle_defect_labels):
        row = [0 for _ in variables]
        for a, b in ((i, j), (j, k), (k, i)):
            if a == b:
                continue
            if a < b:
                row[variables[(a, b)]] = (row[variables[(a, b)]] + 1) % p
            else:
                row[variables[(b, a)]] = (row[variables[(b, a)]] - 1) % p
        rows.append(row)
        rhs.append((int(sign) * int(triangle_defect_labels[(i, j, k)])) % p)
    exact, vector, rank = _rref_solve_mod_p(rows, rhs, p)
    if exact:
        edge_labels = {edge: int(vector[idx]) % p for edge, idx in variables.items()} if vector else {edge: 0 for edge in variables}
    else:
        edge_labels, _ = _least_violation_solution_mod_p(rows, rhs, variables, p)
    residuals = []
    for (i, j, k), label in sorted(triangle_defect_labels.items()):
        lhs = (
            oriented_edge_value(edge_labels, i, j, p)
            + oriented_edge_value(edge_labels, j, k, p)
            + oriented_edge_value(edge_labels, k, i, p)
        ) % p
        residuals.append((lhs - int(sign) * int(label)) % p)
    residual_after = float(np.mean([value % p != 0 for value in residuals])) if residuals else 0.0
    residual_before = quotient_residual_from_labels(triangle_defect_labels, p)
    solved_exact = bool(exact and residual_after <= 0.0)
    status = "exact_quotient_cochain_solve" if solved_exact else "quotient_cochain_inconsistent"
    return EdgeCochainSolution(
        p=p,
        sign=int(sign),
        solved_exact=solved_exact,
        edge_labels=edge_labels,
        quotient_residual_before=float(residual_before),
        quotient_residual_after=float(residual_after),
        n_equations=int(len(rows)),
        n_variables=int(len(variables)),
        rank=int(rank),
        solve_status=f"{status}_sign_{int(sign)}",
    )


def solve_best_edge_cochain_mod_p(
    triangle_defect_labels: dict[tuple[int, int, int], int],
    n_models: int,
    p: int,
) -> EdgeCochainSolution:
    candidates = [
        solve_edge_cochain_mod_p(triangle_defect_labels, n_models, p, sign=1),
        solve_edge_cochain_mod_p(triangle_defect_labels, n_models, p, sign=-1),
    ]
    candidates.sort(key=lambda item: (not item.solved_exact, item.quotient_residual_after, 0 if item.sign == 1 else 1))
    return candidates[0]


def representative_for_cp_label(
    label: int,
    fit,
    observed_holonomies: Iterable[np.ndarray],
    width: int,
    p: int,
) -> RepresentativeChoice:
    p = int(p)
    label = int(label) % p
    identity = np.arange(width, dtype=int)
    if label == 0:
        return RepresentativeChoice(
            label=0,
            representative=identity.copy(),
            representative_label=0,
            generator_label=0,
            disagreement_from_identity=0.0,
            status="identity_representative",
        )
    best: RepresentativeChoice | None = None
    for holonomy in observed_holonomies:
        generator = np.asarray(holonomy, dtype=int)
        if not is_valid_permutation(generator):
            continue
        generator_label = quotient_label_from_fit(generator, fit, p)
        if generator_label % p == 0:
            continue
        try:
            inv_label = pow(generator_label, -1, p)
        except ValueError:
            continue
        exponent = (inv_label * label) % p
        representative = permutation_power(generator, exponent)
        rep_label = (generator_label * exponent) % p
        if rep_label != label:
            continue
        disagreement = permutation_disagreement(representative, identity)
        choice = RepresentativeChoice(
            label=label,
            representative=representative,
            representative_label=int(rep_label),
            generator_label=int(generator_label),
            disagreement_from_identity=float(disagreement),
            status="observed_holonomy_power_representative",
        )
        if best is None or choice.disagreement_from_identity < best.disagreement_from_identity:
            best = choice
    if best is None:
        return RepresentativeChoice(
            label=label,
            representative=None,
            representative_label=None,
            generator_label=None,
            disagreement_from_identity=float("nan"),
            status="no_representative_correction_available",
        )
    return best


def lift_cp_edge_labels_to_permutations(
    edge_labels: dict[tuple[int, int], int],
    fit,
    observed_holonomies: Iterable[np.ndarray],
    width: int,
    p: int,
    n_models: int,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], RepresentativeChoice], str]:
    identity = np.arange(width, dtype=int)
    representatives = {(idx, idx): identity.copy() for idx in range(int(n_models))}
    choices: dict[tuple[int, int], RepresentativeChoice] = {}
    statuses = []
    for i, j in product(range(int(n_models)), repeat=2):
        if i == j:
            choice = RepresentativeChoice(0, identity.copy(), 0, 0, 0.0, "identity_representative")
        else:
            choice = representative_for_cp_label(edge_labels[(i, j)], fit, observed_holonomies, width, p)
        choices[(i, j)] = choice
        statuses.append(choice.status)
        if choice.representative is not None:
            representatives[(i, j)] = choice.representative.copy()
    if any(status == "no_representative_correction_available" for status in statuses):
        for i, j in product(range(int(n_models)), repeat=2):
            representatives.setdefault((i, j), identity.copy())
        return representatives, choices, "no_representative_correction_available"
    return representatives, choices, "representative_correction_available"


def apply_edge_label_corrections(
    pairwise: dict[tuple[int, int], np.ndarray],
    representatives: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    return {
        edge: compose_perm(invert_perm(representatives[edge]), np.asarray(pairwise[edge], dtype=int))
        for edge in pairwise
    }


def inverse_consistency_ok(pairwise: dict[tuple[int, int], np.ndarray], n_models: int) -> bool:
    for i, j in product(range(int(n_models)), repeat=2):
        if not np.array_equal(pairwise[(j, i)], invert_perm(pairwise[(i, j)])):
            return False
    return True


def solve_and_correct_pairwise(
    pairwise: dict[tuple[int, int], np.ndarray],
    fit,
    n_models: int,
    prime: int,
) -> CorrectionResult:
    width = len(next(iter(pairwise.values())))
    identity = np.arange(width, dtype=int)
    defects = triangle_defects_from_pairwise(pairwise, n_models)
    labels = {tri: quotient_label_from_fit(defect, fit, prime) for tri, defect in defects.items()}
    solution = solve_best_edge_cochain_mod_p(labels, n_models, prime)
    edge_labels: dict[tuple[int, int], int] = {}
    representative_choices: dict[tuple[int, int], RepresentativeChoice] = {}
    for i, j in product(range(int(n_models)), repeat=2):
        if i == j:
            edge_labels[(i, j)] = 0
            representative_choices[(i, j)] = RepresentativeChoice(0, identity.copy(), 0, 0, 0.0, "identity_representative")
            continue
        edge_labels[(i, j)] = oriented_edge_value(solution.edge_labels, i, j, prime)
    if not solution.solved_exact:
        corrections = {(idx, idx): identity.copy() for idx in range(int(n_models))}
        corrected = {edge: value.copy() for edge, value in pairwise.items()}
        for i, j in product(range(int(n_models)), repeat=2):
            corrections[(i, j)] = identity.copy()
            representative_choices.setdefault(
                (i, j),
                RepresentativeChoice(
                    label=edge_labels.get((i, j), 0),
                    representative=identity.copy(),
                    representative_label=0,
                    generator_label=0,
                    disagreement_from_identity=0.0,
                    status=solution.solve_status,
                ),
            )
        return CorrectionResult(
            corrections=corrections,
            corrected=corrected,
            edge_labels=edge_labels,
            representative_choices=representative_choices,
            solution=solution,
            edge_cochain_solve_status=solution.solve_status,
            representative_selection_status="quotient_cochain_inconsistent",
            implemented=False,
            inverse_consistency_ok=inverse_consistency_ok(corrected, n_models),
        )

    corrections, representative_choices, representative_status = lift_cp_edge_labels_to_permutations(
        edge_labels,
        fit,
        defects.values(),
        width,
        prime,
        n_models,
    )
    if representative_status == "no_representative_correction_available":
        corrected = {edge: value.copy() for edge, value in pairwise.items()}
        return CorrectionResult(
            corrections=corrections,
            corrected=corrected,
            edge_labels=edge_labels,
            representative_choices=representative_choices,
            solution=solution,
            edge_cochain_solve_status=solution.solve_status,
            representative_selection_status=representative_status,
            implemented=False,
            inverse_consistency_ok=inverse_consistency_ok(corrected, n_models),
        )

    corrected = apply_edge_label_corrections(pairwise, corrections)
    return CorrectionResult(
        corrections=corrections,
        corrected=corrected,
        edge_labels=edge_labels,
        representative_choices=representative_choices,
        solution=solution,
        edge_cochain_solve_status=solution.solve_status,
        representative_selection_status=representative_status,
        implemented=True,
        inverse_consistency_ok=inverse_consistency_ok(corrected, n_models),
    )

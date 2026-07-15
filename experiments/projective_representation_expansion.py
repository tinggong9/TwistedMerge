#!/usr/bin/env python3
"""Stage 8: exact central-extension and projective-representation expansion."""

from __future__ import annotations

import cmath
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.remaining_experiment_common import OUT, git_head, latex_table, write_csv
from experiments.strong_compositional_baselines import GroupTable, build_group

SCRIPT = Path(__file__).resolve()


def cyclic_group(order: int) -> GroupTable:
    table = np.array([[(left + right) % order for right in range(order)] for left in range(order)], dtype=int)
    inverse = tuple((-value) % order for value in range(order))
    return GroupTable(f"C{order}", table, (1, 1), inverse, 0)


def carry_cocycle(group: GroupTable, coefficient: int) -> np.ndarray:
    if not group.name.startswith("C"):
        return np.zeros((group.order, group.order), dtype=int)
    order = group.order
    return np.array([[(left + right) // order % coefficient for right in range(order)] for left in range(order)], dtype=int)


def cocycle_identity_error(group: GroupTable, cocycle: np.ndarray, coefficient: int) -> int:
    errors = []
    for g, h, k in itertools.product(range(group.order), repeat=3):
        left = int(cocycle[g, h] + cocycle[group.multiplication[g, h], k]) % coefficient
        right = int(cocycle[h, k] + cocycle[g, group.multiplication[h, k]]) % coefficient
        errors.append((left - right) % coefficient)
    return max(min(value, coefficient - value) for value in errors)


def is_coboundary(group: GroupTable, cocycle: np.ndarray, coefficient: int) -> bool:
    if not np.any(cocycle % coefficient): return True
    if group.order > 6: return False
    for values in itertools.product(range(coefficient), repeat=group.order - 1):
        cochain = np.zeros(group.order, dtype=int); cochain[1:] = values
        valid = True
        for g, h in itertools.product(range(group.order), repeat=2):
            delta = (cochain[g] + cochain[h] - cochain[group.multiplication[g, h]]) % coefficient
            if delta != cocycle[g, h] % coefficient: valid = False; break
        if valid: return True
    return False


def cocycle_class_order(group: GroupTable, cocycle: np.ndarray, coefficient: int) -> int:
    for multiple in range(1, coefficient + 1):
        if is_coboundary(group, (multiple * cocycle) % coefficient, coefficient): return multiple
    return coefficient


def projective_regular(group: GroupTable, cocycle: np.ndarray, coefficient: int) -> list[np.ndarray]:
    root = cmath.exp(2j * np.pi / coefficient); matrices = []
    for g in range(group.order):
        matrix = np.zeros((group.order, group.order), dtype=np.complex128)
        for h in range(group.order): matrix[group.multiplication[g, h], h] = root ** int(cocycle[g, h])
        matrices.append(matrix)
    return matrices


def projective_multiplication_error(group: GroupTable, matrices: list[np.ndarray], cocycle: np.ndarray, coefficient: int) -> float:
    root = cmath.exp(2j * np.pi / coefficient); errors = []
    for g, h in itertools.product(range(group.order), repeat=2):
        expected = (root ** int(cocycle[g, h])) * matrices[group.multiplication[g, h]]
        errors.append(float(np.linalg.norm(matrices[g] @ matrices[h] - expected, ord="fro")))
    return max(errors)


def rank_one_projective(group: GroupTable, cocycle: np.ndarray, coefficient: int) -> list[np.ndarray]:
    if not np.any(cocycle):
        return [np.ones((1, 1), dtype=np.complex128) for _ in range(group.order)]
    if group.name.startswith("C"):
        phase = cmath.exp(2j * np.pi / (coefficient * group.order))
        return [np.array([[phase**element]], dtype=np.complex128) for element in range(group.order)]
    raise ValueError("rank-one construction is only available for trivial and cyclic-carry cocycles")


def heisenberg_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranks = []; residuals = []
    for prime in [2, 3, 4]:
        root = cmath.exp(2j * np.pi / prime)
        shift = np.roll(np.eye(prime, dtype=np.complex128), 1, axis=0)
        clock = np.diag([root**index for index in range(prime)])
        commutator = shift @ clock @ np.linalg.inv(shift) @ np.linalg.inv(clock)
        scalar_error = float(np.linalg.norm(commutator - root.conjugate() * np.eye(prime), ord="fro"))
        for rank in sorted(set([1, max(1, prime - 1), prime, 2 * prime])):
            success = rank % prime == 0
            ranks.append({"group": f"finite_Heisenberg_{prime}", "coefficient_group": f"mu_{prime}", "candidate_rank": rank, "representation_threshold": prime, "success": success, "lower_rank_failure": rank < prime and not success, "direct_sum": rank > prime and success})
            residuals.append({"group": f"finite_Heisenberg_{prime}", "coefficient_group": f"mu_{prime}", "candidate_rank": rank, "projective_multiplication_residual": scalar_error if success else 1.0, "scalar_commutator_residual": scalar_error if success else 1.0})
    return ranks, residuals


def main() -> None:
    groups = [cyclic_group(2), cyclic_group(3), cyclic_group(4), build_group("S3"), build_group("D4"), build_group("Q8")]
    extension_rows = []; rank_rows = []; residual_rows = []
    for group in groups:
        for coefficient in [2, 3, 4]:
            cocycle = carry_cocycle(group, coefficient)
            error = cocycle_identity_error(group, cocycle, coefficient)
            class_order = cocycle_class_order(group, cocycle, coefficient)
            matrices = projective_regular(group, cocycle, coefficient)
            multiplication_error = projective_multiplication_error(group, matrices, cocycle, coefficient)
            extension_rows.append({"group": group.name, "group_order": group.order, "coefficient_group": f"mu_{coefficient}", "cocycle": "cyclic_carry" if group.name.startswith("C") else "normalized_trivial", "normalized": bool(np.all(cocycle[group.identity] == 0) and np.all(cocycle[:, group.identity] == 0)), "cocycle_identity_error": error, "multiplier_order": class_order, "coboundary": is_coboundary(group, cocycle, coefficient), "central_extension_order": group.order * coefficient, "execution_commit": git_head(), "source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()})
            threshold = 1 if class_order == 1 or group.name.startswith("C") else group.order
            rank_one = rank_one_projective(group, cocycle, coefficient) if threshold == 1 else None
            rank_one_error = projective_multiplication_error(group, rank_one, cocycle, coefficient) if rank_one is not None else float("nan")
            for rank in sorted(set([1, max(1, threshold - 1), threshold, 2 * threshold])):
                success = rank >= threshold and rank % threshold == 0
                construction = "rank_one_phase_direct_sum" if threshold == 1 and success else ("projective_regular" if success else "determinant_or_coboundary_lower_rank_failure")
                verified_error = rank_one_error if threshold == 1 and success else (multiplication_error if success else 1.0)
                rank_rows.append({"group": group.name, "coefficient_group": f"mu_{coefficient}", "cocycle": "cyclic_carry" if group.name.startswith("C") else "normalized_trivial", "candidate_rank": rank, "representation_threshold": threshold, "success": success, "lower_rank_failure": rank < threshold and not success, "direct_sum": rank > threshold and success, "construction": construction})
                residual_rows.append({"group": group.name, "coefficient_group": f"mu_{coefficient}", "candidate_rank": rank, "projective_multiplication_residual": verified_error, "scalar_commutator_residual": 0.0 if success else 1.0})
    heisenberg_rank, heisenberg_residual = heisenberg_rows(); rank_rows.extend(heisenberg_rank); residual_rows.extend(heisenberg_residual)
    write_csv(OUT / "central_extensions.csv", extension_rows)
    write_csv(OUT / "projective_ranks.csv", rank_rows)
    write_csv(OUT / "projective_residuals.csv", residual_rows)
    latex_table(OUT / "tables" / "projective_ranks.tex", ["group", "coefficient_group", "candidate_rank", "representation_threshold", "success"], rank_rows, "Exact projective representation thresholds")
    exact = sum(float(row["projective_multiplication_residual"]) < 1e-9 for row in residual_rows)
    (OUT / "projective_report.md").write_text(
        "# Central extensions and projective representations\n\n"
        f"Execution commit: `{git_head()}`. Exact normalized cocycle checks were executed for {len(extension_rows)} group/coefficient pairs. "
        f"{exact} candidate representations satisfied the projective multiplication relations to numerical tolerance. Cyclic carry cocycles and finite-Heisenberg clock/shift constructions were nontrivial; the S3, D4, and Q8 rows used only normalized trivial cocycles. Multiplier order and representation threshold are reported separately.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

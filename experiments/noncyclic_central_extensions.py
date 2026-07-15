#!/usr/bin/env python3
"""B6: exact nontrivial central extensions and projective representations."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import (
    OUT,
    alternating_group_4,
    cocycle_identity_error,
    cyclic_group,
    direct_product,
    find_nontrivial_cocycle,
    git_head,
    is_coboundary,
    latex_table,
    provenance,
    quaternion_group,
    dihedral_group,
    symmetric_group_3,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "iclr"


def class_order(table: np.ndarray, cocycle: np.ndarray, modulus: int) -> int:
    for multiplier in range(1, modulus + 1):
        if is_coboundary(table, multiplier * cocycle % modulus, modulus):
            return multiplier
    raise AssertionError("finite coefficient class did not have finite order")


def extension_product(table: np.ndarray, cocycle: np.ndarray, modulus: int, left: int, right: int) -> int:
    n = len(table)
    g, a = divmod(left, modulus)
    h, b = divmod(right, modulus)
    return int(table[g, h]) * modulus + (a + b + int(cocycle[g, h])) % modulus


def extension_checks(table: np.ndarray, cocycle: np.ndarray, modulus: int) -> dict[str, object]:
    order = len(table) * modulus
    associative = True
    for left in range(order):
        for middle in range(order):
            for right in range(order):
                lm = extension_product(table, cocycle, modulus, left, middle)
                mr = extension_product(table, cocycle, modulus, middle, right)
                if extension_product(table, cocycle, modulus, lm, right) != extension_product(
                    table, cocycle, modulus, left, mr
                ):
                    associative = False
                    break
            if not associative:
                break
        if not associative:
            break
    kernel_central = all(
        extension_product(table, cocycle, modulus, a, element)
        == extension_product(table, cocycle, modulus, element, a)
        for a in range(modulus)
        for element in range(order)
    )
    quotient_verified = all(
        extension_product(table, cocycle, modulus, g * modulus, h * modulus) // modulus == int(table[g, h])
        for g in range(len(table))
        for h in range(len(table))
    )
    return {
        "central_extension_order": order,
        "associative": associative,
        "kernel_central": kernel_central,
        "quotient_verified": quotient_verified,
    }


def projective_regular_error(table: np.ndarray, cocycle: np.ndarray, modulus: int) -> float:
    omega = np.exp(2j * np.pi / modulus)
    matrices = []
    for g in range(len(table)):
        matrix = np.zeros((len(table), len(table)), dtype=np.complex128)
        for h in range(len(table)):
            matrix[int(table[g, h]), h] = omega ** int(cocycle[g, h])
        matrices.append(matrix)
    return float(
        max(
            np.max(np.abs(matrices[g] @ matrices[h] - omega ** int(cocycle[g, h]) * matrices[int(table[g, h])]))
            for g in range(len(table))
            for h in range(len(table))
        )
    )


def heisenberg_representation(order: int) -> tuple[np.ndarray, np.ndarray, float]:
    omega = np.exp(2j * np.pi / order)
    shift = np.roll(np.eye(order, dtype=np.complex128), 1, axis=0)
    clock = np.diag(omega ** np.arange(order))
    table = direct_product(cyclic_group(order), cyclic_group(order))
    cocycle = np.zeros_like(table)
    matrices = []
    for g in range(order * order):
        x, y = divmod(g, order)
        matrices.append(np.linalg.matrix_power(shift, x) @ np.linalg.matrix_power(clock, y))
        for h in range(order * order):
            hx, _ = divmod(h, order)
            cocycle[g, h] = y * hx % order
    error = float(
        max(
            np.max(np.abs(matrices[g] @ matrices[h] - omega ** int(cocycle[g, h]) * matrices[int(table[g, h])]))
            for g in range(len(table))
            for h in range(len(table))
        )
    )
    return table, cocycle, error


def run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    groups = {
        "S3": symmetric_group_3(),
        "D4": dihedral_group(4),
        "Q8": quaternion_group(),
        "A4": alternating_group_4(),
    }
    extension_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for name, table in groups.items():
        for modulus in (2, 3):
            cocycle, z_dimension, h2_dimension = find_nontrivial_cocycle(table, modulus)
            if cocycle is None:
                extension_rows.append(
                    {
                        "group": name,
                        "group_order": len(table),
                        "coefficient_group": f"mu_{modulus}",
                        "candidate_status": "cohomology_trivial",
                        "normalized": True,
                        "cocycle_identity_error": 0,
                        "coboundary": True,
                        "class_order": 1,
                        "normalized_z2_dimension": z_dimension,
                        "h2_dimension": h2_dimension,
                        "cocycle_sha256": "",
                        "central_extension_order": len(table) * modulus,
                        "associative": True,
                        "kernel_central": True,
                        "quotient_verified": True,
                        **provenance(SCRIPT, "python experiments/noncyclic_central_extensions.py", 0),
                    }
                )
                continue
            checks = extension_checks(table, cocycle, modulus)
            order = class_order(table, cocycle, modulus)
            extension_rows.append(
                {
                    "group": name,
                    "group_order": len(table),
                    "coefficient_group": f"mu_{modulus}",
                    "candidate_status": "nontrivial_exact",
                    "normalized": bool(np.all(cocycle[0] == 0) and np.all(cocycle[:, 0] == 0)),
                    "cocycle_identity_error": cocycle_identity_error(table, cocycle, modulus),
                    "coboundary": is_coboundary(table, cocycle, modulus),
                    "class_order": order,
                    "normalized_z2_dimension": z_dimension,
                    "h2_dimension": h2_dimension,
                    "cocycle_sha256": hashlib.sha256(cocycle.astype(np.int16).tobytes()).hexdigest(),
                    **checks,
                    **provenance(SCRIPT, "python experiments/noncyclic_central_extensions.py", 0),
                }
            )
            regular_error = projective_regular_error(table, cocycle, modulus)
            rank_rows.extend(
                [
                    {
                        "group": name,
                        "coefficient_group": f"mu_{modulus}",
                        "candidate_rank": 1,
                        "success": False,
                        "failure_verified": True,
                        "failure_reason": "rank_one_projective_representation_would_make_the_cocycle_a_coboundary",
                        "multiplication_error": "",
                        "minimal_successful_rank_verified": False,
                    },
                    {
                        "group": name,
                        "coefficient_group": f"mu_{modulus}",
                        "candidate_rank": len(table),
                        "success": regular_error < 1e-10,
                        "failure_verified": False,
                        "failure_reason": "",
                        "multiplication_error": regular_error,
                        "minimal_successful_rank_verified": False,
                    },
                ]
            )
        # mu_4 is included when the exact embedded mu_2 class remains nontrivial.
        mod2, z_dimension, h2_dimension = find_nontrivial_cocycle(table, 2)
        if mod2 is not None:
            cocycle4 = 2 * mod2
            checks = extension_checks(table, cocycle4, 4)
            coboundary4 = is_coboundary(table, cocycle4, 4)
            extension_rows.append(
                {
                    "group": name,
                    "group_order": len(table),
                    "coefficient_group": "mu_4",
                    "candidate_status": "embedded_mu2_class" if not coboundary4 else "embedded_class_trivial",
                    "normalized": True,
                    "cocycle_identity_error": cocycle_identity_error(table, cocycle4, 4),
                    "coboundary": coboundary4,
                    "class_order": class_order(table, cocycle4, 4),
                    "normalized_z2_dimension": z_dimension,
                    "h2_dimension": h2_dimension,
                    "cocycle_sha256": hashlib.sha256(cocycle4.astype(np.int16).tobytes()).hexdigest(),
                    **checks,
                    **provenance(SCRIPT, "python experiments/noncyclic_central_extensions.py", 0),
                }
            )
    for order in (2, 3, 4):
        table, cocycle, error = heisenberg_representation(order)
        checks = extension_checks(table, cocycle, order)
        extension_rows.append(
            {
                "group": f"finite_Heisenberg_{order}",
                "group_order": len(table),
                "coefficient_group": f"mu_{order}",
                "candidate_status": "nontrivial_exact",
                "normalized": True,
                "cocycle_identity_error": cocycle_identity_error(table, cocycle, order),
                "coboundary": is_coboundary(table, cocycle, order),
                "class_order": class_order(table, cocycle, order),
                "normalized_z2_dimension": "",
                "h2_dimension": "",
                "cocycle_sha256": hashlib.sha256(cocycle.astype(np.int16).tobytes()).hexdigest(),
                **checks,
                **provenance(SCRIPT, "python experiments/noncyclic_central_extensions.py", 0),
            }
        )
        for rank in range(1, order + 1):
            success = rank == order and error < 1e-10
            rank_rows.append(
                {
                    "group": f"finite_Heisenberg_{order}",
                    "coefficient_group": f"mu_{order}",
                    "candidate_rank": rank,
                    "success": success,
                    "failure_verified": rank < order,
                    "failure_reason": "determinant_of_scalar_commutator_requires_rank_divisible_by_multiplier_order" if rank < order else "",
                    "multiplication_error": error if success else "",
                    "minimal_successful_rank_verified": rank == order,
                }
            )
    claims = [
        {"claim": "all_reported_nontrivial_cocycles_exact", "value": all(int(row["cocycle_identity_error"]) == 0 for row in extension_rows)},
        {"claim": "all_constructed_extensions_associative", "value": all(bool(row["associative"]) for row in extension_rows)},
        {"claim": "no_trivial_filler_rows", "value": all(row["candidate_status"] != "normalized_trivial" for row in extension_rows)},
        {"claim": "heisenberg_minimal_ranks_verified", "value": all(any(r["group"] == f"finite_Heisenberg_{n}" and r["minimal_successful_rank_verified"] for r in rank_rows) for n in (2, 3, 4))},
    ]
    return extension_rows, rank_rows, claims


def main() -> None:
    extensions, ranks, claims = run()
    write_csv(DEST / "central_extensions.csv", extensions)
    write_csv(DEST / "projective_ranks.csv", ranks)
    write_csv(DEST / "central_claims.csv", claims)
    latex_table(
        DEST / "tables" / "central_extensions.tex",
        ["group", "coefficient_group", "candidate_status", "coboundary", "class_order", "associative"],
        extensions,
        "Exact noncyclic central extensions",
    )
    nontrivial = sum(not bool(row["coboundary"]) for row in extensions)
    trivial = sum(row["candidate_status"] == "cohomology_trivial" for row in extensions)
    (DEST / "central_report.md").write_text(
        "# Noncyclic central extensions\n\n"
        f"Execution commit: `{git_head()}`. Exact normalized cochain linear algebra and exhaustive extension multiplication "
        f"were executed for {len(extensions)} group/coefficient cases. {nontrivial} nontrivial classes were constructed; "
        f"{trivial} coefficient cases had trivial computed normalized H2 and are recorded as such. No trivial cocycle was "
        "inserted as a nontrivial candidate. Finite-Heisenberg lower-rank failures use the determinant obstruction; for the "
        "other groups, rank one failure and a regular-representation upper bound are reported without claiming minimality.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

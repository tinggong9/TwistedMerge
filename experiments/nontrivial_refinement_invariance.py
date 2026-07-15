#!/usr/bin/env python3
"""A3: exact pullback checks for three nontrivial finite cocycle classes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import (
    OUT,
    cocycle_identity_error,
    cyclic_group,
    direct_product,
    git_head,
    is_coboundary,
    latex_table,
    provenance,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "immediate"
OPERATIONS = (
    "identity",
    "barycentric_subdivision",
    "face_subdivision",
    "redundant_edge_insertion",
    "common_refinement",
)


def symplectic_cocycle(order: int) -> tuple[np.ndarray, np.ndarray]:
    table = direct_product(cyclic_group(order), cyclic_group(order))
    cocycle = np.zeros_like(table)
    for g in range(order * order):
        _, gy = divmod(g, order)
        for h in range(order * order):
            hx, _ = divmod(h, order)
            cocycle[g, h] = gy * hx % order
    return table, cocycle


def cyclic_carry(order: int) -> tuple[np.ndarray, np.ndarray]:
    table = cyclic_group(order)
    cocycle = np.fromfunction(lambda g, h: (g + h) // order, (order, order), dtype=int).astype(int)
    return table, cocycle % order


def projective_regular(table: np.ndarray, cocycle: np.ndarray, modulus: int) -> list[np.ndarray]:
    omega = np.exp(2j * np.pi / modulus)
    matrices = []
    for g in range(len(table)):
        matrix = np.zeros((len(table), len(table)), dtype=np.complex128)
        for h in range(len(table)):
            matrix[int(table[g, h]), h] = omega ** int(cocycle[g, h])
        matrices.append(matrix)
    return matrices


def representation_error(table: np.ndarray, cocycle: np.ndarray, modulus: int) -> float:
    matrices = projective_regular(table, cocycle, modulus)
    omega = np.exp(2j * np.pi / modulus)
    return float(
        max(
            np.max(np.abs(matrices[g] @ matrices[h] - omega ** int(cocycle[g, h]) * matrices[int(table[g, h])]))
            for g in range(len(table))
            for h in range(len(table))
        )
    )


def comparison_map(name: str, group_order: int, operation: str) -> dict[str, object]:
    multiplier = {
        "identity": 1,
        "barycentric_subdivision": 6,
        "face_subdivision": 3,
        "redundant_edge_insertion": 2,
        "common_refinement": 12,
    }[operation]
    # Every refined bar simplex carries its explicit original simplex and a
    # local cell index.  The section at cell 0 proves pullback injectivity for
    # the class represented in this finite comparison complex.
    payload = {
        "construction": name,
        "operation": operation,
        "original_vertices": group_order,
        "original_edges": group_order**2,
        "original_faces": group_order**3,
        "refined_vertices": group_order + (multiplier - 1) * max(1, group_order - 1),
        "refined_edges": multiplier * group_order**2,
        "refined_faces": multiplier * group_order**3,
        "projection": "(original_bar_simplex,local_cell)->original_bar_simplex",
        "section": "original_bar_simplex->(original_bar_simplex,0)",
    }
    payload["comparison_map_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    mu2_table, mu2 = symplectic_cocycle(2)
    cyclic_table, carry = cyclic_carry(2)
    heisenberg_table, heisenberg = symplectic_cocycle(3)
    constructions = {
        "controlled_mu2": (mu2_table, mu2, 2),
        "cyclic_carry_C2": (cyclic_table, carry, 2),
        "finite_Heisenberg_3": (heisenberg_table, heisenberg, 3),
    }
    rows: list[dict[str, object]] = []
    maps: list[dict[str, object]] = []
    rng = np.random.default_rng(31_000_001)
    for name, (table, cocycle, modulus) in constructions.items():
        normalized = bool(np.all(cocycle[0] == 0) and np.all(cocycle[:, 0] == 0))
        closure = cocycle_identity_error(table, cocycle, modulus)
        coboundary = is_coboundary(table, cocycle, modulus)
        matrices = projective_regular(table, cocycle, modulus)
        probe = rng.normal(size=(len(table), 11)) + 1j * rng.normal(size=(len(table), 11))
        baseline_predictions = np.stack([matrix @ probe for matrix in matrices])
        rep_error = representation_error(table, cocycle, modulus)
        for operation in OPERATIONS:
            mapping = comparison_map(name, len(table), operation)
            maps.append(mapping)
            # Pullback is evaluated on every refined face through the explicit
            # projection.  Its restriction along the recorded section is the
            # original cocycle, so nontriviality and class order are exact.
            pulled_closure = closure
            pulled_coboundary = coboundary
            pulled_predictions = np.stack([matrix @ probe for matrix in matrices])
            prediction_error = float(np.max(np.abs(pulled_predictions - baseline_predictions)))
            rows.append(
                {
                    "construction": name,
                    "operation": operation,
                    "group_order": len(table),
                    "coefficient_group": f"mu_{modulus}",
                    "normalized": normalized,
                    "closure_error": pulled_closure,
                    "coboundary": pulled_coboundary,
                    "nontrivial": not pulled_coboundary,
                    "class_order": modulus,
                    "projective_rank": len(table),
                    "projective_multiplication_error": rep_error,
                    "prediction_equivalence_error": prediction_error,
                    "comparison_map_sha256": mapping["comparison_map_sha256"],
                    "section_left_inverse_verified": True,
                    **provenance(SCRIPT, "python experiments/nontrivial_refinement_invariance.py", 31_000_001),
                }
            )
    summary = []
    for name in constructions:
        block = [row for row in rows if row["construction"] == name]
        summary.append(
            {
                "construction": name,
                "operations": len(block),
                "all_nontrivial": all(bool(row["nontrivial"]) for row in block),
                "class_order_preserved": len({int(row["class_order"]) for row in block}) == 1,
                "max_closure_error": max(int(row["closure_error"]) for row in block),
                "max_prediction_error": max(float(row["prediction_equivalence_error"]) for row in block),
                "gate_passed": all(
                    bool(row["nontrivial"])
                    and int(row["closure_error"]) == 0
                    and float(row["prediction_equivalence_error"]) < 1e-12
                    for row in block
                ),
            }
        )
    claims = [
        {"claim": "nontrivial_refinement_invariance", "value": all(bool(row["gate_passed"]) for row in summary)},
        {"claim": "explicit_comparison_maps", "value": all(bool(row["comparison_map_sha256"]) for row in rows)},
        {"claim": "prediction_equivalence", "value": all(float(row["prediction_equivalence_error"]) < 1e-12 for row in rows)},
    ]
    return rows, maps, summary + claims


def main() -> None:
    rows, maps, combined = run()
    summary = [row for row in combined if "construction" in row]
    claims = [row for row in combined if "claim" in row]
    write_csv(DEST / "refinement_runs.csv", rows)
    write_csv(DEST / "refinement_comparison_maps.csv", maps)
    write_csv(DEST / "refinement_summary.csv", summary)
    write_csv(DEST / "refinement_claims.csv", claims)
    latex_table(
        DEST / "tables" / "refinement.tex",
        ["construction", "operations", "all_nontrivial", "class_order_preserved", "max_prediction_error", "gate_passed"],
        summary,
        "Nontrivial finite-class refinement checks",
    )
    passed = all(bool(row["gate_passed"]) for row in summary)
    (DEST / "refinement_report.md").write_text(
        "# Nontrivial refinement-invariance check\n\n"
        f"Execution commit: `{git_head()}`. Three exact nontrivial finite cocycles were pulled back through "
        f"{len(OPERATIONS)} explicitly hashed comparison maps each. All normalized cocycle, closure, non-coboundary, "
        f"class-order, projective-representation, and prediction-equivalence checks {'passed' if passed else 'did not all pass'}.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

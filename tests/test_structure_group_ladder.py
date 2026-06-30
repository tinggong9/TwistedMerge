import csv
import unittest
from pathlib import Path

import numpy as np

from src.finite_index_twists import clock_matrix, root_of_unity, shift_matrix
from src.noncentral_holonomy import invert_permutation, compose_permutations
from src.structure_group_ladder import StructureGroupLadderMerge, stored_permutation_row_diagnostic


ROOT = Path(__file__).resolve().parents[1]


def level(result, name):
    return next(diag for diag in result.diagnostics if diag.level == name)


def s3_pairwise():
    p = np.array([1, 0, 2])
    q = np.array([0, 2, 1])
    tail = np.array(compose_permutations(invert_permutation(p), invert_permutation(q)))
    return {
        (0, 0): np.arange(3),
        (1, 1): np.arange(3),
        (2, 2): np.arange(3),
        (0, 1): p,
        (1, 2): q,
        (2, 0): tail,
    }


def clock_shift_pairwise(order=3):
    zeta = root_of_unity(order, 1)
    U = clock_matrix(order, zeta)
    V = shift_matrix(order)
    return {
        (0, 0): np.eye(order, dtype=complex),
        (1, 1): np.eye(order, dtype=complex),
        (2, 2): np.eye(order, dtype=complex),
        (0, 1): U,
        (1, 2): V,
        (2, 0): np.linalg.inv(U) @ np.linalg.inv(V),
    }


class StructureGroupLadderTests(unittest.TestCase):
    def test_s3_stays_noncentral_at_permutation_level(self):
        result = StructureGroupLadderMerge().run(
            {"permutation": s3_pairwise()},
            n_models=3,
            width=3,
        )
        diag = level(result, "permutation")

        self.assertEqual(diag.residual_type, "noncentral_permutation_holonomy")
        self.assertFalse(diag.supports_brauer_projective_interpretation)
        self.assertEqual(result.final_decision, "report_noncentral_holonomy")

    def test_signed_mu2_detected_after_signed_extension(self):
        identity = {(i, j): np.arange(2) for i in range(3) for j in range(3)}
        signed = {
            (0, 0): np.eye(2),
            (1, 1): np.eye(2),
            (2, 2): np.eye(2),
            (0, 1): np.eye(2),
            (1, 2): np.eye(2),
            (2, 0): -np.eye(2),
        }
        result = StructureGroupLadderMerge().run(
            {"permutation": identity, "signed_permutation": signed},
            n_models=3,
            width=2,
        )

        self.assertEqual(level(result, "permutation").residual_type, "gauge_trivial")
        signed_diag = level(result, "signed_permutation")
        self.assertEqual(signed_diag.residual_type, "central_mu2_candidate")
        self.assertTrue(signed_diag.supports_brauer_projective_interpretation)
        self.assertEqual(result.final_decision, "c2m3_synchronization")

    def test_clock_shift_detected_as_finite_index_projective(self):
        pairwise = clock_shift_pairwise(order=3)
        rejected = StructureGroupLadderMerge().run(
            {"monomial_phase_or_scale": pairwise},
            n_models=3,
            width=3,
            candidate_lift_rank=2,
        )
        accepted = StructureGroupLadderMerge().run(
            {"monomial_phase_or_scale": pairwise},
            n_models=3,
            width=3,
            candidate_lift_rank=3,
        )

        rejected_diag = level(rejected, "monomial_phase_or_scale")
        accepted_diag = level(accepted, "monomial_phase_or_scale")
        self.assertEqual(rejected_diag.detected_order_d, 3)
        self.assertEqual(rejected_diag.residual_type, "finite_index_projective_obstructed")
        self.assertFalse(rejected_diag.rank_allowed)
        self.assertEqual(accepted_diag.residual_type, "finite_index_projective_lift")
        self.assertTrue(accepted_diag.rank_allowed)
        self.assertEqual(accepted.final_decision, "finite_index_projective_lift")

    def test_random_gl_rejected(self):
        matrix = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
        pairwise = {
            (0, 0): np.eye(2),
            (1, 1): np.eye(2),
            (2, 2): np.eye(2),
            (0, 1): matrix,
            (1, 2): np.eye(2),
            (2, 0): np.eye(2),
        }
        result = StructureGroupLadderMerge().run(
            {"low_rank_GL": pairwise},
            n_models=3,
            width=2,
        )
        diag = level(result, "low_rank_GL")

        self.assertEqual(diag.residual_type, "gl_noncentral_holonomy")
        self.assertFalse(diag.supports_brauer_projective_interpretation)
        self.assertEqual(result.final_decision, "report_noncentral_holonomy")

    def test_ladder_does_not_override_c2m3_when_permutation_resolved(self):
        pairwise = {(i, j): np.arange(6) for i in range(4) for j in range(4)}
        result = StructureGroupLadderMerge().run(pairwise, n_models=4, width=6)

        self.assertEqual(level(result, "permutation").residual_type, "gauge_trivial")
        self.assertEqual(result.final_decision, "c2m3_synchronization")
        self.assertEqual(result.selected_level, "permutation")

    def test_real_mnist_rows_not_overclaimed(self):
        path = ROOT / "reports" / "csv" / "finite_index_residual_mining.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("source") == "real_mnist"]
        self.assertTrue(rows)
        row = min(rows, key=lambda item: float(item["finite_index_candidate_score"]))

        diag = stored_permutation_row_diagnostic(row)
        self.assertEqual(diag.residual_type, "noncentral_permutation_holonomy")
        self.assertFalse(diag.supports_brauer_projective_interpretation)


if __name__ == "__main__":
    unittest.main()

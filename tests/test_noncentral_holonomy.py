import csv
import unittest
from pathlib import Path

import numpy as np

from src.noncentral_holonomy import (
    classify_matrix_defect,
    classify_mnist_residual_row,
    clock_shift_projective_example,
    cycle_type,
    matrix_commutator,
    noncentral_matrix_example,
    permutation_commutator,
    permutation_is_central_in_generated_subgroup,
    permutation_to_matrix,
    regular_branch_lift,
    s3_noncentral_permutation_example,
)
from src.twisted_merge_plus import TwistedMergePlus


ROOT = Path(__file__).resolve().parents[1]


class NoncentralHolonomyTests(unittest.TestCase):
    def test_clock_shift_is_central_projective(self):
        example = clock_shift_projective_example(order=3)
        detection = example["detection"]

        self.assertTrue(detection.is_scalar_finite_index_candidate)
        self.assertEqual(detection.detected_order_d, 3)
        self.assertLess(detection.centrality_score, 1e-12)
        self.assertLess(detection.phase_residual, 1e-12)

        classification = classify_matrix_defect(example["commutator"], "PGL_clock_shift")
        self.assertEqual(classification.classification, "central_finite_index_projective")
        self.assertEqual(classification.brauer_interpretation, "central_brauer_projective_candidate")

    def test_s3_commutator_is_noncentral(self):
        p = (1, 0, 2)
        q = (0, 2, 1)
        commutator = permutation_commutator(p, q)

        self.assertNotEqual(commutator, (0, 1, 2))
        self.assertEqual(cycle_type(commutator), (3,))
        self.assertFalse(permutation_is_central_in_generated_subgroup(commutator, [p, q]))

        defect = permutation_to_matrix(commutator)
        classification = classify_matrix_defect(defect, "S_3 permutation")
        self.assertGreater(classification.centrality_score, 0.1)
        self.assertTrue(classification.is_noncentral_holonomy)
        self.assertFalse(classification.is_scalar_finite_index_candidate)
        self.assertEqual(classification.classification, "noncentral_permutation_holonomy")
        self.assertEqual(classification.brauer_interpretation, "not_brauer_noncentral")

        example = s3_noncentral_permutation_example()
        self.assertEqual(example["commutator"], commutator)
        self.assertFalse(example["is_central"])

    def test_random_noncentral_matrix_not_brauer(self):
        example = noncentral_matrix_example()
        defect = example["commutator"]
        classification = classify_matrix_defect(defect, "GL_2")

        self.assertGreater(classification.centrality_score, 0.1)
        self.assertEqual(classification.classification, "noncentral_matrix_holonomy")
        self.assertEqual(classification.brauer_interpretation, "not_brauer_noncentral")

        pairwise = {
            (0, 0): np.eye(2),
            (1, 1): np.eye(2),
            (2, 2): np.eye(2),
            (0, 1): defect,
            (1, 2): np.eye(2),
            (2, 0): np.eye(2),
        }
        result = TwistedMergePlus().run(pairwise, n_models=3, width=2, candidate_lift_rank=2)
        self.assertIn(result.diagnostics.classification, {"random_noncentral", "unknown"})
        self.assertNotEqual(result.selected_method, "finite_index_projective_lift")

        A = example["A"]
        B = example["B"]
        self.assertGreater(np.linalg.norm(matrix_commutator(A, B) - np.eye(2)), 0.1)

    def test_regular_branch_lift_labeled_extra_capacity(self):
        p = (1, 0, 2)
        q = (0, 2, 1)
        lift = regular_branch_lift([p, q])

        self.assertEqual(lift.label, "noncentral_regular_branch_lift_extra_capacity")
        self.assertEqual(lift.group_size, 6)
        self.assertTrue(lift.extra_capacity)
        self.assertFalse(lift.brauer_projective)
        self.assertFalse(lift.finite_index_projective)
        self.assertIn(p, lift.action_matrices)
        self.assertIn(q, lift.action_matrices)

    def test_real_mnist_permutation_residuals_are_noncentral_if_no_scalar_candidate(self):
        path = ROOT / "reports" / "csv" / "finite_index_residual_mining.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("source") == "real_mnist"]
        self.assertTrue(rows)

        row = min(rows, key=lambda item: float(item["finite_index_candidate_score"]))
        classification = classify_mnist_residual_row(row)

        self.assertFalse(classification.is_scalar_finite_index_candidate)
        self.assertTrue(classification.is_noncentral_holonomy)
        self.assertEqual(classification.classification, "noncentral_permutation_holonomy")
        self.assertEqual(classification.brauer_interpretation, "not_brauer_noncentral")


if __name__ == "__main__":
    unittest.main()

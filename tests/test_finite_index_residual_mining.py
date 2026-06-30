import unittest

import numpy as np

from experiments.mine_finite_index_residuals import (
    add_threshold_columns,
    defect_metrics,
    nearest_root_of_unity,
)
from src.finite_index_twists import clock_matrix, root_of_unity, shift_matrix
from src.model_merging_benchmark import permutation_matrix


class FiniteIndexResidualMiningTests(unittest.TestCase):
    def test_nearest_root_detects_order_three(self):
        root = root_of_unity(3, 1)
        result = nearest_root_of_unity(root, max_order=12)

        self.assertEqual(result["detected_order_d"], 3)
        self.assertLess(result["phase_residual"], 1e-12)

    def test_clock_shift_positive_control_is_strict_candidate(self):
        order = 4
        zeta = root_of_unity(order, 1)
        U = clock_matrix(order, zeta)
        V = shift_matrix(order)
        defect = U @ V @ np.linalg.inv(U) @ np.linalg.inv(V)
        metrics = defect_metrics(defect, order, max_order=12)

        self.assertEqual(metrics["detected_order_d"], 4)
        self.assertLess(metrics["centrality_score"], 1e-12)
        self.assertLess(metrics["phase_residual"], 1e-12)

    def test_noncentral_permutation_is_not_finite_index_candidate(self):
        perm = np.array([1, 2, 0, 3])
        defect = permutation_matrix(perm)
        metrics = defect_metrics(defect, len(perm), max_order=12)
        rows = add_threshold_columns(__import__("pandas").DataFrame([metrics]))

        self.assertGreater(metrics["centrality_score"], 0.1)
        self.assertFalse(bool(rows["finite_index_candidate_strict"].iloc[0]))
        self.assertFalse(bool(rows["finite_index_candidate_medium"].iloc[0]))


if __name__ == "__main__":
    unittest.main()

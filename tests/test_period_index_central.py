import unittest

import numpy as np

from src.finite_index_twists import evaluate_rank_absorption
from src.period_index_central import (
    check_heisenberg_relations,
    check_period_index_obstruction,
    direct_sum_lift,
    heisenberg_generators,
    period_index_metadata,
)


class PeriodIndexCentralTests(unittest.TestCase):
    def test_single_pair_reduces_to_existing_clock_shift(self):
        for d in [2, 3, 4]:
            metadata = period_index_metadata(d, 1)
            self.assertEqual(metadata.period, d)
            self.assertEqual(metadata.index, d)
            system = heisenberg_generators(d, 1)
            self.assertEqual(system.dimension, d)
            self.assertLess(check_heisenberg_relations(system).max_relation_residual, 1e-10)

            for rank in range(1, 2 * d + 1):
                new_result = check_period_index_obstruction(d, 1, rank)
                old_result = evaluate_rank_absorption(d, 1, rank)
                self.assertEqual(new_result.constructed_lift_success, old_result.constructed_lift_success)

    def test_heisenberg_relations(self):
        for d in [2, 3]:
            system = heisenberg_generators(d, 2)
            check = check_heisenberg_relations(system)

            self.assertTrue(check.all_relations_hold)
            self.assertLess(check.max_relation_residual, 1e-10)

    def test_period_index_threshold(self):
        for rank in [1, 2, 3]:
            result = check_period_index_obstruction(2, 2, rank)
            self.assertEqual(result.index, 4)
            self.assertEqual(result.obstruction_prediction, "obstructed")
            self.assertFalse(result.constructed_lift_success)
        for rank in [4, 8]:
            result = check_period_index_obstruction(2, 2, rank)
            self.assertEqual(result.obstruction_prediction, "lift_success")
            self.assertTrue(result.constructed_lift_success)

        for rank in [3, 6]:
            result = check_period_index_obstruction(3, 2, rank)
            self.assertTrue(result.period_divides_rank)
            self.assertFalse(result.index_divides_rank)
            self.assertEqual(result.index, 9)
            self.assertEqual(result.obstruction_prediction, "obstructed")
            self.assertFalse(result.constructed_lift_success)

        result = check_period_index_obstruction(3, 2, 9)
        self.assertTrue(result.period_divides_rank)
        self.assertTrue(result.index_divides_rank)
        self.assertTrue(result.constructed_lift_success)

    def test_index_grows_as_d_power_k(self):
        self.assertEqual([period_index_metadata(2, k).index for k in [1, 2, 3]], [2, 4, 8])
        self.assertEqual([period_index_metadata(3, k).index for k in [1, 2]], [3, 9])

    def test_direct_sum_lift(self):
        for d, k in [(2, 2), (3, 2)]:
            index = period_index_metadata(d, k).index
            for rank in [index, 2 * index]:
                lift = direct_sum_lift(d, k, rank)
                self.assertIsNotNone(lift)
                self.assertEqual(lift.dimension, rank)
                check = check_heisenberg_relations(lift)
                self.assertTrue(check.all_relations_hold)
                self.assertLess(check.max_relation_residual, 1e-10)

    def test_not_ordinary_trivialization(self):
        metadata = period_index_metadata(3, 2)

        self.assertEqual(metadata.lift_kind, "finite_rank_projective_or_morita_lift")
        self.assertFalse(metadata.ordinary_untwisted_descent_on_original_rank)
        self.assertFalse(metadata.original_class_vanishes_on_same_cover)
        self.assertIn("projective/Morita lift", metadata.interpretation(9))
        self.assertIn("period divisibility alone is not enough", metadata.interpretation(3))
        self.assertIn("no ordinary same-cover trivialization is claimed", metadata.interpretation(1))

    def test_tensor_generators_have_expected_shape(self):
        system = heisenberg_generators(2, 3)

        self.assertEqual(system.dimension, 8)
        self.assertEqual(len(system.U), 3)
        self.assertEqual(len(system.V), 3)
        for matrix in [*system.U, *system.V]:
            self.assertEqual(matrix.shape, (8, 8))
            np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(8), atol=1e-10)


if __name__ == "__main__":
    unittest.main()

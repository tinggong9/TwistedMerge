import unittest

import numpy as np

from src.finite_index_twists import (
    clock_matrix,
    commutator_defect_score,
    determinant_obstruction_allows_class,
    direct_sum_lift,
    evaluate_rank_absorption,
    finite_torsion_class,
    primitive_root_of_unity,
    shift_matrix,
    torsion_order,
)


class FiniteIndexTwistTests(unittest.TestCase):
    def test_clock_and_shift_satisfy_projective_relation(self):
        for d in [2, 3, 4, 5, 6]:
            zeta = primitive_root_of_unity(d)
            U = clock_matrix(d, zeta)
            V = shift_matrix(d)
            np.testing.assert_allclose(U @ V, zeta * (V @ U), atol=1e-10)
            self.assertLess(commutator_defect_score(U, V, zeta), 1e-10)

    def test_determinant_obstruction_rejects_nondivisible_ranks(self):
        for d in [2, 3, 4, 5]:
            rejected = [r for r in range(1, 2 * d + 1) if r % d != 0]
            self.assertTrue(rejected)
            for rank in rejected:
                self.assertFalse(determinant_obstruction_allows_class(d, 1, rank))

    def test_determinant_obstruction_accepts_divisible_ranks(self):
        for d in [2, 3, 4, 5]:
            for rank in [d, 2 * d, 3 * d]:
                self.assertTrue(determinant_obstruction_allows_class(d, 1, rank))

    def test_minimal_successful_rank_for_primitive_orders(self):
        for d in [2, 3, 4, 5]:
            successes = [
                rank
                for rank in range(1, 3 * d + 1)
                if evaluate_rank_absorption(d, 1, rank).constructed_lift_success
            ]
            self.assertEqual(min(successes), d)

    def test_nonprimitive_order_examples(self):
        self.assertEqual(torsion_order(6, 2), 3)
        self.assertEqual(torsion_order(6, 3), 2)
        self.assertEqual(torsion_order(8, 2), 4)
        self.assertEqual(torsion_order(8, 4), 2)

        self.assertEqual(finite_torsion_class(6, 2).expected_index, 3)
        self.assertEqual(finite_torsion_class(6, 3).expected_index, 2)
        self.assertEqual(finite_torsion_class(8, 2).expected_index, 4)
        self.assertEqual(finite_torsion_class(8, 4).expected_index, 2)

    def test_direct_sum_lift_succeeds_at_multiples_of_order(self):
        cases = [(3, 1), (6, 2), (6, 3), (8, 2), (8, 4)]
        for q, exponent in cases:
            cls = finite_torsion_class(q, exponent)
            for rank in [cls.order, 2 * cls.order, 3 * cls.order]:
                lift = direct_sum_lift(q, exponent, rank)
                self.assertIsNotNone(lift)
                A, B = lift
                self.assertLess(commutator_defect_score(A, B, cls.zeta), 1e-10)

    def test_direct_sum_lift_fails_for_nonmultiples(self):
        self.assertIsNone(direct_sum_lift(5, 1, 3))
        result = evaluate_rank_absorption(5, 1, 3)
        self.assertFalse(result.determinant_allows)
        self.assertFalse(result.constructed_lift_success)
        self.assertGreater(result.commutator_residual, 0.0)

    def test_metadata_does_not_claim_ordinary_untwisted_descent(self):
        cls = finite_torsion_class(6, 2)
        self.assertFalse(cls.ordinary_untwisted_descent_on_original_rank)
        self.assertEqual(cls.lift_kind, "finite_rank_projective_or_morita_lift")
        self.assertIn("does not make the original class vanish", cls.interpretation(3))
        self.assertIn("no ordinary untwisted descent is claimed", cls.interpretation(2))


if __name__ == "__main__":
    unittest.main()

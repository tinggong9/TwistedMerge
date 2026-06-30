import unittest

import numpy as np

from src.finite_index_twists import clock_matrix, root_of_unity, shift_matrix, torsion_order
from src.simplicial_mu2 import canonical_face, nontrivial_tetrahedral_mu2_twist, tetrahedral_sphere
from src.twisted_merge_plus import TwistedMergePlus


def projective_pairwise(q: int, exponent: int) -> tuple[dict[tuple[int, int], np.ndarray], int]:
    order = torsion_order(q, exponent)
    zeta = root_of_unity(q, exponent)
    U = clock_matrix(order, zeta)
    V = shift_matrix(order)
    return {
        (0, 0): np.eye(order, dtype=complex),
        (1, 1): np.eye(order, dtype=complex),
        (2, 2): np.eye(order, dtype=complex),
        (0, 1): U,
        (1, 2): V,
        (2, 0): np.linalg.inv(U) @ np.linalg.inv(V),
    }, order


def identity_permutations(n_models: int, width: int) -> dict[tuple[int, int], np.ndarray]:
    return {
        (i, j): np.arange(width)
        for i in range(n_models)
        for j in range(n_models)
    }


def h2_triples() -> list[tuple[int, int, int]]:
    return [canonical_face(face) for face in tetrahedral_sphere().faces]


def identity_matrices(rank: int) -> dict[tuple[int, int], np.ndarray]:
    return {
        (i, j): np.eye(rank)
        for i in tetrahedral_sphere().vertices
        for j in tetrahedral_sphere().vertices
    }


class TwistedMergePlusFiniteIndexTests(unittest.TestCase):
    def test_order3_insufficient_rank_is_obstructed(self):
        pairwise, width = projective_pairwise(3, 1)
        result = TwistedMergePlus().run(
            pairwise,
            n_models=3,
            width=width,
            candidate_lift_rank=2,
        )

        self.assertEqual(result.diagnostics.classification, "finite_index_projective_obstructed")
        self.assertEqual(result.status, "finite_index_projective_obstructed")
        self.assertNotEqual(result.selected_method, "finite_index_projective_lift")
        self.assertEqual(result.diagnostics.root_order_d, 3)
        self.assertEqual(result.diagnostics.recommended_min_lift_rank, 3)
        self.assertFalse(result.diagnostics.determinant_obstruction_allows)
        self.assertFalse(result.diagnostics.rank_divisible_by_order)
        self.assertIsNone(result.finite_index_lift)

    def test_order3_sufficient_rank_activates_projective_lift(self):
        pairwise, width = projective_pairwise(3, 1)
        result = TwistedMergePlus().run(
            pairwise,
            n_models=3,
            width=width,
            candidate_lift_rank=3,
        )

        self.assertEqual(result.diagnostics.classification, "finite_index_projective_lift")
        self.assertEqual(result.status, "finite_index_projective_lift")
        self.assertEqual(result.selected_method, "finite_index_projective_lift")
        self.assertTrue(result.diagnostics.determinant_obstruction_allows)
        self.assertTrue(result.diagnostics.rank_divisible_by_order)
        self.assertLess(result.diagnostics.phase_residual, 1e-10)
        self.assertLess(result.diagnostics.finite_index_lift_residual, 1e-10)
        self.assertIsNotNone(result.finite_index_lift)

    def test_nonprimitive_q6_a2_has_order3_threshold(self):
        pairwise, width = projective_pairwise(6, 2)
        rejected = TwistedMergePlus().run(pairwise, n_models=3, width=width, candidate_lift_rank=2)
        accepted = TwistedMergePlus().run(pairwise, n_models=3, width=width, candidate_lift_rank=3)

        self.assertEqual(rejected.diagnostics.root_order_d, 3)
        self.assertEqual(rejected.diagnostics.classification, "finite_index_projective_obstructed")
        self.assertFalse(rejected.diagnostics.determinant_obstruction_allows)
        self.assertEqual(accepted.diagnostics.root_order_d, 3)
        self.assertEqual(accepted.diagnostics.classification, "finite_index_projective_lift")
        self.assertTrue(accepted.diagnostics.determinant_obstruction_allows)

    def test_q6_a3_order2_threshold(self):
        pairwise, width = projective_pairwise(6, 3)
        rejected = TwistedMergePlus().run(pairwise, n_models=3, width=width, candidate_lift_rank=1)
        accepted = TwistedMergePlus().run(pairwise, n_models=3, width=width, candidate_lift_rank=2)

        self.assertEqual(rejected.diagnostics.root_order_d, 2)
        self.assertEqual(rejected.diagnostics.classification, "finite_index_projective_obstructed")
        self.assertFalse(rejected.diagnostics.determinant_obstruction_allows)
        self.assertEqual(accepted.diagnostics.root_order_d, 2)
        self.assertEqual(accepted.diagnostics.classification, "finite_index_projective_lift")
        self.assertTrue(accepted.diagnostics.determinant_obstruction_allows)

    def test_random_noncentral_matrix_is_not_finite_index_lift(self):
        matrix = np.array(
            [
                [1.0, 0.4, 0.0],
                [0.0, 1.0, 0.2],
                [0.1, 0.0, 1.0],
            ]
        )
        pairwise = {
            (0, 0): np.eye(3),
            (1, 1): np.eye(3),
            (2, 2): np.eye(3),
            (0, 1): matrix,
            (1, 2): np.eye(3),
            (2, 0): np.eye(3),
        }
        result = TwistedMergePlus().run(pairwise, n_models=3, width=3, candidate_lift_rank=3)

        self.assertIn(result.diagnostics.classification, {"random_noncentral", "unknown"})
        self.assertNotEqual(result.selected_method, "finite_index_projective_lift")
        self.assertIsNone(result.diagnostics.root_order_d)
        self.assertIsNone(result.finite_index_lift)

    def test_c2m3_resolved_case_keeps_c2m3_priority(self):
        result = TwistedMergePlus().run(
            identity_permutations(n_models=4, width=6),
            n_models=4,
            width=6,
            candidate_lift_rank=3,
        )

        self.assertEqual(result.diagnostics.classification, "gauge_trivial")
        self.assertEqual(result.selected_method, "c2m3_cycle_consistent")
        self.assertIsNone(result.diagnostics.root_order_d)
        self.assertIsNone(result.finite_index_lift)

    def test_nonzero_h2_tetrahedral_case_not_mislabeled_finite_index_lift(self):
        result = TwistedMergePlus().run(
            identity_matrices(rank=2),
            n_models=4,
            width=2,
            known_alpha=nontrivial_tetrahedral_mu2_twist(tetrahedral_sphere()),
            triples=h2_triples(),
            candidate_lift_rank=2,
        )

        self.assertEqual(result.diagnostics.classification, "central_non_coboundary_candidate")
        self.assertEqual(result.selected_method, "branch_lift_extra_capacity")
        self.assertNotEqual(result.selected_method, "finite_index_projective_lift")
        self.assertIsNone(result.diagnostics.root_order_d)
        self.assertIsNone(result.finite_index_lift)


if __name__ == "__main__":
    unittest.main()

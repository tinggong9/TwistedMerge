import unittest

import numpy as np

from src.simplicial_mu2 import (
    canonical_face,
    nontrivial_tetrahedral_mu2_twist,
    tetrahedral_sphere,
    trivial_mu2_twist,
)
from src.twisted_merge_algorithm import lift_mu2_transition
from src.twisted_merge_plus import TwistedMergePlus


def triples() -> list[tuple[int, int, int]]:
    return [canonical_face(face) for face in tetrahedral_sphere().faces]


def identity_permutations(n_models: int, width: int) -> dict[tuple[int, int], np.ndarray]:
    return {
        (i, j): np.arange(width)
        for i in range(n_models)
        for j in range(n_models)
    }


def central_alignments(rank: int, finite_twist: bool = True) -> dict[tuple[int, int], np.ndarray]:
    maps = {
        (i, j): np.eye(rank)
        for i in tetrahedral_sphere().vertices
        for j in tetrahedral_sphere().vertices
    }
    if finite_twist:
        maps[(0, 2)] = -np.eye(rank)
        maps[(2, 0)] = -np.eye(rank)
    return maps


def finite_central_twist() -> dict[tuple[int, int, int], int]:
    twist = trivial_mu2_twist(tetrahedral_sphere())
    twist[(0, 1, 2)] = -1
    twist[(0, 2, 3)] = -1
    return twist


class TwistedMergePlusTests(unittest.TestCase):
    def test_zero_defect_reduces_to_untwisted_c2m3(self):
        result = TwistedMergePlus().run(
            identity_permutations(n_models=4, width=6),
            n_models=4,
            width=6,
        )

        self.assertEqual(result.diagnostics.classification, "gauge_trivial")
        self.assertEqual(result.status, "untwisted_c2m3")
        self.assertEqual(result.selected_method, "c2m3_cycle_consistent")
        self.assertAlmostEqual(result.diagnostics.c2m3_residual, 0.0)
        self.assertFalse(result.lifted_transition_maps)

    def test_zero_defect_with_trivial_alpha_still_reduces_to_c2m3(self):
        result = TwistedMergePlus().run(
            central_alignments(rank=2, finite_twist=False),
            n_models=4,
            width=2,
            known_alpha=trivial_mu2_twist(tetrahedral_sphere()),
            triples=triples(),
        )

        self.assertEqual(result.diagnostics.classification, "gauge_trivial")
        self.assertEqual(result.status, "untwisted_c2m3")
        self.assertFalse(result.lifted_transition_maps)

    def test_one_edge_noisy_permutation_is_not_called_central_twist(self):
        pairwise = identity_permutations(n_models=4, width=6)
        swap = np.arange(6)
        swap[0], swap[1] = swap[1], swap[0]
        pairwise[(0, 1)] = swap
        pairwise[(1, 0)] = swap

        result = TwistedMergePlus().run(pairwise, n_models=4, width=6)

        self.assertEqual(result.diagnostics.classification, "edge_outlier_or_noise")
        self.assertEqual(result.status, "untwisted_c2m3")
        self.assertEqual(result.selected_method, "c2m3_cycle_consistent")
        self.assertNotIn("central", result.diagnostics.classification)
        self.assertFalse(result.lifted_transition_maps)

    def test_planted_central_coboundary_sign_defect_builds_nontrivial_lift(self):
        alpha = finite_central_twist()
        result = TwistedMergePlus().run(
            central_alignments(rank=2, finite_twist=True),
            n_models=4,
            width=2,
            known_alpha=alpha,
            triples=triples(),
        )

        self.assertEqual(result.diagnostics.classification, "central_coboundary")
        self.assertEqual(result.status, "central_coboundary_lift")
        self.assertEqual(result.selected_method, "lifted_transition_merge")
        self.assertIsNotNone(result.edge_central_signs)
        self.assertEqual(result.edge_central_signs[(0, 2)], -1)
        expected = lift_mu2_transition(central_alignments(2)[(0, 2)], -1)
        placeholder = lift_mu2_transition(central_alignments(2)[(0, 2)], 1)
        np.testing.assert_allclose(result.lifted_transition_maps[(0, 2)], expected)
        self.assertGreater(np.linalg.norm(result.lifted_transition_maps[(0, 2)] - placeholder), 0.0)

    def test_wrong_supplied_alpha_is_rejected(self):
        wrong_alpha = trivial_mu2_twist(tetrahedral_sphere())
        result = TwistedMergePlus().run(
            central_alignments(rank=2, finite_twist=True),
            n_models=4,
            width=2,
            known_alpha=wrong_alpha,
            triples=triples(),
        )

        self.assertEqual(result.diagnostics.classification, "unknown")
        self.assertEqual(result.status, "failed")
        self.assertNotEqual(result.selected_method, "lifted_transition_merge")
        self.assertGreater(result.diagnostics.alpha_residual, 0.1)
        self.assertFalse(result.lifted_transition_maps)

    def test_random_noncentral_defect_refuses_twist_language(self):
        pairwise = identity_permutations(n_models=4, width=6)
        pairwise[(0, 1)] = np.array([1, 0, 2, 3, 4, 5])
        pairwise[(1, 0)] = np.array([1, 0, 2, 3, 4, 5])
        pairwise[(0, 2)] = np.array([0, 2, 1, 3, 4, 5])
        pairwise[(2, 0)] = np.array([0, 2, 1, 3, 4, 5])
        pairwise[(1, 3)] = np.array([0, 1, 3, 2, 4, 5])
        pairwise[(3, 1)] = np.array([0, 1, 3, 2, 4, 5])

        result = TwistedMergePlus().run(pairwise, n_models=4, width=6)

        self.assertIn(result.diagnostics.classification, {"random_noncentral", "unknown", "edge_outlier_or_noise"})
        self.assertNotIn(result.diagnostics.classification, {"central_coboundary", "central_non_coboundary_candidate"})
        self.assertNotEqual(result.selected_method, "lifted_transition_merge")

    def test_nonzero_h2_tetrahedral_alpha_is_not_silent_ordinary_descent(self):
        alpha = nontrivial_tetrahedral_mu2_twist(tetrahedral_sphere())
        result = TwistedMergePlus().run(
            central_alignments(rank=2, finite_twist=False),
            n_models=4,
            width=2,
            known_alpha=alpha,
            triples=triples(),
        )

        self.assertEqual(result.diagnostics.classification, "central_non_coboundary_candidate")
        self.assertEqual(result.status, "twisted_branch_lift")
        self.assertEqual(result.selected_method, "branch_lift_extra_capacity")
        self.assertGreater(result.diagnostics.alpha_residual, 0.0)
        self.assertFalse(result.lifted_transition_maps)
        self.assertIn("not an ordinary untwisted descent claim", " ".join(result.notes))


if __name__ == "__main__":
    unittest.main()

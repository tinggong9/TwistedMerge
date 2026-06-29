import unittest

import numpy as np

from src.simplicial_mu2 import (
    LinearLocalModel,
    canonical_face,
    is_coboundary_mu2,
    nontrivial_tetrahedral_mu2_twist,
    tetrahedral_sphere,
    trivial_mu2_twist,
)
from src.twisted_merge_algorithm import (
    TwistedMerge,
    TwistedMergeConfig,
    evaluate_vector_model,
    lift_mu2_transition,
    solve_mu2_edge_cochain,
)


def base_weight(rank: int) -> np.ndarray:
    weight = np.zeros(rank)
    weight[0] = 1.0
    return weight


def local_models(rank: int) -> list[LinearLocalModel]:
    return [LinearLocalModel(weight=base_weight(rank).copy()) for _ in tetrahedral_sphere().vertices]


def triples() -> list[tuple[int, int, int]]:
    return [canonical_face(face) for face in tetrahedral_sphere().faces]


def finite_central_twist() -> dict[tuple[int, int, int], int]:
    twist = trivial_mu2_twist(tetrahedral_sphere())
    twist[(0, 1, 2)] = -1
    twist[(0, 2, 3)] = -1
    return twist


def alignments(rank: int, finite_twist: bool, noise: float = 0.0) -> dict[tuple[int, int], np.ndarray]:
    maps = {}
    for i in tetrahedral_sphere().vertices:
        for j in tetrahedral_sphere().vertices:
            maps[(i, j)] = np.eye(rank)
    if finite_twist:
        scale = 1.0 + noise
        maps[(0, 2)] = -scale * np.eye(rank)
        maps[(2, 0)] = -scale * np.eye(rank)
    return maps


def face_data(twist: dict[tuple[int, int, int], int], rank: int):
    x = np.zeros((2, rank))
    x[0, 0] = 1.0
    x[1, 0] = -1.0
    data = {}
    for face in tetrahedral_sphere().faces:
        key = canonical_face(face)
        y = (twist[key] * (x @ base_weight(rank)) >= 0.0).astype(np.int64)
        data[key] = (x.copy(), y)
    return data


def run_tm(rank: int, q: int, alpha, finite_twist: bool, noise: float = 0.0, central_tolerance: float = 1e-5):
    tm = TwistedMerge(TwistedMergeConfig(rank_lift_q=q, central_tolerance=central_tolerance))
    return tm.run(
        local_models(rank),
        pairwise_alignments=alignments(rank, finite_twist=finite_twist, noise=noise),
        alpha=alpha,
        triples=triples(),
    )


class TwistedMergeAlgorithmTests(unittest.TestCase):
    def test_trivial_twist_q1_is_ordinary(self):
        rank = 2
        alpha = trivial_mu2_twist(tetrahedral_sphere())
        result = run_tm(rank=rank, q=1, alpha=alpha, finite_twist=False)
        metrics = TwistedMerge().evaluate(result, face_data(alpha, rank))

        self.assertEqual(result.status, "ordinary")
        self.assertTrue(result.gauge.success)
        self.assertIsNone(result.twisted_model)
        self.assertAlmostEqual(metrics["ordinary_merge"]["zero_one_loss"], 0.0)

    def test_nontrivial_finite_central_twist_q1_fails(self):
        rank = 2
        alpha = finite_central_twist()
        result = run_tm(rank=rank, q=1, alpha=alpha, finite_twist=True)

        self.assertEqual(result.status, "failed")
        self.assertFalse(result.gauge.success)
        self.assertIsNone(result.twisted_model)
        self.assertAlmostEqual(result.twist_residual, 0.0)

    def test_nontrivial_finite_central_twist_q2_builds_branch_lift(self):
        rank = 2
        alpha = finite_central_twist()
        result = run_tm(rank=rank, q=2, alpha=alpha, finite_twist=True)
        metrics = TwistedMerge().evaluate(result, face_data(alpha, rank))

        self.assertEqual(result.status, "twisted_rank_lifted")
        self.assertFalse(result.gauge.success)
        self.assertIsNotNone(result.twisted_model)
        self.assertAlmostEqual(result.twist_residual, 0.0)
        self.assertAlmostEqual(metrics["twisted_merge"]["zero_one_loss"], 0.0)
        self.assertAlmostEqual(metrics["ordinary_merge"]["zero_one_loss"], 0.5)

    def test_nontrivial_finite_central_twist_without_alpha_fails(self):
        result = run_tm(rank=2, q=2, alpha=None, finite_twist=True)

        self.assertEqual(result.status, "failed")
        self.assertFalse(result.gauge.success)
        self.assertIsNone(result.twisted_model)
        self.assertIsNone(result.twist_residual)

    def test_wrong_alpha_fails_with_nonzero_residual(self):
        wrong_alpha = trivial_mu2_twist(tetrahedral_sphere())
        result = run_tm(rank=2, q=2, alpha=wrong_alpha, finite_twist=True)

        self.assertEqual(result.status, "failed")
        self.assertFalse(result.gauge.success)
        self.assertIsNone(result.twisted_model)
        self.assertGreater(result.twist_residual, 0.1)

    def test_noisy_central_defects_respect_tolerance(self):
        alpha = finite_central_twist()
        loose = run_tm(rank=2, q=2, alpha=alpha, finite_twist=True, noise=1e-4, central_tolerance=1e-3)
        strict = run_tm(rank=2, q=2, alpha=alpha, finite_twist=True, noise=1e-4, central_tolerance=1e-6)

        self.assertEqual(loose.status, "twisted_rank_lifted")
        self.assertGreater(loose.twist_residual, 0.0)
        self.assertLess(loose.twist_residual, 1e-3)
        self.assertEqual(strict.status, "failed")
        self.assertGreater(strict.twist_residual, 1e-6)

    def test_h2_nontrivial_tetrahedral_twist_is_not_absorbed_by_algorithm(self):
        rank = 2
        h2_alpha = nontrivial_tetrahedral_mu2_twist(tetrahedral_sphere())
        result = run_tm(rank=rank, q=2, alpha=h2_alpha, finite_twist=False)
        metrics = TwistedMerge().evaluate(result, face_data(h2_alpha, rank))

        self.assertFalse(is_coboundary_mu2(h2_alpha, tetrahedral_sphere()))
        self.assertEqual(result.status, "ordinary")
        self.assertTrue(result.gauge.success)
        self.assertIsNone(result.twisted_model)
        self.assertGreater(result.twist_residual, 0.0)
        self.assertAlmostEqual(metrics["ordinary_merge"]["zero_one_loss"], 0.25)

    def test_lifted_transition_maps_encode_nontrivial_edge_sign(self):
        rank = 2
        alpha = finite_central_twist()
        result = run_tm(rank=rank, q=2, alpha=alpha, finite_twist=True)
        edge_cochain = solve_mu2_edge_cochain(alpha, n_models=4, triples=triples())

        self.assertIsNotNone(edge_cochain)
        self.assertEqual(edge_cochain[(0, 2)], -1)
        expected = lift_mu2_transition(alignments(rank, True)[(0, 2)], -1)
        placeholder = lift_mu2_transition(alignments(rank, True)[(0, 2)], 1)
        self.assertIn((0, 2), result.lifted_transition_maps)
        np.testing.assert_allclose(result.lifted_transition_maps[(0, 2)], expected)
        self.assertGreater(np.linalg.norm(result.lifted_transition_maps[(0, 2)] - placeholder), 0.0)


if __name__ == "__main__":
    unittest.main()

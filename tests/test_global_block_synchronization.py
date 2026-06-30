import unittest

import numpy as np

from src.global_block_synchronization import (
    cycle_score,
    default_triples,
    global_block_spectral_synchronization,
    global_sync_accepted,
    triangle_defects,
)
from src.structure_group_ladder import StructureGroupLadderMerge


def level(result, name):
    return next(diag for diag in result.diagnostics if diag.level == name)


def rotation(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def maps_from_gauges(gauges):
    maps = {}
    for i, qi in gauges.items():
        for j, qj in gauges.items():
            maps[(i, j)] = qi @ qj.T
    return maps


class GlobalBlockSynchronizationTests(unittest.TestCase):
    def test_recovers_planted_globally_consistent_block_gauges(self):
        gauges = {0: np.eye(2), 1: rotation(0.3), 2: rotation(-0.45), 3: rotation(0.8)}
        pairwise = maps_from_gauges(gauges)
        blocks = {idx: [np.array([0, 1])] for idx in gauges}

        result = global_block_spectral_synchronization(pairwise, blocks, n_models=4, width=2)

        self.assertLess(result.connection_residual, 1e-10)
        self.assertTrue(global_sync_accepted(result, tolerance=1e-8))
        self.assertLess(cycle_score(triangle_defects(result.synchronized_maps, default_triples(4))), 1e-10)
        for key, matrix in pairwise.items():
            np.testing.assert_allclose(result.synchronized_maps[key], matrix, atol=1e-10)

    def test_noisy_pairwise_connection_is_projected_but_not_silently_exact(self):
        gauges = {0: np.eye(2), 1: rotation(0.2), 2: rotation(-0.5)}
        pairwise = maps_from_gauges(gauges)
        pairwise[(0, 1)] = pairwise[(0, 1)] @ rotation(0.25)
        pairwise[(1, 0)] = pairwise[(0, 1)].T
        blocks = {idx: [np.array([0, 1])] for idx in gauges}
        original_cycle = cycle_score(triangle_defects(pairwise, [(0, 1, 2)]))

        result = global_block_spectral_synchronization(pairwise, blocks, n_models=3, width=2)

        self.assertGreater(original_cycle, 0.05)
        self.assertLess(cycle_score(triangle_defects(result.synchronized_maps, [(0, 1, 2)])), 1e-10)
        self.assertGreater(result.connection_residual, 0.01)

    def test_noncentral_block_holonomy_is_rejected_by_connection_residual(self):
        reflection = np.array([[0.0, 1.0], [1.0, 0.0]])
        rot = rotation(0.4)
        pairwise = {
            (0, 0): np.eye(2),
            (1, 1): np.eye(2),
            (2, 2): np.eye(2),
            (0, 1): reflection,
            (1, 2): rot,
            (2, 0): np.linalg.inv(reflection) @ np.linalg.inv(rot),
        }
        pairwise[(1, 0)] = pairwise[(0, 1)].T
        pairwise[(2, 1)] = pairwise[(1, 2)].T
        pairwise[(0, 2)] = pairwise[(2, 0)].T
        blocks = {idx: [np.array([0, 1])] for idx in range(3)}

        result = global_block_spectral_synchronization(pairwise, blocks, n_models=3, width=2)
        diag = level(StructureGroupLadderMerge().run({"block_orthogonal": pairwise}, n_models=3, width=2), "block_orthogonal")

        self.assertEqual(diag.residual_type, "block_noncentral_holonomy")
        self.assertGreater(result.connection_residual, 0.1)
        self.assertFalse(global_sync_accepted(result, tolerance=0.05))

    def test_scalar_block_phase_detected_before_projection(self):
        pairwise = {
            (0, 0): np.eye(4),
            (1, 1): np.eye(4),
            (2, 2): np.eye(4),
            (0, 1): np.eye(4),
            (1, 2): np.eye(4),
            (2, 0): -np.eye(4),
            (1, 0): np.eye(4),
            (2, 1): np.eye(4),
            (0, 2): -np.eye(4),
        }

        diag = level(StructureGroupLadderMerge().run({"block_orthogonal": pairwise}, n_models=3, width=4), "block_orthogonal")

        self.assertEqual(diag.residual_type, "central_projective_after_block")
        self.assertEqual(diag.detected_order_d, 2)
        self.assertTrue(diag.supports_brauer_projective_interpretation)


if __name__ == "__main__":
    unittest.main()

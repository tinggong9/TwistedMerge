import unittest

import numpy as np

from src.block_sync_calibration import (
    calibrate_connection_residual_threshold,
    classify_sync_evidence,
)
from src.global_block_synchronization import (
    cycle_score,
    default_triples,
    global_block_spectral_synchronization,
    residual_optimized_global_block_sync,
    triangle_defects,
)


def rotation(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    out = np.zeros((size, size), dtype=float)
    cursor = 0
    for block in blocks:
        n = block.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n
    return out


def maps_from_gauges(gauges: dict[int, np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    return {(i, j): gauges[i] @ gauges[j].T for i in gauges for j in gauges}


class OptimizedGlobalBlockSynchronizationTests(unittest.TestCase):
    def test_residual_optimized_recovers_exact_global_block_gauges(self):
        gauges = {
            0: block_diag([rotation(0.0), rotation(0.0)]),
            1: block_diag([rotation(0.35), rotation(-0.1)]),
            2: block_diag([rotation(-0.25), rotation(0.45)]),
            3: block_diag([rotation(0.6), rotation(-0.3)]),
        }
        blocks = {idx: [np.array([0, 1]), np.array([2, 3])] for idx in gauges}
        pairwise = maps_from_gauges(gauges)

        result = residual_optimized_global_block_sync(pairwise, blocks, n_models=4, width=4, n_restarts=3)

        self.assertLess(result.connection_residual, 1e-10)
        self.assertLess(cycle_score(triangle_defects(result.synchronized_maps, default_triples(4))), 1e-10)
        for key, matrix in pairwise.items():
            np.testing.assert_allclose(result.synchronized_maps[key], matrix, atol=1e-10)

    def test_residual_optimized_can_reduce_connection_residual_beyond_spectral(self):
        rng = np.random.default_rng(0)
        gauges = {
            i: block_diag([rotation(rng.normal()), rotation(rng.normal())])
            for i in range(4)
        }
        blocks = {idx: [np.array([0, 1]), np.array([2, 3])] for idx in gauges}
        pairwise = maps_from_gauges(gauges)
        for i, j in [(0, 1), (1, 2), (2, 3)]:
            perturbation = block_diag([rotation(0.7 * rng.normal()), rotation(0.7 * rng.normal())])
            pairwise[(i, j)] = pairwise[(i, j)] @ perturbation
            pairwise[(j, i)] = pairwise[(i, j)].T

        spectral = global_block_spectral_synchronization(pairwise, blocks, n_models=4, width=4)
        optimized = residual_optimized_global_block_sync(
            pairwise,
            blocks,
            n_models=4,
            width=4,
            lambda_feature=0.0,
            max_iters=100,
            tolerance=1e-8,
            n_restarts=20,
            seed=0,
        )

        self.assertLess(optimized.connection_residual, spectral.connection_residual - 1e-3)
        self.assertLessEqual(optimized.connection_residual, optimized.initial_connection_residual)

    def test_projected_cycle_trap_is_rejected_when_connection_residual_large(self):
        calibration = calibrate_connection_residual_threshold(
            positive_residuals=[0.0, 0.01, 0.02],
            negative_residuals=[0.2, 0.3, 0.5],
            target_false_positive_rate=0.0,
        )

        label = classify_sync_evidence(
            observed_scalar_projective_candidate=False,
            observed_centrality_score=0.8,
            projected_cycle_score=0.0,
            connection_residual=0.25,
            calibration=calibration,
        )

        self.assertEqual(label, "projected_cycle_only_connection_large")


if __name__ == "__main__":
    unittest.main()

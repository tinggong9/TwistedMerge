import unittest

from experiments.block_gauge_phase_diagram import make_family_maps
from src.global_block_synchronization import (
    global_block_spectral_synchronization,
    residual_optimized_global_block_sync,
)


class OptimizedGlobalBlockMultiSeedTests(unittest.TestCase):
    def test_optimized_residual_does_not_worsen_exact_planted_gauges(self):
        for seed in range(5):
            maps, blocks, _label, _accept = make_family_maps("exact_global_block_gauge", 4, 8, 2, 0.0, seed)
            spectral = global_block_spectral_synchronization(maps, blocks, 4, 8)
            optimized = residual_optimized_global_block_sync(
                maps,
                blocks,
                4,
                8,
                lambda_feature=0.0,
                max_iters=8,
                tolerance=1e-7,
                n_restarts=1,
                seed=seed,
            )
            self.assertLessEqual(optimized.connection_residual, spectral.connection_residual + 1e-8)


if __name__ == "__main__":
    unittest.main()

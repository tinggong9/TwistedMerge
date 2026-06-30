import unittest

import numpy as np

from src.improved_monomial_merge import global_log_scale_synchronization, reference_log_scales_from_features


class GlobalMonomialSynchronizationTests(unittest.TestCase):
    def test_recovers_planted_per_model_scales(self):
        rng = np.random.default_rng(42)
        base = rng.uniform(0.1, 2.0, size=(80, 5))
        planted = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, -0.1, 0.4, 0.0, -0.3],
                [-0.5, 0.3, 0.1, -0.2, 0.25],
            ]
        )
        features = {idx: base * np.exp(planted[idx]) for idx in range(3)}
        perms = {idx: np.arange(5) for idx in range(3)}

        result = global_log_scale_synchronization(features, perms, n_models=3, width=5, ref=0)

        np.testing.assert_allclose(result.log_scales, planted, atol=1e-10)
        self.assertLess(result.rms_residual, 1e-10)
        self.assertLess(result.max_residual, 1e-10)

    def test_reference_log_scales_match_reference_ratios(self):
        rng = np.random.default_rng(7)
        base = rng.uniform(0.2, 1.5, size=(40, 3))
        logs = np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.1]])
        features = {0: base, 1: base * np.exp(logs[1])}
        perms = {0: np.arange(3), 1: np.arange(3)}

        estimated = reference_log_scales_from_features(features, perms, ref=0, width=3)

        np.testing.assert_allclose(estimated, logs, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

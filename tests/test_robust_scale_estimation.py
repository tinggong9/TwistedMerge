import unittest

import numpy as np

from src.greedy_aware_monomial import (
    estimate_positive_scale,
    huber_ratio_scale,
    least_squares_scale,
    median_ratio_scale,
    trimmed_mean_ratio_scale,
)


class RobustScaleEstimationTests(unittest.TestCase):
    def test_robust_scale_estimators_return_positive_finite_values(self):
        rng = np.random.default_rng(5)
        source = rng.uniform(0.1, 2.0, size=100)
        target = 1.7 * source + 0.02 * rng.normal(size=100)
        target[0] = 200.0

        for fn in [least_squares_scale, median_ratio_scale, trimmed_mean_ratio_scale, huber_ratio_scale]:
            scale = fn(source, target)
            self.assertTrue(np.isfinite(scale), fn.__name__)
            self.assertGreater(scale, 0.0, fn.__name__)

    def test_dispatcher_rejects_unknown_estimator(self):
        with self.assertRaises(ValueError):
            estimate_positive_scale(np.array([1.0]), np.array([1.0]), estimator="not_real")

    def test_median_ratio_is_less_sensitive_to_outlier_than_least_squares(self):
        source = np.ones(10)
        target = np.array([2.0] * 9 + [100.0])

        self.assertLess(median_ratio_scale(source, target), least_squares_scale(source, target))


if __name__ == "__main__":
    unittest.main()

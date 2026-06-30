import unittest

import numpy as np

from src.greedy_safe_selector import (
    nested_validation_selector,
    regret_bound_selector,
    tau_bootstrap_selector,
    tau_fixed_selector,
    tau_loss_aware_selector,
)


class GreedySafeSelectorTests(unittest.TestCase):
    def test_tau_fixed_keeps_greedy_when_margin_is_too_small(self):
        metrics = {
            "greedy_soup": {"accuracy": 0.8200, "loss": 0.45},
            "union_candidate_soup": {"accuracy": 0.8204, "loss": 0.44},
        }

        choice = tau_fixed_selector(metrics, tau_accuracy=0.0005)

        self.assertEqual(choice.selected, "greedy_soup")
        self.assertEqual(choice.challenger, "union_candidate_soup")
        self.assertFalse(choice.used_test_metrics)

    def test_tau_fixed_accepts_clear_validation_gain(self):
        metrics = {
            "greedy_soup": {"accuracy": 0.8200, "loss": 0.45},
            "optimized_monomial_scale": {"accuracy": 0.8230, "loss": 0.46},
        }

        choice = tau_fixed_selector(metrics, tau_accuracy=0.002)

        self.assertEqual(choice.selected, "optimized_monomial_scale")

    def test_loss_aware_accepts_accuracy_tie_with_loss_improvement(self):
        metrics = {
            "greedy_soup": {"accuracy": 0.8200, "loss": 0.450},
            "union_candidate_soup": {"accuracy": 0.8200, "loss": 0.444},
        }

        choice = tau_loss_aware_selector(metrics, tau_accuracy=0.001, tau_loss=0.002)

        self.assertEqual(choice.selected, "union_candidate_soup")

    def test_bootstrap_selector_uses_correctness_arrays_when_available(self):
        metrics = {
            "greedy_soup": {"accuracy": 0.50, "loss": 0.70},
            "union_candidate_soup": {"accuracy": 0.75, "loss": 0.60},
        }
        correctness = {
            "greedy_soup": np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=bool),
            "union_candidate_soup": np.array([1, 1, 1, 0, 1, 1, 0, 1], dtype=bool),
        }

        choice = tau_bootstrap_selector(metrics, correctness_by_name=correctness, confidence=0.80, n_bootstrap=200, seed=7)

        self.assertEqual(choice.selected, "union_candidate_soup")
        self.assertIsNotNone(choice.lower_confidence_bound)
        self.assertFalse(choice.used_test_metrics)

    def test_nested_validation_accepts_on_accept_split_only(self):
        selector_metrics = {
            "greedy_soup": {"accuracy": 0.820, "loss": 0.45},
            "union_candidate_soup": {"accuracy": 0.830, "loss": 0.44},
        }
        accept_metrics = {
            "greedy_soup": {"accuracy": 0.820, "loss": 0.45},
            "union_candidate_soup": {"accuracy": 0.819, "loss": 0.44},
        }

        choice = nested_validation_selector(selector_metrics, accept_metrics, tau_accuracy=0.0)

        self.assertEqual(choice.challenger, "union_candidate_soup")
        self.assertEqual(choice.selected, "greedy_soup")

    def test_regret_bound_is_conservative_for_uncertain_gain(self):
        metrics = {
            "greedy_soup": {"accuracy": 0.8200, "loss": 0.45},
            "union_candidate_soup": {"accuracy": 0.8201, "loss": 0.44},
        }

        choice = regret_bound_selector(metrics, n_validation=100, regret_threshold=0.0)

        self.assertEqual(choice.selected, "greedy_soup")
        self.assertIsNotNone(choice.predicted_regret_bound)


if __name__ == "__main__":
    unittest.main()

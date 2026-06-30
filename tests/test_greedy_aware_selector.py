import unittest

from src.greedy_aware_monomial import (
    greedy_aware_selector,
    lower_confidence_greedy_aware_selector,
)


class GreedyAwareSelectorTests(unittest.TestCase):
    def test_falls_back_to_greedy_when_margin_below_epsilon(self):
        metrics = {
            "greedy_soup": {"accuracy": 0.8000, "loss": 0.50},
            "union_candidate_soup": {"accuracy": 0.8004, "loss": 0.49},
        }

        choice = greedy_aware_selector(metrics, epsilon=0.001)

        self.assertEqual(choice.selected, "greedy_soup")
        self.assertEqual(choice.challenger, "union_candidate_soup")
        self.assertFalse(choice.used_test_metrics)

    def test_chooses_challenger_when_margin_exceeds_epsilon(self):
        metrics = {
            "greedy_soup": {"accuracy": 0.8000, "loss": 0.50},
            "global_monomial_greedy_soup": {"accuracy": 0.8030, "loss": 0.505},
        }

        choice = greedy_aware_selector(metrics, epsilon=0.002, loss_slack=0.01)

        self.assertEqual(choice.selected, "global_monomial_greedy_soup")
        self.assertFalse(choice.used_test_metrics)

    def test_loss_slack_can_block_high_accuracy_challenger(self):
        metrics = {
            "greedy_soup": {"accuracy": 0.8000, "loss": 0.50},
            "optimized_monomial_greedy_soup": {"accuracy": 0.8050, "loss": 0.60},
        }

        choice = greedy_aware_selector(metrics, epsilon=0.001, loss_slack=0.01)

        self.assertEqual(choice.selected, "greedy_soup")

    def test_lower_confidence_selector_does_not_use_test_metrics(self):
        metrics = {
            "greedy_soup": {"accuracy": 0.8000, "loss": 0.50},
            "union_candidate_soup": {"accuracy": 0.8100, "loss": 0.49},
        }

        choice = lower_confidence_greedy_aware_selector(metrics, n_validation=1000)

        self.assertFalse(choice.used_test_metrics)
        self.assertIsNotNone(choice.lower_confidence_bound)


if __name__ == "__main__":
    unittest.main()

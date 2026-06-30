import unittest

from src.improved_monomial_merge import choose_by_validation, margin_selector


class ValidationSelectorNoLeakageTests(unittest.TestCase):
    def test_regret_selector_uses_validation_metrics_only(self):
        val_metrics = {
            "c2m3_permutation": {"accuracy": 0.80, "loss": 0.55},
            "optimized_monomial_scale": {"accuracy": 0.79, "loss": 0.50},
            "greedy_soup": {"accuracy": 0.78, "loss": 0.45},
        }

        choice = choose_by_validation(val_metrics)

        self.assertEqual(choice.selected, "c2m3_permutation")
        self.assertFalse(choice.used_test_metrics)

    def test_validation_loss_breaks_accuracy_ties(self):
        val_metrics = {
            "c2m3_permutation": {"accuracy": 0.80, "loss": 0.60},
            "shrinkage_monomial_scale": {"accuracy": 0.80, "loss": 0.50},
        }

        choice = choose_by_validation(val_metrics)

        self.assertEqual(choice.selected, "shrinkage_monomial_scale")
        self.assertFalse(choice.used_test_metrics)

    def test_margin_selector_requires_validation_margin(self):
        val_metrics = {
            "c2m3_permutation": {"accuracy": 0.800, "loss": 0.55},
            "monomial_scale": {"accuracy": 0.801, "loss": 0.54},
        }

        conservative = margin_selector("c2m3_permutation", "monomial_scale", val_metrics, epsilon=0.002)
        permissive = margin_selector("c2m3_permutation", "monomial_scale", val_metrics, epsilon=0.001)

        self.assertEqual(conservative.selected, "c2m3_permutation")
        self.assertEqual(permissive.selected, "monomial_scale")
        self.assertFalse(conservative.used_test_metrics)
        self.assertFalse(permissive.used_test_metrics)


if __name__ == "__main__":
    unittest.main()

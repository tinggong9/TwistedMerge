import unittest

import pandas as pd

from src.greedy_aware_monomial import selector_regret_analysis


class SelectorRegretAnalysisTests(unittest.TestCase):
    def test_selector_regret_counts_toy_table(self):
        rows = [
            {
                "setting_id": "a",
                "method": "greedy_soup",
                "accuracy": 0.80,
                "loss": 0.5,
                "val_accuracy": 0.80,
                "val_loss": 0.5,
            },
            {
                "setting_id": "a",
                "method": "challenger",
                "accuracy": 0.82,
                "loss": 0.4,
                "val_accuracy": 0.82,
                "val_loss": 0.4,
            },
            {
                "setting_id": "a",
                "method": "selector",
                "accuracy": 0.82,
                "loss": 0.4,
                "val_accuracy": 0.82,
                "val_loss": 0.4,
                "selector_chose": "challenger",
            },
            {
                "setting_id": "b",
                "method": "greedy_soup",
                "accuracy": 0.85,
                "loss": 0.3,
                "val_accuracy": 0.84,
                "val_loss": 0.3,
            },
            {
                "setting_id": "b",
                "method": "challenger",
                "accuracy": 0.83,
                "loss": 0.4,
                "val_accuracy": 0.86,
                "val_loss": 0.2,
            },
            {
                "setting_id": "b",
                "method": "selector",
                "accuracy": 0.83,
                "loss": 0.4,
                "val_accuracy": 0.86,
                "val_loss": 0.2,
                "selector_chose": "challenger",
            },
        ]

        result = selector_regret_analysis(
            pd.DataFrame(rows),
            selector_methods=["selector"],
            candidate_methods=["greedy_soup", "challenger"],
        )

        row = result.iloc[0]
        self.assertEqual(int(row["beats_greedy"]), 1)
        self.assertEqual(int(row["loses_to_greedy"]), 1)
        self.assertEqual(float(row["false_challenger_rate"]), 0.5)
        self.assertFalse(bool(row["used_test_metrics"]))


if __name__ == "__main__":
    unittest.main()

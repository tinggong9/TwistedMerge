import unittest

import numpy as np
import pandas as pd

from experiments.model_merging_fixed_setting_verification import (
    PREDICTION_TARGETS,
    add_paired_deltas,
    compute_predictor_regressions,
    triangle_predictor_summary,
)


class RealObstructionPredictorTests(unittest.TestCase):
    def test_triangle_predictor_summary_identity(self):
        width = 4
        pairwise = {(i, j): np.arange(width) for i in range(3) for j in range(3)}

        summary = triangle_predictor_summary(pairwise, n_models=3, width=width)

        self.assertEqual(summary["mean_cycle_score"], 0.0)
        self.assertEqual(summary["max_cycle_score"], 0.0)
        self.assertEqual(summary["median_cycle_score"], 0.0)
        self.assertEqual(summary["nonidentity_triangle_fraction"], 0.0)

    def test_triangle_predictor_summary_nonidentity(self):
        width = 4
        identity = np.arange(width)
        swap = np.array([1, 0, 2, 3])
        pairwise = {(i, j): identity.copy() for i in range(3) for j in range(3)}
        pairwise[(0, 1)] = swap

        summary = triangle_predictor_summary(pairwise, n_models=3, width=width)

        self.assertGreater(summary["mean_cycle_score"], 0.0)
        self.assertEqual(summary["nonidentity_triangle_fraction"], 1.0)

    def test_response_variables_are_added_to_every_method_row(self):
        rows = [
            {"method": "weight_average", "test_accuracy": 0.70},
            {"method": "git_rebasin_pairwise_ref0", "test_accuracy": 0.75},
            {"method": "c2m3_synchronized", "test_accuracy": 0.80},
            {"method": "greedy_soup", "test_accuracy": 0.82},
            {"method": "twisted_rank_lift_2", "test_accuracy": 0.85},
        ]

        add_paired_deltas(rows, single_best_accuracy=0.90, mean_individual_accuracy=0.78)

        for row in rows:
            for target in PREDICTION_TARGETS:
                self.assertIn(target, row)
        self.assertAlmostEqual(rows[0]["weight_average_degradation_vs_best_single"], 0.20)
        self.assertAlmostEqual(rows[0]["git_rebasin_degradation_vs_best_single"], 0.15)
        self.assertAlmostEqual(rows[0]["c2m3_delta_vs_git_rebasin"], 0.05)
        self.assertAlmostEqual(rows[0]["rank_lift_delta_vs_c2m3"], 0.05)
        self.assertAlmostEqual(rows[0]["greedy_soup_delta_vs_weight_average"], 0.12)

    def test_predictor_regression_marks_positive_observed_target_supported(self):
        rng = np.random.default_rng(123)
        rows = []
        for seed in range(30):
            x = seed / 29.0
            mean_acc = 0.70 + 0.04 * rng.random()
            residual = 0.05 + 0.10 * rng.random()
            outcome = 0.10 + 0.60 * x + 0.03 * mean_acc - 0.02 * residual + 0.005 * rng.normal()
            row = {
                "method": "weight_average",
                "dataset": "toy",
                "architecture": "mlp",
                "n_models": 3,
                "width": 4,
                "domain_shift": "none",
                "matching": "weight",
                "alignment_source": "observed",
                "alignment_noise_fraction": 0.0,
                "seed": seed,
                "mean_cycle_score": x,
                "combined_obstruction_score": x,
                "sync_disagreement": x,
                "mean_individual_accuracy": mean_acc,
                "pairwise_alignment_residual_mean": residual,
            }
            for target in PREDICTION_TARGETS:
                row[target] = outcome
            rows.append(row)

        regressions = compute_predictor_regressions(pd.DataFrame(rows), bootstrap_samples=50)
        match = regressions[
            (regressions["outcome"] == "weight_average_degradation_vs_best_single")
            & (regressions["predictor"] == "mean_cycle_score")
        ].iloc[0]

        self.assertGreater(match["predictor_beta"], 0.0)
        self.assertEqual(match["claim_status"], "supported_positive_predictor_coefficient")
        self.assertTrue(match["claim_supported"])


if __name__ == "__main__":
    unittest.main()

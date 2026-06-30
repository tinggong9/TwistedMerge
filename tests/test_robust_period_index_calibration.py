import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class RobustPeriodIndexCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.reports_dir = Path(cls.tmp.name)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "robust_period_index_calibration.py"),
                "--reports-dir",
                str(cls.reports_dir),
                "--seeds",
                "2",
                "--noise-levels",
                "0",
                "1e-6",
                "--skip-plots",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cls.rows = pd.read_csv(cls.reports_dir / "csv" / "robust_period_index_calibration.csv")
        cls.summary = pd.read_csv(cls.reports_dir / "csv" / "robust_period_index_calibration_summary.csv")
        cls.policies = pd.read_csv(
            cls.reports_dir / "csv" / "robust_period_index_calibration_threshold_policies.csv"
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_calibration_experiment_produces_nonempty_csvs(self):
        self.assertGreater(len(self.rows), 0)
        self.assertGreater(len(self.summary), 0)
        self.assertGreater(len(self.policies), 0)

    def test_summary_contains_positive_and_negative_rows(self):
        self.assertIn("central_positive", set(self.summary["source"]))
        self.assertIn("noncentral_negative", set(self.summary["source"]))

    def test_noise_zero_exact_central_cases_are_certified(self):
        expected = {
            ("heisenberg_d2_k2", 4): (2, 4),
            ("heisenberg_d3_k2", 9): (3, 9),
            ("heisenberg_d2_k3", 8): (2, 8),
            ("rank_deficient_d3_one_pair", 3): (3, 3),
            ("composite_d4_k1", 4): (4, 4),
        }
        for (case_id, rank), (period, index) in expected.items():
            subset = self.rows[
                (self.rows["case_id"] == case_id)
                & (self.rows["candidate_rank"] == rank)
                & (self.rows["noise_level"] == 0)
            ]
            self.assertFalse(subset.empty)
            self.assertTrue((subset["detector_status"] == "certified").all())
            self.assertTrue((subset["detected_period"] == period).all())
            self.assertTrue((subset["detected_index"] == index).all())

    def test_period_divisible_but_index_obstructed_ranks_never_select_lift(self):
        obstructed = self.rows[self.rows["expected_decision"] == "period_divisible_index_obstructed"]
        self.assertFalse(obstructed.empty)
        self.assertTrue((obstructed["selected_method"] == "none").all())

    def test_noncentral_negative_controls_never_select_lift(self):
        negative = self.rows[self.rows["source"] == "noncentral_negative"]
        self.assertFalse(negative.empty)
        self.assertNotIn("period_index_projective_morita_lift", set(negative["selected_method"]))

    def test_trivial_abelian_controls_not_labeled_nontrivial_period_index(self):
        trivial = self.rows[self.rows["source"] == "trivial_abelian_negative"]
        self.assertFalse(trivial.empty)
        self.assertTrue((trivial["decision"] == "not_central_projective").all())
        self.assertTrue((trivial["selected_method"] == "none").all())

    def test_threshold_recommendation_preserves_certified_only_lift_policy(self):
        recommended = self.policies[self.policies["recommended"]]
        self.assertEqual(len(recommended), 1)
        row = recommended.iloc[0]
        self.assertEqual(row["false_positive_lift_rate"], 0)
        false_lifts = self.rows[
            (self.rows["selected_method"] == "period_index_projective_morita_lift")
            & (self.rows["expected_decision"] != "period_index_lift_success")
        ]
        self.assertTrue(false_lifts.empty)


if __name__ == "__main__":
    unittest.main()

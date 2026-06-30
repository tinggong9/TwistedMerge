import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.period_index_mining import project_to_nearest_unitary
from src.time_frequency_benchmark import generate_paired_time_frequency_chart_dataset
from src.time_frequency_chart_denoising import (
    cycle_consistency_residual,
    fit_complex_unitary_projection,
    fit_nearest_unitary_projection,
    fit_unitary_global_chart_synchronization,
    nearest_orthogonal_projection,
    synchronize_pairwise_complex_maps,
)
from src.time_frequency_learned_charts import (
    LIFT_METHOD,
    detect_recovered_chart_generators,
    random_noncentral_chart_generators,
    selected_method_for,
)


ROOT = Path(__file__).resolve().parents[1]


class TimeFrequencyChartDenoisingTests(unittest.TestCase):
    def test_nearest_unitary_projection_returns_orthogonal_map(self):
        rng = np.random.default_rng(10)
        noisy = np.eye(8) + 0.05 * rng.normal(size=(8, 8))
        projected = nearest_orthogonal_projection(noisy)

        self.assertTrue(np.allclose(projected.T @ projected, np.eye(8), atol=1e-10))

    def test_unitary_projection_does_not_false_lift_noncentral(self):
        projected = {
            name: project_to_nearest_unitary(matrix)
            for name, matrix in random_noncentral_chart_generators(4, count=4, seed=11).items()
        }
        detection = detect_recovered_chart_generators(projected, candidate_rank=4)

        self.assertEqual(detection.status, "rejected_noncentral")
        self.assertEqual(selected_method_for(detection), "none")

    def test_global_chart_synchronization_reduces_pairwise_inconsistency(self):
        rng = np.random.default_rng(12)
        chart_names = ("I", "A", "B", "C")
        exact_gauges = {
            "I": np.eye(4, dtype=complex),
            "A": project_to_nearest_unitary(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))),
            "B": project_to_nearest_unitary(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))),
            "C": project_to_nearest_unitary(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))),
        }
        pairwise = {}
        for target in chart_names:
            for source in chart_names:
                exact = exact_gauges[target] @ np.linalg.inv(exact_gauges[source])
                noise = 0.02 * (rng.normal(size=exact.shape) + 1j * rng.normal(size=exact.shape))
                pairwise[(target, source)] = exact if target == source else exact + noise

        before = cycle_consistency_residual(pairwise, chart_names)
        sync = synchronize_pairwise_complex_maps(pairwise, chart_names, project_gauges_unitary=True)

        self.assertLess(sync.cycle_residual_after, before)
        self.assertLess(sync.cycle_residual_after, 1e-10)

    def test_denoised_recovery_low_noise_d2_k2(self):
        dataset = generate_paired_time_frequency_chart_dataset(
            2,
            2,
            train_samples=200,
            validation_samples=60,
            test_samples=60,
            noise_level=3e-5,
            seed=0,
        )
        recoveries = [
            fit_nearest_unitary_projection(dataset),
            fit_complex_unitary_projection(dataset),
            fit_unitary_global_chart_synchronization(dataset),
        ]
        detections = [
            detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank=4)
            for recovery in recoveries
        ]

        self.assertTrue(any(d.status == "certified" and d.period == 2 and d.index == 4 for d in detections))

    def test_period_divisible_rank_still_rejected(self):
        dataset = generate_paired_time_frequency_chart_dataset(
            3,
            2,
            train_samples=180,
            validation_samples=60,
            test_samples=60,
            noise_level=1e-4,
            seed=0,
        )
        recovery = fit_unitary_global_chart_synchronization(dataset)

        for rank in (3, 6):
            detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank=rank)
            self.assertEqual(detection.status, "certified")
            self.assertEqual(detection.period, 3)
            self.assertEqual(detection.index, 9)
            self.assertEqual(detection.decision, "period_divisible_index_obstructed")
            self.assertEqual(selected_method_for(detection), "none")

    def test_index_rank_lifts_when_certified(self):
        dataset = generate_paired_time_frequency_chart_dataset(
            3,
            2,
            train_samples=180,
            validation_samples=60,
            test_samples=60,
            noise_level=1e-4,
            seed=0,
        )
        recovery = fit_unitary_global_chart_synchronization(dataset)
        detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank=9)

        self.assertEqual(detection.status, "certified")
        self.assertEqual(detection.period, 3)
        self.assertEqual(detection.index, 9)
        self.assertEqual(detection.decision, "period_index_lift_success")
        self.assertEqual(selected_method_for(detection), LIFT_METHOD)

    def test_no_lift_for_uncertain_or_rejected(self):
        detection = detect_recovered_chart_generators(
            random_noncentral_chart_generators(4, count=4, seed=13),
            candidate_rank=4,
        )

        self.assertIn(detection.status, {"candidate_uncertain", "rejected_noncentral", "unknown_index"})
        self.assertEqual(selected_method_for(detection), "none")

    def test_report_outputs_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            subprocess.check_call(
                [
                    sys.executable,
                    str(ROOT / "experiments" / "time_frequency_denoised_chart_benchmark.py"),
                    "--reports-dir",
                    str(reports_dir),
                    "--seeds",
                    "1",
                    "--train-samples",
                    "80",
                    "--validation-samples",
                    "30",
                    "--test-samples",
                    "30",
                    "--noise-levels",
                    "0",
                    "0.0001",
                ],
                cwd=ROOT,
            )

            self.assertTrue((reports_dir / "time_frequency_denoised_chart_report.md").exists())
            self.assertTrue((reports_dir / "csv" / "time_frequency_denoised_chart_benchmark.csv").exists())
            self.assertTrue((reports_dir / "csv" / "time_frequency_denoised_chart_summary.csv").exists())
            self.assertTrue((reports_dir / "plots" / "time_frequency_denoised_certification_rate.pdf").exists())
            self.assertTrue((reports_dir / "plots" / "time_frequency_denoised_operator_error.pdf").exists())
            self.assertTrue((reports_dir / "plots" / "time_frequency_denoised_rank_threshold.pdf").exists())


if __name__ == "__main__":
    unittest.main()

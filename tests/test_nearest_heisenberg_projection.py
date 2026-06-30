import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.nearest_heisenberg_projection import (
    HEISENBERG_PROJECTION_METHOD,
    canonical_heisenberg_generators,
    project_to_nearest_finite_heisenberg,
)
from src.period_index_mining import add_entrywise_noise
from src.time_frequency_learned_charts import random_noncentral_chart_generators


ROOT = Path(__file__).resolve().parents[1]


def _named_random_noncentral(d: int, k: int, seed: int = 0) -> dict[str, np.ndarray]:
    names = tuple(canonical_heisenberg_generators(d, k))
    random_maps = random_noncentral_chart_generators(d**k, count=len(names), seed=seed)
    return {name: matrix for name, matrix in zip(names, random_maps.values(), strict=True)}


class NearestHeisenbergProjectionTests(unittest.TestCase):
    def test_projection_recovers_exact_heisenberg(self):
        generators = canonical_heisenberg_generators(2, 2)
        result = project_to_nearest_finite_heisenberg(
            generators,
            expected_d=2,
            expected_k=2,
            candidate_rank=4,
            projection_residual_threshold=1e-4,
        )

        self.assertTrue(result.projection_accepted)
        self.assertLess(result.projection_residual, 1e-12)
        self.assertEqual(result.detector_after_projection.status, "certified")
        self.assertEqual(result.detector_after_projection.period, 2)
        self.assertEqual(result.detector_after_projection.index, 4)
        self.assertEqual(result.decision, "heisenberg_projection_lift_success")

    def test_projection_recovers_small_noisy_heisenberg(self):
        exact = canonical_heisenberg_generators(2, 2)
        noisy = {
            name: add_entrywise_noise(matrix, 1e-4, project_unitary=True, seed=idx)
            for idx, (name, matrix) in enumerate(exact.items())
        }
        result = project_to_nearest_finite_heisenberg(
            noisy,
            expected_d=2,
            expected_k=2,
            candidate_rank=4,
            projection_residual_threshold=1e-3,
        )

        self.assertTrue(result.projection_accepted)
        self.assertEqual(result.detector_after_projection.period, 2)
        self.assertEqual(result.detector_after_projection.index, 4)
        self.assertEqual(result.selected_method, HEISENBERG_PROJECTION_METHOD)

    def test_projection_rejects_large_residual(self):
        exact = canonical_heisenberg_generators(2, 2)
        noisy = {
            name: add_entrywise_noise(matrix, 1e-2, project_unitary=True, seed=idx)
            for idx, (name, matrix) in enumerate(exact.items())
        }
        result = project_to_nearest_finite_heisenberg(
            noisy,
            expected_d=2,
            expected_k=2,
            candidate_rank=4,
            projection_residual_threshold=1e-4,
        )

        self.assertFalse(result.projection_accepted)
        self.assertEqual(result.selected_method, "none")
        self.assertGreater(result.projection_residual, 1e-4)

    def test_projection_does_not_false_lift_noncentral(self):
        result = project_to_nearest_finite_heisenberg(
            _named_random_noncentral(2, 2, seed=4),
            expected_d=2,
            expected_k=2,
            candidate_rank=4,
            projection_residual_threshold=1e-2,
        )

        self.assertFalse(result.projection_accepted)
        self.assertEqual(result.selected_method, "none")
        self.assertIn(result.decision, {"heisenberg_projection_rejected", "heisenberg_projection_uncertain"})

    def test_wrong_period_rejected(self):
        result = project_to_nearest_finite_heisenberg(
            canonical_heisenberg_generators(3, 2),
            expected_d=2,
            expected_k=2,
            candidate_rank=4,
            projection_residual_threshold=1e-2,
        )

        self.assertFalse(result.projection_accepted)
        self.assertEqual(result.selected_method, "none")

    def test_period_divisible_rank_still_rejected_after_projection(self):
        generators = canonical_heisenberg_generators(3, 2)
        for rank in (3, 6):
            result = project_to_nearest_finite_heisenberg(
                generators,
                expected_d=3,
                expected_k=2,
                candidate_rank=rank,
                projection_residual_threshold=1e-4,
            )
            self.assertTrue(result.projection_accepted)
            self.assertEqual(result.detector_after_projection.status, "certified")
            self.assertEqual(result.detector_after_projection.index, 9)
            self.assertEqual(result.decision, "heisenberg_projection_index_obstructed")
            self.assertEqual(result.selected_method, "none")

    def test_index_rank_lifts_after_projection_when_certified(self):
        result = project_to_nearest_finite_heisenberg(
            canonical_heisenberg_generators(3, 2),
            expected_d=3,
            expected_k=2,
            candidate_rank=9,
            projection_residual_threshold=1e-4,
        )

        self.assertTrue(result.projection_accepted)
        self.assertEqual(result.detector_after_projection.status, "certified")
        self.assertEqual(result.detector_after_projection.period, 3)
        self.assertEqual(result.detector_after_projection.index, 9)
        self.assertEqual(result.decision, "heisenberg_projection_lift_success")
        self.assertEqual(result.selected_method, HEISENBERG_PROJECTION_METHOD)

    def test_benchmark_outputs_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            subprocess.check_call(
                [
                    sys.executable,
                    str(ROOT / "experiments" / "time_frequency_heisenberg_projection_benchmark.py"),
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
                    "0.001",
                    "--projection-thresholds",
                    "0.01",
                ],
                cwd=ROOT,
            )

            self.assertTrue((reports_dir / "time_frequency_heisenberg_projection_report.md").exists())
            self.assertTrue((reports_dir / "csv" / "time_frequency_heisenberg_projection_benchmark.csv").exists())
            self.assertTrue((reports_dir / "csv" / "time_frequency_heisenberg_projection_summary.csv").exists())
            self.assertTrue((reports_dir / "plots" / "time_frequency_heisenberg_projection_certification_rate.pdf").exists())
            self.assertTrue((reports_dir / "plots" / "time_frequency_heisenberg_projection_residual.pdf").exists())
            self.assertTrue((reports_dir / "plots" / "time_frequency_heisenberg_projection_false_lift.pdf").exists())


if __name__ == "__main__":
    unittest.main()

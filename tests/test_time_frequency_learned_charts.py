import json
import unittest
from pathlib import Path

import numpy as np

from src.time_frequency_benchmark import generate_paired_time_frequency_chart_dataset
from src.time_frequency_learned_charts import (
    LIFT_METHOD,
    detect_recovered_chart_generators,
    fit_input_least_squares_chart,
    fit_linear_autoencoder_chart,
    fit_supervised_encoder_chart,
    random_noncentral_chart_generators,
    selected_method_for,
)


ROOT = Path(__file__).resolve().parents[1]


class TimeFrequencyLearnedChartsTests(unittest.TestCase):
    def test_paired_chart_dataset_has_matching_sample_ids(self):
        dataset = generate_paired_time_frequency_chart_dataset(
            2,
            2,
            train_samples=9,
            validation_samples=5,
            test_samples=4,
            seed=1,
        )

        expected = np.arange(dataset.train.n_samples)
        for chart_name in dataset.chart_names:
            self.assertTrue(np.array_equal(dataset.train.chart_sample_ids(chart_name), expected))
            self.assertEqual(dataset.train.chart_rows(chart_name).shape, (9, dataset.dimension_real))

    def test_input_least_squares_recovers_known_chart_low_noise(self):
        dataset = generate_paired_time_frequency_chart_dataset(
            2,
            2,
            train_samples=80,
            validation_samples=30,
            test_samples=30,
            noise_level=0.0,
            seed=2,
        )
        recovery = fit_input_least_squares_chart(dataset, ridge=1e-10)
        detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank=4)

        self.assertLess(recovery.learned_operator_error_max, 1e-8)
        self.assertLess(recovery.pair_reconstruction_residual_test, 1e-8)
        self.assertEqual(detection.status, "certified")
        self.assertEqual(detection.period, 2)
        self.assertEqual(detection.index, 4)
        self.assertEqual(detection.decision, "period_index_lift_success")
        self.assertEqual(selected_method_for(detection), LIFT_METHOD)

    def test_input_least_squares_period_divisible_rank_rejected(self):
        dataset = generate_paired_time_frequency_chart_dataset(
            3,
            2,
            train_samples=180,
            validation_samples=40,
            test_samples=40,
            noise_level=0.0,
            seed=3,
        )
        recovery = fit_input_least_squares_chart(dataset, ridge=1e-10)
        for rank in [3, 6]:
            detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank=rank)
            self.assertEqual(detection.status, "certified")
            self.assertEqual(detection.period, 3)
            self.assertEqual(detection.index, 9)
            self.assertEqual(detection.decision, "period_divisible_index_obstructed")
            self.assertEqual(selected_method_for(detection), "none")

        accepted = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank=9)
        self.assertEqual(accepted.status, "certified")
        self.assertEqual(accepted.decision, "period_index_lift_success")
        self.assertEqual(selected_method_for(accepted), LIFT_METHOD)

    def test_input_least_squares_noise_uncertain_or_rejected(self):
        dataset = generate_paired_time_frequency_chart_dataset(
            2,
            2,
            train_samples=60,
            validation_samples=30,
            test_samples=30,
            noise_level=0.5,
            seed=4,
        )
        recovery = fit_input_least_squares_chart(dataset, ridge=1e-3)
        detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank=4)

        if detection.status == "certified":
            self.assertNotEqual(detection.decision, "period_index_lift_success")
        self.assertEqual(selected_method_for(detection), "none")

    def test_linear_autoencoder_chart_runs(self):
        dataset = generate_paired_time_frequency_chart_dataset(
            2,
            2,
            train_samples=40,
            validation_samples=20,
            test_samples=20,
            noise_level=0.01,
            seed=5,
        )
        recovery = fit_linear_autoencoder_chart(dataset, latent_dimension=dataset.dimension_real)
        detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank=4)

        self.assertEqual(recovery.latent_dimension, dataset.dimension_real)
        self.assertEqual(len(recovery.candidate_generators), 4)
        self.assertIn(detection.status, {"certified", "candidate_uncertain", "rejected_noncentral", "unknown_index"})
        if selected_method_for(detection) == LIFT_METHOD:
            self.assertEqual(detection.status, "certified")
            self.assertEqual(detection.decision, "period_index_lift_success")

    def test_supervised_encoder_chart_no_overclaim(self):
        dataset = generate_paired_time_frequency_chart_dataset(
            2,
            2,
            train_samples=60,
            validation_samples=20,
            test_samples=20,
            noise_level=0.02,
            seed=6,
        )
        recovery = fit_supervised_encoder_chart(dataset)
        detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank=2)

        if detection.decision != "period_index_lift_success" or detection.status != "certified":
            self.assertEqual(selected_method_for(detection), "none")

    def test_noncentral_negative_chart_control(self):
        detection = detect_recovered_chart_generators(
            random_noncentral_chart_generators(4, count=4, seed=7),
            candidate_rank=4,
        )

        self.assertEqual(detection.status, "rejected_noncentral")
        self.assertEqual(detection.decision, "not_central_projective")
        self.assertEqual(selected_method_for(detection), "none")

    def test_report_scope_note(self):
        report = ROOT / "reports" / "time_frequency_learned_chart_report.md"
        config = ROOT / "reports" / "configs" / "time_frequency_learned_chart_config.json"
        self.assertTrue(report.exists())
        self.assertTrue(config.exists())

        text = report.read_text(encoding="utf-8")
        config_payload = json.loads(config.read_text(encoding="utf-8"))
        self.assertIn("known-operator", text)
        self.assertIn("input least-squares", text)
        self.assertIn("linear autoencoder", text)
        self.assertIn("supervised encoder", text)
        self.assertIn("No MNIST/CIFAR residual", text)
        self.assertEqual(config_payload["scope"]["mnist_cifar_claim"], "not_claimed")
        self.assertEqual(config_payload["scope"]["uncertain_lift_policy"], "no_lift")


if __name__ == "__main__":
    unittest.main()

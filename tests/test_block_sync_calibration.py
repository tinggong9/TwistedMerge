import unittest

from src.block_sync_calibration import (
    accepted_sync_from_calibration,
    calibrate_connection_residual_threshold,
    classify_sync_evidence,
)


class BlockSyncCalibrationTests(unittest.TestCase):
    def test_calibration_accepts_positive_controls_and_rejects_negative_controls(self):
        calibration = calibrate_connection_residual_threshold(
            positive_residuals=[0.0, 0.005, 0.02, 0.03],
            negative_residuals=[0.18, 0.24, 0.5, 0.7],
            target_false_positive_rate=0.0,
        )

        self.assertEqual(calibration.accepted_negative_count, 0)
        self.assertEqual(calibration.observed_false_positive_rate, 0.0)
        self.assertTrue(accepted_sync_from_calibration(0.02, calibration))
        self.assertFalse(accepted_sync_from_calibration(0.18, calibration))

    def test_calibration_records_false_positive_tradeoff_when_controls_overlap(self):
        calibration = calibrate_connection_residual_threshold(
            positive_residuals=[0.0, 0.02, 0.08],
            negative_residuals=[0.04, 0.3, 0.5],
            target_false_positive_rate=0.0,
        )

        self.assertLessEqual(calibration.observed_false_positive_rate, calibration.target_false_positive_rate)
        self.assertFalse(accepted_sync_from_calibration(0.04, calibration))

    def test_classification_separates_true_sync_from_projection_trap(self):
        calibration = calibrate_connection_residual_threshold(
            positive_residuals=[0.0, 0.01],
            negative_residuals=[0.3, 0.4],
            target_false_positive_rate=0.0,
        )

        self.assertEqual(
            classify_sync_evidence(
                observed_scalar_projective_candidate=False,
                observed_centrality_score=0.0,
                projected_cycle_score=0.0,
                connection_residual=0.005,
                calibration=calibration,
            ),
            "global_gauge_consistent",
        )
        self.assertEqual(
            classify_sync_evidence(
                observed_scalar_projective_candidate=False,
                observed_centrality_score=0.7,
                projected_cycle_score=0.0,
                connection_residual=0.35,
                calibration=calibration,
            ),
            "projected_cycle_only_connection_large",
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from experiments.block_gauge_phase_diagram import (
    build_calibration_controls,
    defect_summary,
    make_family_maps,
    sync_row,
)
from src.block_sync_calibration import apply_block_sync_policy, calibrate_block_sync_policies


class Args:
    synthetic_seeds = 4
    max_order = 12
    max_iters = 8
    tolerance = 1e-7
    n_restarts = 1
    calibration_floor = 1e-12


class BlockSyncCalibrationFloorTests(unittest.TestCase):
    def test_effective_threshold_uses_numerical_floor(self):
        policies = calibrate_block_sync_policies(
            positive_residuals=[1e-16, 4e-13, 8e-13],
            negative_residuals=[0.1, 0.2],
            numerical_floor=1e-12,
        )
        strict = {policy.name: policy for policy in policies}["strict"]

        self.assertLess(strict.raw_calibrated_threshold, strict.effective_threshold)
        self.assertGreaterEqual(strict.effective_threshold, 1e-12)
        self.assertEqual(apply_block_sync_policy(8e-13, strict), "accept")
        self.assertEqual(apply_block_sync_policy(0.1, strict), "reject")

    def test_floor_accepts_exact_and_keeps_negative_controls_rejected(self):
        args = Args()
        calibration, policies = build_calibration_controls(args)
        exact = sync_row(
            args,
            family="exact_global_block_gauge",
            n_models=3,
            width=4,
            block_size=2,
            noise_level=0.0,
            seed=2,
            calibration=calibration,
            policies=policies,
        )
        fake = sync_row(
            args,
            family="fake_projection_trap",
            n_models=3,
            width=4,
            block_size=2,
            noise_level=0.4,
            seed=2,
            calibration=calibration,
            policies=policies,
        )
        noncentral = sync_row(
            args,
            family="noncentral_block_holonomy",
            n_models=3,
            width=4,
            block_size=2,
            noise_level=0.4,
            seed=2,
            calibration=calibration,
            policies=policies,
        )

        self.assertEqual(exact["strict_policy_decision"], "accept")
        self.assertNotEqual(fake["strict_policy_decision"], "accept")
        self.assertNotEqual(noncentral["strict_policy_decision"], "accept")
        self.assertLess(exact["raw_calibrated_threshold"], exact["effective_threshold"])
        self.assertGreaterEqual(exact["effective_threshold"], args.calibration_floor)

    def test_scalar_mu2_phase_is_detected_before_projection(self):
        maps, _blocks, _label, _accept = make_family_maps("scalar_block_phase_mu2", 3, 4, 2, 0.0, 2)
        summary = defect_summary(maps, 3, 12)

        self.assertTrue(summary["scalar_projective_candidate"])
        self.assertIn("2", summary["detected_orders"])


if __name__ == "__main__":
    unittest.main()

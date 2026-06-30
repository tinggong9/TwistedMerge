import unittest

from src.block_sync_calibration import (
    apply_block_sync_policy,
    calibrate_block_sync_policies,
)


class BlockSyncCalibrationPhaseDiagramTests(unittest.TestCase):
    def test_strict_balanced_and_loose_policies_are_ordered(self):
        policies = calibrate_block_sync_policies(
            positive_residuals=[0.0, 0.01, 0.02],
            negative_residuals=[0.2, 0.3, 0.5],
        )
        by_name = {policy.name: policy for policy in policies}

        self.assertIn("strict", by_name)
        self.assertIn("balanced", by_name)
        self.assertIn("loose_diagnostic", by_name)
        self.assertEqual(by_name["strict"].observed_false_positive_rate, 0.0)
        self.assertEqual(apply_block_sync_policy(0.01, by_name["strict"]), "accept")
        self.assertEqual(apply_block_sync_policy(0.2, by_name["strict"]), "reject")
        self.assertGreater(by_name["loose_diagnostic"].uncertain_band, 0.0)


if __name__ == "__main__":
    unittest.main()

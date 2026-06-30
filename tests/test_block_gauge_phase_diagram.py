import unittest

import numpy as np

from experiments.block_gauge_phase_diagram import (
    build_calibration_controls,
    defect_summary,
    make_family_maps,
    sync_row,
)


class Args:
    synthetic_seeds = 4
    max_order = 12
    max_iters = 8
    tolerance = 1e-7
    n_restarts = 1


class BlockGaugePhaseDiagramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = Args()
        cls.calibration, cls.policies = build_calibration_controls(cls.args)

    def row(self, family: str):
        return sync_row(
            self.args,
            family=family,
            n_models=3,
            width=4,
            block_size=2,
            noise_level=0.0 if family == "exact_global_block_gauge" else 0.4,
            seed=1,
            calibration=self.calibration,
            policies=self.policies,
        )

    def test_exact_global_gauges_accepted_under_strict_calibration(self):
        row = self.row("exact_global_block_gauge")
        self.assertEqual(row["strict_policy_decision"], "accept")
        self.assertEqual(row["evidence_label"], "global_gauge_consistent")
        self.assertLess(row["optimized_connection_residual"], 1e-8)

    def test_noncentral_holonomy_rejected_under_strict_calibration(self):
        row = self.row("noncentral_block_holonomy")
        self.assertNotEqual(row["strict_policy_decision"], "accept")
        self.assertTrue(row["post_projection_cycle_only_warning"])

    def test_fake_projection_trap_rejected_despite_zero_projected_cycle(self):
        row = self.row("fake_projection_trap")
        self.assertLess(row["projected_cycle_score"], 1e-8)
        self.assertNotEqual(row["strict_policy_decision"], "accept")
        self.assertTrue(row["post_projection_cycle_only_warning"])

    def test_scalar_block_phase_detected_before_projection(self):
        maps, _blocks, _label, _accept = make_family_maps("scalar_block_phase_mu2", 3, 4, 2, 0.0, 2)
        summary = defect_summary(maps, 3, 12)
        self.assertTrue(summary["scalar_projective_candidate"])
        self.assertIn("2", summary["detected_orders"])


if __name__ == "__main__":
    unittest.main()

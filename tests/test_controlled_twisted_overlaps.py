import unittest

import numpy as np

from src.controlled_twisted_overlaps import (
    build_controlled_case,
    defect_rows_for_case,
    evaluate_methods,
    pairwise_rows_for_case,
)


def method_lookup(case, extra_controls=None):
    return {row["method"]: row for row in evaluate_methods(case, extra_controls=extra_controls)}


class ControlledTwistedOverlapTests(unittest.TestCase):
    def test_coboundary_is_solved_by_cycle_synchronization(self):
        case = build_controlled_case(
            "mu2_coboundary",
            width=32,
            n_models=4,
            seed=11,
            samples_per_chart=256,
            samples_per_overlap=256,
            branch_count=2,
        )
        rows = method_lookup(case)

        self.assertTrue(case.is_coboundary)
        self.assertGreater(rows["c2m3_synchronized"]["test_accuracy"], 0.99)
        self.assertGreater(rows["c2m3_synchronized"]["test_accuracy"], rows["ordinary_weight_average"]["test_accuracy"] + 0.40)
        self.assertAlmostEqual(
            rows["twisted_q2_branch"]["test_accuracy"],
            rows["c2m3_synchronized"]["test_accuracy"],
            places=6,
        )

    def test_nontrivial_mu2_h2_uses_q2_branch_and_beats_matched_baselines(self):
        case = build_controlled_case(
            "mu2_nontrivial_h2",
            width=32,
            n_models=4,
            seed=12,
            samples_per_chart=256,
            samples_per_overlap=256,
            branch_count=2,
        )
        rows = method_lookup(case)

        self.assertFalse(case.is_coboundary)
        self.assertGreater(rows["twisted_q2_branch"]["test_accuracy"], 0.99)
        self.assertLess(rows["c2m3_synchronized"]["test_accuracy"], 0.80)
        for baseline in [
            "random_branch_ensemble",
            "validation_selected_branch_ensemble",
            "c2m3_cluster_branch_ensemble",
        ]:
            self.assertGreater(rows["twisted_q2_branch"]["test_accuracy"], rows[baseline]["test_accuracy"] + 0.20)
            self.assertTrue(rows[baseline]["capacity_matched_to_rank_lift"])

    def test_pairwise_and_triangle_defects_are_exact(self):
        case = build_controlled_case(
            "mu2_nontrivial_h2",
            width=32,
            n_models=4,
            seed=13,
            samples_per_chart=128,
            samples_per_overlap=128,
            branch_count=2,
        )
        pairwise = pairwise_rows_for_case(case)
        triangles = defect_rows_for_case(case)

        self.assertTrue(all(np.isclose(row["pairwise_alignment_residual"], 0.0) for row in pairwise))
        self.assertTrue(all(np.isclose(row["defect_to_true_twist_residual"], 0.0) for row in triangles))
        self.assertEqual(sum(1 for row in triangles if row["observed_triangle_sign"] < 0), 1)

    def test_random_noncentral_control_is_not_promoted(self):
        case = build_controlled_case(
            "random_noncentral",
            width=32,
            n_models=4,
            seed=14,
            samples_per_chart=128,
            samples_per_overlap=128,
            branch_count=2,
        )
        rows = method_lookup(case)
        triangles = defect_rows_for_case(case)

        self.assertFalse(case.central_twist_claim_allowed)
        self.assertTrue(all(row["defect_type"] == "noncentral_permutation" for row in triangles))
        self.assertTrue(all(row["centrality_residual"] > 0.0 for row in triangles))
        self.assertEqual(rows["twisted_q2_branch"]["claim_role"], "noncentral_control_not_mu2_claim")

    def test_nontrivial_h2_hardening_controls(self):
        case = build_controlled_case(
            "mu2_nontrivial_h2",
            width=32,
            n_models=4,
            seed=15,
            samples_per_chart=256,
            samples_per_overlap=256,
            branch_count=2,
        )
        rows = method_lookup(
            case,
            extra_controls=[
                "wrong_twist",
                "wrong_context",
                "learned_router",
                "distilled_single",
                "parameter_matched_wide",
                "no_twist_branch",
            ],
        )

        self.assertGreater(rows["twisted_q2_branch"]["test_accuracy"], 0.99)
        self.assertGreater(rows["twisted_q2_branch"]["test_accuracy"], rows["wrong_twist_control"]["test_accuracy"] + 0.20)
        self.assertGreater(rows["twisted_q2_branch"]["test_accuracy"], rows["wrong_context_control"]["test_accuracy"] + 0.20)
        self.assertGreater(rows["twisted_q2_branch"]["test_accuracy"], rows["no_twist_branch_control"]["test_accuracy"] + 0.20)
        self.assertAlmostEqual(rows["learned_context_router"]["test_accuracy"], rows["twisted_q2_branch"]["test_accuracy"], places=6)
        self.assertLess(rows["distilled_twisted_single_model"]["test_accuracy"], rows["twisted_q2_branch"]["test_accuracy"] - 0.20)
        self.assertLess(rows["parameter_matched_wide_control"]["test_accuracy"], rows["twisted_q2_branch"]["test_accuracy"] - 0.20)
        self.assertTrue(rows["parameter_matched_wide_control"]["capacity_matched_to_rank_lift"])
        self.assertTrue(rows["distilled_twisted_single_model"]["is_single_model"])


if __name__ == "__main__":
    unittest.main()

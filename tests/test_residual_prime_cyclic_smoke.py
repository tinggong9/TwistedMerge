import numpy as np

from experiments.residual_prime_cyclic_smoke import (
    CertificationEvidence,
    capacity_multiplier_for_plan,
    certify_prime_residual,
    enumerate_prime_residual_paths,
    is_prime,
    missing_metric_na_reason,
    planned_case_decision,
    selection_decision,
)


def test_prime_residual_candidates_are_detected_correctly_for_mnist_order():
    paths = enumerate_prime_residual_paths(6176520, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 43])
    prime_rows = {(row["peeled_primes"], int(row["residual_prime"])) for row in paths if row["residual_order_is_prime"]}

    assert ("2,3,5,7,19", 43) in prime_rows
    assert ("2,3,5,7,43", 19) in prime_rows


def test_817_is_not_marked_prime():
    paths = enumerate_prime_residual_paths(6176520, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 43])
    row_817 = [row for row in paths if row["residual_order_candidate"] == 817][0]

    assert is_prime(817) is False
    assert row_817["residual_order_is_prime"] is False
    assert row_817["residual_factorization"] == "19 * 43"


def test_residual_19_is_marked_prime():
    assert is_prime(19) is True
    paths = enumerate_prime_residual_paths(813960, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 43])

    assert any(row["residual_order_is_prime"] and int(row["residual_prime"]) == 19 for row in paths)


def test_observed_prime_lcm_alone_does_not_certify_full_residual_cp():
    cert = certify_prime_residual(CertificationEvidence(residual_order_candidate=19, group_closure_status="truncated_max_group_order"))

    assert cert["certification_status"] == "observed_prime_lcm_only"
    assert cert["full_residual_group_order_certified"] is False
    assert cert["cyclic_quotient_certified"] is False


def test_exact_group_order_p_certifies_cp():
    cert = certify_prime_residual(
        CertificationEvidence(
            residual_order_candidate=19,
            group_closure_status="exact_closure",
            group_order_if_exact=19,
        )
    )

    assert cert["certification_status"] == "certified_full_residual_Cp"
    assert cert["certification_method"] == "exact_group_order_prime"
    assert cert["full_residual_group_order_certified"] is True


def test_certified_quotient_is_distinct_from_full_residual_certification():
    cert = certify_prime_residual(
        CertificationEvidence(
            residual_order_candidate=43,
            group_closure_status="truncated_max_group_order",
            quotient_relation_violation_rate=0.0,
            quotient_nontrivial_rate=1.0,
            quotient_entropy=0.5,
            quotient_confidence=0.95,
        )
    )

    assert cert["certification_status"] == "certified_cyclic_Cp_quotient"
    assert cert["full_residual_group_order_certified"] is False
    assert cert["cyclic_quotient_certified"] is True


def test_unimplemented_planned_cyclic_prime_methods_cannot_be_selected():
    selectable, status = planned_case_decision("certified_cyclic_Cp_quotient", False, False)

    assert selectable is False
    assert status == "certified_but_merge_rerun_not_implemented"


def test_test_accuracy_is_never_used_for_selection():
    selected, reason = selection_decision(
        {
            "uses_test_for_selection": False,
            "planned_case_status": "implemented_candidate_available_requires_validation_gate",
            "validation_accuracy": 0.79,
            "test_accuracy": 0.99,
            "baseline_validation_accuracy": 0.8,
            "wrong_prime_control_validation_accuracy": 0.1,
            "shuffled_control_validation_accuracy": 0.1,
            "random_residual_control_validation_accuracy": 0.1,
        }
    )

    assert selected is False
    assert reason == "not_selected_fails_unpeeled_baseline_gate"

    selected, reason = selection_decision(
        {
            "uses_test_for_selection": True,
            "planned_case_status": "implemented_candidate_available_requires_validation_gate",
            "validation_accuracy": 0.99,
            "test_accuracy": 0.99,
            "baseline_validation_accuracy": 0.8,
            "wrong_prime_control_validation_accuracy": 0.1,
            "shuffled_control_validation_accuracy": 0.1,
            "random_residual_control_validation_accuracy": 0.1,
        }
    )
    assert selected is False
    assert reason == "blocked_test_metric_selection_forbidden"


def test_every_missing_metric_row_has_na_reason():
    row = {
        "validation_accuracy": np.nan,
        "test_accuracy": np.nan,
        "validation_delta_vs_baseline": np.nan,
        "test_delta_vs_baseline": np.nan,
        "wrong_prime_control_validation_accuracy": np.nan,
        "shuffled_control_validation_accuracy": np.nan,
        "random_residual_control_validation_accuracy": np.nan,
        "planned_case_status": "observed_lcm_prime_but_not_certified",
        "certification_status": "observed_prime_lcm_only",
    }

    assert missing_metric_na_reason(row) == "observed_lcm_prime_but_not_certified"


def test_capacity_multiplier_for_no_lift_and_branch_lift():
    assert capacity_multiplier_for_plan("no_lift_cyclic_prime_correction", 19) == 1.0
    assert capacity_multiplier_for_plan("branch_lift", 19) == 19.0

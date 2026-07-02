import numpy as np
import pandas as pd

from experiments.small_prime_peeling_smoke import (
    SelectedSetting,
    build_prime_rows,
    factor_axis_label,
    p_adic_multiplicity,
    prime_peeling_plan,
    selection_decision,
)


def test_primes_are_only_eligible_when_they_divide_source_order():
    rows = {row["prime"]: row for row in prime_peeling_plan(2 * 2 * 3 * 5, [2, 3, 7])}

    assert rows[2]["eligible"] is True
    assert rows[2]["p_adic_multiplicity"] == 2
    assert rows[2]["remaining_order_after"] == 15
    assert rows[3]["eligible"] is True
    assert rows[7]["eligible"] is False
    assert rows[7]["skip_reason"] == "p_not_dividing_remaining_order"


def test_c6_is_mixed_not_a_primary_axis():
    assert factor_axis_label(2) == "C2_prime_axis"
    assert factor_axis_label(3) == "C3_prime_axis"
    assert factor_axis_label(6) == "mixed_2_plus_3_not_primary_axis"


def test_unimplemented_prime_candidates_cannot_be_selected():
    selected, reason = selection_decision(
        {
            "eligible": True,
            "implemented_real_lift": False,
            "relation_count_status": "sufficient",
            "uses_test_for_selection": False,
            "validation_accuracy": 0.99,
            "best_fallback_validation_accuracy": 0.8,
            "random_same_branch_control_validation_accuracy": 0.7,
            "wrong_prime_control_validation_accuracy": 0.7,
        }
    )

    assert selected is False
    assert reason == "diagnostic_only_no_real_prediction"


def test_test_accuracy_is_not_used_for_selection():
    selected, reason = selection_decision(
        {
            "eligible": True,
            "implemented_real_lift": True,
            "relation_count_status": "sufficient",
            "uses_test_for_selection": False,
            "validation_accuracy": 0.79,
            "test_accuracy": 0.99,
            "best_fallback_validation_accuracy": 0.8,
            "random_same_branch_control_validation_accuracy": 0.1,
            "wrong_prime_control_validation_accuracy": 0.1,
        }
    )

    assert selected is False
    assert reason == "not_selected_fails_best_fallback_gate"


def test_same_branch_and_wrong_prime_controls_can_block_selection():
    row = {
        "eligible": True,
        "implemented_real_lift": True,
        "relation_count_status": "sufficient",
        "uses_test_for_selection": False,
        "validation_accuracy": 0.84,
        "best_fallback_validation_accuracy": 0.8,
        "random_same_branch_control_validation_accuracy": 0.85,
        "wrong_prime_control_validation_accuracy": 0.7,
    }
    selected, reason = selection_decision(row)
    assert selected is False
    assert reason == "not_selected_fails_random_same_branch_control"

    row["random_same_branch_control_validation_accuracy"] = 0.7
    row["wrong_prime_control_validation_accuracy"] = 0.85
    selected, reason = selection_decision(row)
    assert selected is False
    assert reason == "not_selected_fails_wrong_prime_control"


def test_large_prime_capacity_multiplier_is_the_prime():
    setting = SelectedSetting(
        dataset="mnist",
        run_id="r0",
        setting_id="s0",
        architecture="mlp",
        n_models=4,
        width=64,
        domain_shift="none",
        matching="activation",
        seed=0,
        relation_count=4,
        relation_count_status="sufficient",
        observed_holonomy_order_lcm=29,
        group_closure_status="truncated_max_group_order",
        group_exponent_if_exact=None,
        primary_source_order=29,
        primary_source_order_source="observed_holonomy_order_lcm",
    )
    run_rows = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "method": "greedy_soup",
                "val_accuracy": 0.8,
                "val_loss": 0.4,
                "test_accuracy": 0.79,
                "test_loss": 0.42,
                "is_ensemble_or_extra_capacity": False,
            }
        ]
    )
    row = build_prime_rows(setting, run_rows, [29])[0]

    assert row["capacity_multiplier"] == 29.0
    assert row["inference_multiplier"] == 29.0
    assert p_adic_multiplicity(29 * 29 * 2, 29) == 2

    selected, reason = selection_decision(
        {
            "eligible": True,
            "implemented_real_lift": True,
            "relation_count_status": "sufficient",
            "uses_test_for_selection": False,
            "validation_accuracy": 0.9,
            "best_fallback_validation_accuracy": 0.8,
            "random_same_branch_control_validation_accuracy": np.nan,
            "wrong_prime_control_validation_accuracy": 0.7,
        }
    )
    assert selected is False
    assert reason == "not_selected_missing_random_same_branch_control"

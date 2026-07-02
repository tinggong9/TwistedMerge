import numpy as np

from experiments.primary_residual_peeling_smoke import (
    correction_is_safe,
    na_reason_for_metrics,
    no_lift_capacity_metadata,
    prime_peeling_plan,
    selection_decision,
)


def test_primes_eligible_only_when_dividing_remaining_order():
    rows = {row["prime"]: row for row in prime_peeling_plan(2 * 3 * 5, [2, 3, 7])}

    assert rows[2]["eligible"] is True
    assert rows[3]["eligible"] is True
    assert rows[7]["eligible"] is False


def test_peeling_removes_full_p_adic_multiplicity():
    rows = {row["prime"]: row for row in prime_peeling_plan(2 * 2 * 2 * 3 * 3 * 5, [2, 3, 5])}

    assert rows[2]["p_adic_multiplicity"] == 3
    assert rows[2]["remaining_order_after"] == 45
    assert rows[3]["p_adic_multiplicity"] == 2
    assert rows[3]["remaining_order_after"] == 5


def test_ineligible_primes_cannot_be_selected():
    selected, reason = selection_decision(
        {
            "eligible": False,
            "correction_reduces_residual": True,
            "implemented_corrected_merge": True,
            "uses_test_for_selection": False,
            "validation_accuracy": 0.9,
            "baseline_validation_accuracy": 0.8,
            "wrong_prime_control_validation_accuracy": 0.7,
            "shuffled_control_validation_accuracy": 0.7,
            "random_residual_control_validation_accuracy": 0.7,
        }
    )

    assert selected is False
    assert reason == "prime_not_eligible"


def test_unimplemented_corrected_merges_cannot_be_selected():
    selected, reason = selection_decision(
        {
            "eligible": True,
            "correction_reduces_residual": True,
            "implemented_corrected_merge": False,
            "uses_test_for_selection": False,
            "validation_accuracy": 0.9,
            "baseline_validation_accuracy": 0.8,
            "wrong_prime_control_validation_accuracy": 0.7,
            "shuffled_control_validation_accuracy": 0.7,
            "random_residual_control_validation_accuracy": 0.7,
        }
    )

    assert selected is False
    assert reason == "merge_rerun_not_implemented"


def test_test_accuracy_is_never_used_for_selection():
    selected, reason = selection_decision(
        {
            "eligible": True,
            "correction_reduces_residual": True,
            "implemented_corrected_merge": True,
            "uses_test_for_selection": False,
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


def test_correction_must_reduce_residual_before_safe():
    assert correction_is_safe(True, 0.0, 0.75, 0.0)
    assert not correction_is_safe(True, 0.0, 0.25, 0.25)
    assert not correction_is_safe(True, 0.2, 0.75, 0.2)
    assert not correction_is_safe(False, 0.0, 0.75, 0.0)


def test_wrong_shuffled_and_random_controls_can_block_selection():
    base = {
        "eligible": True,
        "correction_reduces_residual": True,
        "implemented_corrected_merge": True,
        "uses_test_for_selection": False,
        "validation_accuracy": 0.84,
        "baseline_validation_accuracy": 0.8,
        "wrong_prime_control_validation_accuracy": 0.7,
        "shuffled_control_validation_accuracy": 0.7,
        "random_residual_control_validation_accuracy": 0.7,
    }
    assert selection_decision(base)[0] is True

    for control in [
        "wrong_prime_control_validation_accuracy",
        "shuffled_control_validation_accuracy",
        "random_residual_control_validation_accuracy",
    ]:
        row = dict(base)
        row[control] = 0.85
        selected, reason = selection_decision(row)
        assert selected is False
        assert "control" in reason


def test_no_lift_capacity_and_inference_multipliers_are_one():
    meta = no_lift_capacity_metadata()

    assert meta["capacity_multiplier"] == 1.0
    assert meta["inference_multiplier"] == 1.0


def test_na_reason_populated_when_metrics_missing():
    row = {"validation_accuracy": np.nan, "test_accuracy": np.nan}

    assert na_reason_for_metrics(row, "merge_rerun_not_implemented") == "merge_rerun_not_implemented"
    assert na_reason_for_metrics({"validation_accuracy": 0.8, "test_accuracy": 0.7}, "x") == ""

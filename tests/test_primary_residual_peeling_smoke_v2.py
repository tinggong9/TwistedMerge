from types import SimpleNamespace

import numpy as np

from experiments.primary_residual_peeling_smoke_v2 import (
    SelectedSetting,
    accuracy_row,
    cycle_residual,
    edge_self_corrected_maps,
    is_valid_permutation,
    no_lift_capacity_metadata,
    prime_peeling_plan,
    row_missing_reason,
    selection_decision,
)


def _setting() -> SelectedSetting:
    return SelectedSetting(
        dataset="mnist",
        run_id="r0",
        setting_id="s0",
        architecture="mlp",
        n_models=3,
        width=3,
        domain_shift="none",
        matching="activation",
        seed=10,
        relation_count=1,
        relation_count_status="underconstrained",
        observed_holonomy_order_lcm=6,
        group_closure_status="mock",
        group_exponent_if_exact=None,
        primary_source_order=6,
        primary_source_order_source="observed_holonomy_order_lcm",
        model_source="mock",
    )


def _selectable_row(**updates):
    row = {
        "eligible": True,
        "implemented_corrected_merge": True,
        "uses_test_for_selection": False,
        "correction_reduces_residual": True,
        "validation_accuracy": 0.84,
        "test_accuracy": 0.10,
        "baseline_validation_accuracy": 0.80,
        "wrong_prime_control_validation_accuracy": 0.81,
        "shuffled_control_validation_accuracy": 0.82,
        "random_residual_control_validation_accuracy": 0.83,
        "capacity_multiplier": 1.0,
        "inference_multiplier": 1.0,
    }
    row.update(updates)
    return row


def test_eligible_primes_are_computed_from_remaining_order():
    rows = {row["prime"]: row for row in prime_peeling_plan(2 * 2 * 3 * 5, [2, 3, 7])}

    assert rows[2]["eligible"] is True
    assert rows[3]["eligible"] is True
    assert rows[7]["eligible"] is False


def test_peeling_removes_full_p_adic_multiplicity():
    rows = {row["prime"]: row for row in prime_peeling_plan(2 * 2 * 2 * 3 * 3 * 5, [2, 3, 5])}

    assert rows[2]["p_adic_multiplicity"] == 3
    assert rows[2]["remaining_order_after"] == 45
    assert rows[3]["p_adic_multiplicity"] == 2
    assert rows[3]["remaining_order_after"] == 5


def test_corrected_maps_are_valid_permutations_where_claimed():
    swap = np.array([1, 0, 2])
    identity = np.arange(3)
    pairwise = {
        (0, 0): identity,
        (1, 1): identity,
        (2, 2): identity,
        (0, 1): swap,
        (1, 0): swap,
        (1, 2): identity,
        (2, 1): identity,
        (2, 0): identity,
        (0, 2): identity,
    }
    fit = SimpleNamespace(assignment={tuple(swap.tolist()): 1})

    corrections, corrected, status = edge_self_corrected_maps(pairwise, fit, 3, 2)

    assert status == "observed_edge_representative_permutation"
    assert all(is_valid_permutation(correction) for correction in corrections.values())
    assert all(is_valid_permutation(perm) for perm in corrected.values())


def test_cycle_residual_after_correction_is_lower_for_safe_rows():
    swap = np.array([1, 0, 2])
    identity = np.arange(3)
    pairwise = {
        (0, 0): identity,
        (1, 1): identity,
        (2, 2): identity,
        (0, 1): swap,
        (1, 0): swap,
        (1, 2): identity,
        (2, 1): identity,
        (2, 0): identity,
        (0, 2): identity,
    }
    fit = SimpleNamespace(assignment={tuple(swap.tolist()): 1})

    before = cycle_residual(pairwise, 3)
    _corrections, corrected, _status = edge_self_corrected_maps(pairwise, fit, 3, 2)
    after = cycle_residual(corrected, 3)

    assert after < before


def test_ineligible_primes_cannot_be_selected():
    selected, reason = selection_decision(_selectable_row(eligible=False))

    assert selected is False
    assert reason == "prime_not_eligible"


def test_diagnostic_only_rows_cannot_be_selected():
    selected, reason = selection_decision(
        _selectable_row(implemented_corrected_merge=False, na_reason="diagnostic_only")
    )

    assert selected is False
    assert reason == "diagnostic_only"


def test_test_accuracy_is_never_used_for_selection():
    selected, reason = selection_decision(
        _selectable_row(validation_accuracy=0.79, test_accuracy=0.99)
    )

    assert selected is False
    assert reason == "not_selected_fails_unpeeled_baseline_gate"


def test_capacity_and_inference_multipliers_remain_one():
    meta = no_lift_capacity_metadata()

    assert meta["capacity_multiplier"] == 1.0
    assert meta["inference_multiplier"] == 1.0


def test_every_missing_metric_has_na_reason():
    assert row_missing_reason({"validation_accuracy": np.nan, "test_accuracy": 0.8}, "missing_metric") == "missing_metric"
    assert row_missing_reason({"validation_accuracy": 0.8, "test_accuracy": np.nan}, "missing_metric") == "missing_metric"
    assert row_missing_reason({"validation_accuracy": 0.8, "test_accuracy": 0.7}, "missing_metric") == ""


def test_mock_path_can_produce_implemented_corrected_merge_row():
    peel = {
        "prime": 2,
        "prime_index": 0,
        "p_adic_multiplicity": 1,
        "eligible": True,
        "remaining_order_before": 2,
        "remaining_order_after": 1,
    }
    fit = SimpleNamespace(quotient_fit_status="mock_fit", relation_violation_rate=0.0)
    metrics = {"validation_accuracy": 0.84, "test_accuracy": 0.82}
    baseline = {"validation_accuracy": 0.80, "test_accuracy": 0.79}
    controls = {
        "wrong_prime_control": {"validation_accuracy": 0.81, "test_accuracy": 0.80},
        "shuffled_control": {"validation_accuracy": 0.82, "test_accuracy": 0.81},
        "random_residual_control": {"validation_accuracy": 0.83, "test_accuracy": 0.80},
    }

    row = accuracy_row(
        _setting(),
        peel,
        "peeled_p_c2m3_permutation",
        "baseline_c2m3_permutation",
        metrics,
        baseline,
        before=1.0,
        after=0.0,
        reduces=True,
        fit=fit,
        peel_mode="peel_p_only",
        cumulative_primes="2",
        representative_status="observed_edge_representative_permutation",
        implemented=True,
        control_metrics=controls,
    )

    assert row["implemented_corrected_merge"] is True
    assert row["selected_by_validation"] is True
    assert row["capacity_multiplier"] == 1.0
    assert row["inference_multiplier"] == 1.0

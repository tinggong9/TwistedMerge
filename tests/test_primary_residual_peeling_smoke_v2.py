from itertools import permutations
from types import SimpleNamespace

import numpy as np

from experiments.primary_residual_peeling_smoke_v2 import (
    SelectedSetting,
    accuracy_row,
    cycle_residual,
    no_lift_capacity_metadata,
    permutation_json,
    prime_peeling_plan,
    row_missing_reason,
    selection_decision,
)
from src.primary_residual_peeling import (
    compose_perm,
    invert_perm,
    is_valid_permutation,
    permutation_power,
    representative_for_cp_label,
    solve_and_correct_pairwise,
    solve_edge_cochain_mod_p,
    triangle_defects_from_pairwise,
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
        "validation_accuracy": 0.84,
        "test_accuracy": 0.10,
        "baseline_validation_accuracy": 0.80,
        "wrong_prime_control_validation_accuracy": 0.81,
        "shuffled_control_validation_accuracy": 0.82,
        "random_residual_control_validation_accuracy": 0.83,
        "capacity_multiplier": 1.0,
        "inference_multiplier": 1.0,
        "quotient_residual_before": 1.0,
        "quotient_residual_after": 0.0,
        "permutation_cycle_residual_before": 0.25,
        "permutation_cycle_residual_after": 0.25,
    }
    row.update(updates)
    return row


def _oriented(edge_labels, i, j, p):
    if i < j:
        return edge_labels[(i, j)] % p
    return (-edge_labels[(j, i)]) % p


def _coboundary_labels(edge_labels, n_models, p):
    labels = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            for k in range(j + 1, n_models):
                labels[(i, j, k)] = (_oriented(edge_labels, i, j, p) + _oriented(edge_labels, j, k, p) + _oriented(edge_labels, k, i, p)) % p
    return labels


def _complete_pairwise(p01, p12, p20):
    identity = np.arange(len(p01), dtype=int)
    return {
        (0, 0): identity,
        (1, 1): identity,
        (2, 2): identity,
        (0, 1): np.asarray(p01, dtype=int),
        (1, 0): invert_perm(np.asarray(p01, dtype=int)),
        (1, 2): np.asarray(p12, dtype=int),
        (2, 1): invert_perm(np.asarray(p12, dtype=int)),
        (2, 0): np.asarray(p20, dtype=int),
        (0, 2): invert_perm(np.asarray(p20, dtype=int)),
    }


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


def test_solve_edge_cochain_mod_p_recovers_c2_coboundary():
    labels = {(0, 1, 2): 1}

    solution = solve_edge_cochain_mod_p(labels, n_models=3, p=2, sign=1)

    assert solution.solved_exact is True
    assert solution.quotient_residual_before == 1.0
    assert solution.quotient_residual_after == 0.0
    lhs = (_oriented(solution.edge_labels, 0, 1, 2) + _oriented(solution.edge_labels, 1, 2, 2) + _oriented(solution.edge_labels, 2, 0, 2)) % 2
    assert lhs == 1


def test_solve_edge_cochain_mod_p_recovers_c3_coboundary():
    true_edges = {
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 0,
        (1, 2): 2,
        (1, 3): 1,
        (2, 3): 2,
    }
    labels = _coboundary_labels(true_edges, 4, 3)

    solution = solve_edge_cochain_mod_p(labels, n_models=4, p=3, sign=1)

    assert solution.solved_exact is True
    assert solution.quotient_residual_after == 0.0
    for (i, j, k), label in labels.items():
        lhs = (_oriented(solution.edge_labels, i, j, 3) + _oriented(solution.edge_labels, j, k, 3) + _oriented(solution.edge_labels, k, i, 3)) % 3
        assert lhs == label


def test_inconsistent_non_coboundary_data_is_not_marked_exact():
    labels = {
        (0, 1, 2): 1,
        (0, 1, 3): 0,
        (0, 2, 3): 0,
        (1, 2, 3): 0,
    }

    solution = solve_edge_cochain_mod_p(labels, n_models=4, p=2, sign=1)

    assert solution.solved_exact is False
    assert "quotient_cochain_inconsistent" in solution.solve_status


def test_label_zero_representative_is_identity():
    choice = representative_for_cp_label(0, SimpleNamespace(assignment={}), [], width=4, p=3)

    assert choice.status == "identity_representative"
    assert np.array_equal(choice.representative, np.arange(4))


def test_representative_generator_normalization_uses_inverse_label():
    cycle = np.array([1, 2, 0])
    fit = SimpleNamespace(assignment={tuple(cycle.tolist()): 2})

    choice = representative_for_cp_label(1, fit, [cycle], width=3, p=3)

    assert choice.status == "observed_holonomy_power_representative"
    assert choice.generator_label == 2
    assert choice.representative_label == 1
    assert np.array_equal(choice.representative, permutation_power(cycle, 2))


def test_correction_original_shortcut_is_forbidden_by_representative_lift():
    swap = np.array([1, 0, 2])
    identity = np.arange(3)
    cycle = np.array([1, 2, 0])
    p20 = None
    for candidate in permutations(range(3)):
        candidate = np.asarray(candidate, dtype=int)
        if np.array_equal(compose_perm(compose_perm(swap, identity), candidate), cycle):
            p20 = candidate
            break
    assert p20 is not None
    pairwise = _complete_pairwise(swap, identity, p20)
    fit = SimpleNamespace(assignment={tuple(cycle.tolist()): 1})

    result = solve_and_correct_pairwise(pairwise, fit, n_models=3, prime=3)

    assert result.implemented is True
    assert not np.array_equal(result.corrections[(0, 1)], pairwise[(0, 1)])
    assert np.array_equal(result.corrections[(0, 1)], cycle)


def test_corrected_quotient_residual_decreases_in_synthetic_positive_control():
    identity = np.arange(3)
    cycle = np.array([1, 2, 0])
    pairwise = _complete_pairwise(identity, identity, cycle)
    fit = SimpleNamespace(assignment={tuple(cycle.tolist()): 1})

    result = solve_and_correct_pairwise(pairwise, fit, n_models=3, prime=3)

    assert result.solution.solved_exact is True
    assert result.solution.quotient_residual_before == 1.0
    assert result.solution.quotient_residual_after == 0.0
    assert all(is_valid_permutation(perm) for perm in result.corrected.values())


def test_safe_correction_rows_cannot_have_increased_permutation_cycle_residual():
    selected, reason = selection_decision(
        _selectable_row(permutation_cycle_residual_before=0.1, permutation_cycle_residual_after=0.2)
    )

    assert selected is False
    assert reason == "quotient_peel_not_permutation_safe"


def test_triangle_defects_are_recomputed_from_corrected_maps():
    identity = np.arange(3)
    cycle = np.array([1, 2, 0])
    pairwise = _complete_pairwise(identity, identity, cycle)
    before = triangle_defects_from_pairwise(pairwise, 3)
    pairwise[(2, 0)] = identity
    pairwise[(0, 2)] = identity
    after = triangle_defects_from_pairwise(pairwise, 3)

    assert not np.array_equal(before[(0, 1, 2)], after[(0, 1, 2)])


def test_test_accuracy_is_never_used_for_selection():
    selected, reason = selection_decision(
        _selectable_row(validation_accuracy=0.79, test_accuracy=0.99)
    )

    assert selected is False
    assert reason == "not_selected_fails_unpeeled_baseline_gate"


def test_no_lift_rows_have_capacity_and_inference_multipliers_one():
    meta = no_lift_capacity_metadata()

    assert meta["capacity_multiplier"] == 1.0
    assert meta["inference_multiplier"] == 1.0


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


def test_every_missing_metric_has_na_reason():
    assert row_missing_reason({"validation_accuracy": np.nan, "test_accuracy": 0.8}, "missing_metric") == "missing_metric"
    assert row_missing_reason({"validation_accuracy": 0.8, "test_accuracy": np.nan}, "missing_metric") == "missing_metric"
    assert row_missing_reason({"validation_accuracy": 0.8, "test_accuracy": 0.7}, "missing_metric") == ""


def test_mock_path_can_produce_selected_no_lift_row():
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
        after=1.0,
        reduces=True,
        fit=fit,
        peel_mode="peel_p_only",
        cumulative_primes="2",
        representative_status="representative_correction_available",
        implemented=True,
        control_metrics=controls,
        quotient_before=1.0,
        quotient_after=0.0,
        edge_cochain_solve_status="exact_quotient_cochain_solve_sign_1",
        representative_selection_status="representative_correction_available",
    )

    assert row["implemented_corrected_merge"] is True
    assert row["selected_by_validation"] is True
    assert row["capacity_multiplier"] == 1.0
    assert row["inference_multiplier"] == 1.0
    assert permutation_json(np.arange(3)) == "[0,1,2]"

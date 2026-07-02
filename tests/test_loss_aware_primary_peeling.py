from types import SimpleNamespace

import numpy as np

from src.loss_aware_primary_peeling import (
    LossAwareObjectiveWeights,
    assemble_loss_aware_corrections,
    build_representative_bank,
    cumulative_update_allowed_loss_aware,
    no_lift_capacity_metadata,
    validation_selection_decision,
)
from src.primary_residual_peeling import invert_perm, is_valid_permutation, permutation_power


def _identity(width: int) -> np.ndarray:
    return np.arange(width, dtype=int)


def _complete_identity_pairwise(n_models: int, width: int) -> dict[tuple[int, int], np.ndarray]:
    identity = _identity(width)
    return {(i, j): identity.copy() for i in range(n_models) for j in range(n_models)}


def _directed_labels(n_models: int, updates: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    labels = {(i, j): 0 for i in range(n_models) for j in range(n_models)}
    labels.update(updates)
    return labels


def _selectable_row(**updates) -> dict:
    row = {
        "uses_test_for_selection": False,
        "implemented_corrected_merge": True,
        "validation_accuracy": 0.84,
        "test_accuracy": 0.01,
        "baseline_validation_accuracy": 0.80,
        "wrong_prime_control_validation_accuracy": 0.81,
        "shuffled_control_validation_accuracy": 0.82,
        "random_control_validation_accuracy": 0.83,
        "no_quotient_control_validation_accuracy": 0.79,
        "quotient_residual_before": 1.0,
        "quotient_residual_after": 0.0,
        "permutation_cycle_residual_before": 0.5,
        "permutation_cycle_residual_after": 0.4,
    }
    row.update(updates)
    return row


def test_representative_bank_includes_identity_for_label_zero():
    bank = build_representative_bank(SimpleNamespace(assignment={}), {}, [], width=4, p=3)

    assert any(np.array_equal(candidate.perm, _identity(4)) for candidate in bank[0])
    assert all(candidate.is_valid_permutation for candidates in bank.values() for candidate in candidates)


def test_representative_bank_rejects_wrong_verified_label():
    cycle = np.array([1, 2, 0])
    cycle_squared = permutation_power(cycle, 2)
    fit = SimpleNamespace(assignment={tuple(cycle.tolist()): 1, tuple(cycle_squared.tolist()): 1})

    bank = build_representative_bank(fit, {}, [cycle], width=3, p=3)

    assert not any(np.array_equal(candidate.perm, cycle_squared) for candidate in bank[2])


def test_representative_bank_keeps_generator_powers_with_correct_labels():
    cycle = np.array([1, 2, 0])
    cycle_squared = permutation_power(cycle, 2)
    fit = SimpleNamespace(assignment={tuple(cycle.tolist()): 1, tuple(cycle_squared.tolist()): 2})

    bank = build_representative_bank(fit, {}, [cycle], width=3, p=3)

    assert any(np.array_equal(candidate.perm, cycle_squared) for candidate in bank[2])


def test_beam_search_returns_valid_representative_for_every_edge_label_when_available():
    cycle = np.array([1, 2, 0])
    cycle_squared = permutation_power(cycle, 2)
    fit = SimpleNamespace(assignment={tuple(cycle.tolist()): 1, tuple(cycle_squared.tolist()): 2})
    bank = build_representative_bank(fit, {}, [cycle], width=3, p=3)
    pairwise = _complete_identity_pairwise(n_models=3, width=3)
    labels = _directed_labels(3, {(0, 1): 1, (1, 0): 2})

    result = assemble_loss_aware_corrections(pairwise, labels, bank, n_models=3, p=3, max_beam_size=32)

    assert result.implemented is True
    for edge, label in labels.items():
        assert result.selected_candidates[edge].label == label
        assert is_valid_permutation(result.corrections[edge])


def test_beam_search_prefers_lower_displacement_when_quotient_residual_ties():
    low = np.array([1, 0, 2, 3])
    high = np.array([1, 2, 3, 0])
    bank = {
        0: [],
        1: [
            SimpleNamespace(label=1, perm=high, disagreement_from_identity=0.75, source="high", order=4, is_valid_permutation=True),
            SimpleNamespace(label=1, perm=low, disagreement_from_identity=0.50, source="low", order=2, is_valid_permutation=True),
        ],
    }
    pairwise = _complete_identity_pairwise(n_models=2, width=4)
    labels = _directed_labels(2, {(0, 1): 1, (1, 0): 1})

    result = assemble_loss_aware_corrections(
        pairwise,
        labels,
        bank,
        n_models=2,
        p=2,
        objective_weights=LossAwareObjectiveWeights(quotient_residual=1.0, permutation_cycle_residual=0.0, representative_displacement=1.0, inverse_consistency=0.0),
        max_beam_size=8,
    )

    assert result.implemented is True
    assert np.array_equal(result.corrections[(0, 1)], low)


def test_beam_search_blocks_candidates_that_increase_permutation_residual_beyond_tolerance():
    cycle = np.array([1, 2, 0])
    cycle_squared = permutation_power(cycle, 2)
    fit = SimpleNamespace(assignment={tuple(cycle.tolist()): 1, tuple(cycle_squared.tolist()): 2})
    bank = build_representative_bank(fit, {}, [cycle], width=3, p=3)
    pairwise = _complete_identity_pairwise(n_models=3, width=3)
    labels = _directed_labels(3, {(0, 1): 1, (1, 0): 2})

    result = assemble_loss_aware_corrections(
        pairwise,
        labels,
        bank,
        n_models=3,
        p=3,
        permutation_residual_tolerance=0.0,
    )

    assert result.implemented is False
    assert result.status == "permutation_residual_tolerance_blocked_all_candidates"


def test_cumulative_peeling_requires_all_three_gates():
    assert cumulative_update_allowed_loss_aware(True, True, True) is True
    assert cumulative_update_allowed_loss_aware(False, True, True) is False
    assert cumulative_update_allowed_loss_aware(True, False, True) is False
    assert cumulative_update_allowed_loss_aware(True, True, False) is False


def test_validation_selection_does_not_use_test_accuracy():
    selected, reason = validation_selection_decision(_selectable_row(validation_accuracy=0.79, test_accuracy=1.0))

    assert selected is False
    assert reason == "not_selected_fails_unpeeled_baseline_gate"


def test_validation_selection_explicitly_blocks_test_metric_selection():
    selected, reason = validation_selection_decision(_selectable_row(uses_test_for_selection=True, validation_accuracy=0.99, test_accuracy=1.0))

    assert selected is False
    assert reason == "blocked_test_metric_selection_forbidden"


def test_validation_selection_requires_quotient_reduction_and_permutation_safety():
    selected_no_reduction, reason_no_reduction = validation_selection_decision(_selectable_row(quotient_residual_after=1.0))
    selected_unsafe, reason_unsafe = validation_selection_decision(_selectable_row(permutation_cycle_residual_after=0.6))

    assert selected_no_reduction is False
    assert reason_no_reduction == "metric_produced_but_not_claimable"
    assert selected_unsafe is False
    assert reason_unsafe == "quotient_peel_not_permutation_safe"


def test_no_lift_candidates_have_capacity_and_inference_multiplier_one():
    assert no_lift_capacity_metadata() == {"capacity_multiplier": 1.0, "inference_multiplier": 1.0}


def test_no_quotient_or_random_control_can_block_selection():
    selected_random, reason_random = validation_selection_decision(_selectable_row(random_control_validation_accuracy=0.85))
    selected_no_quotient, reason_no_quotient = validation_selection_decision(_selectable_row(no_quotient_control_validation_accuracy=0.85))

    assert selected_random is False
    assert reason_random == "not_selected_fails_random_control"
    assert selected_no_quotient is False
    assert reason_no_quotient == "not_selected_fails_no_quotient_control"

import numpy as np

from src.executed_two_loop_holonomy import (
    METHODS,
    build_case,
    executed_candidate_logits,
    make_dataset,
    structural_certificates,
)


def test_two_independent_loops_have_noncommuting_generators():
    for group_name in ("S3", "D4"):
        case = build_case(group_name, 32, 0)
        cert = structural_certificates(case)
        assert cert["generators_recovered"]
        assert cert["generators_noncommute"]
        assert cert["commutator_residual"] > 0.0


def test_local_models_are_exact_executed_reparameterizations():
    case = build_case("S3", 32, 1)
    assert structural_certificates(case)["local_equivalence_passed"]


def test_regular_action_and_pooling_certificates_hold_for_both_generators():
    case = build_case("D4", 32, 2)
    cert = structural_certificates(case)
    assert cert["group_action_certificate_passed"]
    assert cert["pooling_certificate_passed"]
    assert cert["pooling_residual_gamma_1"] <= 1e-12
    assert cert["pooling_residual_gamma_2"] <= 1e-12


def test_wrong_and_random_controls_fail_structural_certificate():
    case = build_case("S3", 32, 3)
    cert = structural_certificates(case)
    assert cert["wrong_generator_recovery_residual"] > 0.0
    assert cert["random_action_multiplication_residual"] > 0.0
    assert cert["wrong_controls_rejected_structurally"]


def test_every_required_candidate_is_executed_and_has_no_label_input():
    case = build_case("S3", 32, 4)
    val_x, val_y, val_context = make_dataset(case, "validation", 64)
    test_x, _test_y, test_context = make_dataset(case, "test", 96)
    logits = executed_candidate_logits(
        case, test_x, test_context, validation_inputs=val_x, validation_labels=val_y
    )
    assert set(logits) == set(METHODS)
    assert all(value.shape == (96, case.base_model.n_classes) for value in logits.values())


def test_label_permutation_cannot_change_saved_candidate_logits():
    case = build_case("D4", 32, 5)
    val_x, val_y, _ = make_dataset(case, "validation", 64)
    test_x, test_y, test_context = make_dataset(case, "test", 96)
    before = executed_candidate_logits(
        case, test_x, test_context, validation_inputs=val_x, validation_labels=val_y
    )
    saved = {method: value.copy() for method, value in before.items()}
    _permuted_labels = np.random.default_rng(99).permutation(test_y)
    after = executed_candidate_logits(
        case, test_x, test_context, validation_inputs=val_x, validation_labels=val_y
    )
    assert all(np.array_equal(saved[method], after[method]) for method in METHODS)

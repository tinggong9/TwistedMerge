"""Saved-logit helpers for the clean controlled central reproduction."""

from __future__ import annotations

import numpy as np

from src.controlled_twisted_overlaps import (
    ControlledCase,
    align_hidden_to_base,
    average_models,
    branch_sign_from_assignment,
    distilled_twisted_scale,
    global_validation_branch_assignment,
    learned_context_router_assignment,
    make_parameter_matched_wide_model,
    no_twist_branch_assignment,
    random_branch_assignment,
    twisted_branch_assignment,
    wrong_context_assignment,
    wrong_twist_assignment,
)


def _assignment_logits(case: ControlledCase, model, assignment) -> np.ndarray:
    outputs = []
    for face in sorted(case.test_face_data):
        inputs, _evaluation_labels = case.test_face_data[face]
        sign = branch_sign_from_assignment(assignment[face])
        outputs.append(sign * model.logits(inputs))
    return np.concatenate(outputs)


def _predictor_logits(case: ControlledCase, predictor) -> np.ndarray:
    return np.concatenate([predictor(face, case.test_face_data[face][0]) for face in sorted(case.test_face_data)])


def central_candidate_predictors(case: ControlledCase):
    """Return zero-argument candidate executors that never read test labels."""
    aligned_models = [align_hidden_to_base(model) for model in case.local_models]
    ordinary = average_models(case.local_models)
    aligned = average_models(aligned_models)
    rng = np.random.default_rng(case.seed + 991 * case.width)
    supplied = twisted_branch_assignment(case)
    random_assignment = random_branch_assignment(case, rng)
    global_validation = global_validation_branch_assignment(case, aligned)
    face_table, _details = learned_context_router_assignment(case, aligned)
    wrong_twist = wrong_twist_assignment(case)
    wrong_context = wrong_context_assignment(case, rng)
    no_twist = no_twist_branch_assignment(case)
    distilled_scale, _ = distilled_twisted_scale(case, aligned)
    wide = make_parameter_matched_wide_model(aligned, aligned.parameter_count * case.branch_count)
    return {
        "ordinary_weight_average": lambda: _predictor_logits(case, lambda _face, x: ordinary.logits(x)),
        "git_rebasin_pairwise": lambda: _predictor_logits(case, lambda _face, x: aligned.logits(x)),
        "c2m3_synchronized": lambda: _predictor_logits(case, lambda _face, x: aligned.logits(x)),
        "supplied_context_q2_branch_predictor": lambda: _assignment_logits(case, aligned, supplied),
        "random_branch_control": lambda: _assignment_logits(case, aligned, random_assignment),
        "validation_global_branch_selector": lambda: _assignment_logits(case, aligned, global_validation),
        "validation_face_table_router": lambda: _assignment_logits(case, aligned, face_table),
        "wrong_twist_control": lambda: _assignment_logits(case, aligned, wrong_twist),
        "wrong_context_control": lambda: _assignment_logits(case, aligned, wrong_context),
        "no_twist_branch_control": lambda: _assignment_logits(case, aligned, no_twist),
        "distilled_single_model_control": lambda: _predictor_logits(case, lambda _face, x: distilled_scale * aligned.logits(x)),
        "parameter_matched_wide_control": lambda: _predictor_logits(case, lambda _face, x: wide.logits(x)),
        "ensemble_reference": lambda: _predictor_logits(
            case, lambda _face, x: np.stack([model.logits(x) for model in case.local_models]).mean(axis=0)
        ),
    }


def executed_central_candidate_logits(case: ControlledCase) -> dict[str, np.ndarray]:
    """Execute central candidates without reading any test labels."""

    return {method: predictor() for method, predictor in central_candidate_predictors(case).items()}


def concatenated_test_labels(case: ControlledCase) -> np.ndarray:
    return np.concatenate([case.test_face_data[face][1] for face in sorted(case.test_face_data)])

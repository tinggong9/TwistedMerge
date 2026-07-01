from __future__ import annotations

import numpy as np

from src.method_family_baselines import (
    dare_merge_delta,
    sequential_slerp,
    slerp_vector,
    structural_coverage_matrix,
    task_arithmetic_merge,
    task_vector,
    ties_merge_delta,
)


def test_slerp_preserves_parameter_shapes_and_endpoints():
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([0.0, 1.0, 0.0])
    assert slerp_vector(v0, v1, 0.0).shape == v0.shape
    np.testing.assert_allclose(slerp_vector(v0, v1, 0.0), v0)
    np.testing.assert_allclose(slerp_vector(v0, v1, 1.0), v1)
    assert sequential_slerp([v0, v1, np.array([0.0, 0.0, 1.0])]).shape == v0.shape


def test_task_vector_reconstructs_base_plus_delta():
    base = np.array([1.0, -2.0, 0.5])
    target = np.array([2.5, -1.0, 3.0])
    tau = task_vector(base, target)
    np.testing.assert_allclose(base + tau, target)
    np.testing.assert_allclose(task_arithmetic_merge(base, [tau], alpha=1.0), target)


def test_ties_sign_election_keeps_consistent_coordinates():
    deltas = [
        np.array([1.0, -2.0, 0.1]),
        np.array([2.0, 3.0, -0.2]),
        np.array([-1.0, 4.0, -0.3]),
    ]
    merged = ties_merge_delta(deltas, keep_rate=1.0)
    np.testing.assert_allclose(merged, np.array([1.5, 3.5, -0.25]))


def test_dare_rescaling_approximately_preserves_expected_delta():
    delta = np.ones(512)
    merged = dare_merge_delta([delta], drop_rate=0.5, seed=123, n_masks=400)
    assert abs(float(merged.mean()) - 1.0) < 0.08


def test_structural_coverage_matrix_has_required_methods_without_missing_flags():
    rows = structural_coverage_matrix()
    methods = {row["method"] for row in rows}
    assert {
        "weight_average",
        "greedy_soup",
        "c2m3_permutation",
        "twistedmerge_selector",
        "slerp",
        "task_arithmetic",
        "ties",
        "dare",
    }.issubset(methods)
    required = [
        "validation_selection",
        "pairwise_gauge_synchronization",
        "permutation_gauge_handling",
        "monomial_relu_scaling_gauge_handling",
        "coordinatewise_sign_or_sparsity",
        "cycle_holonomy_diagnostic",
        "central_projective_obstruction_detection",
        "conservative_rejection_no_lift",
    ]
    for row in rows:
        for key in required:
            assert row[key] in {"yes", "no", "partial"}

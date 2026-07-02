from __future__ import annotations

import numpy as np
import pandas as pd

from src.small_order_torsion import (
    DEFAULT_POLICIES,
    analyze_permutation_residual,
    analyze_residual_matrix,
    permutation_matrix,
    policy_passes,
)
from src.validation_gated_period_index_lift import (
    SelectorPolicy,
    classify_rank,
    torsion_safe_selector,
)


def test_exact_minus_identity_is_strict_torsion():
    metrics = analyze_residual_matrix(-np.eye(4), orders=(2, 3, 4, 5, 6, 8))
    assert metrics["detected_order"] == 2
    assert metrics["centrality_residual"] == 0.0
    assert metrics["scalar_residual_best"] < 1e-12
    assert policy_passes(metrics, DEFAULT_POLICIES[0])


def test_noncentral_finite_order_permutation_is_rejected():
    perm = np.array([1, 0, 2, 3])
    metrics = analyze_permutation_residual(perm, orders=(2, 3, 4, 5, 6, 8))
    assert metrics["finite_order_residual_d2"] == 0.0
    assert metrics["centrality_residual"] > 0.1
    assert metrics["explained_as_noncentral_holonomy"]
    assert not policy_passes(metrics, DEFAULT_POLICIES[0])


def test_period_index_rank_gate_rejects_period_divisible_index_obstructed_rank():
    assert classify_rank(2, 4, 1) == "rank_not_period_divisible"
    assert classify_rank(2, 4, 2) == "period_divisible_index_obstructed"
    assert classify_rank(2, 4, 4) == "index_divisible_lift_allowed"


def test_torsion_safe_selector_uses_validation_fallback_without_test_tie_break():
    rows = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "method": "greedy_soup",
                "candidate_method": "fallback_greedy_soup",
                "val_accuracy": 0.80,
                "val_loss": 0.5,
                "test_accuracy": 0.10,
                "test_loss": 2.0,
            },
            {
                "run_id": "r0",
                "method": "c2m3_synchronized",
                "candidate_method": "fallback_c2m3",
                "val_accuracy": 0.79,
                "val_loss": 0.4,
                "test_accuracy": 0.99,
                "test_loss": 0.1,
            },
        ]
    )
    selected = torsion_safe_selector(rows, pd.DataFrame(), SelectorPolicy())
    assert selected.iloc[0]["method"] == "greedy_soup"
    assert bool(selected.iloc[0]["selector_no_test_leakage"])
    assert not bool(selected.iloc[0]["selected_lift"])


def test_permutation_matrix_shape_for_bootstrap_inputs():
    perm = np.array([1, 2, 0])
    matrix = permutation_matrix(perm)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix.sum(axis=0), np.ones(3))
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3))

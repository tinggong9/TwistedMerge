from __future__ import annotations

import numpy as np
import torch

from src.holonomy_application_corpus import LowRankChartAdapter
from src.lineage_merge_audit import (
    align_components,
    aligned_rank_bounded_merge,
    binary_prediction_metrics,
    components_logits,
    double_holdout_logistic,
    double_holdout_ridge,
    harmful_merge_label,
    model_components,
    raw_parameter_average,
    validation_selected_interpolation,
)


def random_orthogonal(dimension: int, seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(dimension, dimension))
    left, _singular, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right


def test_representation_alignment_preserves_logits_before_rank_compression() -> None:
    torch.manual_seed(5)
    model = LowRankChartAdapter(feature_dim=7, rank=3, classes=4)
    with torch.no_grad():
        model.up.weight.normal_(std=0.1)
    features = np.random.default_rng(8).normal(size=(20, 7))
    components = model_components(model)
    aligned = align_components(components, random_orthogonal(7, 9))
    np.testing.assert_allclose(
        components_logits(components, features),
        components_logits(aligned, features),
        atol=1e-10,
        rtol=1e-10,
    )


def test_merge_methods_keep_rank_and_fixed_interpolation_budget() -> None:
    torch.manual_seed(10)
    left = LowRankChartAdapter(feature_dim=6, rank=2, classes=3)
    right = LowRankChartAdapter(feature_dim=6, rank=2, classes=3)
    with torch.no_grad():
        left.up.weight.normal_(std=0.1)
        right.up.weight.normal_(std=0.1)
    raw = raw_parameter_average((left, right))
    aligned = aligned_rank_bounded_merge((left, right), (np.eye(6), random_orthogonal(6, 2)))
    selected, weight, evaluations = validation_selected_interpolation(
        left, right, lambda model: -float(torch.linalg.norm(model.up.weight.detach()))
    )
    assert raw.rank == aligned.rank == selected.rank == 2
    assert 0.0 <= weight <= 1.0
    assert evaluations == 11


def test_double_holdout_excludes_seed_family_and_loop() -> None:
    rows = [(seed, family) for seed in range(5) for family in ("AB", "AC", "BC")]
    seeds = np.asarray([row[0] for row in rows])
    families = np.asarray([row[1] for row in rows])
    loops = np.asarray([f"{seed}:{family}" for seed, family in rows])
    features = np.column_stack([seeds, np.asarray([len(value) for value in families])])
    targets = ((seeds + np.asarray([0 if value == "AB" else 1 for value in families])) % 2).astype(int)
    probabilities, logistic_audit = double_holdout_logistic(features, targets, seeds, families, loops)
    predictions, ridge_audit = double_holdout_ridge(features, targets.astype(float), seeds, families, loops)
    assert np.isfinite(probabilities).all() and np.isfinite(predictions).all()
    assert all(row["seed_excluded_from_train"] for row in logistic_audit)
    assert all(row["family_excluded_from_train"] for row in logistic_audit)
    assert all(row["loop_excluded_from_train"] for row in ridge_audit)
    assert 0.0 <= binary_prediction_metrics(targets, probabilities).brier <= 1.0


def test_harmful_merge_threshold_is_frozen() -> None:
    assert harmful_merge_label(0.65, 0.50, 0.70, 0.68, 0.60, 0.58)
    assert not harmful_merge_label(0.685, 0.575, 0.70, 0.68, 0.60, 0.58)

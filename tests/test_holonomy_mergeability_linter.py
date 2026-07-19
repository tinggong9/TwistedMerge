from __future__ import annotations

import numpy as np

from src.holonomy_mergeability_linter import (
    coverage_risk_rows,
    double_holdout_predictions,
    linter_metrics,
)


def test_double_holdout_excludes_seed_and_family() -> None:
    rows = []
    for seed in range(4):
        for family in ("a", "b", "c"):
            for value in range(6):
                rows.append((seed, family, value))
    seeds = np.asarray([row[0] for row in rows])
    families = np.asarray([row[1] for row in rows])
    values = np.asarray([row[2] for row in rows], dtype=float)
    features = np.column_stack([values, seeds])
    targets = ((values + seeds) % 3 == 0).astype(int)
    probabilities, folds, _coefficients = double_holdout_predictions(
        features, targets, seeds, families, random_state=7
    )
    assert np.isfinite(probabilities).all()
    assert len(folds) == 12
    assert all(row["seed_excluded_from_train"] for row in folds)
    assert all(row["family_excluded_from_train"] for row in folds)


def test_metrics_and_coverage_are_well_formed() -> None:
    targets = np.asarray([0, 0, 0, 1, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    metrics = linter_metrics(targets, probabilities)
    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0
    assert metrics.false_lift_rate == 0.0
    rows = coverage_risk_rows(targets, probabilities)
    assert len(rows) == 5
    assert rows[0]["coverage"] == 1.0

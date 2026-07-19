"""Grouped logistic-regression utilities for the mergeability linter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LinterMetrics:
    auroc: float
    auprc: float
    brier: float
    ece: float
    accuracy: float
    harmful_avoidance: float
    false_lift_rate: float
    missed_lift_rate: float


def double_holdout_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    seeds: np.ndarray,
    families: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, list[dict[str, object]], list[np.ndarray]]:
    """Hold out both a corpus seed and an entire setting family for each test cell."""

    probabilities = np.full(len(targets), np.nan, dtype=np.float64)
    folds: list[dict[str, object]] = []
    coefficients: list[np.ndarray] = []
    for seed in np.unique(seeds):
        for family in np.unique(families):
            test = (seeds == seed) & (families == family)
            if not test.any():
                continue
            train = (seeds != seed) & (families != family)
            train_targets = targets[train]
            if len(np.unique(train_targets)) < 2:
                probability = float(train_targets.mean()) if len(train_targets) else float(targets.mean())
                probabilities[test] = probability
                coefficients.append(np.zeros(features.shape[1], dtype=np.float64))
                fit_status = "constant_prior_single_class_train"
            else:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                )
                model.fit(features[train], train_targets)
                probabilities[test] = model.predict_proba(features[test])[:, 1]
                coefficients.append(model[-1].coef_[0].copy())
                fit_status = "logistic_regression"
            folds.append(
                {
                    "heldout_seed": int(seed),
                    "heldout_family": str(family),
                    "train_rows": int(train.sum()),
                    "test_rows": int(test.sum()),
                    "train_positive": int(train_targets.sum()),
                    "test_positive": int(targets[test].sum()),
                    "seed_excluded_from_train": bool(not np.any(seeds[train] == seed)),
                    "family_excluded_from_train": bool(not np.any(families[train] == family)),
                    "fit_status": fit_status,
                }
            )
    if np.isnan(probabilities).any():
        raise RuntimeError("double-holdout prediction left unscored rows")
    return probabilities, folds, coefficients


def expected_calibration_error(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lower) & (
            (probabilities <= upper) if upper == 1.0 else (probabilities < upper)
        )
        if mask.any():
            result += float(mask.mean() * abs(probabilities[mask].mean() - targets[mask].mean()))
    return result


def linter_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> LinterMetrics:
    predictions = probabilities >= threshold
    positives = targets == 1
    negatives = ~positives
    true_positive = int((predictions & positives).sum())
    false_positive = int((predictions & negatives).sum())
    false_negative = int((~predictions & positives).sum())
    true_negative = int((~predictions & negatives).sum())
    return LinterMetrics(
        auroc=float(roc_auc_score(targets, probabilities)),
        auprc=float(average_precision_score(targets, probabilities)),
        brier=float(brier_score_loss(targets, probabilities)),
        ece=expected_calibration_error(targets, probabilities),
        accuracy=float((predictions == targets).mean()),
        harmful_avoidance=true_positive / max(true_positive + false_negative, 1),
        false_lift_rate=false_positive / max(false_positive + true_negative, 1),
        missed_lift_rate=false_negative / max(true_positive + false_negative, 1),
    )


def coverage_risk_rows(targets: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    rows = []
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predictions = probabilities >= 0.5
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        covered = confidence >= threshold
        rows.append(
            {
                "confidence_threshold": threshold,
                "coverage": float(covered.mean()),
                "risk": float((predictions[covered] != targets[covered]).mean()) if covered.any() else float("nan"),
            }
        )
    return rows


def reliability_rows(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    fraction, mean = calibration_curve(targets, probabilities, n_bins=bins, strategy="uniform")
    return [
        {
            "mean_predicted_probability": float(predicted),
            "observed_positive_fraction": float(observed),
        }
        for predicted, observed in zip(mean, fraction, strict=True)
    ]

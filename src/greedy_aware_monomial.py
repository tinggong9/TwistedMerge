"""Greedy-aware monomial selectors and soup diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, isfinite, sqrt
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .improved_monomial_merge import ValidationChoice
from .model_merging_benchmark import require_torch


DEFAULT_GREEDY_AWARE_POOL = (
    "c2m3_greedy_soup",
    "monomial_scaled_greedy_soup",
    "shrinkage_monomial_greedy_soup",
    "global_monomial_greedy_soup",
    "optimized_monomial_greedy_soup",
    "union_candidate_soup",
    "c2m3_permutation",
    "shrinkage_monomial_scale",
    "global_monomial_scale",
    "optimized_monomial_scale",
)


@dataclass(frozen=True)
class GreedyAwareChoice(ValidationChoice):
    baseline: str = "greedy_soup"
    challenger: str = ""
    epsilon: float = 0.0
    loss_slack: float = float("inf")
    validation_accuracy_delta: float = 0.0
    validation_loss_delta: float = 0.0
    lower_confidence_bound: float | None = None
    selection_rule: str = "greedy_aware_selector"


def _candidate_key(metrics: Mapping[str, float]) -> tuple[float, float]:
    return float(metrics["accuracy"]), -float(metrics["loss"])


def best_validation_challenger(
    val_metrics_by_name: Mapping[str, Mapping[str, float]],
    *,
    baseline: str = "greedy_soup",
    challenger_pool: Sequence[str] | None = None,
) -> str:
    names = list(challenger_pool) if challenger_pool is not None else [
        name for name in val_metrics_by_name if name != baseline
    ]
    names = [name for name in names if name in val_metrics_by_name and name != baseline]
    if not names:
        raise ValueError("at least one non-baseline challenger is required")
    return max(names, key=lambda name: (_candidate_key(val_metrics_by_name[name]), name))


def greedy_aware_selector(
    val_metrics_by_name: Mapping[str, Mapping[str, float]],
    *,
    baseline: str = "greedy_soup",
    challenger_pool: Sequence[str] | None = None,
    epsilon: float = 0.0,
    loss_slack: float = float("inf"),
) -> GreedyAwareChoice:
    """Conservative selector that treats greedy soup as the fallback."""

    if baseline not in val_metrics_by_name:
        raise ValueError(f"baseline {baseline!r} is missing")
    challenger = best_validation_challenger(
        val_metrics_by_name,
        baseline=baseline,
        challenger_pool=challenger_pool,
    )
    base_metrics = val_metrics_by_name[baseline]
    challenger_metrics = val_metrics_by_name[challenger]
    acc_delta = float(challenger_metrics["accuracy"] - base_metrics["accuracy"])
    loss_delta = float(challenger_metrics["loss"] - base_metrics["loss"])
    choose_challenger = acc_delta >= float(epsilon) and loss_delta <= float(loss_slack)
    selected = challenger if choose_challenger else baseline
    selected_metrics = val_metrics_by_name[selected]
    return GreedyAwareChoice(
        selected=selected,
        selected_val_accuracy=float(selected_metrics["accuracy"]),
        selected_val_loss=float(selected_metrics["loss"]),
        margin_to_runner_up=acc_delta if choose_challenger else -acc_delta,
        used_test_metrics=False,
        baseline=baseline,
        challenger=challenger,
        epsilon=float(epsilon),
        loss_slack=float(loss_slack),
        validation_accuracy_delta=acc_delta,
        validation_loss_delta=loss_delta,
        lower_confidence_bound=None,
        selection_rule="greedy_aware_selector",
    )


def _normal_quantile_for_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    # The experiment uses 95% by default; keep the implementation dependency
    # free and fall back to the common two-sided normal values.
    if abs(confidence - 0.95) < 1e-12:
        return 1.96
    if abs(confidence - 0.90) < 1e-12:
        return 1.645
    if abs(confidence - 0.99) < 1e-12:
        return 2.576
    # Acklam's approximation would be overkill here.  This monotone fallback is
    # accurate enough for a conservative selector sanity check.
    return sqrt(2.0) * _inverse_erf(confidence)


def _inverse_erf(x: float) -> float:
    # Winitzki approximation.
    a = 0.147
    sign = -1.0 if x < 0 else 1.0
    ln = np.log(max(1.0 - x * x, 1e-12))
    first = 2.0 / (np.pi * a) + ln / 2.0
    return float(sign * np.sqrt(np.sqrt(first * first - ln / a) - first))


def lower_confidence_greedy_aware_selector(
    val_metrics_by_name: Mapping[str, Mapping[str, float]],
    *,
    baseline: str = "greedy_soup",
    challenger_pool: Sequence[str] | None = None,
    n_validation: int,
    confidence: float = 0.95,
    loss_slack: float = float("inf"),
) -> GreedyAwareChoice:
    """Choose challenger only if a validation-accuracy LCB beats zero."""

    if n_validation <= 0:
        raise ValueError("n_validation must be positive")
    challenger = best_validation_challenger(
        val_metrics_by_name,
        baseline=baseline,
        challenger_pool=challenger_pool,
    )
    base_metrics = val_metrics_by_name[baseline]
    challenger_metrics = val_metrics_by_name[challenger]
    acc_delta = float(challenger_metrics["accuracy"] - base_metrics["accuracy"])
    loss_delta = float(challenger_metrics["loss"] - base_metrics["loss"])
    p_base = float(np.clip(base_metrics["accuracy"], 0.0, 1.0))
    p_chal = float(np.clip(challenger_metrics["accuracy"], 0.0, 1.0))
    se = sqrt(max(p_base * (1.0 - p_base) + p_chal * (1.0 - p_chal), 0.0) / float(n_validation))
    lcb = acc_delta - _normal_quantile_for_confidence(confidence) * se
    choose_challenger = lcb > 0.0 and loss_delta <= float(loss_slack)
    selected = challenger if choose_challenger else baseline
    selected_metrics = val_metrics_by_name[selected]
    return GreedyAwareChoice(
        selected=selected,
        selected_val_accuracy=float(selected_metrics["accuracy"]),
        selected_val_loss=float(selected_metrics["loss"]),
        margin_to_runner_up=acc_delta if choose_challenger else -acc_delta,
        used_test_metrics=False,
        baseline=baseline,
        challenger=challenger,
        epsilon=0.0,
        loss_slack=float(loss_slack),
        validation_accuracy_delta=acc_delta,
        validation_loss_delta=loss_delta,
        lower_confidence_bound=float(lcb),
        selection_rule="lower_confidence_greedy_aware_selector",
    )


def tune_greedy_aware_thresholds(
    settings: Sequence[Mapping[str, Mapping[str, float]]],
    *,
    epsilon_grid: Sequence[float],
    loss_slack_grid: Sequence[float],
    baseline: str = "greedy_soup",
    challenger_pool: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Choose epsilon/loss_slack using validation metrics only."""

    if not settings:
        raise ValueError("at least one setting is required")
    best = None
    for epsilon in epsilon_grid:
        for loss_slack in loss_slack_grid:
            accuracies = []
            losses = []
            for metrics in settings:
                choice = greedy_aware_selector(
                    metrics,
                    baseline=baseline,
                    challenger_pool=challenger_pool,
                    epsilon=float(epsilon),
                    loss_slack=float(loss_slack),
                )
                accuracies.append(choice.selected_val_accuracy)
                losses.append(choice.selected_val_loss)
            key = (float(np.mean(accuracies)), -float(np.mean(losses)), -float(epsilon), -float(loss_slack))
            if best is None or key > best[0]:
                best = (key, float(epsilon), float(loss_slack))
    assert best is not None
    return best[1], best[2]


def least_squares_scale(source: np.ndarray, target: np.ndarray, min_scale: float = 1e-3, max_scale: float = 1e3) -> float:
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    denom = max(float(np.dot(src, src)), 1e-12)
    scale = float(np.dot(src, tgt) / denom)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(np.clip(scale, min_scale, max_scale))


def _positive_ratios(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    mask = (src > 1e-8) & np.isfinite(src) & np.isfinite(tgt)
    ratios = tgt[mask] / src[mask]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    return ratios


def median_ratio_scale(source: np.ndarray, target: np.ndarray, min_scale: float = 1e-3, max_scale: float = 1e3) -> float:
    ratios = _positive_ratios(source, target)
    scale = 1.0 if ratios.size == 0 else float(np.median(ratios))
    return float(np.clip(scale, min_scale, max_scale))


def trimmed_mean_ratio_scale(
    source: np.ndarray,
    target: np.ndarray,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
    trim_fraction: float = 0.1,
) -> float:
    ratios = np.sort(_positive_ratios(source, target))
    if ratios.size == 0:
        scale = 1.0
    else:
        trim = int(np.floor(float(trim_fraction) * ratios.size))
        kept = ratios[trim : ratios.size - trim] if trim > 0 and ratios.size > 2 * trim else ratios
        scale = float(np.mean(kept))
    return float(np.clip(scale, min_scale, max_scale))


def huber_ratio_scale(
    source: np.ndarray,
    target: np.ndarray,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
    delta: float = 1.5,
    max_iters: int = 20,
) -> float:
    ratios = _positive_ratios(source, target)
    if ratios.size == 0:
        return 1.0
    estimate = float(np.median(ratios))
    for _ in range(int(max_iters)):
        residual = ratios - estimate
        scale = max(float(np.median(np.abs(residual))) / 0.6745, 1e-8)
        weights = np.minimum(1.0, float(delta) * scale / np.maximum(np.abs(residual), 1e-12))
        new_estimate = float(np.sum(weights * ratios) / max(float(np.sum(weights)), 1e-12))
        if abs(new_estimate - estimate) <= 1e-8 * max(abs(estimate), 1.0):
            estimate = new_estimate
            break
        estimate = new_estimate
    return float(np.clip(estimate, min_scale, max_scale))


def estimate_positive_scale(
    source: np.ndarray,
    target: np.ndarray,
    *,
    estimator: str = "least_squares_scale",
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
) -> float:
    if estimator == "least_squares_scale":
        return least_squares_scale(source, target, min_scale, max_scale)
    if estimator == "median_ratio_scale":
        return median_ratio_scale(source, target, min_scale, max_scale)
    if estimator == "trimmed_mean_ratio_scale":
        return trimmed_mean_ratio_scale(source, target, min_scale, max_scale)
    if estimator == "huber_ratio_scale":
        return huber_ratio_scale(source, target, min_scale, max_scale)
    raise ValueError(f"unknown scale estimator: {estimator}")


def nested_validation_split(dataset, *, val_model_fraction: float, val_selector_fraction: float, seed: int):
    """Split into train_inner, val_model, val_selector with disjoint indices."""

    torch, _, _ = require_torch()
    n = len(dataset)
    if n < 3:
        raise ValueError("dataset must contain at least three samples")
    if val_model_fraction <= 0.0 or val_selector_fraction <= 0.0:
        raise ValueError("validation fractions must be positive")
    n_val_model = max(1, int(round(n * float(val_model_fraction))))
    n_val_selector = max(1, int(round(n * float(val_selector_fraction))))
    if n_val_model + n_val_selector >= n:
        raise ValueError("validation splits leave no training data")
    n_train = n - n_val_model - n_val_selector
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.random_split(dataset, [n_train, n_val_model, n_val_selector], generator=generator)


def selector_regret_analysis(
    df: pd.DataFrame,
    *,
    selector_methods: Sequence[str],
    candidate_methods: Sequence[str],
    baseline: str = "greedy_soup",
) -> pd.DataFrame:
    """Compute selector regret and false/missed challenger counts."""

    rows = []
    pivot = df.pivot_table(
        index=["setting_id"],
        columns="method",
        values=["accuracy", "loss", "val_accuracy", "val_loss", "selector_chose"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{method}" for metric, method in pivot.columns]
    pivot = pivot.reset_index()
    for selector in selector_methods:
        selected_col = f"selector_chose__{selector}"
        if selected_col not in pivot:
            continue
        clean = pivot.dropna(subset=[selected_col, f"accuracy__{baseline}"]).copy()
        deltas = []
        regret_best = []
        regret_greedy = []
        false_challenger = 0
        missed_challenger = 0
        choices: dict[str, int] = {}
        for _, row in clean.iterrows():
            selected = str(row[selected_col])
            choices[selected] = choices.get(selected, 0) + 1
            selected_acc = float(row.get(f"accuracy__{selected}", np.nan))
            greedy_acc = float(row[f"accuracy__{baseline}"])
            candidate_accs = {
                method: float(row.get(f"accuracy__{method}", np.nan))
                for method in candidate_methods
                if np.isfinite(float(row.get(f"accuracy__{method}", np.nan)))
            }
            best_acc = max(candidate_accs.values()) if candidate_accs else greedy_acc
            deltas.append(selected_acc - greedy_acc)
            regret_best.append(best_acc - selected_acc)
            regret_greedy.append(greedy_acc - selected_acc)
            if selected != baseline and selected_acc < greedy_acc:
                false_challenger += 1
            best_challenger = max(
                (acc for method, acc in candidate_accs.items() if method != baseline),
                default=float("-inf"),
            )
            if selected == baseline and best_challenger > greedy_acc:
                missed_challenger += 1
        arr = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "selector": selector,
                "n_rows": int(len(clean)),
                "selected_method_distribution": choices,
                "mean_test_delta_vs_greedy": float(np.mean(arr)) if arr.size else float("nan"),
                "beats_greedy": int(np.sum(arr > 0)),
                "ties_greedy": int(np.sum(arr == 0)),
                "loses_to_greedy": int(np.sum(arr < 0)),
                "mean_regret_vs_best_candidate": float(np.mean(regret_best)) if regret_best else float("nan"),
                "mean_regret_vs_greedy": float(np.mean(regret_greedy)) if regret_greedy else float("nan"),
                "false_challenger_rate": float(false_challenger / max(len(clean), 1)),
                "missed_challenger_rate": float(missed_challenger / max(len(clean), 1)),
                "used_test_metrics": False,
            }
        )
    return pd.DataFrame(rows)

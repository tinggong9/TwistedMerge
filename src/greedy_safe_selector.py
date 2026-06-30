"""Greedy-safe validation selectors.

These helpers treat greedy soup as the default action.  A TwistedMerge
candidate replaces it only when validation evidence clears a conservative
mode-specific gate.  Test metrics are deliberately absent from the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Mapping, Sequence

import numpy as np


MetricMap = Mapping[str, Mapping[str, float]]


DEFAULT_GREEDY_SAFE_POOL = (
    "c2m3_greedy_soup",
    "monomial_scaled_greedy_soup",
    "shrinkage_monomial_greedy_soup",
    "global_monomial_greedy_soup",
    "optimized_monomial_greedy_soup",
    "union_candidate_soup",
    "improved_validated_selector",
    "optimized_monomial_scale",
    "global_monomial_scale",
    "shrinkage_monomial_scale",
    "monomial_scale",
    "c2m3_permutation",
)


@dataclass(frozen=True)
class GreedySafeChoice:
    selected: str
    challenger: str
    baseline: str
    mode: str
    selected_val_accuracy: float
    selected_val_loss: float
    challenger_val_accuracy: float
    challenger_val_loss: float
    baseline_val_accuracy: float
    baseline_val_loss: float
    validation_accuracy_delta: float
    validation_loss_delta: float
    tau_accuracy: float = 0.0
    tau_loss: float = 0.0
    confidence: float | None = None
    lower_confidence_bound: float | None = None
    predicted_regret_bound: float | None = None
    used_test_metrics: bool = False


def _normal_quantile_for_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    common = {0.80: 1.281552, 0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}
    for key, value in common.items():
        if abs(float(confidence) - key) < 1e-12:
            return value
    return sqrt(2.0) * _inverse_erf(confidence)


def _inverse_erf(x: float) -> float:
    # Winitzki approximation, sufficient for conservative selector gates.
    a = 0.147
    sign = -1.0 if x < 0 else 1.0
    ln = np.log(max(1.0 - x * x, 1e-12))
    first = 2.0 / (np.pi * a) + ln / 2.0
    return float(sign * np.sqrt(np.sqrt(first * first - ln / a) - first))


def _metric_key(metrics: Mapping[str, float]) -> tuple[float, float]:
    return float(metrics["accuracy"]), -float(metrics["loss"])


def best_challenger(
    val_metrics_by_name: MetricMap,
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
    return max(names, key=lambda name: (_metric_key(val_metrics_by_name[name]), name))


def _choice(
    val_metrics_by_name: MetricMap,
    *,
    baseline: str,
    challenger: str,
    mode: str,
    choose_challenger: bool,
    tau_accuracy: float = 0.0,
    tau_loss: float = 0.0,
    confidence: float | None = None,
    lower_confidence_bound: float | None = None,
    predicted_regret_bound: float | None = None,
) -> GreedySafeChoice:
    base = val_metrics_by_name[baseline]
    chal = val_metrics_by_name[challenger]
    selected = challenger if choose_challenger else baseline
    selected_metrics = val_metrics_by_name[selected]
    return GreedySafeChoice(
        selected=selected,
        challenger=challenger,
        baseline=baseline,
        mode=mode,
        selected_val_accuracy=float(selected_metrics["accuracy"]),
        selected_val_loss=float(selected_metrics["loss"]),
        challenger_val_accuracy=float(chal["accuracy"]),
        challenger_val_loss=float(chal["loss"]),
        baseline_val_accuracy=float(base["accuracy"]),
        baseline_val_loss=float(base["loss"]),
        validation_accuracy_delta=float(chal["accuracy"] - base["accuracy"]),
        validation_loss_delta=float(chal["loss"] - base["loss"]),
        tau_accuracy=float(tau_accuracy),
        tau_loss=float(tau_loss),
        confidence=None if confidence is None else float(confidence),
        lower_confidence_bound=lower_confidence_bound,
        predicted_regret_bound=predicted_regret_bound,
        used_test_metrics=False,
    )


def tau_fixed_selector(
    val_metrics_by_name: MetricMap,
    *,
    baseline: str = "greedy_soup",
    challenger_pool: Sequence[str] | None = None,
    tau_accuracy: float = 0.0,
) -> GreedySafeChoice:
    challenger = best_challenger(val_metrics_by_name, baseline=baseline, challenger_pool=challenger_pool)
    delta = float(val_metrics_by_name[challenger]["accuracy"] - val_metrics_by_name[baseline]["accuracy"])
    return _choice(
        val_metrics_by_name,
        baseline=baseline,
        challenger=challenger,
        mode="tau_fixed",
        choose_challenger=delta > float(tau_accuracy),
        tau_accuracy=tau_accuracy,
    )


def tau_loss_aware_selector(
    val_metrics_by_name: MetricMap,
    *,
    baseline: str = "greedy_soup",
    challenger_pool: Sequence[str] | None = None,
    tau_accuracy: float = 0.0,
    tau_loss: float = 0.0,
) -> GreedySafeChoice:
    challenger = best_challenger(val_metrics_by_name, baseline=baseline, challenger_pool=challenger_pool)
    acc_delta = float(val_metrics_by_name[challenger]["accuracy"] - val_metrics_by_name[baseline]["accuracy"])
    loss_improvement = float(val_metrics_by_name[baseline]["loss"] - val_metrics_by_name[challenger]["loss"])
    choose = acc_delta > float(tau_accuracy) or (acc_delta >= 0.0 and loss_improvement > float(tau_loss))
    return _choice(
        val_metrics_by_name,
        baseline=baseline,
        challenger=challenger,
        mode="tau_loss_aware",
        choose_challenger=choose,
        tau_accuracy=tau_accuracy,
        tau_loss=tau_loss,
    )


def bootstrap_lcb_from_correctness(
    challenger_correct: Sequence[bool] | np.ndarray,
    baseline_correct: Sequence[bool] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 12345,
) -> float:
    challenger = np.asarray(challenger_correct, dtype=float)
    baseline = np.asarray(baseline_correct, dtype=float)
    if challenger.shape != baseline.shape:
        raise ValueError("correctness arrays must have the same shape")
    if challenger.size == 0:
        raise ValueError("correctness arrays must be nonempty")
    rng = np.random.default_rng(seed)
    deltas = challenger - baseline
    means = [float(rng.choice(deltas, size=deltas.size, replace=True).mean()) for _ in range(int(n_bootstrap))]
    alpha = 100.0 * (1.0 - float(confidence))
    return float(np.percentile(means, alpha))


def normal_lcb_from_accuracies(challenger_accuracy: float, baseline_accuracy: float, n_validation: int, confidence: float) -> float:
    if n_validation <= 0:
        raise ValueError("n_validation must be positive")
    p_chal = float(np.clip(challenger_accuracy, 0.0, 1.0))
    p_base = float(np.clip(baseline_accuracy, 0.0, 1.0))
    delta = p_chal - p_base
    se = sqrt(max(p_chal * (1.0 - p_chal) + p_base * (1.0 - p_base), 0.0) / float(n_validation))
    return float(delta - _normal_quantile_for_confidence(confidence) * se)


def tau_bootstrap_selector(
    val_metrics_by_name: MetricMap,
    *,
    baseline: str = "greedy_soup",
    challenger_pool: Sequence[str] | None = None,
    n_validation: int | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 12345,
    correctness_by_name: Mapping[str, Sequence[bool] | np.ndarray] | None = None,
) -> GreedySafeChoice:
    challenger = best_challenger(val_metrics_by_name, baseline=baseline, challenger_pool=challenger_pool)
    if correctness_by_name is not None and challenger in correctness_by_name and baseline in correctness_by_name:
        lcb = bootstrap_lcb_from_correctness(
            correctness_by_name[challenger],
            correctness_by_name[baseline],
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    else:
        if n_validation is None:
            raise ValueError("n_validation is required when correctness arrays are unavailable")
        lcb = normal_lcb_from_accuracies(
            val_metrics_by_name[challenger]["accuracy"],
            val_metrics_by_name[baseline]["accuracy"],
            int(n_validation),
            confidence,
        )
    return _choice(
        val_metrics_by_name,
        baseline=baseline,
        challenger=challenger,
        mode="tau_bootstrap",
        choose_challenger=lcb > 0.0,
        confidence=confidence,
        lower_confidence_bound=lcb,
    )


def nested_validation_selector(
    selector_metrics_by_name: MetricMap,
    accept_metrics_by_name: MetricMap,
    *,
    baseline: str = "greedy_soup",
    challenger_pool: Sequence[str] | None = None,
    tau_accuracy: float = 0.0,
    tau_loss: float = 0.0,
) -> GreedySafeChoice:
    challenger = best_challenger(selector_metrics_by_name, baseline=baseline, challenger_pool=challenger_pool)
    if baseline not in accept_metrics_by_name or challenger not in accept_metrics_by_name:
        raise ValueError("accept metrics must include baseline and selected challenger")
    acc_delta = float(accept_metrics_by_name[challenger]["accuracy"] - accept_metrics_by_name[baseline]["accuracy"])
    loss_improvement = float(accept_metrics_by_name[baseline]["loss"] - accept_metrics_by_name[challenger]["loss"])
    choose = acc_delta > float(tau_accuracy) or (acc_delta >= 0.0 and loss_improvement > float(tau_loss))
    return _choice(
        accept_metrics_by_name,
        baseline=baseline,
        challenger=challenger,
        mode="nested_validation",
        choose_challenger=choose,
        tau_accuracy=tau_accuracy,
        tau_loss=tau_loss,
    )


def regret_bound_selector(
    val_metrics_by_name: MetricMap,
    *,
    baseline: str = "greedy_soup",
    challenger_pool: Sequence[str] | None = None,
    regret_threshold: float = 0.0,
    confidence: float = 0.90,
    n_validation: int = 1,
) -> GreedySafeChoice:
    challenger = best_challenger(val_metrics_by_name, baseline=baseline, challenger_pool=challenger_pool)
    lcb = normal_lcb_from_accuracies(
        val_metrics_by_name[challenger]["accuracy"],
        val_metrics_by_name[baseline]["accuracy"],
        int(n_validation),
        confidence,
    )
    predicted_regret = max(0.0, -float(lcb))
    acc_delta = float(val_metrics_by_name[challenger]["accuracy"] - val_metrics_by_name[baseline]["accuracy"])
    choose = acc_delta >= 0.0 and predicted_regret <= float(regret_threshold)
    return _choice(
        val_metrics_by_name,
        baseline=baseline,
        challenger=challenger,
        mode="regret_bound",
        choose_challenger=choose,
        confidence=confidence,
        lower_confidence_bound=lcb,
        predicted_regret_bound=predicted_regret,
    )

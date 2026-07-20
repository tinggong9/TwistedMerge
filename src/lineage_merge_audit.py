"""Capacity-matched branch merging and held-out lineage prediction helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.holonomy_application_corpus import LowRankChartAdapter, classification_metrics
from src.lora_gauge_alignment import canonical_svd_factors


Array = np.ndarray


@dataclass(frozen=True)
class BinaryPredictionMetrics:
    auroc: float
    auprc: float
    brier: float
    ece: float
    accuracy: float
    harmful_avoidance: float


@dataclass(frozen=True)
class MergeComponents:
    effective_adapter: Array
    head_weight: Array
    head_bias: Array


def clone_model(model: LowRankChartAdapter) -> LowRankChartAdapter:
    output = LowRankChartAdapter(model.feature_dim, model.rank, model.classes)
    output.load_state_dict(copy.deepcopy(model.state_dict()))
    output.eval()
    return output


def model_components(model: LowRankChartAdapter) -> MergeComponents:
    return MergeComponents(
        effective_adapter=model.effective_adapter().detach().cpu().double().numpy(),
        head_weight=model.head.weight.detach().cpu().double().numpy(),
        head_bias=model.head.bias.detach().cpu().double().numpy(),
    )


def align_components(components: MergeComponents, row_map_to_common: Array) -> MergeComponents:
    """Re-express a hidden representation while preserving its logits.

    If row activations transform as ``h_common = h @ R``, then the effective
    feature map becomes ``R.T @ E`` and the classifier becomes ``W @ R^-T``.
    """

    row_map = np.asarray(row_map_to_common, dtype=np.float64)
    if row_map.shape != components.effective_adapter.shape:
        raise ValueError("representation map has the wrong shape")
    if not np.isfinite(row_map).all() or np.linalg.matrix_rank(row_map) < row_map.shape[0]:
        raise ValueError("representation map must be finite and invertible")
    return MergeComponents(
        effective_adapter=row_map.T @ components.effective_adapter,
        head_weight=components.head_weight @ np.linalg.inv(row_map).T,
        head_bias=components.head_bias.copy(),
    )


def components_logits(components: MergeComponents, features: Array) -> Array:
    features = np.asarray(features, dtype=np.float64)
    hidden = features @ components.effective_adapter.T
    return hidden @ components.head_weight.T + components.head_bias


def model_from_components(
    template: LowRankChartAdapter,
    components: MergeComponents,
    *,
    rank: int | None = None,
) -> LowRankChartAdapter:
    output_rank = int(rank if rank is not None else template.rank)
    dimension = template.feature_dim
    delta = components.effective_adapter - np.eye(dimension)
    b_factor, a_factor = canonical_svd_factors(delta, output_rank)
    output = LowRankChartAdapter(dimension, output_rank, template.classes)
    with torch.no_grad():
        output.up.weight.copy_(torch.from_numpy(b_factor).to(output.up.weight))
        output.down.weight.copy_(torch.from_numpy(a_factor).to(output.down.weight))
        output.head.weight.copy_(torch.from_numpy(components.head_weight).to(output.head.weight))
        output.head.bias.copy_(torch.from_numpy(components.head_bias).to(output.head.bias))
    output.eval()
    return output


def raw_parameter_average(models: Sequence[LowRankChartAdapter], weights: Sequence[float] | None = None) -> LowRankChartAdapter:
    if not models:
        raise ValueError("at least one model is required")
    if weights is None:
        weights_array = np.full(len(models), 1.0 / len(models))
    else:
        weights_array = np.asarray(weights, dtype=np.float64)
        if len(weights_array) != len(models) or np.any(weights_array < 0) or weights_array.sum() <= 0:
            raise ValueError("merge weights must be nonnegative and aligned")
        weights_array = weights_array / weights_array.sum()
    output = clone_model(models[0])
    state = {}
    for name in output.state_dict():
        state[name] = sum(
            float(weight) * model.state_dict()[name].detach().cpu()
            for weight, model in zip(weights_array, models, strict=True)
        )
    output.load_state_dict(state)
    output.eval()
    return output


def aligned_rank_bounded_merge(
    models: Sequence[LowRankChartAdapter],
    row_maps_to_common: Sequence[Array],
) -> LowRankChartAdapter:
    if len(models) == 0 or len(models) != len(row_maps_to_common):
        raise ValueError("models and maps must be nonempty and aligned")
    aligned = [
        align_components(model_components(model), map_value)
        for model, map_value in zip(models, row_maps_to_common, strict=True)
    ]
    mean = MergeComponents(
        effective_adapter=np.mean([value.effective_adapter for value in aligned], axis=0),
        head_weight=np.mean([value.head_weight for value in aligned], axis=0),
        head_bias=np.mean([value.head_bias for value in aligned], axis=0),
    )
    return model_from_components(models[0], mean, rank=models[0].rank)


def validation_selected_interpolation(
    left: LowRankChartAdapter,
    right: LowRankChartAdapter,
    score: Callable[[LowRankChartAdapter], float],
    grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11)),
) -> tuple[LowRankChartAdapter, float, int]:
    candidates = []
    for right_weight in grid:
        model = raw_parameter_average((left, right), weights=(1.0 - right_weight, right_weight))
        candidates.append((float(score(model)), float(right_weight), model))
    best_score, best_weight, best_model = max(candidates, key=lambda row: (row[0], -row[1]))
    _ = best_score
    return best_model, best_weight, len(candidates)


def logits_for_domains(
    model: LowRankChartAdapter, domain_features: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        return {task: model(features).detach().cpu() for task, features in domain_features.items()}


def score_domain_logits(
    logits: Mapping[str, torch.Tensor], labels: torch.Tensor
) -> dict[str, float]:
    per_domain = {task: classification_metrics(value, labels) for task, value in logits.items()}
    accuracies = [row["accuracy"] for row in per_domain.values()]
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "worst_domain_accuracy": float(np.min(accuracies)),
        "mean_nll": float(np.mean([row["nll"] for row in per_domain.values()])),
        "mean_brier": float(np.mean([row["brier"] for row in per_domain.values()])),
        "mean_ece": float(np.mean([row["ece"] for row in per_domain.values()])),
        **{
            f"{task.lower()}_{metric}": float(value)
            for task, row in per_domain.items()
            for metric, value in row.items()
        },
    }


def harmful_merge_label(
    merge_mean_accuracy: float,
    merge_worst_accuracy: float,
    left_mean_accuracy: float,
    right_mean_accuracy: float,
    left_worst_accuracy: float,
    right_worst_accuracy: float,
) -> bool:
    average_mean = 0.5 * (left_mean_accuracy + right_mean_accuracy)
    average_worst = 0.5 * (left_worst_accuracy + right_worst_accuracy)
    return bool(
        merge_mean_accuracy < average_mean - 0.01
        or merge_worst_accuracy < average_worst - 0.02
    )


def expected_calibration_error(targets: Array, probabilities: Array, bins: int = 10) -> float:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    value = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if np.any(mask):
            value += float(mask.mean() * abs(targets[mask].mean() - probabilities[mask].mean()))
    return value


def binary_prediction_metrics(targets: Array, probabilities: Array) -> BinaryPredictionMetrics:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= 0.5
    auroc = float(roc_auc_score(targets, probabilities)) if len(np.unique(targets)) > 1 else float("nan")
    auprc = (
        float(average_precision_score(targets, probabilities))
        if np.any(targets == 1)
        else float("nan")
    )
    harmful = targets == 1
    avoidance = float(np.mean(predictions[harmful])) if np.any(harmful) else float("nan")
    return BinaryPredictionMetrics(
        auroc=auroc,
        auprc=auprc,
        brier=float(brier_score_loss(targets, probabilities)),
        ece=expected_calibration_error(targets, probabilities),
        accuracy=float(np.mean(predictions == targets)),
        harmful_avoidance=avoidance,
    )


def _fold_masks(
    seeds: Array,
    families: Array,
    loop_ids: Array | None = None,
) -> list[tuple[Array, Array, dict[str, object]]]:
    seeds = np.asarray(seeds)
    families = np.asarray(families)
    loops = np.asarray(loop_ids) if loop_ids is not None else None
    folds = []
    for seed, family in sorted(set(zip(seeds.tolist(), families.tolist()))):
        test = (seeds == seed) & (families == family)
        train = (seeds != seed) & (families != family)
        if loops is not None:
            heldout_loops = set(loops[test].tolist())
            train &= ~np.isin(loops, list(heldout_loops))
        folds.append(
            (
                train,
                test,
                {
                    "heldout_seed": seed,
                    "heldout_family": family,
                    "train_rows": int(train.sum()),
                    "test_rows": int(test.sum()),
                    "seed_excluded_from_train": not np.any(seeds[train] == seed),
                    "family_excluded_from_train": not np.any(families[train] == family),
                    "loop_excluded_from_train": (
                        True
                        if loops is None
                        else not bool(set(loops[train].tolist()).intersection(loops[test].tolist()))
                    ),
                },
            )
        )
    return folds


def double_holdout_logistic(
    features: Array,
    targets: Array,
    seeds: Array,
    families: Array,
    loop_ids: Array | None = None,
) -> tuple[Array, list[dict[str, object]]]:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=int)
    probabilities = np.full(len(targets), np.nan)
    audit = []
    for train, test, row in _fold_masks(seeds, families, loop_ids):
        if not np.any(test) or not np.any(train):
            row["status"] = "insufficient_double_holdout_rows"
            probabilities[test] = float(targets[train].mean()) if np.any(train) else float(targets.mean())
        elif len(np.unique(targets[train])) < 2:
            row["status"] = "constant_training_prevalence"
            probabilities[test] = float(targets[train].mean())
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
            )
            model.fit(features[train], targets[train])
            probabilities[test] = model.predict_proba(features[test])[:, 1]
            row["status"] = "fitted"
        audit.append(row)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("double-holdout logistic prediction left unscored rows")
    return probabilities, audit


def double_holdout_ridge(
    features: Array,
    targets: Array,
    seeds: Array,
    families: Array,
    loop_ids: Array | None = None,
) -> tuple[Array, list[dict[str, object]]]:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.full(len(targets), np.nan)
    audit = []
    for train, test, row in _fold_masks(seeds, families, loop_ids):
        if not np.any(test) or not np.any(train):
            row["status"] = "insufficient_double_holdout_rows"
            predictions[test] = float(targets[train].mean()) if np.any(train) else float(targets.mean())
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(features[train], targets[train])
            predictions[test] = model.predict(features[test])
            row["status"] = "fitted"
        audit.append(row)
    if not np.isfinite(predictions).all():
        raise RuntimeError("double-holdout ridge prediction left unscored rows")
    return predictions, audit


def seed_bootstrap_interval(
    values_by_seed: Mapping[int, float], *, samples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    keys = np.asarray(sorted(values_by_seed), dtype=int)
    values = np.asarray([values_by_seed[int(key)] for key in keys], dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(int(samples))]
    )
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

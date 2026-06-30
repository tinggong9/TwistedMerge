#!/usr/bin/env python
"""Fixed-setting repeated-seed verification for small model merging.

This experiment is deliberately stricter than the earlier smoke-scale
model-merging benchmark.  It keeps dataset, architecture, model count, width,
domain shift, and matching protocol separated, then asks whether observed
cycle/cocycle residuals predict ordinary weight-average merge degradation.
Injected permutation noise is recorded as a control diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    DomainShiftDataset,
    average_models,
    collect_features,
    compose_perm,
    compute_layerwise_pairwise_permutations,
    cycle_score,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    format_markdown_table,
    greedy_soup,
    inject_pairwise_permutation_noise,
    load_dataset,
    make_loader,
    make_model,
    model_layer_widths,
    permutation_disagreement,
    permute_model_to_reference,
    primary_alignment_layer,
    primary_pairwise_permutations,
    rank_lifted_branch_models,
    require_torch,
    save_checkpoint,
    set_seed,
    synchronize_layerwise_permutations,
    synchronize_permutations,
    train_model,
)
from src.monomial_gauge_alignment import (  # noqa: E402
    apply_monomial_alignment_to_reference,
    average_monomial_defect_score,
    compare_function_before_after_alignment,
    estimate_pairwise_monomial_alignments,
    monomial_scaling_statistics,
)
from src.rank_lift_baselines import (  # noqa: E402
    c2m3_cluster_branch_ensemble,
    method_capacity_metadata,
    random_branch_ensemble,
    validation_branch_ensemble,
)


RUNS_CSV = "fixed_setting_verification_runs.csv"
STATS_CSV = "fixed_setting_verification_stats.csv"
TRIANGLES_CSV = "fixed_setting_triangle_defects.csv"
INDIVIDUALS_CSV = "fixed_setting_individual_models.csv"
REAL_OBSTRUCTION_RUNS_CSV = "real_obstruction_degradation.csv"
REAL_OBSTRUCTION_SUMMARY_CSV = "real_obstruction_summary.csv"
REAL_OBSTRUCTION_TRIANGLES_CSV = "real_obstruction_triangle_defects.csv"
REAL_OBSTRUCTION_INDIVIDUALS_CSV = "real_obstruction_individual_models.csv"
REAL_OBSTRUCTION_PAIRED_DELTAS_CSV = "real_obstruction_paired_deltas.csv"
REAL_OBSTRUCTION_REGRESSIONS_CSV = "real_obstruction_predictor_regressions.csv"
MONOMIAL_RUNS_CSV = "monomial_fixed_setting_runs.csv"
MONOMIAL_TRIANGLES_CSV = "monomial_triangle_defects.csv"
BRANCH_CAPACITY_BASELINES = (
    "random_branch_ensemble",
    "validation_branch_ensemble",
    "c2m3_cluster_branch_ensemble",
)
MONOMIAL_MATCHINGS = {"monomial_activation", "monomial_weight"}
PREDICTION_TARGETS = (
    "weight_average_degradation_vs_best_single",
    "git_rebasin_degradation_vs_best_single",
    "c2m3_degradation_vs_best_single",
    "c2m3_delta_vs_git_rebasin",
    "c2m3_delta_vs_weight_average",
    "rank_lift_delta_vs_weight_average",
    "rank_lift_delta_vs_c2m3",
    "greedy_soup_delta_vs_weight_average",
)
REGRESSION_PREDICTORS = (
    "mean_cycle_score",
    "combined_obstruction_score",
    "sync_disagreement",
)


def parse_csv(text: str, cast=str) -> list:
    if text is None:
        return []
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_float_csv(text: str) -> list[float]:
    return [float(item) for item in parse_csv(text, str)]


def parse_seeds(text: str) -> list[int]:
    text = str(text).strip()
    if not text:
        return []
    if "," in text:
        return parse_csv(text, int)
    if ":" in text:
        start_text, end_text = text.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        step = 1 if end >= start else -1
        return list(range(start, end + step, step))
    return [int(text)]


def split_indices(n_items: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    torch, _, _ = require_torch()
    n_val = max(1, int(round(n_items * val_fraction)))
    n_train = max(1, n_items - n_val)
    if n_train + n_val > n_items:
        n_val = n_items - n_train
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(n_items, generator=generator).tolist()
    return indices[:n_train], indices[n_train : n_train + n_val]


def fixed_setting_id(dataset: str, architecture: str, n_models: int, width: int, domain_shift: str, matching: str) -> str:
    return f"{dataset}_{architecture}_N{n_models}_W{width}_{domain_shift}_{matching}"


def is_monomial_matching(matching: str) -> bool:
    return str(matching).strip().lower() in MONOMIAL_MATCHINGS


def permutation_matching_for(matching: str) -> str:
    name = str(matching).strip().lower()
    if name == "monomial_activation":
        return "activation"
    if name == "monomial_weight":
        return "weight"
    return name


def run_id_for(setting_id: str, seed: int) -> str:
    return f"{setting_id}_seed{seed}"


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_mean(values) -> float:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def safe_min(values) -> float:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if len(arr) else float("nan")


def safe_max(values) -> float:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if len(arr) else float("nan")


def safe_std(values) -> float:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=1)) if len(arr) > 1 else 0.0 if len(arr) == 1 else float("nan")


def safe_pearson(x_values, y_values) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x_values, y_values) -> float:
    x = pd.Series(np.asarray(x_values, dtype=float)).rank(method="average").to_numpy()
    y = pd.Series(np.asarray(y_values, dtype=float)).rank(method="average").to_numpy()
    return safe_pearson(x, y)


def bootstrap_corr_ci(x_values, y_values, corr_fn, n_boot: int, seed: int) -> tuple[float, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        value = corr_fn(x[idx], y[idx])
        if math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def bootstrap_mean_ci(values, n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1 or n_boot <= 0:
        mean = float(arr.mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        estimates.append(float(arr[idx].mean()))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def residualize(target: np.ndarray, controls: np.ndarray) -> np.ndarray:
    x = np.asarray(controls, dtype=float)
    y = np.asarray(target, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def partial_correlation(x_values, y_values, control_columns: list[np.ndarray]) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    controls = np.column_stack([np.asarray(col, dtype=float) for col in control_columns])
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    x = x[mask]
    y = y[mask]
    controls = controls[mask]
    if len(x) <= controls.shape[1] + 2:
        return float("nan")
    return safe_pearson(residualize(x, controls), residualize(y, controls))


def regression_cycle_beta(x_values, y_values, control_columns: list[np.ndarray]) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    controls = np.column_stack([np.asarray(col, dtype=float) for col in control_columns])
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    x = x[mask]
    y = y[mask]
    controls = controls[mask]
    if len(x) <= controls.shape[1] + 2 or np.std(x) <= 1e-12:
        return float("nan")
    design = np.column_stack([np.ones(len(x)), x, controls])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1])


def regression_predictor_beta(predictor_values, outcome_values, control_columns: list[np.ndarray]) -> float:
    x = np.asarray(predictor_values, dtype=float)
    y = np.asarray(outcome_values, dtype=float)
    controls = np.column_stack([np.asarray(col, dtype=float) for col in control_columns])
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    x = x[mask]
    y = y[mask]
    controls = controls[mask]
    if len(x) <= controls.shape[1] + 2 or np.std(x) <= 1e-12:
        return float("nan")
    design = np.column_stack([np.ones(len(x)), x, controls])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1])


def bootstrap_regression_beta_ci(
    predictor_values,
    outcome_values,
    control_columns: list[np.ndarray],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    x = np.asarray(predictor_values, dtype=float)
    y = np.asarray(outcome_values, dtype=float)
    controls = np.column_stack([np.asarray(col, dtype=float) for col in control_columns])
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    x = x[mask]
    y = y[mask]
    controls = controls[mask]
    if len(x) <= controls.shape[1] + 2 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        value = regression_predictor_beta(x[idx], y[idx], [controls[idx, col] for col in range(controls.shape[1])])
        if math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize_seed_list(seeds: list[int]) -> str:
    if not seeds:
        return ""
    if len(seeds) > 4 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}:{seeds[-1]}"
    return ",".join(str(seed) for seed in seeds)


def permutation_json(pairwise_perms: dict[tuple[int, int], np.ndarray]) -> str:
    payload = {f"{i}->{j}": pairwise_perms[(i, j)].astype(int).tolist() for i, j in sorted(pairwise_perms)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def layerwise_permutation_json(pairwise_by_layer: dict[str, dict[tuple[int, int], np.ndarray]]) -> str:
    payload = {
        layer: {f"{i}->{j}": pairwise_by_layer[layer][(i, j)].astype(int).tolist() for i, j in sorted(pairwise_by_layer[layer])}
        for layer in sorted(pairwise_by_layer)
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def layer_reference_perms(pairwise_by_layer: dict[str, dict[tuple[int, int], np.ndarray]], ref: int, idx: int) -> dict[str, np.ndarray]:
    return {layer: pairwise[(ref, idx)] for layer, pairwise in pairwise_by_layer.items()}


def synced_layer_perms(synced_by_layer: dict[str, dict[int, np.ndarray]], idx: int) -> dict[str, np.ndarray]:
    return {layer: synced[idx] for layer, synced in synced_by_layer.items()}


def permutation_arg_for_architecture(architecture: str, layer_perms: dict[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
    if architecture in {"mlp", "cnn"}:
        return layer_perms[primary_alignment_layer(architecture)]
    return layer_perms


def synchronize_alignment_bundle(
    pairwise_by_layer: dict[str, dict[tuple[int, int], np.ndarray]],
    n_models: int,
) -> tuple[str, dict[str, dict[int, np.ndarray]], float]:
    if len(pairwise_by_layer) == 1:
        layer, pairwise = next(iter(pairwise_by_layer.items()))
        ref, synced, residual = synchronize_permutations(pairwise, n_models)
        return f"{layer}:{ref}", {layer: synced}, residual
    return synchronize_layerwise_permutations(pairwise_by_layer, n_models)


def inject_layerwise_permutation_noise(
    pairwise_by_layer: dict[str, dict[tuple[int, int], np.ndarray]],
    n_models: int,
    widths_by_layer: dict[str, int],
    swap_fraction: float,
    seed: int,
) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    return {
        layer: inject_pairwise_permutation_noise(
            pairwise,
            n_models,
            int(widths_by_layer[layer]),
            swap_fraction,
            seed + 104729 * layer_idx,
        )
        for layer_idx, (layer, pairwise) in enumerate(pairwise_by_layer.items())
    }


def triangle_predictor_summary(pairwise_perms: dict[tuple[int, int], np.ndarray], n_models: int, width: int) -> dict:
    _score, rows = cycle_score(pairwise_perms, n_models, width)
    cycle_values = [float(row["cycle_defect"]) for row in rows]
    identity = np.arange(width)
    nonidentity = []
    defect_rates = []
    for i, j, k in combinations(range(n_models), 3):
        triangle_perm = compose_perm(compose_perm(pairwise_perms[(i, j)], pairwise_perms[(j, k)]), pairwise_perms[(k, i)])
        defect_rate = permutation_disagreement(triangle_perm, identity)
        defect_rates.append(defect_rate)
        nonidentity.append(float(defect_rate > 0.0))
    return {
        "mean_cycle_score": safe_mean(cycle_values),
        "max_cycle_score": max(cycle_values, default=float("nan")),
        "median_cycle_score": float(np.median(cycle_values)) if cycle_values else float("nan"),
        "nonidentity_triangle_fraction": safe_mean(nonidentity),
        "mean_triangle_defect_rate": safe_mean(defect_rates),
        "max_triangle_defect_rate": max(defect_rates, default=float("nan")),
    }


def triangle_rows(
    base_row: dict,
    pairwise_perms: dict[tuple[int, int], np.ndarray],
    n_models: int,
    width: int,
    source: str,
    noise_fraction: float,
) -> tuple[float, list[dict]]:
    score, rows = cycle_score(pairwise_perms, n_models, width)
    out = []
    identity = np.arange(width)
    score_lookup = {(int(row["i"]), int(row["j"]), int(row["k"])): float(row["cycle_defect"]) for row in rows}
    for i, j, k in combinations(range(n_models), 3):
        p_ij = pairwise_perms[(i, j)]
        p_jk = pairwise_perms[(j, k)]
        p_ki = pairwise_perms[(k, i)]
        triangle_perm = compose_perm(compose_perm(p_ij, p_jk), p_ki)
        defect_rate = permutation_disagreement(triangle_perm, identity)
        out.append(
            {
                **base_row,
                "alignment_source": source,
                "alignment_noise_fraction": noise_fraction,
                "triangle_type": "permutation",
                "triangle": f"{i}-{j}-{k}",
                "i": i,
                "j": j,
                "k": k,
                "p_ij": json.dumps(p_ij.astype(int).tolist(), separators=(",", ":")),
                "p_jk": json.dumps(p_jk.astype(int).tolist(), separators=(",", ":")),
                "p_ki": json.dumps(p_ki.astype(int).tolist(), separators=(",", ":")),
                "triangle_perm": json.dumps(triangle_perm.astype(int).tolist(), separators=(",", ":")),
                "triangle_defect_count": int(np.sum(triangle_perm != identity)),
                "triangle_defect_rate": float(defect_rate),
                "cycle_defect": score_lookup[(i, j, k)],
                "cycle_score": score,
            }
        )
    return score, out


def monomial_triangle_rows(
    base_row: dict,
    monomial_alignments: dict,
    n_models: int,
    alignment_source: str,
    noise_fraction: float,
) -> tuple[float, list[dict]]:
    score, rows = average_monomial_defect_score(monomial_alignments, n_models)
    out = []
    for row in rows:
        out.append(
            {
                **base_row,
                "alignment_source": alignment_source,
                "alignment_noise_fraction": noise_fraction,
                "is_injected_alignment_control": alignment_source != "observed",
                "triangle_type": "monomial",
                "i": int(row["i"]),
                "j": int(row["j"]),
                "k": int(row["k"]),
                "monomial_defect_score": row["monomial_defect_score"],
                "monomial_cycle_trace": row["monomial_cycle_trace"],
                "monomial_cycle_determinant": row["monomial_cycle_determinant"],
                "monomial_average_defect_score": score,
            }
        )
    return score, out


def _center_normalize_features(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=0, keepdims=True), 1e-12)


def activation_assignment_similarity(features_i: np.ndarray, features_j: np.ndarray, perm: np.ndarray) -> dict:
    xi = _center_normalize_features(features_i)
    xj = _center_normalize_features(features_j)
    similarity = xi.T @ xj
    assigned = similarity[np.arange(len(perm)), perm]
    return {
        "assigned_similarity_mean": safe_mean(assigned),
        "assigned_similarity_min": min([safe_float(value) for value in assigned], default=float("nan")),
    }


def pairwise_alignment_residuals(models: list, pairwise_perms: dict[tuple[int, int], np.ndarray], loader, device, max_batches: int) -> dict:
    features = [collect_features(model, loader, device, max_batches=max_batches) for model in models]
    rows = []
    for i, j in combinations(range(len(models)), 2):
        perm = pairwise_perms[(i, j)]
        fi = features[i] - features[i].mean(axis=0, keepdims=True)
        fj = features[j][:, perm] - features[j][:, perm].mean(axis=0, keepdims=True)
        denom = float(np.linalg.norm(fi) + np.linalg.norm(fj) + 1e-12)
        residual = float(np.linalg.norm(fi - fj) / denom)
        sim = activation_assignment_similarity(features[i], features[j], perm)
        rows.append({"pair": f"{i}-{j}", "residual": residual, **sim})
    return {
        "pairwise_alignment_residual_mean": safe_mean([row["residual"] for row in rows]),
        "pairwise_alignment_residual_max": max([row["residual"] for row in rows], default=float("nan")),
        "activation_assignment_similarity_mean": safe_mean([row["assigned_similarity_mean"] for row in rows]),
        "activation_assignment_similarity_min": safe_min([row["assigned_similarity_min"] for row in rows]),
        "pairwise_alignment_residual_json": json.dumps(rows, sort_keys=True, separators=(",", ":")),
    }


def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    value = safe_float(value)
    if not math.isfinite(value):
        return float("nan")
    return float(min(high, max(low, value)))


def obstruction_predictor_columns(
    pairwise_perms: dict[tuple[int, int], np.ndarray],
    n_models: int,
    width: int,
    residuals: dict,
    sync_disagreement: float,
) -> dict:
    predictors = triangle_predictor_summary(pairwise_perms, n_models, width)
    residual_mean = bounded(residuals.get("pairwise_alignment_residual_mean", float("nan")))
    similarity_mean = safe_float(residuals.get("activation_assignment_similarity_mean", float("nan")))
    similarity_confidence = bounded((similarity_mean + 1.0) / 2.0)
    residual_confidence = bounded(1.0 - residual_mean)
    alignment_confidence = (
        float(similarity_confidence * residual_confidence)
        if math.isfinite(similarity_confidence) and math.isfinite(residual_confidence)
        else float("nan")
    )
    obstruction_terms = [
        bounded(predictors["mean_cycle_score"]),
        bounded(predictors["max_cycle_score"]),
        bounded(predictors["median_cycle_score"]),
        bounded(predictors["nonidentity_triangle_fraction"]),
        bounded(sync_disagreement),
        residual_mean,
        bounded(1.0 - alignment_confidence),
    ]
    predictors.update(
        {
            "alignment_confidence_score": alignment_confidence,
            "combined_obstruction_score": safe_mean(obstruction_terms),
        }
    )
    return predictors


def baseline_record(
    *,
    method: str,
    val_metrics: dict,
    test_metrics: dict,
    base: dict,
    selection_val_accuracy: float = float("nan"),
    selection_indices: list[int] | None = None,
    is_single_model: bool,
    exact_relu_symmetry: bool,
    is_soup: bool,
    is_ensemble_or_extra_capacity: bool,
    capacity_matched: bool,
    parameter_multiplier: float,
    inference_multiplier: float,
    uses_validation_data: bool,
    method_note: str,
    capacity_metadata: dict | None = None,
) -> dict:
    if capacity_metadata is None:
        branch_count = 1 if is_single_model else max(1, int(round(inference_multiplier)))
        capacity_metadata = {
            "method_note": method_note,
            "is_single_model": bool(is_single_model),
            "branch_count": int(branch_count),
            "parameter_count": float("nan"),
            "parameter_multiplier": float(parameter_multiplier),
            "inference_multiplier": float(inference_multiplier),
            "capacity_matched_to_weight_average": bool(capacity_matched),
            "capacity_matched_to_rank_lift": False,
            "uses_obstruction_residual": False,
            "uses_validation_data": bool(uses_validation_data),
            "uses_distillation": False,
        }
    else:
        is_single_model = bool(capacity_metadata["is_single_model"])
        capacity_matched = bool(capacity_metadata["capacity_matched_to_weight_average"])
        parameter_multiplier = float(capacity_metadata["parameter_multiplier"])
        inference_multiplier = float(capacity_metadata["inference_multiplier"])
        uses_validation_data = bool(capacity_metadata["uses_validation_data"])
        method_note = str(capacity_metadata["method_note"])
    return {
        **base,
        "method": method,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_loss": float(val_metrics["loss"]),
        "validation_accuracy_used_for_selection": selection_val_accuracy,
        "selection_indices": json.dumps(selection_indices or [], separators=(",", ":")),
        "uses_validation_data": bool(uses_validation_data),
        "is_single_model": bool(is_single_model),
        "exact_relu_symmetry": bool(exact_relu_symmetry),
        "is_soup": bool(is_soup),
        "is_ensemble_or_extra_capacity": bool(is_ensemble_or_extra_capacity),
        "capacity_matched_to_weight_average": bool(capacity_matched),
        "capacity_matched_to_rank_lift": bool(capacity_metadata["capacity_matched_to_rank_lift"]),
        "branch_count": int(capacity_metadata["branch_count"]),
        "parameter_count": capacity_metadata["parameter_count"],
        "parameter_multiplier": float(parameter_multiplier),
        "inference_multiplier": float(inference_multiplier),
        "parameter_count_multiplier": float(parameter_multiplier),
        "inference_time_multiplier": float(inference_multiplier),
        "uses_obstruction_residual": bool(capacity_metadata["uses_obstruction_residual"]),
        "uses_distillation": bool(capacity_metadata["uses_distillation"]),
        "method_note": method_note,
    }


def evaluate_methods(
    args,
    *,
    models: list,
    architecture: str,
    spec,
    width: int,
    pairwise_perms: dict[tuple[int, int], np.ndarray],
    pairwise_by_layer: dict[str, dict[tuple[int, int], np.ndarray]],
    monomial_alignments: dict | None = None,
    val_loader,
    test_loader,
    match_loader=None,
    device,
    base: dict,
) -> list[dict]:
    rows: list[dict] = []
    base_model = models[0]

    weight_model = average_models(models, architecture, spec, width)
    rows.append(
        baseline_record(
            method="weight_average",
            val_metrics=evaluate_model(weight_model, val_loader, device),
            test_metrics=evaluate_model(weight_model, test_loader, device),
            base=base,
            is_single_model=True,
            exact_relu_symmetry=False,
            is_soup=False,
            is_ensemble_or_extra_capacity=False,
            capacity_matched=True,
            parameter_multiplier=1.0,
            inference_multiplier=1.0,
            uses_validation_data=False,
            method_note="ordinary parameter average without alignment",
            capacity_metadata=method_capacity_metadata("weight_average", weight_model, base_model),
        )
    )

    soup_model, soup_indices, soup_test = greedy_soup(models, val_loader, test_loader, device, architecture, spec, width)
    rows.append(
        baseline_record(
            method="greedy_soup",
            val_metrics=evaluate_model(soup_model, val_loader, device),
            test_metrics=soup_test,
            base=base,
            selection_val_accuracy=evaluate_model(soup_model, val_loader, device)["accuracy"],
            selection_indices=soup_indices,
            is_single_model=True,
            exact_relu_symmetry=False,
            is_soup=True,
            is_ensemble_or_extra_capacity=False,
            capacity_matched=True,
            parameter_multiplier=1.0,
            inference_multiplier=1.0,
            uses_validation_data=True,
            method_note="faithful greedy Model Soups-style validation-selected soup",
            capacity_metadata=method_capacity_metadata("greedy_soup", soup_model, base_model),
        )
    )

    pairwise_aligned = []
    for idx in range(len(models)):
        if idx == 0:
            pairwise_aligned.append(models[0])
        else:
            pairwise_aligned.append(
                permute_model_to_reference(
                    models[idx],
                    architecture,
                    spec,
                    width,
                    permutation_arg_for_architecture(
                        architecture,
                        layer_reference_perms(pairwise_by_layer, 0, idx),
                    ),
                )
            )
    pairwise_merged = average_models(pairwise_aligned, architecture, spec, width)
    rows.append(
        baseline_record(
            method="git_rebasin_pairwise_ref0",
            val_metrics=evaluate_model(pairwise_merged, val_loader, device),
            test_metrics=evaluate_model(pairwise_merged, test_loader, device),
            base=base,
            is_single_model=True,
            exact_relu_symmetry=True,
            is_soup=False,
            is_ensemble_or_extra_capacity=False,
            capacity_matched=True,
            parameter_multiplier=1.0,
            inference_multiplier=1.0,
            uses_validation_data=False,
            method_note="faithful Git-ReBasin-style pairwise hidden-unit alignment to model 0",
            capacity_metadata=method_capacity_metadata("git_rebasin_pairwise_ref0", pairwise_merged, base_model),
        )
    )

    if architecture == "mlp" and monomial_alignments is not None:
        monomial_aligned = [models[0]]
        preservation_rows = []
        for idx in range(1, len(models)):
            aligned = apply_monomial_alignment_to_reference(models[idx], spec, width, monomial_alignments[(0, idx)])
            if match_loader is not None:
                preservation_rows.append(
                    compare_function_before_after_alignment(
                        models[idx],
                        aligned,
                        match_loader,
                        device,
                        max_batches=args.feature_batches,
                    )
                )
            monomial_aligned.append(aligned)
        preservation_summary = {
            "functional_preservation_error": safe_max(
                [row["functional_preservation_error"] for row in preservation_rows]
            )
            if preservation_rows
            else float("nan"),
            "functional_preservation_mean_abs_error": safe_mean(
                [row["functional_preservation_mean_abs_error"] for row in preservation_rows]
            )
            if preservation_rows
            else float("nan"),
            "functional_preservation_prediction_disagreement": safe_mean(
                [row["functional_preservation_prediction_disagreement"] for row in preservation_rows]
            )
            if preservation_rows
            else float("nan"),
        }
        monomial_model = average_models(monomial_aligned, architecture, spec, width)
        monomial_base = {
            **base,
            **monomial_scaling_statistics(monomial_alignments),
            **preservation_summary,
        }
        rows.append(
            baseline_record(
                method="monomial_gauge_ref0",
                val_metrics=evaluate_model(monomial_model, val_loader, device),
                test_metrics=evaluate_model(monomial_model, test_loader, device),
                base=monomial_base,
                is_single_model=True,
                exact_relu_symmetry=True,
                is_soup=False,
                is_ensemble_or_extra_capacity=False,
                capacity_matched=True,
                parameter_multiplier=1.0,
                inference_multiplier=1.0,
                uses_validation_data=False,
                method_note="ReLU-compatible monomial alignment to model 0 before same-capacity averaging",
                capacity_metadata=method_capacity_metadata("monomial_gauge_ref0", monomial_model, base_model),
            )
        )

    sync_ref, synced_perms_by_layer, sync_disagreement = synchronize_alignment_bundle(pairwise_by_layer, len(models))
    c2m3_aligned = [
        permute_model_to_reference(
            models[idx],
            architecture,
            spec,
            width,
            permutation_arg_for_architecture(architecture, synced_layer_perms(synced_perms_by_layer, idx)),
        )
        for idx in range(len(models))
    ]
    c2m3_model = average_models(c2m3_aligned, architecture, spec, width)
    c2m3_base = {**base, "sync_reference_model": sync_ref, "sync_disagreement": sync_disagreement}
    rows.append(
        baseline_record(
            method="c2m3_synchronized",
            val_metrics=evaluate_model(c2m3_model, val_loader, device),
            test_metrics=evaluate_model(c2m3_model, test_loader, device),
            base=c2m3_base,
            is_single_model=True,
            exact_relu_symmetry=True,
            is_soup=False,
            is_ensemble_or_extra_capacity=False,
            capacity_matched=True,
            parameter_multiplier=1.0,
            inference_multiplier=1.0,
            uses_validation_data=False,
            method_note="internal C2M3-style global permutation synchronization before averaging",
            capacity_metadata=method_capacity_metadata("c2m3_synchronized", c2m3_model, base_model),
        )
    )

    ensemble_metrics = evaluate_ensemble(models, test_loader, device)
    rows.append(
        baseline_record(
            method="ensemble_upper_bound",
            val_metrics=evaluate_ensemble(models, val_loader, device),
            test_metrics=ensemble_metrics,
            base=base,
            is_single_model=False,
            exact_relu_symmetry=False,
            is_soup=False,
            is_ensemble_or_extra_capacity=True,
            capacity_matched=False,
            parameter_multiplier=float(len(models)),
            inference_multiplier=float(len(models)),
            uses_validation_data=False,
            method_note="extra-capacity ensemble upper bound over all local models",
            capacity_metadata=method_capacity_metadata("ensemble_upper_bound", models, base_model),
        )
    )

    branches = rank_lifted_branch_models(
        c2m3_aligned,
        pairwise_perms,
        args.rank_lift_branches,
        architecture,
        spec,
        width,
    )
    branch_count = max(1, len(branches))
    rows.append(
        baseline_record(
            method=f"twisted_rank_lift_{branch_count}",
            val_metrics=evaluate_ensemble(branches, val_loader, device),
            test_metrics=evaluate_ensemble(branches, test_loader, device),
            base=base,
            is_single_model=False,
            exact_relu_symmetry=True,
            is_soup=False,
            is_ensemble_or_extra_capacity=True,
            capacity_matched=False,
            parameter_multiplier=float(branch_count),
            inference_multiplier=float(branch_count),
            uses_validation_data=False,
            method_note="rank-lift branch ensemble; extra capacity, not a single merged model",
            capacity_metadata=method_capacity_metadata(f"twisted_rank_lift_{branch_count}", branches, base_model),
        )
    )

    random_branches = random_branch_ensemble(
        c2m3_aligned,
        branch_count,
        architecture,
        spec,
        width,
        seed=int(base["seed"]) + 7919 + 97 * branch_count,
    )
    random_branch_count = max(1, len(random_branches))
    rows.append(
        baseline_record(
            method=f"random_branch_ensemble_{random_branch_count}",
            val_metrics=evaluate_ensemble(random_branches, val_loader, device),
            test_metrics=evaluate_ensemble(random_branches, test_loader, device),
            base=base,
            is_single_model=False,
            exact_relu_symmetry=True,
            is_soup=False,
            is_ensemble_or_extra_capacity=True,
            capacity_matched=False,
            parameter_multiplier=float(random_branch_count),
            inference_multiplier=float(random_branch_count),
            uses_validation_data=False,
            method_note="random branch ensemble matched to rank-lift branch count; non-obstruction control",
            capacity_metadata=method_capacity_metadata(
                f"random_branch_ensemble_{random_branch_count}",
                random_branches,
                base_model,
            ),
        )
    )

    validation_branches = validation_branch_ensemble(
        models,
        val_loader,
        test_loader,
        branch_count,
        architecture,
        spec,
        width,
        device,
    )
    validation_branch_count = max(1, len(validation_branches))
    rows.append(
        baseline_record(
            method=f"validation_branch_ensemble_{validation_branch_count}",
            val_metrics=evaluate_ensemble(validation_branches, val_loader, device),
            test_metrics=evaluate_ensemble(validation_branches, test_loader, device),
            base=base,
            is_single_model=False,
            exact_relu_symmetry=False,
            is_soup=False,
            is_ensemble_or_extra_capacity=True,
            capacity_matched=False,
            parameter_multiplier=float(validation_branch_count),
            inference_multiplier=float(validation_branch_count),
            uses_validation_data=True,
            method_note="validation-selected branch ensemble matched to rank-lift branch count; non-obstruction control",
            capacity_metadata=method_capacity_metadata(
                f"validation_branch_ensemble_{validation_branch_count}",
                validation_branches,
                base_model,
            ),
        )
    )

    c2m3_cluster_branches = c2m3_cluster_branch_ensemble(
        c2m3_aligned,
        pairwise_perms,
        branch_count,
        architecture,
        spec,
        width,
    )
    c2m3_cluster_branch_count = max(1, len(c2m3_cluster_branches))
    rows.append(
        baseline_record(
            method=f"c2m3_cluster_branch_ensemble_{c2m3_cluster_branch_count}",
            val_metrics=evaluate_ensemble(c2m3_cluster_branches, val_loader, device),
            test_metrics=evaluate_ensemble(c2m3_cluster_branches, test_loader, device),
            base=base,
            is_single_model=False,
            exact_relu_symmetry=True,
            is_soup=False,
            is_ensemble_or_extra_capacity=True,
            capacity_matched=False,
            parameter_multiplier=float(c2m3_cluster_branch_count),
            inference_multiplier=float(c2m3_cluster_branch_count),
            uses_validation_data=False,
            method_note="C2M3-distance branch ensemble matched to rank-lift branch count; no obstruction residual used",
            capacity_metadata=method_capacity_metadata(
                f"c2m3_cluster_branch_ensemble_{c2m3_cluster_branch_count}",
                c2m3_cluster_branches,
                base_model,
            ),
        )
    )
    return rows


def add_paired_deltas(rows: list[dict], single_best_accuracy: float, mean_individual_accuracy: float) -> None:
    lookup = {row["method"]: row for row in rows}
    weight = lookup.get("weight_average", {})
    greedy = lookup.get("greedy_soup", {})
    git_rebasin = lookup.get("git_rebasin_pairwise_ref0", {})
    c2m3 = lookup.get("c2m3_synchronized", {})
    rank_lift = next((row for row in rows if str(row["method"]).startswith("twisted_rank_lift_")), {})
    weight_acc = float(weight.get("test_accuracy", float("nan")))
    greedy_acc = float(greedy.get("test_accuracy", float("nan")))
    git_acc = float(git_rebasin.get("test_accuracy", float("nan")))
    c2m3_acc = float(c2m3.get("test_accuracy", float("nan")))
    rank_lift_acc = float(rank_lift.get("test_accuracy", float("nan")))
    target_values = {
        "weight_average_degradation_vs_best_single": single_best_accuracy - weight_acc,
        "git_rebasin_degradation_vs_best_single": single_best_accuracy - git_acc,
        "c2m3_degradation_vs_best_single": single_best_accuracy - c2m3_acc,
        "c2m3_delta_vs_git_rebasin": c2m3_acc - git_acc,
        "c2m3_delta_vs_weight_average": c2m3_acc - weight_acc,
        "rank_lift_delta_vs_weight_average": rank_lift_acc - weight_acc,
        "rank_lift_delta_vs_c2m3": rank_lift_acc - c2m3_acc,
        "greedy_soup_delta_vs_weight_average": greedy_acc - weight_acc,
    }
    for row in rows:
        row["single_best_merge_degradation"] = single_best_accuracy - float(row["test_accuracy"])
        row["mean_individual_merge_degradation"] = mean_individual_accuracy - float(row["test_accuracy"])
        row["weight_merge_degradation"] = (
            single_best_accuracy - weight_acc
            if row["method"] == "weight_average"
            else float("nan")
        )
        row["delta_vs_weight_average"] = float(row["test_accuracy"]) - weight_acc
        row["delta_vs_greedy_soup"] = float(row["test_accuracy"]) - greedy_acc
        row["delta_vs_c2m3_synchronized"] = float(row["test_accuracy"]) - c2m3_acc
        row.update(target_values)


def run_one_seed(args, dataset_name: str, architecture: str, n_models: int, width: int, domain_shift: str, matching: str, seed: int):
    torch, _, _ = require_torch()
    device = device_from_arg(args.device)
    if n_models < 3:
        raise ValueError("fixed-setting verification requires N>=3 because N=2 has no triangle obstruction")

    setting_id = fixed_setting_id(dataset_name, architecture, n_models, width, domain_shift, matching)
    run_id = run_id_for(setting_id, seed)
    spec, train_base, test_base = load_dataset(
        dataset_name,
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
        augmentation=args.augmentation,
    )
    train_indices, val_indices = split_indices(len(train_base), args.val_fraction, args.dataset_seed + 17)
    val_subset = torch.utils.data.Subset(train_base, val_indices)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=args.dataset_seed + 100)
    test_loader = make_loader(test_base, args.batch_size, shuffle=False, seed=args.dataset_seed + 200)
    match_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=args.dataset_seed + 300)

    models = []
    individual_rows = []
    for model_idx in range(n_models):
        local_seed = seed + 1009 * model_idx + 37 * width + 101 * n_models
        set_seed(local_seed)
        shifted_train = DomainShiftDataset(train_base, domain_shift, model_idx, n_models)
        train_subset = torch.utils.data.Subset(shifted_train, train_indices)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=local_seed + 1)
        model = make_model(architecture, spec, width)
        train_model(
            model,
            train_loader,
            args.epochs,
            args.lr,
            device,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            step_size=args.step_size,
            gamma=args.gamma,
        )
        val_metrics = evaluate_model(model, val_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)
        model.to("cpu")
        checkpoint_path = args.reports_dir / "checkpoints" / "fixed_setting_verification" / setting_id / f"seed{seed}_model{model_idx}.pt"
        checkpoint_metadata = {
            "dataset": dataset_name,
            "architecture": architecture,
            "n_models": n_models,
            "width": width,
            "domain_shift": domain_shift,
            "matching": matching,
            "experiment_seed": seed,
            "local_seed": local_seed,
            "model_index": model_idx,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "augmentation": args.augmentation,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "dataset_seed": args.dataset_seed,
            "train_split_seed": args.dataset_seed + 17,
            "checkpoint_saved": bool(args.save_checkpoints),
        }
        if args.save_checkpoints:
            save_checkpoint(model, checkpoint_path, checkpoint_metadata)
        models.append(model)
        individual_rows.append(
            {
                "setting_id": setting_id,
                "run_id": run_id,
                "dataset": dataset_name,
                "architecture": architecture,
                "n_models": n_models,
                "width": width,
                "domain_shift": domain_shift,
                "matching": matching,
                "seed": seed,
                "model_index": model_idx,
                "local_seed": local_seed,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "checkpoint_saved": bool(args.save_checkpoints),
                "checkpoint_path": str(checkpoint_path) if args.save_checkpoints else "",
                "checkpoint_metadata_json": json.dumps(checkpoint_metadata, sort_keys=True, separators=(",", ":")),
            }
        )

    mean_individual_accuracy = safe_mean([row["test_accuracy"] for row in individual_rows])
    single_best_accuracy = max(row["test_accuracy"] for row in individual_rows)
    single_worst_accuracy = min(row["test_accuracy"] for row in individual_rows)

    if is_monomial_matching(matching) and architecture != "mlp":
        raise ValueError("monomial gauge alignment is currently implemented only for one-hidden-layer mlp")
    if is_monomial_matching(matching):
        monomial_alignments = estimate_pairwise_monomial_alignments(
            models,
            match_loader,
            device,
            matching=matching,
            max_batches=args.feature_batches,
        )
        pairwise_by_layer = {
            primary_alignment_layer(architecture): {
                pair: alignment.permutation for pair, alignment in monomial_alignments.items()
            }
        }
    else:
        monomial_alignments = None
        pairwise_by_layer = compute_layerwise_pairwise_permutations(
            models,
            architecture,
            match_loader,
            device,
            permutation_matching_for(matching),
        )
    pairwise = primary_pairwise_permutations(pairwise_by_layer, architecture)
    primary_layer = primary_alignment_layer(architecture)
    widths_by_layer = model_layer_widths(models[0], architecture)
    primary_width = int(widths_by_layer[primary_layer])
    residuals = pairwise_alignment_residuals(models, pairwise, match_loader, device, args.feature_batches)
    observed_sync_ref, _observed_synced, observed_sync_disagreement = synchronize_alignment_bundle(pairwise_by_layer, n_models)
    shared_base = {
        "setting_id": setting_id,
        "run_id": run_id,
        "dataset": dataset_name,
        "architecture": architecture,
        "n_models": n_models,
        "width": width,
        "domain_shift": domain_shift,
        "matching": matching,
        "seed": seed,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "augmentation": args.augmentation,
        "dataset_seed": args.dataset_seed,
        "val_fraction": args.val_fraction,
        "mean_individual_accuracy": mean_individual_accuracy,
        "single_best_accuracy": single_best_accuracy,
        "single_worst_accuracy": single_worst_accuracy,
        "individual_accuracy_spread": single_best_accuracy - single_worst_accuracy,
        "primary_alignment_layer": primary_layer,
        "pairwise_alignment_permutations_json": permutation_json(pairwise),
        "layerwise_alignment_permutations_json": layerwise_permutation_json(pairwise_by_layer),
        **residuals,
    }

    run_rows = []
    triangle_out = []
    observed_cycle_score, observed_triangles = triangle_rows(shared_base, pairwise, n_models, primary_width, "observed", 0.0)
    triangle_out.extend(observed_triangles)
    monomial_observed_score = float("nan")
    if monomial_alignments is not None:
        monomial_observed_score, monomial_triangles = monomial_triangle_rows(
            shared_base,
            monomial_alignments,
            n_models,
            "observed",
            0.0,
        )
        triangle_out.extend(monomial_triangles)
    observed_predictors = obstruction_predictor_columns(
        pairwise,
        n_models,
        primary_width,
        residuals,
        observed_sync_disagreement,
    )
    observed_base = {
        **shared_base,
        "alignment_source": "observed",
        "alignment_noise_fraction": 0.0,
        "is_injected_alignment_control": False,
        "cycle_score": observed_cycle_score,
        "monomial_defect_score": monomial_observed_score,
        **observed_predictors,
        "sync_reference_model": observed_sync_ref,
        "sync_disagreement": observed_sync_disagreement,
        "evidence_role": "primary_observed_alignment" if n_models >= 3 else "not_primary_no_triangle",
    }
    rows = evaluate_methods(
        args,
        models=models,
        architecture=architecture,
        spec=spec,
        width=width,
        pairwise_perms=pairwise,
        pairwise_by_layer=pairwise_by_layer,
        monomial_alignments=monomial_alignments,
        val_loader=val_loader,
        test_loader=test_loader,
        match_loader=match_loader,
        device=device,
        base=observed_base,
    )
    add_paired_deltas(rows, single_best_accuracy, mean_individual_accuracy)
    run_rows.extend(rows)

    for noise_fraction in parse_float_csv(args.alignment_noise_levels):
        if noise_fraction <= 0:
            continue
        noisy_by_layer = inject_layerwise_permutation_noise(
            pairwise_by_layer,
            n_models,
            widths_by_layer,
            noise_fraction,
            seed + int(round(10000 * noise_fraction)),
        )
        noisy = primary_pairwise_permutations(noisy_by_layer, architecture)
        noisy_residuals = pairwise_alignment_residuals(models, noisy, match_loader, device, args.feature_batches)
        sync_ref, _synced, sync_disagreement = synchronize_alignment_bundle(noisy_by_layer, n_models)
        noisy_cycle_score, noisy_triangles = triangle_rows(shared_base, noisy, n_models, primary_width, "injected_noise", noise_fraction)
        triangle_out.extend(noisy_triangles)
        noisy_predictors = obstruction_predictor_columns(
            noisy,
            n_models,
            primary_width,
            noisy_residuals,
            sync_disagreement,
        )
        noisy_base = {
            **shared_base,
            **noisy_residuals,
            "pairwise_alignment_permutations_json": permutation_json(noisy),
            "layerwise_alignment_permutations_json": layerwise_permutation_json(noisy_by_layer),
            "alignment_source": "injected_noise",
            "alignment_noise_fraction": noise_fraction,
            "is_injected_alignment_control": True,
            "cycle_score": noisy_cycle_score,
            "monomial_defect_score": float("nan"),
            **noisy_predictors,
            "sync_reference_model": sync_ref,
            "sync_disagreement": sync_disagreement,
            "evidence_role": "negative_control_injected_alignment_noise",
        }
        rows = evaluate_methods(
            args,
            models=models,
            architecture=architecture,
            spec=spec,
            width=width,
            pairwise_perms=noisy,
            pairwise_by_layer=noisy_by_layer,
            monomial_alignments=None,
            val_loader=val_loader,
            test_loader=test_loader,
            match_loader=match_loader,
            device=device,
            base=noisy_base,
        )
        add_paired_deltas(rows, single_best_accuracy, mean_individual_accuracy)
        run_rows.extend(rows)

    for model in models:
        model.to("cpu")
    return run_rows, individual_rows, triangle_out


def compute_stats(runs: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows = []
    weight = runs[runs["method"] == "weight_average"].copy()
    group_cols = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "matching",
        "alignment_source",
        "alignment_noise_fraction",
    ]
    for key, group in weight.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        x = pd.to_numeric(group["cycle_score"], errors="coerce").to_numpy()
        y = pd.to_numeric(group["single_best_merge_degradation"], errors="coerce").to_numpy()
        mean_acc = pd.to_numeric(group["mean_individual_accuracy"], errors="coerce").to_numpy()
        align_resid = pd.to_numeric(group["pairwise_alignment_residual_mean"], errors="coerce").to_numpy()
        pearson = safe_pearson(x, y)
        spearman = safe_spearman(x, y)
        pearson_low, pearson_high = bootstrap_corr_ci(x, y, safe_pearson, bootstrap_samples, seed=271828)
        spearman_low, spearman_high = bootstrap_corr_ci(x, y, safe_spearman, bootstrap_samples, seed=314159)
        partial = partial_correlation(x, y, [mean_acc, align_resid])
        beta = regression_cycle_beta(x, y, [mean_acc, align_resid])
        n_rows = int(len(group))
        n_unique_seeds = int(group["seed"].nunique())
        is_observed = str(meta["alignment_source"]) == "observed" and safe_float(meta["alignment_noise_fraction"]) == 0.0
        n_models = int(meta["n_models"])
        supported = (
            n_models >= 3
            and is_observed
            and n_rows >= 20
            and math.isfinite(pearson)
            and math.isfinite(spearman)
            and pearson > 0
            and spearman > 0
            and math.isfinite(pearson_low)
            and pearson_low > 0
        )
        if n_models < 3:
            status = "unsupported_no_triangle_obstruction"
        elif not is_observed:
            status = "negative_control_not_primary_evidence"
        elif n_rows < 20:
            status = "unsupported_descriptive_n_below_20"
        elif supported:
            status = "supported_fixed_setting_observed"
        else:
            status = "unsupported_descriptive"
        rows.append(
            {
                **meta,
                "fixed_setting_id": fixed_setting_id(
                    str(meta["dataset"]),
                    str(meta["architecture"]),
                    n_models,
                    int(meta["width"]),
                    str(meta["domain_shift"]),
                    str(meta["matching"]),
                ),
                "n_rows": n_rows,
                "n_unique_seeds": n_unique_seeds,
                "mean_cycle_score": safe_mean(x),
                "std_cycle_score": safe_std(x),
                "mean_weight_merge_degradation": safe_mean(y),
                "std_weight_merge_degradation": safe_std(y),
                "pearson_cycle_vs_weight_degradation": pearson,
                "pearson_ci_low": pearson_low,
                "pearson_ci_high": pearson_high,
                "spearman_cycle_vs_weight_degradation": spearman,
                "spearman_ci_low": spearman_low,
                "spearman_ci_high": spearman_high,
                "partial_pearson_control_mean_acc_alignment_residual": partial,
                "regression_cycle_beta_control_mean_acc_alignment_residual": beta,
                "mean_individual_accuracy": safe_mean(mean_acc),
                "mean_pairwise_alignment_residual": safe_mean(align_resid),
                "claim_status": status,
                "claim_supported": bool(supported),
                "primary_evidence": bool(is_observed and n_models >= 3),
            }
        )

    method_rows = []
    method_group_cols = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "matching",
        "alignment_source",
        "alignment_noise_fraction",
        "method",
    ]
    for key, group in runs.groupby(method_group_cols, dropna=False):
        meta = dict(zip(method_group_cols, key))
        method_rows.append(
            {
                **meta,
                "fixed_setting_id": fixed_setting_id(
                    str(meta["dataset"]),
                    str(meta["architecture"]),
                    int(meta["n_models"]),
                    int(meta["width"]),
                    str(meta["domain_shift"]),
                    str(meta["matching"]),
                ),
                "n_rows": int(len(group)),
                "n_unique_seeds": int(group["seed"].nunique()),
                "mean_test_accuracy": safe_mean(group["test_accuracy"]),
                "std_test_accuracy": safe_std(group["test_accuracy"]),
                "mean_delta_vs_weight_average": safe_mean(group["delta_vs_weight_average"]),
                "mean_delta_vs_greedy_soup": safe_mean(group["delta_vs_greedy_soup"]),
                "mean_delta_vs_c2m3_synchronized": safe_mean(group["delta_vs_c2m3_synchronized"]),
                "claim_status": "method_summary_not_obstruction_correlation",
                "claim_supported": False,
                "primary_evidence": str(meta["alignment_source"]) == "observed" and int(meta["n_models"]) >= 3,
            }
        )
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(method_rows)], ignore_index=True, sort=False)


def target_family(outcome: str) -> str:
    if outcome == "weight_average_degradation_vs_best_single":
        return "raw_weight_average"
    if outcome == "greedy_soup_delta_vs_weight_average":
        return "validation_soup_vs_weight_average"
    return "alignment_conditioned"


def compute_predictor_regressions(runs: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    if runs.empty or "method" not in runs:
        return pd.DataFrame()
    base_rows = runs[runs["method"] == "weight_average"].copy()
    group_cols = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "matching",
        "alignment_source",
        "alignment_noise_fraction",
    ]
    rows = []
    for key, group in base_rows.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        n_rows = int(len(group))
        n_unique_seeds = int(group["seed"].nunique())
        mean_acc = pd.to_numeric(group["mean_individual_accuracy"], errors="coerce").to_numpy()
        residual = pd.to_numeric(group["pairwise_alignment_residual_mean"], errors="coerce").to_numpy()
        is_observed = str(meta["alignment_source"]) == "observed" and safe_float(meta["alignment_noise_fraction"]) == 0.0
        for outcome in PREDICTION_TARGETS:
            if outcome not in group:
                continue
            y = pd.to_numeric(group[outcome], errors="coerce").to_numpy()
            for predictor in REGRESSION_PREDICTORS:
                if predictor not in group:
                    continue
                x = pd.to_numeric(group[predictor], errors="coerce").to_numpy()
                beta = regression_predictor_beta(x, y, [mean_acc, residual])
                ci_low, ci_high = bootstrap_regression_beta_ci(
                    x,
                    y,
                    [mean_acc, residual],
                    bootstrap_samples,
                    seed=8111 + len(rows) * 37,
                )
                supported = bool(is_observed and n_rows >= 20 and math.isfinite(ci_low) and ci_low > 0.0)
                if not is_observed:
                    status = "negative_control_not_primary_evidence"
                elif n_rows < 20:
                    status = "unsupported_descriptive_n_below_20"
                elif supported:
                    status = "supported_positive_predictor_coefficient"
                elif math.isfinite(ci_high) and ci_high < 0.0:
                    status = "negative_association"
                else:
                    status = "unsupported_ci_crosses_zero_or_unstable"
                rows.append(
                    {
                        **meta,
                        "fixed_setting_id": fixed_setting_id(
                            str(meta["dataset"]),
                            str(meta["architecture"]),
                            int(meta["n_models"]),
                            int(meta["width"]),
                            str(meta["domain_shift"]),
                            str(meta["matching"]),
                        ),
                        "outcome": outcome,
                        "outcome_family": target_family(outcome),
                        "predictor": predictor,
                        "regression_formula": (
                            f"{outcome} ~ {predictor} + mean_individual_accuracy "
                            "+ pairwise_alignment_residual_mean"
                        ),
                        "n_rows": n_rows,
                        "n_unique_seeds": n_unique_seeds,
                        "predictor_beta": beta,
                        "predictor_beta_ci_low": ci_low,
                        "predictor_beta_ci_high": ci_high,
                        "mean_outcome": safe_mean(y),
                        "mean_predictor": safe_mean(x),
                        "mean_individual_accuracy": safe_mean(mean_acc),
                        "mean_pairwise_alignment_residual": safe_mean(residual),
                        "claim_status": status,
                        "claim_supported": supported,
                        "primary_evidence": bool(is_observed and int(meta["n_models"]) >= 3),
                    }
                )
    return pd.DataFrame(rows)


def compute_branch_paired_deltas(runs: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    if runs.empty or "method" not in runs:
        return pd.DataFrame()
    group_cols = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "matching",
        "alignment_source",
        "alignment_noise_fraction",
    ]
    rows = []
    for key, group in runs.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        methods = set(group["method"].astype(str))
        rank_methods = sorted(method for method in methods if method.startswith("twisted_rank_lift_"))
        for rank_method in rank_methods:
            branch_count = int(rank_method.rsplit("_", 1)[-1])
            rank = group[group["method"] == rank_method][["run_id", "seed", "test_accuracy"]].rename(
                columns={"test_accuracy": "rank_lift_test_accuracy"}
            )
            for baseline_prefix in BRANCH_CAPACITY_BASELINES:
                baseline_method = f"{baseline_prefix}_{branch_count}"
                if baseline_method not in methods:
                    continue
                baseline = group[group["method"] == baseline_method][["run_id", "seed", "test_accuracy"]].rename(
                    columns={"test_accuracy": "baseline_test_accuracy"}
                )
                paired = rank.merge(baseline, on=["run_id", "seed"], how="inner")
                deltas = paired["rank_lift_test_accuracy"].astype(float) - paired["baseline_test_accuracy"].astype(float)
                ci_low, ci_high = bootstrap_mean_ci(
                    deltas,
                    bootstrap_samples,
                    seed=57721 + branch_count * 101 + len(rows),
                )
                n_paired = int(paired["seed"].nunique())
                wins = int((deltas > 1e-12).sum())
                ties = int((np.abs(deltas) <= 1e-12).sum())
                losses = int((deltas < -1e-12).sum())
                rows.append(
                    {
                        **meta,
                        "fixed_setting_id": fixed_setting_id(
                            str(meta["dataset"]),
                            str(meta["architecture"]),
                            int(meta["n_models"]),
                            int(meta["width"]),
                            str(meta["domain_shift"]),
                            str(meta["matching"]),
                        ),
                        "rank_method": rank_method,
                        "baseline_method": baseline_method,
                        "comparison": f"{rank_method} - {baseline_method}",
                        "branch_count": branch_count,
                        "mean_delta": safe_mean(deltas),
                        "std_delta": safe_std(deltas),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "n_paired_seeds": n_paired,
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "baseline_ci_lower_positive": bool(math.isfinite(ci_low) and ci_low > 0.0),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    support_cols = group_cols + ["rank_method", "branch_count"]
    out["rank_lift_capacity_matched_claim_supported"] = False
    out["claim_status"] = "unsupported_missing_capacity_matched_controls"
    for key, group in out.groupby(support_cols, dropna=False):
        idx = group.index
        observed = str(group["alignment_source"].iloc[0]) == "observed" and safe_float(group["alignment_noise_fraction"].iloc[0]) == 0.0
        has_all = set(group["baseline_method"]) == {
            f"{prefix}_{int(group['branch_count'].iloc[0])}" for prefix in BRANCH_CAPACITY_BASELINES
        }
        enough_pairs = bool((group["n_paired_seeds"] >= 20).all())
        all_positive = bool(group["baseline_ci_lower_positive"].all())
        supported = bool(observed and has_all and enough_pairs and all_positive)
        if not observed:
            status = "negative_control_not_primary_evidence"
        elif not has_all:
            status = "unsupported_missing_capacity_matched_controls"
        elif not enough_pairs:
            status = "unsupported_descriptive_n_below_20"
        elif supported:
            status = "supported_vs_all_capacity_matched_branch_baselines"
        else:
            status = "unsupported_ci_crosses_zero_or_negative"
        out.loc[idx, "rank_lift_capacity_matched_claim_supported"] = supported
        out.loc[idx, "claim_status"] = status
    return out


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    rows = df.head(max_rows).copy()
    for col in columns:
        if col not in rows.columns:
            rows[col] = ""
    return format_markdown_table(rows[columns].to_dict("records"), columns)


def plot_cycle_vs_degradation(runs: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = runs[(runs["method"] == "weight_average") & (runs["alignment_source"] == "observed")].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if data.empty:
        ax.text(0.5, 0.5, "No observed weight-average rows", ha="center", va="center")
    else:
        for (dataset, n_models, width), group in data.groupby(["dataset", "n_models", "width"]):
            ax.scatter(
                group["cycle_score"],
                group["single_best_merge_degradation"],
                s=36,
                alpha=0.75,
                label=f"{dataset} N={n_models} W={width}",
            )
        ax.set_xlabel("observed cycle score")
        ax.set_ylabel("single-best accuracy minus weight-average accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_by_n_width(stats: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = stats[
        (stats.get("alignment_source", "") == "observed")
        & stats["pearson_cycle_vs_weight_degradation"].notna()
    ].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if data.empty:
        ax.text(0.5, 0.5, "No correlation rows", ha="center", va="center")
    else:
        labels = [
            f"{row.dataset}\nN={int(row.n_models)} W={int(row.width)}\n{row.domain_shift}"
            for row in data.itertuples()
        ]
        x = np.arange(len(data))
        ax.bar(x, data["pearson_cycle_vs_weight_degradation"], color="tab:blue", alpha=0.75)
        if {"pearson_ci_low", "pearson_ci_high"}.issubset(data.columns):
            low = data["pearson_cycle_vs_weight_degradation"] - data["pearson_ci_low"]
            high = data["pearson_ci_high"] - data["pearson_cycle_vs_weight_degradation"]
            ax.errorbar(x, data["pearson_cycle_vs_weight_degradation"], yerr=[low, high], fmt="none", ecolor="black", capsize=3)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Pearson r")
        ax.set_title("Cycle score vs weight-average degradation by fixed setting")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_delta_methods(runs: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = runs[runs["alignment_source"] == "observed"].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if data.empty:
        ax.text(0.5, 0.5, "No observed method rows", ha="center", va="center")
    else:
        summary = (
            data.groupby("method")["delta_vs_weight_average"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        x = np.arange(len(summary))
        err = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
        ax.bar(x, summary["mean"], yerr=err, color="tab:green", alpha=0.75, capsize=3)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["method"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean accuracy delta vs weight average")
        ax.set_title("Observed fixed-setting method deltas")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def target_prediction_summary(regressions: pd.DataFrame) -> str:
    if regressions.empty:
        return "No regression rows were produced, so no obstruction target is supported."
    observed = regressions[regressions["alignment_source"] == "observed"].copy()
    supported = observed[observed["claim_supported"] == True].copy()  # noqa: E712
    if supported.empty:
        return (
            "No target is supported in this run. Raw weight-average prediction is not claimed, "
            "and alignment-conditioned targets are not claimed."
        )
    pairs = sorted({f"{row.outcome} via {row.predictor}" for row in supported.itertuples()})
    raw_supported = any(supported["outcome"] == "weight_average_degradation_vs_best_single")
    alignment_supported = any(supported["outcome_family"] == "alignment_conditioned")
    lines = ["Supported observed target/predictor pairs:"]
    lines.extend([f"- {pair}" for pair in pairs])
    if alignment_supported and not raw_supported:
        lines.append(
            "Only alignment-conditioned targets are supported; raw weight-average prediction is not claimed."
        )
    elif not raw_supported:
        lines.append("Raw weight-average prediction is not supported in this run.")
    return "\n".join(lines)


def write_report(
    args,
    runs: pd.DataFrame,
    stats: pd.DataFrame,
    individuals: pd.DataFrame,
    triangles: pd.DataFrame,
    paired_deltas: pd.DataFrame,
    regressions: pd.DataFrame,
    report_path: Path,
    title: str = "Fixed-Setting Model-Merging Verification",
) -> None:
    observed_stats = stats[
        (stats["claim_status"].astype(str).str.contains("supported|unsupported|negative_control", na=False))
        & stats["pearson_cycle_vs_weight_degradation"].notna()
    ].copy()
    observed_corr = observed_stats[observed_stats["alignment_source"] == "observed"].copy()
    supported = observed_corr[observed_corr["claim_supported"] == True]  # noqa: E712
    method_summary = stats[stats["claim_status"] == "method_summary_not_obstruction_correlation"].copy()
    individual_summary = (
        individuals.groupby(["dataset", "architecture", "n_models", "width", "domain_shift", "matching"])["test_accuracy"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    default_command = (
        ".venv/bin/python experiments/model_merging_fixed_setting_verification.py "
        "--datasets mnist,fashion_mnist --architecture mlp2 --model-counts 3,4 --widths 128 "
        "--domain-shifts none --seeds 2000:2029 --epochs 10 --max-train-samples 10000 "
        "--max-test-samples 5000 --batch-size 128 --lr 0.001 --optimizer adamw "
        "--scheduler cosine --weight-decay 0.0001 --device cpu --matching activation "
        "--bootstrap-samples 5000 --alignment-noise-levels 0.15"
    )
    claim_text = (
        "At least one fixed observed setting passes the correlation gate."
        if not supported.empty
        else "No fixed observed setting in this run passes the n>=20 positive-correlation gate; results are descriptive."
    )
    report = f"""# {title}

This report is generated by `experiments/model_merging_fixed_setting_verification.py`.

## Exact Command

```bash
{args.command_string}
```

## Default Full Command

The intended full repeated-seed protocol is documented here and is not run automatically:

```bash
{default_command}
```

## Scope And Controls

- Fixed settings are kept separate by dataset, architecture, `N`, width, domain shift, and matching protocol.
- The main obstruction-correlation claim uses only `N>=3` observed-alignment rows; `N=2` is rejected because it has no triangle obstruction.
- The validation and test partitions are shared across all methods within each seed and setting. The test set is evaluation-only.
- Injected alignment-noise rows, when present, are labeled `injected_noise` and are negative/control diagnostics, not primary evidence.
- CIFAR is not part of the default run. No CIFAR success claim is made by this artifact.

## Outputs

- `reports/csv/{RUNS_CSV}`
- `reports/csv/{STATS_CSV}`
- `reports/csv/{TRIANGLES_CSV}`
- `reports/csv/{INDIVIDUALS_CSV}`
- `reports/csv/{REAL_OBSTRUCTION_RUNS_CSV}`
- `reports/csv/{REAL_OBSTRUCTION_SUMMARY_CSV}`
- `reports/csv/{REAL_OBSTRUCTION_TRIANGLES_CSV}`
- `reports/csv/{REAL_OBSTRUCTION_INDIVIDUALS_CSV}`
- `reports/csv/{REAL_OBSTRUCTION_PAIRED_DELTAS_CSV}`
- `reports/csv/{REAL_OBSTRUCTION_REGRESSIONS_CSV}`
- `reports/plots/fixed_setting_cycle_vs_degradation.pdf`
- `reports/plots/fixed_setting_by_N_width.pdf`
- `reports/plots/fixed_setting_delta_methods.pdf`

## Claim Gate

{claim_text}

A fixed setting is marked supported only when `n_rows >= 20`, Pearson and Spearman are both positive, the bootstrap Pearson CI lower bound is positive, and the rows are observed alignments rather than injected controls.

## Fixed-Setting Correlations

{md_table(observed_corr, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "alignment_source", "n_rows", "n_unique_seeds", "mean_cycle_score", "mean_weight_merge_degradation", "pearson_cycle_vs_weight_degradation", "pearson_ci_low", "pearson_ci_high", "spearman_cycle_vs_weight_degradation", "partial_pearson_control_mean_acc_alignment_residual", "regression_cycle_beta_control_mean_acc_alignment_residual", "claim_status"], 30)}

## Method Deltas

{md_table(method_summary, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "alignment_source", "method", "n_rows", "mean_test_accuracy", "mean_delta_vs_weight_average", "mean_delta_vs_greedy_soup", "mean_delta_vs_c2m3_synchronized"], 40)}

## Which target does the obstruction predict?

{target_prediction_summary(regressions)}

{md_table(regressions[regressions["alignment_source"] == "observed"] if not regressions.empty else regressions, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "outcome", "predictor", "n_rows", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "claim_status"], 60)}

## Capacity matching and extra capacity

The branch rank-lift row is not a single merged model. It is an extra-capacity branch ensemble with `branch_count > 1`, `parameter_multiplier > 1`, and `inference_multiplier > 1`. The branch-capacity controls below match that branch count and inference multiplier: random branch partitioning, validation-selected branches, and C2M3-distance cluster branches. A rank-lift improvement is marked supported only when the observed paired bootstrap CI lower bound is positive against all three controls with at least 20 paired seeds.

{md_table(paired_deltas, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "alignment_source", "comparison", "branch_count", "n_paired_seeds", "mean_delta", "std_delta", "bootstrap_ci_low", "bootstrap_ci_high", "wins", "ties", "losses", "claim_status"], 40)}

## Individual Model Accuracy

{md_table(individual_summary, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "mean", "min", "max"], 30)}

## Triangle Defects

Triangle/cocycle defects are written one row per triangle to `reports/csv/{TRIANGLES_CSV}`. The smoke or full run currently produced `{len(triangles)}` triangle rows.

## Interpretation Boundary

This artifact tests whether observed cycle residuals predict ordinary weight-average degradation under fixed small-network settings. It does not claim that TwistedMerge beats Git Re-Basin, C2M3, Model Soups, or all model-merging baselines. Rank-lift rows are branch ensembles with extra capacity and are labeled as such.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def write_full_run_interpretation(
    args,
    runs: pd.DataFrame,
    stats: pd.DataFrame,
    individuals: pd.DataFrame,
    regressions: pd.DataFrame,
    report_path: Path,
) -> None:
    if individuals.empty:
        quality = pd.DataFrame()
    else:
        best_by_run = (
            individuals.groupby(["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "run_id"])["test_accuracy"]
            .max()
            .reset_index(name="best_individual_accuracy")
        )
        quality = (
            best_by_run.groupby(["dataset", "architecture", "n_models", "width", "domain_shift", "matching"])["best_individual_accuracy"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
        )
        quality["gate"] = quality.apply(
            lambda row: (
                "passes_preferred"
                if (row["dataset"] == "mnist" and row["mean"] > 0.90)
                or (row["dataset"] == "fashion_mnist" and row["mean"] > 0.80)
                else "passes_minimum"
                if (row["dataset"] == "mnist" and row["mean"] > 0.85)
                or (row["dataset"] == "fashion_mnist" and row["mean"] > 0.75)
                else "below_quality_gate"
            ),
            axis=1,
        )

    corr_rows = stats[
        (stats["claim_status"].astype(str) != "method_summary_not_obstruction_correlation")
        & (stats["alignment_source"].astype(str) == "observed")
        & (pd.to_numeric(stats["alignment_noise_fraction"], errors="coerce").fillna(1.0) == 0.0)
    ].copy()
    supported_corr = corr_rows[corr_rows.get("claim_supported", False) == True].copy()  # noqa: E712
    observed_methods = runs[
        (runs["alignment_source"].astype(str) == "observed")
        & (pd.to_numeric(runs["alignment_noise_fraction"], errors="coerce").fillna(1.0) == 0.0)
    ].copy()
    method_summary = (
        observed_methods.groupby(["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "method"])[
            ["test_accuracy", "delta_vs_weight_average", "delta_vs_greedy_soup", "delta_vs_c2m3_synchronized"]
        ]
        .mean()
        .reset_index()
        if not observed_methods.empty
        else pd.DataFrame()
    )
    regression_supported = regressions[
        (regressions["alignment_source"].astype(str) == "observed")
        & (regressions.get("claim_supported", False) == True)
    ].copy() if not regressions.empty else pd.DataFrame()
    conclusion = (
        "At least one primary observed fixed setting passes the strict obstruction-correlation gate."
        if not supported_corr.empty
        else "No primary observed fixed setting passes the strict obstruction-correlation gate; the full run is negative-but-useful descriptive evidence."
    )
    regression_conclusion = (
        "Some controlled predictor-regression rows pass their positive-coefficient gate, but these are secondary diagnostics."
        if not regression_supported.empty
        else "No secondary predictor-regression row is promoted as a supported obstruction claim."
    )
    report = f"""# Fixed-Setting Full Run Interpretation

Generated by `experiments/model_merging_fixed_setting_verification.py`.

## Run Command

```bash
{args.command_string}
```

## Selected Setting

The full verification used the strongest CPU-feasible setting from `reports/training_quality_sweep_report.md`: `mlp2`, width `128`, AdamW, cosine scheduling, and MNIST/Fashion-MNIST training quality checks. This run keeps dataset, architecture, `N`, width, domain shift, and matching protocol separated. It excludes `N=2` and does not include CIFAR.

## Local Model Quality Gate

{md_table(quality, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "count", "mean", "min", "max", "gate"], 20)}

## Primary Obstruction-Correlation Gate

{conclusion}

A setting is supported only when it has at least 20 observed rows, positive Pearson and Spearman correlations, and a positive bootstrap lower bound for Pearson. Injected alignment-noise rows are controls and are not primary evidence.

{md_table(corr_rows, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "n_rows", "n_unique_seeds", "mean_cycle_score", "mean_weight_merge_degradation", "pearson_cycle_vs_weight_degradation", "pearson_ci_low", "pearson_ci_high", "spearman_cycle_vs_weight_degradation", "spearman_ci_low", "spearman_ci_high", "partial_pearson_control_mean_acc_alignment_residual", "claim_status"], 30)}

## Method Boundary

{md_table(method_summary, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "method", "test_accuracy", "delta_vs_weight_average", "delta_vs_greedy_soup", "delta_vs_c2m3_synchronized"], 40)}

## Secondary Diagnostics

{regression_conclusion}

{md_table(regression_supported, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "outcome", "predictor", "n_rows", "predictor_beta", "predictor_beta_ci_low", "predictor_beta_ci_high", "claim_status"], 30)}

## Claim Boundary

This report tests whether observed permutation/cycle residuals predict ordinary merge degradation for fixed small neural-network settings. It does not update the paper abstract, does not make a CIFAR claim, and does not claim that TwistedMerge beats Git Re-Basin, C2M3, Model Soups, or general model-merging baselines.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def write_monomial_report(args, runs: pd.DataFrame, triangles: pd.DataFrame, report_path: Path) -> None:
    if runs.empty or "matching" not in runs:
        monomial_runs = pd.DataFrame()
    else:
        monomial_runs = runs[runs["matching"].astype(str).isin(MONOMIAL_MATCHINGS)].copy()
    if monomial_runs.empty or "alignment_source" not in monomial_runs:
        observed = pd.DataFrame()
    else:
        observed = monomial_runs[monomial_runs["alignment_source"].astype(str) == "observed"].copy()
    if observed.empty or "method" not in observed:
        monomial_methods = pd.DataFrame()
    else:
        monomial_methods = observed[observed["method"].astype(str) == "monomial_gauge_ref0"].copy()

    preservation = pd.DataFrame()
    if not monomial_methods.empty:
        preservation = (
            monomial_methods.groupby(["dataset", "architecture", "n_models", "width", "domain_shift", "matching"], dropna=False)
            .agg(
                n_rows=("run_id", "count"),
                max_functional_preservation_error=("functional_preservation_error", "max"),
                mean_functional_preservation_error=("functional_preservation_error", "mean"),
                mean_prediction_disagreement=("functional_preservation_prediction_disagreement", "mean"),
                mean_abs_log_scale=("monomial_mean_abs_log_scale", "mean"),
                max_abs_log_scale=("monomial_max_abs_log_scale", "max"),
            )
            .reset_index()
        )

    method_summary = pd.DataFrame()
    if not observed.empty:
        method_summary = (
            observed.groupby(["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "method"], dropna=False)
            .agg(
                n_rows=("run_id", "count"),
                mean_test_accuracy=("test_accuracy", "mean"),
                mean_delta_vs_weight_average=("delta_vs_weight_average", "mean"),
                mean_delta_vs_greedy_soup=("delta_vs_greedy_soup", "mean"),
                mean_delta_vs_c2m3_synchronized=("delta_vs_c2m3_synchronized", "mean"),
            )
            .reset_index()
        )

    corr_rows = []
    if not observed.empty and "method" in observed:
        weight_rows = observed[observed["method"].astype(str) == "weight_average"].copy()
    else:
        weight_rows = pd.DataFrame()
    group_cols = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching"]
    if not weight_rows.empty and "monomial_defect_score" in weight_rows:
        for key, group in weight_rows.groupby(group_cols, dropna=False):
            meta = dict(zip(group_cols, key))
            y = pd.to_numeric(group["single_best_merge_degradation"], errors="coerce").to_numpy()
            permutation_x = pd.to_numeric(group["cycle_score"], errors="coerce").to_numpy()
            monomial_x = pd.to_numeric(group["monomial_defect_score"], errors="coerce").to_numpy()
            permutation_pearson = safe_pearson(permutation_x, y)
            monomial_pearson = safe_pearson(monomial_x, y)
            corr_rows.append(
                {
                    **meta,
                    "n_rows": int(len(group)),
                    "permutation_pearson_vs_degradation": permutation_pearson,
                    "permutation_spearman_vs_degradation": safe_spearman(permutation_x, y),
                    "monomial_pearson_vs_degradation": monomial_pearson,
                    "monomial_spearman_vs_degradation": safe_spearman(monomial_x, y),
                    "monomial_more_predictive_by_abs_pearson": bool(abs(monomial_pearson) > abs(permutation_pearson))
                    if math.isfinite(monomial_pearson) and math.isfinite(permutation_pearson)
                    else False,
                    "claim_status": "descriptive_n_below_20" if len(group) < 20 else "descriptive_no_strict_gate",
                }
            )
    corr = pd.DataFrame(corr_rows)

    if triangles.empty or "matching" not in triangles:
        monomial_triangles = pd.DataFrame()
    else:
        monomial_triangles = triangles[triangles["matching"].astype(str).isin(MONOMIAL_MATCHINGS)].copy()
    triangle_summary = pd.DataFrame()
    if not monomial_triangles.empty and "triangle_type" in monomial_triangles:
        triangle_summary = (
            monomial_triangles.groupby(["matching", "triangle_type"], dropna=False)
            .agg(
                n_rows=("run_id", "count"),
                mean_cycle_defect=("cycle_defect", "mean"),
                mean_monomial_defect_score=("monomial_defect_score", "mean"),
            )
            .reset_index()
        )

    if monomial_runs.empty:
        conclusion = "No monomial matching rows were generated in this run."
    elif corr.empty or int(corr["n_rows"].max()) < 20:
        conclusion = (
            "The monomial run is implementation/descriptive evidence only; no predictor claim is supported because "
            "each fixed setting has fewer than 20 observed seeds."
        )
    elif bool(corr["monomial_more_predictive_by_abs_pearson"].any()):
        conclusion = (
            "At least one descriptive fixed setting has a larger absolute Pearson correlation for the monomial residual "
            "than for the permutation cycle score, but this report does not promote that to a paper claim without the "
            "same bootstrap gate used by the main verifier."
        )
    else:
        conclusion = "The generated rows do not show a clearer monomial residual predictor than the permutation cycle score."

    report = f"""# ReLU-Compatible Monomial Gauge Alignment

Generated by `experiments/model_merging_fixed_setting_verification.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- Monomial matching is implemented only for one-hidden-layer ReLU `mlp` models.
- The gauge is permutation plus positive hidden-unit scaling with inverse outgoing classifier scaling, so it is an exact ReLU reparameterization before averaging.
- The monomial merge row is a single same-capacity merged model, not an ensemble and not a period-index lift.
- The main fixed-setting obstruction claim gate is unchanged. This report only compares permutation and monomial diagnostics in rows generated with `--matching monomial_activation,monomial_weight`.

## Outputs

- `reports/csv/{MONOMIAL_RUNS_CSV}`
- `reports/csv/{MONOMIAL_TRIANGLES_CSV}`
- `reports/monomial_gauge_alignment_report.md`

## Conclusion

{conclusion}

## Functional Preservation

{md_table(preservation, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "n_rows", "max_functional_preservation_error", "mean_functional_preservation_error", "mean_prediction_disagreement", "mean_abs_log_scale", "max_abs_log_scale"], 30)}

## Method Comparison

{md_table(method_summary, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "method", "n_rows", "mean_test_accuracy", "mean_delta_vs_weight_average", "mean_delta_vs_greedy_soup", "mean_delta_vs_c2m3_synchronized"], 40)}

## Residual Correlations

{md_table(corr, ["dataset", "architecture", "n_models", "width", "domain_shift", "matching", "n_rows", "permutation_pearson_vs_degradation", "permutation_spearman_vs_degradation", "monomial_pearson_vs_degradation", "monomial_spearman_vs_degradation", "monomial_more_predictive_by_abs_pearson", "claim_status"], 30)}

## Triangle Diagnostics

{md_table(triangle_summary, ["matching", "triangle_type", "n_rows", "mean_cycle_defect", "mean_monomial_defect_score"], 30)}

## Claim Boundary

This artifact supports the implementation claim that positive monomial gauges can be estimated and applied as exact ReLU reparameterizations in the one-hidden-layer MLP path. It does not by itself support a claim that monomial obstruction residuals are more predictive, or that monomial-aligned merging beats greedy soup, C2M3, or external baselines.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def update_claims_audit(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    unsupported_old = (
        "| Cycle obstruction score predicts weight-average merge degradation beyond the trivial number-of-models confound. "
        "| Not yet supported | In `reports/csv/model_merging_stats.csv`, fixed-`N` observed correlations are marked unsupported: "
        "`N=3` Pearson `-0.0347`, `N=4` Pearson `-0.3622`, and bootstrap intervals cross zero. |"
    )
    unsupported_new = (
        "| Cycle obstruction score predicts weight-average merge degradation beyond the trivial number-of-models confound. "
        "| Not yet supported | `reports/fixed_setting_verification_report.md` adds a stricter fixed-setting protocol separating dataset, architecture, `N`, width, domain shift, and matching. The current generated artifact is smoke-scale/descriptive unless a setting reaches `n_rows >= 20` with positive Pearson/Spearman and a positive Pearson bootstrap lower bound. |"
    )
    if unsupported_old in text:
        text = text.replace(unsupported_old, unsupported_new)
    supported_marker = (
        "| The model-merging benchmark now includes fixed-`N` repeated-seed MNIST checks and controlled injected-alignment negative controls. "
        "| Supported | `reports/model_merging_verification_report.md` and `reports/csv/model_merging_verification.csv` cover MNIST MLP, `N=3,4`, widths `16,32`, five seeds, and injected pairwise alignment noise. |"
    )
    supported_new = (
        supported_marker
        + "\n| The fixed-setting verification script implements the stronger repeated-seed obstruction-correlation gate for real small neural networks. | Supported implementation | `experiments/model_merging_fixed_setting_verification.py` writes fixed-setting run, statistics, triangle-defect, and individual-model CSVs plus plots/report; claims remain gated by `n_rows >= 20` observed rows and bootstrap CIs. |"
        + "\n| Rank-lift branch evidence is separated from branch-capacity matched non-obstruction controls. | Supported implementation | `src/rank_lift_baselines.py` adds random, validation-selected, and C2M3-cluster branch ensembles. `reports/csv/real_obstruction_paired_deltas.csv` marks rank-lift support only when observed paired CI lower bounds are positive against all three branch controls with at least 20 paired seeds. |"
    )
    if supported_marker in text and "fixed-setting verification script implements the stronger repeated-seed" not in text:
        text = text.replace(supported_marker, supported_new)
    elif "Rank-lift branch evidence is separated from branch-capacity matched non-obstruction controls." not in text:
        text += (
            "\n| Rank-lift branch evidence is separated from branch-capacity matched non-obstruction controls. | Supported implementation | `src/rank_lift_baselines.py` adds random, validation-selected, and C2M3-cluster branch ensembles. `reports/csv/real_obstruction_paired_deltas.csv` marks rank-lift support only when observed paired CI lower bounds are positive against all three branch controls with at least 20 paired seeds. |"
        )
    artifact_marker = "| `reports/csv/model_merging_stats.csv` | Correlations, bootstrap intervals, deltas, and negative-result labels for verification settings. |"
    artifact_new = (
        artifact_marker
        + "\n| `reports/fixed_setting_verification_report.md` | Stronger fixed-setting repeated-seed verification report for cycle residual versus ordinary merge degradation. |"
        + "\n| `reports/real_obstruction_degradation_report.md` | Paper-facing real obstruction-degradation report with capacity-matched rank-lift branch controls. |"
        + f"\n| `reports/csv/{RUNS_CSV}` | Per-method fixed-setting rows including observed/injected alignment labels and method-capacity metadata. |"
        + f"\n| `reports/csv/{REAL_OBSTRUCTION_RUNS_CSV}` | Paper-facing alias for per-method real obstruction-degradation rows, including capacity metadata and branch controls. |"
        + f"\n| `reports/csv/{STATS_CSV}` | Fixed-setting Pearson/Spearman/bootstrap and controlled regression statistics. |"
        + f"\n| `reports/csv/{TRIANGLES_CSV}` | Per-triangle permutation/cocycle defect rows. |"
        + f"\n| `reports/csv/{INDIVIDUALS_CSV}` | Per-local-model validation/test accuracy and checkpoint metadata. |"
        + f"\n| `reports/csv/{REAL_OBSTRUCTION_PAIRED_DELTAS_CSV}` | Paired rank-lift minus branch-capacity matched baseline deltas with bootstrap confidence intervals. |"
        + f"\n| `reports/csv/{REAL_OBSTRUCTION_REGRESSIONS_CSV}` | Predictor-target regressions for obstruction diagnostics, with controls and bootstrap coefficient intervals. |"
    )
    if artifact_marker in text and "fixed_setting_verification_report.md" not in text:
        text = text.replace(artifact_marker, artifact_new)
    elif "real_obstruction_paired_deltas.csv" not in text:
        text += (
            "\n| `reports/real_obstruction_degradation_report.md` | Paper-facing real obstruction-degradation report with capacity-matched rank-lift branch controls. |"
            f"\n| `reports/csv/{REAL_OBSTRUCTION_RUNS_CSV}` | Paper-facing alias for per-method real obstruction-degradation rows, including capacity metadata and branch controls. |"
            f"\n| `reports/csv/{REAL_OBSTRUCTION_PAIRED_DELTAS_CSV}` | Paired rank-lift minus branch-capacity matched baseline deltas with bootstrap confidence intervals. |"
            f"\n| `reports/csv/{REAL_OBSTRUCTION_REGRESSIONS_CSV}` | Predictor-target regressions for obstruction diagnostics, with controls and bootstrap coefficient intervals. |"
        )
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--architecture", default="mlp2", choices=["mlp", "mlp2", "cnn", "small_cnn"])
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="128")
    parser.add_argument("--domain-shifts", default="none")
    parser.add_argument("--seeds", default="2000:2029")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--augmentation", default="none", choices=["none", "light"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--matching", default="activation")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--alignment-noise-levels", default="0.15")
    parser.add_argument("--rank-lift-branches", type=int, default=2)
    parser.add_argument("--feature-batches", type=int, default=8)
    parser.add_argument("--dataset-seed", type=int, default=314159)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--update-claims-audit", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    datasets = parse_csv(args.datasets, str)
    model_counts = parse_csv(args.model_counts, int)
    widths = parse_csv(args.widths, int)
    domain_shifts = parse_csv(args.domain_shifts, str)
    matchings = parse_csv(args.matching, str)
    seeds = parse_seeds(args.seeds)
    valid_matchings = {"activation", "weight", *MONOMIAL_MATCHINGS}
    unknown_matchings = sorted(set(matchings) - valid_matchings)
    if unknown_matchings:
        raise ValueError(f"unknown matching protocols: {unknown_matchings}")
    if any(is_monomial_matching(matching) for matching in matchings) and args.architecture != "mlp":
        raise ValueError("monomial_activation/monomial_weight currently require --architecture mlp")
    if any(n < 3 for n in model_counts):
        raise ValueError("Do not include N=2 in fixed-setting verification: N=2 has no triangle obstruction.")

    all_runs: list[dict] = []
    all_individuals: list[dict] = []
    all_triangles: list[dict] = []
    for dataset_name in datasets:
        if dataset_name == "cifar10":
            raise ValueError("CIFAR-10 is intentionally excluded here unless a separate gate establishes strong individual accuracy.")
        for n_models in model_counts:
            for width in widths:
                for domain_shift in domain_shifts:
                    for matching in matchings:
                        for seed in seeds:
                            print(
                                f"running dataset={dataset_name} arch={args.architecture} N={n_models} "
                                f"W={width} shift={domain_shift} matching={matching} seed={seed}",
                                flush=True,
                            )
                            run_rows, individual_rows, triangle_rows_out = run_one_seed(
                                args,
                                dataset_name,
                                args.architecture,
                                n_models,
                                width,
                                domain_shift,
                                matching,
                                seed,
                            )
                            all_runs.extend(run_rows)
                            all_individuals.extend(individual_rows)
                            all_triangles.extend(triangle_rows_out)

    runs = pd.DataFrame(all_runs)
    individuals = pd.DataFrame(all_individuals)
    triangles = pd.DataFrame(all_triangles)
    monomial_runs = (
        runs[runs["matching"].astype(str).isin(MONOMIAL_MATCHINGS)].copy()
        if not runs.empty and "matching" in runs
        else pd.DataFrame()
    )
    monomial_triangles = (
        triangles[triangles["matching"].astype(str).isin(MONOMIAL_MATCHINGS)].copy()
        if not triangles.empty and "matching" in triangles
        else pd.DataFrame()
    )
    stats = compute_stats(runs, args.bootstrap_samples)
    paired_deltas = compute_branch_paired_deltas(runs, args.bootstrap_samples)
    regressions = compute_predictor_regressions(runs, args.bootstrap_samples)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    runs_path = csv_dir / RUNS_CSV
    stats_path = csv_dir / STATS_CSV
    triangles_path = csv_dir / TRIANGLES_CSV
    individuals_path = csv_dir / INDIVIDUALS_CSV
    paired_deltas_path = csv_dir / REAL_OBSTRUCTION_PAIRED_DELTAS_CSV
    regressions_path = csv_dir / REAL_OBSTRUCTION_REGRESSIONS_CSV
    monomial_runs_path = csv_dir / MONOMIAL_RUNS_CSV
    monomial_triangles_path = csv_dir / MONOMIAL_TRIANGLES_CSV
    runs.to_csv(runs_path, index=False, lineterminator="\n")
    stats.to_csv(stats_path, index=False, lineterminator="\n")
    triangles.to_csv(triangles_path, index=False, lineterminator="\n")
    individuals.to_csv(individuals_path, index=False, lineterminator="\n")
    paired_deltas.to_csv(paired_deltas_path, index=False, lineterminator="\n")
    regressions.to_csv(regressions_path, index=False, lineterminator="\n")
    monomial_runs.to_csv(monomial_runs_path, index=False, lineterminator="\n")
    monomial_triangles.to_csv(monomial_triangles_path, index=False, lineterminator="\n")
    runs.to_csv(csv_dir / REAL_OBSTRUCTION_RUNS_CSV, index=False, lineterminator="\n")
    stats.to_csv(csv_dir / REAL_OBSTRUCTION_SUMMARY_CSV, index=False, lineterminator="\n")
    triangles.to_csv(csv_dir / REAL_OBSTRUCTION_TRIANGLES_CSV, index=False, lineterminator="\n")
    individuals.to_csv(csv_dir / REAL_OBSTRUCTION_INDIVIDUALS_CSV, index=False, lineterminator="\n")

    plot_cycle_vs_degradation(runs, plot_dir / "fixed_setting_cycle_vs_degradation.pdf")
    plot_by_n_width(stats, plot_dir / "fixed_setting_by_N_width.pdf")
    plot_delta_methods(runs, plot_dir / "fixed_setting_delta_methods.pdf")
    write_report(
        args,
        runs,
        stats,
        individuals,
        triangles,
        paired_deltas,
        regressions,
        args.reports_dir / "fixed_setting_verification_report.md",
    )
    write_report(
        args,
        runs,
        stats,
        individuals,
        triangles,
        paired_deltas,
        regressions,
        args.reports_dir / "real_obstruction_degradation_report.md",
        title="Real Obstruction Degradation Verification",
    )
    write_full_run_interpretation(
        args,
        runs,
        stats,
        individuals,
        regressions,
        args.reports_dir / "fixed_setting_full_run_interpretation.md",
    )
    write_monomial_report(
        args,
        runs,
        triangles,
        args.reports_dir / "monomial_gauge_alignment_report.md",
    )
    save_json(
        args.reports_dir / "configs" / "fixed_setting_verification_config.json",
        {
            "argv": sys.argv,
            "parsed_seeds": summarize_seed_list(seeds),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "environment": capture_environment(),
        },
    )
    if args.update_claims_audit:
        update_claims_audit(args.reports_dir / "claims_audit.md")
    print(f"wrote {runs_path}")
    print(f"wrote {stats_path}")
    print(f"wrote {triangles_path}")
    print(f"wrote {individuals_path}")
    print(f"wrote {paired_deltas_path}")
    print(f"wrote {regressions_path}")
    print(f"wrote {monomial_runs_path}")
    print(f"wrote {monomial_triangles_path}")
    print(f"wrote {args.reports_dir / 'fixed_setting_verification_report.md'}")
    print(f"wrote {args.reports_dir / 'real_obstruction_degradation_report.md'}")
    print(f"wrote {args.reports_dir / 'fixed_setting_full_run_interpretation.md'}")
    print(f"wrote {args.reports_dir / 'monomial_gauge_alignment_report.md'}")


if __name__ == "__main__":
    main()

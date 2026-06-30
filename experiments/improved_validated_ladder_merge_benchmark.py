#!/usr/bin/env python
"""Validation-optimized monomial scaling and soup-competitive benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.improved_monomial_merge import (  # noqa: E402
    build_scaled_models,
    choose_by_validation,
    global_log_scale_synchronization,
    greedy_soup_with_metadata,
    log_scale_diagnostics,
    optimize_log_scales_for_validation,
    reference_log_scales_from_features,
    shrink_log_scales,
)
from src.ladder_merge_methods import METHOD_METADATA, MethodMetadata  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    collect_features,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
    set_seed,
    synchronize_permutations,
    train_model,
)
from src.structure_group_ladder import StructureGroupLadderMerge, estimate_pairwise_permutations_from_activations  # noqa: E402


EXTRA_METADATA: dict[str, MethodMetadata] = {
    "validated_ladder_selector": MethodMetadata(
        "validated_ladder_selector",
        "validation_selected_exact_relu_single_model",
        True,
        True,
        "Validation-only selector choosing between C2M3 and raw positive monomial scaling.",
    ),
    "shrinkage_monomial_scale": MethodMetadata(
        "shrinkage_monomial_scale",
        "validation_selected_exact_relu_positive_scale",
        True,
        True,
        "Reference-based positive scales with validation-selected alpha and clipping.",
    ),
    "global_monomial_scale": MethodMetadata(
        "global_monomial_scale",
        "validation_selected_global_exact_relu_positive_scale",
        True,
        True,
        "Least-squares global positive scale synchronization with validation-selected shrinkage.",
    ),
    "optimized_monomial_scale": MethodMetadata(
        "optimized_monomial_scale",
        "validation_optimized_exact_relu_positive_scale",
        True,
        True,
        "Validation-loss optimization over exact positive ReLU log-scale gauges before averaging.",
    ),
    "improved_validated_selector": MethodMetadata(
        "improved_validated_selector",
        "validation_selected_single_model_or_soup",
        True,
        True,
        "Validation-only regret-minimizing selector over exact single-model merge/soup candidates.",
    ),
    "c2m3_greedy_soup": MethodMetadata(
        "c2m3_greedy_soup",
        "exact_relu_permutation_soup",
        True,
        True,
        "Greedy soup over C2M3-aligned models using validation accuracy.",
    ),
    "monomial_scaled_greedy_soup": MethodMetadata(
        "monomial_scaled_greedy_soup",
        "exact_relu_positive_scale_soup",
        True,
        True,
        "Greedy soup over raw monomial-scaled aligned models using validation accuracy.",
    ),
    "shrinkage_monomial_greedy_soup": MethodMetadata(
        "shrinkage_monomial_greedy_soup",
        "exact_relu_positive_scale_soup",
        True,
        True,
        "Greedy soup over shrinkage-monomial aligned models using validation accuracy.",
    ),
    "global_monomial_greedy_soup": MethodMetadata(
        "global_monomial_greedy_soup",
        "global_exact_relu_positive_scale_soup",
        True,
        True,
        "Greedy soup over global-monomial aligned models using validation accuracy.",
    ),
    "optimized_monomial_greedy_soup": MethodMetadata(
        "optimized_monomial_greedy_soup",
        "optimized_exact_relu_positive_scale_soup",
        True,
        True,
        "Greedy soup over validation-optimized monomial aligned models.",
    ),
    "union_candidate_soup": MethodMetadata(
        "union_candidate_soup",
        "validation_selected_union_candidate_single_model_soup",
        True,
        True,
        "Greedy soup over original, C2M3, shrinkage, and global monomial candidates; output is one averaged MLP.",
    ),
}

METHODS = {**METHOD_METADATA, **EXTRA_METADATA}
INT_COLUMNS = {
    "n_rows",
    "n_seeds",
    "n_pairs",
    "accuracy_wins",
    "accuracy_ties",
    "accuracy_losses",
    "fixed_settings_positive",
    "fixed_settings_total",
    "selector_chosen_test_better",
    "selector_chosen_test_tied",
    "selector_chosen_test_worse",
    "soup_mean_ingredient_count",
}


@dataclass(frozen=True)
class PairComparison:
    name: str
    method: str
    baseline: str
    claim_label: str


PAIR_COMPARISONS = (
    PairComparison("improved_validated_selector_vs_c2m3_permutation", "improved_validated_selector", "c2m3_permutation", "improved selector over C2M3"),
    PairComparison("improved_validated_selector_vs_greedy_soup", "improved_validated_selector", "greedy_soup", "improved selector over greedy soup"),
    PairComparison("shrinkage_monomial_scale_vs_monomial_scale", "shrinkage_monomial_scale", "monomial_scale", "shrinkage monomial over raw monomial"),
    PairComparison("global_monomial_scale_vs_monomial_scale", "global_monomial_scale", "monomial_scale", "global monomial over raw monomial"),
    PairComparison("optimized_monomial_scale_vs_monomial_scale", "optimized_monomial_scale", "monomial_scale", "optimized monomial over raw monomial"),
    PairComparison("union_candidate_soup_vs_greedy_soup", "union_candidate_soup", "greedy_soup", "union candidate soup over greedy soup"),
    PairComparison("monomial_scaled_greedy_soup_vs_greedy_soup", "monomial_scaled_greedy_soup", "greedy_soup", "raw monomial soup over greedy soup"),
    PairComparison("shrinkage_monomial_greedy_soup_vs_greedy_soup", "shrinkage_monomial_greedy_soup", "greedy_soup", "shrinkage monomial soup over greedy soup"),
    PairComparison("global_monomial_greedy_soup_vs_greedy_soup", "global_monomial_greedy_soup", "greedy_soup", "global monomial soup over greedy soup"),
    PairComparison("optimized_monomial_greedy_soup_vs_greedy_soup", "optimized_monomial_greedy_soup", "greedy_soup", "optimized monomial soup over greedy soup"),
)


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_worktree_dirty() -> bool | str:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(status.strip())
    except Exception:
        return "unknown"


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def bootstrap_mean_ci(values, n_bootstrap: int = 2000, seed: int = 12345) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def standard_error(values) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return float("nan")
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def safe_corr(x, y, method: str = "pearson") -> float:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce")
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x_arr.notna() & y_arr.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(x_arr[mask].corr(y_arr[mask], method=method))


def feature_alignment_residual(features: dict[int, np.ndarray], synced: dict[int, np.ndarray], n_models: int) -> float:
    aligned = {idx: np.asarray(features[idx])[:, np.asarray(synced[idx], dtype=int)] for idx in range(n_models)}
    residuals = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            a = aligned[i] - aligned[i].mean(axis=0, keepdims=True)
            b = aligned[j] - aligned[j].mean(axis=0, keepdims=True)
            denom = max(float(np.linalg.norm(a, ord="fro")), float(np.linalg.norm(b, ord="fro")), 1e-12)
            residuals.append(float(np.linalg.norm(a - b, ord="fro") / denom))
    return float(np.mean(residuals)) if residuals else 0.0


def diag_by_level(ladder_result) -> dict[str, object]:
    out: dict[str, object] = {
        "ladder_final_decision": ladder_result.final_decision,
        "ladder_selected_level": ladder_result.selected_level,
        "supports_brauer_projective_interpretation": any(
            diag.supports_brauer_projective_interpretation
            for diag in ladder_result.diagnostics
        ),
        "has_finite_index_candidate": any(diag.is_finite_index_candidate for diag in ladder_result.diagnostics),
    }
    for diag in ladder_result.diagnostics:
        prefix = diag.level
        out[f"{prefix}_centrality"] = diag.centrality_score
        out[f"{prefix}_cycle_score"] = diag.cycle_score
        out[f"{prefix}_residual_type"] = diag.residual_type
        out[f"{prefix}_phase_residual"] = diag.phase_residual
        out[f"{prefix}_detected_order_d"] = diag.detected_order_d
    perm = out.get("permutation_centrality", float("nan"))
    mono = out.get("monomial_phase_or_scale_centrality", float("nan"))
    out["monomial_centrality_improvement_from_permutation"] = (
        float(perm - mono) if np.isfinite(perm) and np.isfinite(mono) else float("nan")
    )
    return out


def evaluate_on_val_and_test(model, val_loader, test_loader, device) -> tuple[dict[str, float], dict[str, float]]:
    return evaluate_model(model, val_loader, device), evaluate_model(model, test_loader, device)


def add_method_row(
    rows: list[dict],
    *,
    base: dict,
    method: str,
    test_metrics: dict[str, float],
    val_metrics: dict[str, float],
    single_best_accuracy: float,
    extra: dict | None = None,
) -> None:
    meta = METHODS[method]
    row = {
        **base,
        "method": method,
        "loss": float(test_metrics["loss"]),
        "accuracy": float(test_metrics["accuracy"]),
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "single_best_accuracy": single_best_accuracy,
        "merge_degradation": single_best_accuracy - float(test_metrics["accuracy"]),
        "symmetry_status": meta.symmetry_status,
        "is_single_model": meta.is_single_model,
        "capacity_matched_to_weight_average": meta.capacity_matched_to_weight_average,
        "method_notes": meta.notes,
        "selector_no_test_leakage": True,
        "evaluation_status": "evaluated",
    }
    if extra:
        row.update(extra)
    rows.append(row)


def select_scale_grid(
    *,
    models,
    spec,
    width: int,
    synced: dict[int, np.ndarray],
    raw_log_scales: np.ndarray,
    alpha_grid: list[float],
    tau_grid: list[float],
    val_loader,
    device,
) -> tuple[np.ndarray, object, dict[str, float], float, float]:
    best = None
    for alpha in alpha_grid:
        for tau in tau_grid:
            logs = shrink_log_scales(raw_log_scales, alpha=alpha, tau=tau)
            scaled_models = build_scaled_models(models, spec, width, synced, logs)
            candidate = average_models(scaled_models, "mlp", spec, width)
            val = evaluate_model(candidate, val_loader, device)
            key = (float(val["accuracy"]), -float(val["loss"]))
            if best is None or key > best[0]:
                best = (key, logs, candidate, val, float(alpha), float(tau))
    assert best is not None
    _key, logs, model, val, alpha, tau = best
    return logs, model, val, alpha, tau


def add_scaled_method(
    rows,
    *,
    base,
    method,
    models,
    spec,
    width,
    synced,
    log_scales,
    val_loader,
    test_loader,
    device,
    single_best_accuracy,
    scale_source,
    synchronization_disagreement,
    extra=None,
):
    scaled_models = build_scaled_models(models, spec, width, synced, log_scales)
    model = average_models(scaled_models, "mlp", spec, width)
    val, test = evaluate_on_val_and_test(model, val_loader, test_loader, device)
    diagnostics = log_scale_diagnostics(log_scales, synchronization_disagreement)
    data = {
        "scale_source": scale_source,
        "mean_abs_log_scale": diagnostics.mean_abs_log_scale,
        "max_abs_log_scale": diagnostics.max_abs_log_scale,
        "log_scale_variance": diagnostics.log_scale_variance,
        "scale_synchronization_disagreement": diagnostics.synchronization_disagreement,
    }
    if extra:
        data.update(extra)
    add_method_row(
        rows,
        base=base,
        method=method,
        test_metrics=test,
        val_metrics=val,
        single_best_accuracy=single_best_accuracy,
        extra=data,
    )
    return scaled_models, model, val, test


def add_soup_row(
    rows,
    *,
    base,
    method,
    candidate_models,
    candidate_labels,
    val_loader,
    test_loader,
    device,
    spec,
    width,
    single_best_accuracy,
    extra=None,
):
    soup = greedy_soup_with_metadata(candidate_models, candidate_labels, val_loader, test_loader, device, "mlp", spec, width)
    data = {
        "soup_indices": json.dumps(soup.selected_indices),
        "soup_selected_labels": json.dumps(soup.selected_labels),
        "soup_ingredient_count": int(len(soup.selected_indices)),
        "soup_selected_types": json.dumps(sorted({label.split(":")[0] for label in soup.selected_labels})),
    }
    if extra:
        data.update(extra)
    add_method_row(
        rows,
        base=base,
        method=method,
        test_metrics=soup.test_metrics,
        val_metrics=soup.val_metrics,
        single_best_accuracy=single_best_accuracy,
        extra=data,
    )
    return soup


def run_setting(args, spec, train_data, test_data, seed: int, n_models: int, width: int) -> list[dict]:
    device = device_from_arg(args.device)
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 77)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 700)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 999)
    match_loader = make_loader(train_subset, args.batch_size, shuffle=False, seed=seed + 501)

    models = []
    individual_accuracies = []
    individual_losses = []
    for model_idx in range(n_models):
        model_seed = seed + 1000 * model_idx + 17 * width + n_models
        set_seed(model_seed)
        model = make_model("mlp", spec, width)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed + 11)
        train_model(model, train_loader, args.epochs, args.lr, device)
        test_metrics = evaluate_model(model, test_loader, device)
        individual_accuracies.append(float(test_metrics["accuracy"]))
        individual_losses.append(float(test_metrics["loss"]))
        model.to("cpu")
        models.append(model)

    features = {idx: collect_features(model, match_loader, device) for idx, model in enumerate(models)}
    pairwise = estimate_pairwise_permutations_from_activations(features, n_models, width)
    ref, synced, sync_disagreement = synchronize_permutations(pairwise, n_models)
    ladder_result = StructureGroupLadderMerge(max_order=args.max_order).run(
        {"permutation": pairwise},
        n_models=n_models,
        width=width,
        activations=features,
        candidate_lift_rank=width,
    )
    diagnostics = diag_by_level(ladder_result)
    reference_logs = reference_log_scales_from_features(features, synced, ref=ref, width=width)
    global_sync = global_log_scale_synchronization(features, synced, n_models=n_models, width=width, ref=ref)
    pairwise_residual = feature_alignment_residual(features, synced, n_models)
    setting_id = f"mnist_mlp_N{n_models}_W{width}_S{seed}"
    single_best_accuracy = float(max(individual_accuracies))
    base = {
        "setting_id": setting_id,
        "dataset": "mnist",
        "architecture": "mlp_relu",
        "n_models": n_models,
        "width": width,
        "seed": seed,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "matching": "activation",
        "sync_reference": ref,
        "sync_disagreement": sync_disagreement,
        "global_scale_sync_rms_residual": global_sync.rms_residual,
        "global_scale_sync_max_residual": global_sync.max_residual,
        "pairwise_alignment_residual": pairwise_residual,
        "individual_accuracy_mean": float(np.mean(individual_accuracies)),
        "individual_accuracy_min": float(np.min(individual_accuracies)),
        "individual_accuracy_max": single_best_accuracy,
        "individual_accuracy_variance": float(np.var(individual_accuracies)),
        "individual_loss_mean": float(np.mean(individual_losses)),
        **diagnostics,
    }
    rows: list[dict] = []

    weight_avg = average_models(models, "mlp", spec, width)
    weight_val, weight_test = evaluate_on_val_and_test(weight_avg, val_loader, test_loader, device)
    add_method_row(rows, base=base, method="weight_average", test_metrics=weight_test, val_metrics=weight_val, single_best_accuracy=single_best_accuracy)

    aligned_c2m3 = [permute_model_to_reference(model, "mlp", spec, width, synced[idx]) for idx, model in enumerate(models)]
    c2m3_model = average_models(aligned_c2m3, "mlp", spec, width)
    c2m3_val, c2m3_test = evaluate_on_val_and_test(c2m3_model, val_loader, test_loader, device)
    add_method_row(rows, base=base, method="c2m3_permutation", test_metrics=c2m3_test, val_metrics=c2m3_val, single_best_accuracy=single_best_accuracy)

    raw_monomial_models, _model, monomial_val, monomial_test = add_scaled_method(
        rows,
        base=base,
        method="monomial_scale",
        models=models,
        spec=spec,
        width=width,
        synced=synced,
        log_scales=reference_logs,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        single_best_accuracy=single_best_accuracy,
        scale_source="reference_raw",
        synchronization_disagreement=sync_disagreement,
        extra={"selected_alpha": 1.0, "selected_tau": float("inf")},
    )

    alpha_grid = parse_csv(args.alpha_grid, float)
    tau_grid = [float("inf") if item.lower() in {"inf", "infinity"} else float(item) for item in parse_csv(args.tau_grid, str)]
    shrink_logs, shrink_model, shrink_val, shrink_alpha, shrink_tau = select_scale_grid(
        models=models,
        spec=spec,
        width=width,
        synced=synced,
        raw_log_scales=reference_logs,
        alpha_grid=alpha_grid,
        tau_grid=tau_grid,
        val_loader=val_loader,
        device=device,
    )
    shrink_test = evaluate_model(shrink_model, test_loader, device)
    shrink_models = build_scaled_models(models, spec, width, synced, shrink_logs)
    shrink_diag = log_scale_diagnostics(shrink_logs, sync_disagreement)
    add_method_row(
        rows,
        base=base,
        method="shrinkage_monomial_scale",
        test_metrics=shrink_test,
        val_metrics=shrink_val,
        single_best_accuracy=single_best_accuracy,
        extra={
            "scale_source": "reference_shrinkage_validation_grid",
            "selected_alpha": shrink_alpha,
            "selected_tau": shrink_tau,
            "mean_abs_log_scale": shrink_diag.mean_abs_log_scale,
            "max_abs_log_scale": shrink_diag.max_abs_log_scale,
            "log_scale_variance": shrink_diag.log_scale_variance,
            "scale_synchronization_disagreement": shrink_diag.synchronization_disagreement,
        },
    )

    global_logs, global_model, global_val, global_alpha, global_tau = select_scale_grid(
        models=models,
        spec=spec,
        width=width,
        synced=synced,
        raw_log_scales=global_sync.log_scales,
        alpha_grid=alpha_grid,
        tau_grid=tau_grid,
        val_loader=val_loader,
        device=device,
    )
    global_test = evaluate_model(global_model, test_loader, device)
    global_models = build_scaled_models(models, spec, width, synced, global_logs)
    global_diag = log_scale_diagnostics(global_logs, global_sync.rms_residual)
    add_method_row(
        rows,
        base=base,
        method="global_monomial_scale",
        test_metrics=global_test,
        val_metrics=global_val,
        single_best_accuracy=single_best_accuracy,
        extra={
            "scale_source": "global_least_squares_shrinkage_validation_grid",
            "selected_alpha": global_alpha,
            "selected_tau": global_tau,
            "mean_abs_log_scale": global_diag.mean_abs_log_scale,
            "max_abs_log_scale": global_diag.max_abs_log_scale,
            "log_scale_variance": global_diag.log_scale_variance,
            "scale_synchronization_disagreement": global_diag.synchronization_disagreement,
        },
    )

    optimized_logs = optimize_log_scales_for_validation(
        models,
        spec,
        width,
        synced,
        global_logs,
        val_loader,
        device,
        ref=ref,
        steps=args.optimization_steps,
        lr=args.optimization_lr,
        bound=args.optimization_bound,
        l2=args.optimization_l2,
    )
    optimized_models, _optimized_model, optimized_val, optimized_test = add_scaled_method(
        rows,
        base=base,
        method="optimized_monomial_scale",
        models=models,
        spec=spec,
        width=width,
        synced=synced,
        log_scales=optimized_logs,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        single_best_accuracy=single_best_accuracy,
        scale_source="validation_loss_optimized_global_initialization",
        synchronization_disagreement=global_sync.rms_residual,
        extra={
            "selected_alpha": global_alpha,
            "selected_tau": global_tau,
            "optimization_steps": args.optimization_steps,
            "optimization_bound": args.optimization_bound,
            "optimization_l2": args.optimization_l2,
        },
    )

    previous_selected = choose_by_validation(
        {
            "c2m3_permutation": c2m3_val,
            "monomial_scale": monomial_val,
        }
    )
    previous_test = c2m3_test if previous_selected.selected == "c2m3_permutation" else monomial_test
    previous_val = c2m3_val if previous_selected.selected == "c2m3_permutation" else monomial_val
    previous_alternative_test = monomial_test if previous_selected.selected == "c2m3_permutation" else c2m3_test
    add_method_row(
        rows,
        base=base,
        method="validated_ladder_selector",
        test_metrics=previous_test,
        val_metrics=previous_val,
        single_best_accuracy=single_best_accuracy,
        extra={
            "selector_chose": previous_selected.selected,
            "selector_val_margin": previous_selected.margin_to_runner_up,
            "selector_val_accuracy_delta_monomial_minus_c2m3": monomial_val["accuracy"] - c2m3_val["accuracy"],
            "selector_val_loss_delta_monomial_minus_c2m3": monomial_val["loss"] - c2m3_val["loss"],
            "selector_test_accuracy_delta_monomial_minus_c2m3": monomial_test["accuracy"] - c2m3_test["accuracy"],
            "selector_chosen_test_better": bool(previous_test["accuracy"] > previous_alternative_test["accuracy"]),
            "selector_chosen_test_tied": bool(previous_test["accuracy"] == previous_alternative_test["accuracy"]),
            "selector_chosen_test_worse": bool(previous_test["accuracy"] < previous_alternative_test["accuracy"]),
            "selector_behavior_reference": "c2m3_or_monomial_alternative",
        },
    )

    original_soup = add_soup_row(
        rows,
        base=base,
        method="greedy_soup",
        candidate_models=models,
        candidate_labels=[f"original:{idx}" for idx in range(n_models)],
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        spec=spec,
        width=width,
        single_best_accuracy=single_best_accuracy,
    )
    c2m3_soup = add_soup_row(
        rows,
        base=base,
        method="c2m3_greedy_soup",
        candidate_models=aligned_c2m3,
        candidate_labels=[f"c2m3:{idx}" for idx in range(n_models)],
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        spec=spec,
        width=width,
        single_best_accuracy=single_best_accuracy,
    )
    raw_soup = add_soup_row(
        rows,
        base=base,
        method="monomial_scaled_greedy_soup",
        candidate_models=raw_monomial_models,
        candidate_labels=[f"monomial:{idx}" for idx in range(n_models)],
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        spec=spec,
        width=width,
        single_best_accuracy=single_best_accuracy,
    )
    shrink_soup = add_soup_row(
        rows,
        base=base,
        method="shrinkage_monomial_greedy_soup",
        candidate_models=shrink_models,
        candidate_labels=[f"shrinkage:{idx}" for idx in range(n_models)],
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        spec=spec,
        width=width,
        single_best_accuracy=single_best_accuracy,
        extra={"selected_alpha": shrink_alpha, "selected_tau": shrink_tau},
    )
    global_soup = add_soup_row(
        rows,
        base=base,
        method="global_monomial_greedy_soup",
        candidate_models=global_models,
        candidate_labels=[f"global:{idx}" for idx in range(n_models)],
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        spec=spec,
        width=width,
        single_best_accuracy=single_best_accuracy,
        extra={"selected_alpha": global_alpha, "selected_tau": global_tau},
    )
    optimized_soup = add_soup_row(
        rows,
        base=base,
        method="optimized_monomial_greedy_soup",
        candidate_models=optimized_models,
        candidate_labels=[f"optimized:{idx}" for idx in range(n_models)],
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        spec=spec,
        width=width,
        single_best_accuracy=single_best_accuracy,
    )
    union_models = [*models, *aligned_c2m3, *shrink_models, *global_models]
    union_labels = (
        [f"original:{idx}" for idx in range(n_models)]
        + [f"c2m3:{idx}" for idx in range(n_models)]
        + [f"shrinkage:{idx}" for idx in range(n_models)]
        + [f"global:{idx}" for idx in range(n_models)]
    )
    union_soup = add_soup_row(
        rows,
        base=base,
        method="union_candidate_soup",
        candidate_models=union_models,
        candidate_labels=union_labels,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        spec=spec,
        width=width,
        single_best_accuracy=single_best_accuracy,
        extra={"union_candidate_count": len(union_models)},
    )

    ensemble_test = evaluate_ensemble(models, test_loader, device)
    ensemble_val = evaluate_ensemble(models, val_loader, device)
    add_method_row(rows, base=base, method="ensemble_upper_bound", test_metrics=ensemble_test, val_metrics=ensemble_val, single_best_accuracy=single_best_accuracy)

    by_method = {row["method"]: row for row in rows}
    selector_pool = [
        "c2m3_permutation",
        "monomial_scale",
        "shrinkage_monomial_scale",
        "global_monomial_scale",
        "optimized_monomial_scale",
        "greedy_soup",
        "c2m3_greedy_soup",
        "monomial_scaled_greedy_soup",
        "shrinkage_monomial_greedy_soup",
        "global_monomial_greedy_soup",
        "optimized_monomial_greedy_soup",
        "union_candidate_soup",
    ]
    selector_choice = choose_by_validation(
        {name: {"accuracy": by_method[name]["val_accuracy"], "loss": by_method[name]["val_loss"]} for name in selector_pool},
        allowed_methods=selector_pool,
    )
    selected_row = by_method[selector_choice.selected]
    greedy_acc = by_method["greedy_soup"]["accuracy"]
    greedy_loss = by_method["greedy_soup"]["loss"]
    c2m3_acc = by_method["c2m3_permutation"]["accuracy"]
    c2m3_loss = by_method["c2m3_permutation"]["loss"]
    add_method_row(
        rows,
        base=base,
        method="improved_validated_selector",
        test_metrics={"accuracy": selected_row["accuracy"], "loss": selected_row["loss"]},
        val_metrics={"accuracy": selected_row["val_accuracy"], "loss": selected_row["val_loss"]},
        single_best_accuracy=single_best_accuracy,
        extra={
            "selector_chose": selector_choice.selected,
            "selector_val_margin": selector_choice.margin_to_runner_up,
            "selector_chosen_test_better_than_greedy": bool(selected_row["accuracy"] > greedy_acc),
            "selector_chosen_test_tied_with_greedy": bool(selected_row["accuracy"] == greedy_acc),
            "selector_chosen_test_better_than_c2m3": bool(selected_row["accuracy"] > c2m3_acc),
            "selector_chosen_test_better": bool(selected_row["accuracy"] > greedy_acc),
            "selector_chosen_test_tied": bool(selected_row["accuracy"] == greedy_acc),
            "selector_chosen_test_worse": bool(selected_row["accuracy"] < greedy_acc),
            "selector_behavior_reference": "greedy_soup",
            "selector_pool": json.dumps(selector_pool),
        },
    )

    by_method = {row["method"]: row for row in rows}
    c2m3_acc = by_method["c2m3_permutation"]["accuracy"]
    c2m3_loss = by_method["c2m3_permutation"]["loss"]
    greedy_acc = by_method["greedy_soup"]["accuracy"]
    greedy_loss = by_method["greedy_soup"]["loss"]
    for row in rows:
        row["accuracy_delta_vs_c2m3"] = row["accuracy"] - c2m3_acc
        row["loss_delta_vs_c2m3"] = row["loss"] - c2m3_loss
        row["accuracy_delta_vs_greedy_soup"] = row["accuracy"] - greedy_acc
        row["loss_delta_vs_greedy_soup"] = row["loss"] - greedy_loss
        row["validation_delta_vs_c2m3"] = row["val_accuracy"] - by_method["c2m3_permutation"]["val_accuracy"]
        row["validation_delta_vs_greedy_soup"] = row["val_accuracy"] - by_method["greedy_soup"]["val_accuracy"]
    return rows


def summarize_methods(df: pd.DataFrame) -> list[dict]:
    rows = []
    for scope, keys in [("overall", ["method"]), ("fixed_setting", ["n_models", "width", "method"])]:
        for key_values, group in df.groupby(keys, dropna=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            key_map = dict(zip(keys, key_values, strict=True))
            rows.append(
                {
                    "summary_type": "method_summary",
                    "scope": scope,
                    "n_models": key_map.get("n_models", "all"),
                    "width": key_map.get("width", "all"),
                    "method": key_map["method"],
                    "baseline": "",
                    "n_rows": int(len(group)),
                    "n_seeds": int(group["seed"].nunique()),
                    "mean_accuracy": float(pd.to_numeric(group["accuracy"], errors="coerce").mean()),
                    "accuracy_standard_error": standard_error(group["accuracy"]),
                    "mean_loss": float(pd.to_numeric(group["loss"], errors="coerce").mean()),
                    "mean_merge_degradation": float(pd.to_numeric(group["merge_degradation"], errors="coerce").mean()),
                    "mean_accuracy_delta_vs_c2m3": float(pd.to_numeric(group["accuracy_delta_vs_c2m3"], errors="coerce").mean()),
                    "mean_accuracy_delta_vs_greedy_soup": float(pd.to_numeric(group["accuracy_delta_vs_greedy_soup"], errors="coerce").mean()),
                    "symmetry_status": str(group["symmetry_status"].iloc[0]),
                    "is_single_model": bool(group["is_single_model"].iloc[0]),
                    "capacity_matched_to_weight_average": bool(group["capacity_matched_to_weight_average"].iloc[0]),
                }
            )
    return rows


def paired_rows(df: pd.DataFrame, n_bootstrap: int) -> list[dict]:
    rows = []
    fixed = df.pivot_table(
        index=["n_models", "width", "seed", "setting_id"],
        columns="method",
        values=["accuracy", "loss"],
        aggfunc="first",
    )
    fixed.columns = [f"{metric}__{method}" for metric, method in fixed.columns]
    fixed = fixed.reset_index()
    for comparison in PAIR_COMPARISONS:
        acc_method = f"accuracy__{comparison.method}"
        acc_base = f"accuracy__{comparison.baseline}"
        loss_method = f"loss__{comparison.method}"
        loss_base = f"loss__{comparison.baseline}"
        clean = fixed[[acc_method, acc_base, loss_method, loss_base, "n_models", "width"]].dropna()
        accuracy_delta = clean[acc_method] - clean[acc_base]
        loss_delta = clean[loss_method] - clean[loss_base]
        wins = int((accuracy_delta > 0).sum())
        ties = int((accuracy_delta == 0).sum())
        losses = int((accuracy_delta < 0).sum())
        ci_low, ci_high = bootstrap_mean_ci(accuracy_delta, n_bootstrap=n_bootstrap, seed=9910 + len(rows))
        fixed_positive = 0
        fixed_total = 0
        for (_n, _w), group in clean.groupby(["n_models", "width"], dropna=False):
            fixed_total += 1
            if float((group[acc_method] - group[acc_base]).mean()) > 0:
                fixed_positive += 1
        rows.append(
            {
                "summary_type": "paired_comparison",
                "scope": "overall_paired",
                "comparison": comparison.name,
                "claim_label": comparison.claim_label,
                "method": comparison.method,
                "baseline": comparison.baseline,
                "n_pairs": int(len(clean)),
                "paired_mean_accuracy_delta": float(accuracy_delta.mean()) if len(clean) else float("nan"),
                "paired_accuracy_delta_ci_low": ci_low,
                "paired_accuracy_delta_ci_high": ci_high,
                "paired_mean_loss_delta": float(loss_delta.mean()) if len(clean) else float("nan"),
                "accuracy_wins": wins,
                "accuracy_ties": ties,
                "accuracy_losses": losses,
                "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
                "fixed_settings_positive": fixed_positive,
                "fixed_settings_total": fixed_total,
                "selector_no_test_leakage": True,
            }
        )
    return rows


def selector_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for method in ["validated_ladder_selector", "improved_validated_selector"]:
        selector = df[df["method"] == method].copy()
        for scope, group in [("overall", selector), *[(f"N{n}_W{w}", g) for (n, w), g in selector.groupby(["n_models", "width"])]]:
            choices = group["selector_chose"].value_counts(dropna=False)
            better = group["selector_chosen_test_better"].fillna(False).astype(bool)
            tied = group["selector_chosen_test_tied"].fillna(False).astype(bool)
            worse = group["selector_chosen_test_worse"].fillna(False).astype(bool)
            rows.append(
                {
                    "summary_type": "selector_behavior",
                    "scope": scope,
                    "method": method,
                    "n_rows": int(len(group)),
                    "selector_choice_counts": json.dumps({str(k): int(v) for k, v in choices.items()}),
                    "selector_behavior_reference": str(group["selector_behavior_reference"].dropna().iloc[0]) if group["selector_behavior_reference"].notna().any() else "",
                    "selector_chosen_test_better": int(better.sum()),
                    "selector_chosen_test_tied": int(tied.sum()),
                    "selector_chosen_test_worse": int(worse.sum()),
                    "selector_no_test_leakage": bool(group["selector_no_test_leakage"].fillna(True).astype(bool).all()),
                }
            )
    return rows


def alpha_tau_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    subset = df[df["method"].isin(["shrinkage_monomial_scale", "global_monomial_scale", "optimized_monomial_scale"])].copy()
    for method, group in subset.groupby("method"):
        alpha = pd.to_numeric(group["selected_alpha"], errors="coerce")
        tau = pd.to_numeric(group["selected_tau"].replace(float("inf"), np.nan), errors="coerce")
        rows.append(
            {
                "summary_type": "alpha_tau_selection",
                "scope": "overall",
                "method": method,
                "n_rows": int(len(group)),
                "mean_selected_alpha": float(alpha.mean()),
                "fraction_alpha_zero": float((alpha == 0.0).mean()),
                "fraction_alpha_one_or_more": float((alpha >= 1.0).mean()),
                "mean_finite_selected_tau": float(tau.mean()),
                "fraction_tau_infinite": float(np.isinf(pd.to_numeric(group["selected_tau"], errors="coerce")).mean()),
            }
        )
    return rows


def soup_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    soup_methods = [method for method in df["method"].unique() if "soup" in str(method)]
    for method in sorted(soup_methods):
        group = df[df["method"] == method].copy()
        rows.append(
            {
                "summary_type": "soup_ingredient_behavior",
                "scope": "overall",
                "method": method,
                "n_rows": int(len(group)),
                "soup_mean_ingredient_count": float(pd.to_numeric(group["soup_ingredient_count"], errors="coerce").mean()),
                "selected_type_examples": "; ".join(str(item) for item in group["soup_selected_types"].dropna().head(3).tolist()),
                "capacity_matched_to_weight_average": bool(group["capacity_matched_to_weight_average"].fillna(True).astype(bool).all()),
            }
        )
    return rows


def residual_correlation_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    pivot = df.pivot_table(index=["n_models", "width", "seed", "setting_id"], columns="method", values="accuracy", aggfunc="first").reset_index()
    base = df[df["method"] == "monomial_scale"].drop_duplicates("setting_id")
    merged = base.merge(pivot, on=["n_models", "width", "seed", "setting_id"], suffixes=("", "__pivot"))
    predictors = [
        "monomial_centrality_improvement_from_permutation",
        "log_scale_variance",
        "mean_abs_log_scale",
        "max_abs_log_scale",
        "validation_delta_vs_c2m3",
        "scale_synchronization_disagreement",
        "pairwise_alignment_residual",
        "sync_disagreement",
        "individual_accuracy_variance",
    ]
    target_defs = {
        "monomial_gain_vs_c2m3": merged["monomial_scale"] - merged["c2m3_permutation"],
        "shrinkage_gain_vs_monomial": merged["shrinkage_monomial_scale"] - merged["monomial_scale"],
        "global_gain_vs_monomial": merged["global_monomial_scale"] - merged["monomial_scale"],
        "optimized_gain_vs_monomial": merged["optimized_monomial_scale"] - merged["monomial_scale"],
        "union_soup_gain_vs_greedy": merged["union_candidate_soup"] - merged["greedy_soup"],
    }
    for target_name, target in target_defs.items():
        for predictor in predictors:
            rows.append(
                {
                    "summary_type": "residual_diagnostic_correlation",
                    "scope": "overall",
                    "target": target_name,
                    "predictor": predictor,
                    "n_rows": int(len(merged)),
                    "pearson": safe_corr(merged[predictor], target, method="pearson"),
                    "spearman": safe_corr(merged[predictor], target, method="spearman"),
                }
            )
    union = df[df["method"] == "union_candidate_soup"].drop_duplicates("setting_id")
    if not union.empty:
        rows.append(
            {
                "summary_type": "residual_diagnostic_correlation",
                "scope": "overall",
                "target": "union_soup_gain_vs_greedy",
                "predictor": "soup_ingredient_count",
                "n_rows": int(len(union)),
                "pearson": safe_corr(union["soup_ingredient_count"], union["accuracy_delta_vs_greedy_soup"], method="pearson"),
                "spearman": safe_corr(union["soup_ingredient_count"], union["accuracy_delta_vs_greedy_soup"], method="spearman"),
            }
        )
    return rows


def claim_decision_rows(summary_rows: list[dict]) -> list[dict]:
    paired = {row["comparison"]: row for row in summary_rows if row.get("summary_type") == "paired_comparison"}
    method_summary = [row for row in summary_rows if row.get("summary_type") == "method_summary" and row.get("scope") == "overall"]
    n_seeds = int(max((row.get("n_seeds", 0) for row in method_summary), default=0))
    rows = []

    def label(comparison: str, supported_label: str, descriptive_label: str, negative_label: str) -> tuple[str, str]:
        row = paired[comparison]
        mean_delta = float(row["paired_mean_accuracy_delta"])
        ci_low = float(row["paired_accuracy_delta_ci_low"])
        fixed_positive = int(row["fixed_settings_positive"])
        fixed_total = int(row["fixed_settings_total"])
        if mean_delta > 0.0 and np.isfinite(ci_low) and ci_low > 0.0 and fixed_positive > fixed_total / 2 and n_seeds >= 20:
            return supported_label, "paired validation-clean statistics have positive mean and positive bootstrap CI"
        if mean_delta > 0.0:
            return descriptive_label, "mean delta is positive but confidence interval or fixed-setting support is insufficient"
        return negative_label, "paired mean accuracy delta is not positive"

    decisions = [
        ("improved_selector_over_c2m3", *label("improved_validated_selector_vs_c2m3_permutation", "Supported limited", "Supported descriptive", "Supported negative")),
        ("improved_selector_over_greedy_soup", *label("improved_validated_selector_vs_greedy_soup", "Supported limited", "Supported descriptive", "Supported negative")),
        ("shrinkage_over_raw_monomial", *label("shrinkage_monomial_scale_vs_monomial_scale", "Supported limited", "Supported descriptive", "Supported negative")),
        ("global_over_raw_monomial", *label("global_monomial_scale_vs_monomial_scale", "Supported limited", "Supported descriptive", "Supported negative")),
        ("optimized_over_raw_monomial", *label("optimized_monomial_scale_vs_monomial_scale", "Supported limited", "Supported descriptive", "Supported negative")),
        ("union_soup_over_greedy_soup", *label("union_candidate_soup_vs_greedy_soup", "Supported limited", "Supported descriptive", "Supported negative")),
        ("monomial_soup_over_greedy_soup", *label("monomial_scaled_greedy_soup_vs_greedy_soup", "Supported limited", "Supported descriptive", "Supported negative")),
    ]
    for claim, decision, reason in decisions:
        rows.append(
            {
                "summary_type": "claim_decision",
                "scope": "overall",
                "claim": claim,
                "claim_decision": decision,
                "claim_reason": reason,
                "n_seeds": n_seeds,
            }
        )
    return rows


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(summarize_methods(df))
    rows.extend(paired_rows(df, n_bootstrap))
    rows.extend(selector_rows(df))
    rows.extend(alpha_tau_rows(df))
    rows.extend(soup_rows(df))
    rows.extend(residual_correlation_rows(df))
    rows.extend(claim_decision_rows(rows))
    return pd.DataFrame(rows)


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if value == "":
                values.append("")
            elif isinstance(value, bool):
                values.append(str(value))
            elif pd.isna(value):
                values.append("nan")
            elif col in INT_COLUMNS:
                values.append(str(int(round(float(value)))))
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_delta_plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    methods = ["improved_validated_selector", "union_candidate_soup", "shrinkage_monomial_scale", "global_monomial_scale", "optimized_monomial_scale"]
    for method in methods:
        group = df[df["method"] == method]
        if group.empty:
            continue
        ax.scatter(group["val_accuracy"], group["accuracy_delta_vs_greedy_soup"], label=method, alpha=0.68, s=22)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Validation accuracy")
    ax.set_ylabel("Test accuracy delta vs greedy soup")
    ax.set_title("Improved Validated Ladder: Delta vs Greedy Soup")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_alpha_plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = df[df["method"].isin(["shrinkage_monomial_scale", "global_monomial_scale", "optimized_monomial_scale"])].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method, group in subset.groupby("method"):
        ax.hist(pd.to_numeric(group["selected_alpha"], errors="coerce").dropna(), bins=np.linspace(0, 1.5, 16), alpha=0.5, label=method)
    ax.set_xlabel("Validation-selected alpha")
    ax.set_ylabel("Count")
    ax.set_title("Scale Shrinkage Alpha Selection")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, delta_plot: Path, alpha_plot: Path, path: Path) -> None:
    method_list = [
        "weight_average",
        "c2m3_permutation",
        "monomial_scale",
        "shrinkage_monomial_scale",
        "global_monomial_scale",
        "optimized_monomial_scale",
        "validated_ladder_selector",
        "improved_validated_selector",
        "greedy_soup",
        "c2m3_greedy_soup",
        "monomial_scaled_greedy_soup",
        "shrinkage_monomial_greedy_soup",
        "global_monomial_greedy_soup",
        "optimized_monomial_greedy_soup",
        "union_candidate_soup",
        "ensemble_upper_bound",
    ]
    method_rows = [
        {
            "method": method,
            "symmetry_status": METHODS[method].symmetry_status,
            "is_single_model": METHODS[method].is_single_model,
            "capacity_matched": METHODS[method].capacity_matched_to_weight_average,
        }
        for method in method_list
    ]
    overall = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "overall")].to_dict("records")
    fixed = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "fixed_setting")].to_dict("records")
    paired = summary[summary["summary_type"] == "paired_comparison"].to_dict("records")
    selectors = summary[summary["summary_type"] == "selector_behavior"].to_dict("records")
    alpha_tau = summary[summary["summary_type"] == "alpha_tau_selection"].to_dict("records")
    soup = summary[summary["summary_type"] == "soup_ingredient_behavior"].to_dict("records")
    residual = summary[summary["summary_type"] == "residual_diagnostic_correlation"].head(24).to_dict("records")
    claims = summary[summary["summary_type"] == "claim_decision"].to_dict("records")
    claim_text = "\n".join(f"- `{row['claim']}`: {row['claim_decision']} ({row['claim_reason']})." for row in claims)
    report = f"""# Improved Validated Ladder Merge Report

This report is generated by `experiments/improved_validated_ladder_merge_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State At Report Generation

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_worktree_dirty()}`

## Dataset And Grid

- Dataset: MNIST
- Architecture: one-hidden-layer ReLU MLP
- Model counts: `{args.model_counts}`
- Widths: `{args.widths}`
- Seeds: `{args.seeds}`
- Epochs: `{args.epochs}`
- Train samples before validation split: `{args.max_train_samples}`
- Validation fraction: `{args.val_fraction}`
- Test samples: `{args.max_test_samples}` (`0` means full dataset)
- Alpha grid: `{args.alpha_grid}`
- Tau grid: `{args.tau_grid}`
- Optimization steps: `{args.optimization_steps}`
- Matching: activation

All method choices are made using validation accuracy/loss only. Test metrics are computed after selection.

## Method Labels

{table(method_rows, ["method", "symmetry_status", "is_single_model", "capacity_matched"])}

## Main Performance Table

{table(overall, ["method", "n_rows", "n_seeds", "mean_accuracy", "accuracy_standard_error", "mean_loss", "mean_merge_degradation", "mean_accuracy_delta_vs_c2m3", "mean_accuracy_delta_vs_greedy_soup", "symmetry_status"])}

## Fixed-Setting Performance

{table(fixed, ["n_models", "width", "method", "n_seeds", "mean_accuracy", "mean_accuracy_delta_vs_c2m3", "mean_accuracy_delta_vs_greedy_soup"])}

## Paired Comparisons

{table(paired, ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "paired_mean_loss_delta", "accuracy_wins", "accuracy_ties", "accuracy_losses", "sign_test_two_sided_p", "fixed_settings_positive", "fixed_settings_total"])}

## Selector Behavior

{table(selectors, ["method", "scope", "n_rows", "selector_choice_counts", "selector_behavior_reference", "selector_chosen_test_better", "selector_chosen_test_tied", "selector_chosen_test_worse", "selector_no_test_leakage"])}

The improved selector is a validation-only regret-minimizing selector over C2M3,
raw monomial, shrinkage monomial, global monomial, optimized monomial, greedy
soup variants, and union candidate soup.

## Alpha And Tau Selection

{table(alpha_tau, ["method", "n_rows", "mean_selected_alpha", "fraction_alpha_zero", "fraction_alpha_one_or_more", "mean_finite_selected_tau", "fraction_tau_infinite"])}

Plot: `reports/plots/{alpha_plot.name}`.

## Soup Ingredient Behavior

{table(soup, ["method", "n_rows", "soup_mean_ingredient_count", "selected_type_examples", "capacity_matched_to_weight_average"])}

Union candidate soup is still one averaged MLP with the original architecture and parameter count; it is not an ensemble.

## Residual Diagnostic Correlations

{table(residual, ["target", "predictor", "n_rows", "pearson", "spearman"])}

Plot: `reports/plots/{delta_plot.name}`.

## Claim Decision Table

{claim_text}

## Negative Boundaries

- This is an MNIST one-hidden-layer ReLU MLP benchmark, not a broad external-baseline claim.
- This does not compare against external Git Re-Basin or external C2M3 code.
- This experiment is about exact positive monomial ReLU gauges, not Brauer/projective residuals.
- The ensemble remains extra capacity and is not used for single-model claims.
- A validation-selected soup remains one averaged model, but it is selected from a larger candidate pool; the report records that candidate pool explicitly.
- No method choice uses test-set metrics.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_config(args, path: Path) -> None:
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "dirty_worktree": git_worktree_dirty(),
        "dataset": "mnist",
        "model_counts": parse_csv(args.model_counts, int),
        "widths": parse_csv(args.widths, int),
        "seeds": parse_csv(args.seeds, int),
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "alpha_grid": parse_csv(args.alpha_grid, float),
        "tau_grid": parse_csv(args.tau_grid, str),
        "optimization_steps": args.optimization_steps,
        "optimization_lr": args.optimization_lr,
        "optimization_bound": args.optimization_bound,
        "optimization_l2": args.optimization_l2,
        "environment": capture_environment(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(1800, 1820)))
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="16,32,64")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--alpha-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5")
    parser.add_argument("--tau-grid", default="0.25,0.5,1.0,2.0,inf")
    parser.add_argument("--optimization-steps", type=int, default=30)
    parser.add_argument("--optimization-lr", type=float, default=0.05)
    parser.add_argument("--optimization-bound", type=float, default=1.0)
    parser.add_argument("--optimization-l2", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    spec, train_data, test_data = load_dataset(
        "mnist",
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
    )
    rows = []
    for seed in parse_csv(args.seeds, int):
        for n_models in parse_csv(args.model_counts, int):
            for width in parse_csv(args.widths, int):
                print(f"running seed={seed} n_models={n_models} width={width}", flush=True)
                rows.extend(run_setting(args, spec, train_data, test_data, seed, n_models, width))

    df = pd.DataFrame(rows)
    summary = summarize(df, args.bootstrap_samples)
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "improved_validated_ladder_merge_benchmark.csv"
    summary_path = csv_dir / "improved_validated_ladder_merge_summary.csv"
    delta_plot = plot_dir / "improved_validated_ladder_delta_vs_greedy.pdf"
    alpha_plot = plot_dir / "scale_shrinkage_alpha_selection.pdf"
    report_path = args.reports_dir / "improved_validated_ladder_merge_report.md"
    config_path = config_dir / "improved_validated_ladder_merge_config.json"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_delta_plot(df, delta_plot)
    write_alpha_plot(df, alpha_plot)
    write_report(args, df, summary, delta_plot, alpha_plot, report_path)
    write_config(args, config_path)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {delta_plot}")
    print(f"wrote {alpha_plot}")
    print(f"wrote {report_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()

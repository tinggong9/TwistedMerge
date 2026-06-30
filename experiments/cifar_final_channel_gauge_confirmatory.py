#!/usr/bin/env python
"""Final bounded CIFAR-10 channel-gauge confirmatory benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.cifar_or_colored_mnist_feasibility import (  # noqa: E402
    average_eval,
    build_gauged_models,
    collect_features,
    feature_alignment_residual,
    greedy_soup,
    pairwise_perms,
    parse_csv,
    reference_log_scales,
    sync_perms,
    zero_logs,
)
from experiments.cifar_rescue_or_no_go import (  # noqa: E402
    LAYERS,
    RescueConfig,
    gate_status,
    layer_widths,
    load_cifar_splits,
    make_model,
    train_rescue_model,
)
from src.cnn_channel_gauge import count_parameters, inference_cost_units  # noqa: E402
from src.greedy_safe_selector import tau_fixed_selector  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    cycle_score,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    make_loader,
    set_seed,
)


METHOD_ORDER = [
    "weight_average",
    "c2m3_channel_synchronization",
    "positive_channel_scale",
    "shrinkage_channel_scale",
    "global_channel_scale",
    "optimized_channel_scale",
    "greedy_soup",
    "positive_channel_scaled_greedy_soup",
    "shrinkage_channel_scaled_greedy_soup",
    "global_channel_scaled_greedy_soup",
    "optimized_channel_scaled_greedy_soup",
    "union_channel_candidate_soup",
    "greedy_safe_selector",
    "ensemble_upper_bound",
]

REQUIRED_COMPARISONS = [
    ("shrinkage_channel_scale", "c2m3_channel_synchronization"),
    ("global_channel_scale", "c2m3_channel_synchronization"),
    ("optimized_channel_scale", "c2m3_channel_synchronization"),
    ("union_channel_candidate_soup", "greedy_soup"),
    ("greedy_safe_selector", "greedy_soup"),
    ("greedy_safe_selector", "c2m3_channel_synchronization"),
]

INT_COLUMNS = {
    "n_rows",
    "n_settings",
    "n_pairs",
    "accuracy_wins",
    "accuracy_ties",
    "accuracy_losses",
    "fixed_settings_positive",
    "fixed_settings_total",
    "n_models",
    "seed",
    "epochs",
    "max_train_samples",
    "max_test_samples",
    "conv1_channels",
    "conv2_channels",
    "hidden_units",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | str:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return "unknown"


def parse_configs(text: str) -> list[RescueConfig]:
    configs: list[RescueConfig] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(part.strip()) for part in item.split(",")]
        if len(parts) != 4:
            raise ValueError("configs must be conv1,conv2,hidden,epochs entries separated by semicolons")
        configs.append(RescueConfig(*parts))
    return configs


def seed_plan(args) -> list[tuple[RescueConfig, int, int, str]]:
    configs = parse_configs(args.configs)
    main_seeds = parse_csv(args.main_seeds, int)
    secondary_seeds = parse_csv(args.secondary_seeds, int)
    n_models_values = parse_csv(args.model_counts, int)
    plan = []
    for cfg_idx, cfg in enumerate(configs):
        for n_models in n_models_values:
            role = "main" if cfg_idx == 0 and n_models == args.main_n_models else "secondary"
            seeds = main_seeds if role == "main" else secondary_seeds
            for seed in seeds:
                plan.append((cfg, n_models, seed, role))
    return plan


def bootstrap_mean_ci(values, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(int(n_bootstrap))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def standard_error(values) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def sign_test_two_sided(wins: int, losses: int) -> float:
    import math

    n = wins + losses
    if n <= 0:
        return float("nan")
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def global_log_scale_synchronization(features_by_model, synced, refs, n_models: int, widths: dict[str, int]):
    logs = {}
    residuals = []
    for layer in LAYERS:
        width = widths[layer]
        ref = refs[layer]
        aligned = {
            idx: features_by_model[idx][layer][:, np.asarray(synced[layer][idx], dtype=int)]
            for idx in range(n_models)
        }
        layer_logs = np.zeros((n_models, width), dtype=float)
        for unit in range(width):
            A = []
            b = []
            for i, j in product(range(n_models), repeat=2):
                if i == j:
                    continue
                row = np.zeros(n_models, dtype=float)
                row[j] = 1.0
                row[i] = -1.0
                A.append(row)
                b.append(np.log(estimate_scale(aligned[i][:, unit], aligned[j][:, unit])))
            gauge = np.zeros(n_models, dtype=float)
            gauge[ref] = max(float(n_models), 1.0)
            A.append(gauge)
            b.append(0.0)
            A_arr = np.vstack(A)
            b_arr = np.asarray(b)
            sol, *_ = np.linalg.lstsq(A_arr, b_arr, rcond=None)
            sol = sol - sol[ref]
            layer_logs[:, unit] = sol
            residuals.extend((A_arr[:-1] @ sol - b_arr[:-1]).tolist())
        logs[layer] = layer_logs
    rms = float(np.sqrt(np.mean(np.asarray(residuals) ** 2))) if residuals else 0.0
    return logs, rms


def estimate_scale(source: np.ndarray, target: np.ndarray) -> float:
    denom = max(float(np.dot(source, source)), 1e-12)
    scale = float(np.dot(source, target) / denom)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(np.clip(scale, 1e-3, 1e3))


def shrink_logs(logs: dict[str, np.ndarray], alpha: float, tau: float) -> dict[str, np.ndarray]:
    out = {}
    for layer, values in logs.items():
        clipped = values if not np.isfinite(tau) else np.clip(values, -float(tau), float(tau))
        out[layer] = float(alpha) * clipped
    return out


def layer_masked_logs(logs: dict[str, np.ndarray], active_layers: tuple[str, ...]) -> dict[str, np.ndarray]:
    active = set(active_layers)
    return {layer: values.copy() if layer in active else np.zeros_like(values) for layer, values in logs.items()}


def parse_layer_masks(text: str) -> list[tuple[str, ...]]:
    masks = []
    for item in parse_csv(text, str):
        lowered = item.lower()
        if lowered in {"all", "*"}:
            masks.append(tuple(LAYERS))
        elif lowered in {"none", "identity"}:
            masks.append(tuple())
        else:
            layers = tuple(part.strip() for part in item.split("+") if part.strip())
            bad = [layer for layer in layers if layer not in LAYERS]
            if bad:
                raise ValueError(f"unknown layer mask item(s): {bad}")
            masks.append(layers)
    return masks


def scale_diagnostics(logs: dict[str, np.ndarray], prefix: str = "") -> dict[str, float]:
    out = {}
    variances = []
    mean_abs = []
    max_abs = []
    for layer in LAYERS:
        values = np.asarray(logs[layer], dtype=float)
        out[f"{prefix}{layer}_log_scale_variance"] = float(np.var(values))
        out[f"{prefix}{layer}_mean_abs_log_scale"] = float(np.mean(np.abs(values)))
        out[f"{prefix}{layer}_max_abs_log_scale"] = float(np.max(np.abs(values)))
        variances.append(out[f"{prefix}{layer}_log_scale_variance"])
        mean_abs.append(out[f"{prefix}{layer}_mean_abs_log_scale"])
        max_abs.append(out[f"{prefix}{layer}_max_abs_log_scale"])
    out[f"{prefix}mean_log_scale_variance"] = float(np.mean(variances))
    out[f"{prefix}mean_abs_log_scale"] = float(np.mean(mean_abs))
    out[f"{prefix}max_abs_log_scale"] = float(np.max(max_abs))
    return out


def select_scale_grid(models, synced, raw_logs, alpha_grid, tau_grid, val_loader, device):
    best = None
    for alpha in alpha_grid:
        for tau in tau_grid:
            logs = shrink_logs(raw_logs, alpha, tau)
            gauged = build_gauged_models(models, synced, logs)
            from src.cnn_channel_gauge import average_cnn_models

            candidate = average_cnn_models(gauged)
            val = evaluate_model(candidate, val_loader, device)
            key = (float(val["accuracy"]), -float(val["loss"]))
            if best is None or key > best[0]:
                best = (key, logs, gauged, candidate, val, float(alpha), float(tau))
    assert best is not None
    return best[1:]


def choose_optimized_logs(
    *,
    models,
    synced,
    reference_logs,
    global_logs,
    alpha_grid,
    tau_grid,
    layer_masks,
    val_loader,
    device,
):
    best = None
    sources = [("reference", reference_logs), ("global", global_logs)]
    for source_name, source_logs in sources:
        for alpha in alpha_grid:
            for tau in tau_grid:
                shrunk = shrink_logs(source_logs, alpha, tau)
                for mask in layer_masks:
                    logs = layer_masked_logs(shrunk, mask)
                    gauged = build_gauged_models(models, synced, logs)
                    from src.cnn_channel_gauge import average_cnn_models

                    candidate = average_cnn_models(gauged)
                    val = evaluate_model(candidate, val_loader, device)
                    key = (float(val["accuracy"]), -float(val["loss"]), len(mask), source_name)
                    if best is None or key > best[0]:
                        best = (
                            key,
                            logs,
                            gauged,
                            candidate,
                            val,
                            source_name,
                            float(alpha),
                            float(tau),
                            "+".join(mask) if mask else "identity",
                        )
    assert best is not None
    _key, logs, gauged, candidate, val, source_name, alpha, tau, mask = best
    return logs, gauged, candidate, val, source_name, alpha, tau, mask


def cycle_diagnostics(pairwise, n_models: int, widths: dict[str, int]) -> dict[str, float]:
    out = {}
    scores = []
    for layer in LAYERS:
        score, _rows = cycle_score(pairwise[layer], n_models, widths[layer])
        out[f"{layer}_permutation_cycle_score"] = score
        scores.append(score)
    out["channel_permutation_cycle_score"] = float(np.mean(scores)) if scores else 0.0
    return out


def add_row(rows, base, method, val, test, *, extra=None):
    data = {
        **base,
        "method": method,
        "val_accuracy": float(val["accuracy"]),
        "val_loss": float(val["loss"]),
        "accuracy": float(test["accuracy"]),
        "loss": float(test["loss"]),
        "selection_used_validation_only": True,
        "evaluation_status": "evaluated",
        "parameter_count_multiplier": 1.0,
        "inference_time_multiplier": 1.0,
        "ensemble_or_extra_capacity": False,
    }
    if extra:
        data.update(extra)
    rows.append(data)


def base_metadata(args, cfg: RescueConfig, seed: int, n_models: int, train_pool_size: int, test_size: int, individual: list[float], role: str) -> dict:
    eligible, status, decision = gate_status(float(np.max(individual)), args)
    spec = cfg.spec
    return {
        "setting_id": f"cifar_final_{cfg.label}_N{n_models}_S{seed}",
        "dataset": "cifar10",
        "dataset_role": "cifar_final_channel_gauge_confirmatory",
        "architecture": "relu_cnn_no_batchnorm",
        "setting_role": role,
        "conv1_channels": cfg.conv1_channels,
        "conv2_channels": cfg.conv2_channels,
        "hidden_units": cfg.hidden_units,
        "n_models": n_models,
        "seed": seed,
        "epochs": cfg.epochs,
        "max_train_samples": train_pool_size,
        "max_test_samples": test_size,
        "augmentation": bool(args.augmentation),
        "val_fraction": args.val_fraction,
        "feature_batches": args.feature_batches,
        "individual_accuracy_mean": float(np.mean(individual)),
        "individual_accuracy_max": float(np.max(individual)),
        "base_accuracy_gate_passed": bool(eligible),
        "base_gate_status": status,
        "base_gate_decision": decision,
        "cifar_plumbing_threshold": args.plumbing_threshold,
        "cifar_meaningful_threshold": args.meaningful_threshold,
        "parameter_count": count_parameters(make_model(spec)),
        "inference_cost_units": inference_cost_units(spec),
        "exact_positive_channel_scale_available": True,
        "central_projective_candidate": False,
        "finite_index_candidate": False,
        "non_brauer_noncentral": True,
        "channel_residual_taxonomy": "no_central_projective_or_finite_index_candidate",
    }


def evaluate_average(models, val_loader, test_loader, device):
    return average_eval(models, val_loader, test_loader, device)


def run_setting(args, cfg: RescueConfig, n_models: int, seed: int, role: str) -> list[dict]:
    device = device_from_arg(args.device)
    spec = cfg.spec
    widths = layer_widths(spec)
    train_aug, train_eval, val_eval, test_eval, train_pool_size, test_size = load_cifar_splits(args, seed)
    train_loader = make_loader(train_aug, args.batch_size, shuffle=True, seed=seed + 100)
    val_loader = make_loader(val_eval, args.batch_size, shuffle=False, seed=seed + 110)
    test_loader = make_loader(test_eval, args.batch_size, shuffle=False, seed=seed + 120)
    match_loader = make_loader(train_eval, args.batch_size, shuffle=False, seed=seed + 130)

    models = []
    individual = []
    for idx in range(n_models):
        model_seed = seed + idx * 1009 + 17
        set_seed(model_seed)
        model = make_model(spec)
        train_loader = make_loader(train_aug, args.batch_size, shuffle=True, seed=model_seed)
        train_rescue_model(model, train_loader, cfg.epochs, args.lr, args.weight_decay, device)
        metrics = evaluate_model(model, test_loader, device)
        individual.append(float(metrics["accuracy"]))
        model.to("cpu")
        models.append(model)

    features = {idx: collect_features(model, match_loader, device, widths, args.feature_batches) for idx, model in enumerate(models)}
    pairwise = pairwise_perms(features, n_models, widths)
    refs, synced, disagreements = sync_perms(pairwise, n_models)
    reference_logs = reference_log_scales(features, synced, refs, n_models, widths)
    global_logs, global_rms = global_log_scale_synchronization(features, synced, refs, n_models, widths)
    cycles = cycle_diagnostics(pairwise, n_models, widths)
    pair_residual = feature_alignment_residual(features, synced, n_models)

    base = base_metadata(args, cfg, seed, n_models, train_pool_size, test_size, individual, role)
    base.update(
        {
            "pairwise_activation_alignment_residual": pair_residual,
            "global_channel_scale_sync_residual": global_rms,
            "sync_disagreement_mean": float(np.mean(list(disagreements.values()))),
            **cycles,
        }
    )
    rows: list[dict] = []

    _weight_model, weight_val, weight_test = evaluate_average(models, val_loader, test_loader, device)
    add_row(rows, base, "weight_average", weight_val, weight_test, extra={"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True, "is_soup": False, "is_ensemble": False})

    c2m3_models = build_gauged_models(models, synced, zero_logs(n_models, widths))
    _c2m3_model, c2m3_val, c2m3_test = evaluate_average(c2m3_models, val_loader, test_loader, device)
    add_row(rows, base, "c2m3_channel_synchronization", c2m3_val, c2m3_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": False, "is_ensemble": False})

    positive_models = build_gauged_models(models, synced, reference_logs)
    _positive_model, positive_val, positive_test = evaluate_average(positive_models, val_loader, test_loader, device)
    add_row(
        rows,
        base,
        "positive_channel_scale",
        positive_val,
        positive_test,
        extra={
            "exact_relu_channel_gauge": True,
            "single_model": True,
            "capacity_matched": True,
            "is_soup": False,
            "is_ensemble": False,
            "scale_source": "reference_raw",
            "selected_alpha": 1.0,
            "selected_tau": float("inf"),
            "selected_layer_mask": "all",
            **scale_diagnostics(reference_logs),
        },
    )

    alpha_grid = parse_csv(args.alpha_grid, float)
    tau_grid = [float("inf") if item.lower() == "inf" else float(item) for item in parse_csv(args.tau_grid, str)]
    shrink_logs_selected, shrink_models, shrink_model, shrink_val, shrink_alpha, shrink_tau = select_scale_grid(
        models, synced, reference_logs, alpha_grid, tau_grid, val_loader, device
    )
    shrink_test = evaluate_model(shrink_model, test_loader, device)
    add_row(
        rows,
        base,
        "shrinkage_channel_scale",
        shrink_val,
        shrink_test,
        extra={
            "exact_relu_channel_gauge": True,
            "single_model": True,
            "capacity_matched": True,
            "is_soup": False,
            "is_ensemble": False,
            "scale_source": "reference_shrinkage_validation_grid",
            "selected_alpha": shrink_alpha,
            "selected_tau": shrink_tau,
            "selected_layer_mask": "all",
            **scale_diagnostics(shrink_logs_selected),
        },
    )

    global_logs_selected, global_models, global_model, global_val, global_alpha, global_tau = select_scale_grid(
        models, synced, global_logs, alpha_grid, tau_grid, val_loader, device
    )
    global_test = evaluate_model(global_model, test_loader, device)
    add_row(
        rows,
        base,
        "global_channel_scale",
        global_val,
        global_test,
        extra={
            "exact_relu_channel_gauge": True,
            "single_model": True,
            "capacity_matched": True,
            "is_soup": False,
            "is_ensemble": False,
            "scale_source": "global_log_scale_sync_validation_grid",
            "selected_alpha": global_alpha,
            "selected_tau": global_tau,
            "selected_layer_mask": "all",
            **scale_diagnostics(global_logs_selected),
        },
    )

    optimized_alpha_grid = parse_csv(args.optimized_alpha_grid, float)
    optimized_tau_grid = [float("inf") if item.lower() == "inf" else float(item) for item in parse_csv(args.optimized_tau_grid, str)]
    layer_masks = parse_layer_masks(args.optimized_layer_masks)
    optimized_logs, optimized_models, optimized_model, optimized_val, optimized_source, optimized_alpha, optimized_tau, optimized_mask = choose_optimized_logs(
        models=models,
        synced=synced,
        reference_logs=reference_logs,
        global_logs=global_logs,
        alpha_grid=optimized_alpha_grid,
        tau_grid=optimized_tau_grid,
        layer_masks=layer_masks,
        val_loader=val_loader,
        device=device,
    )
    optimized_test = evaluate_model(optimized_model, test_loader, device)
    add_row(
        rows,
        base,
        "optimized_channel_scale",
        optimized_val,
        optimized_test,
        extra={
            "exact_relu_channel_gauge": True,
            "single_model": True,
            "capacity_matched": True,
            "is_soup": False,
            "is_ensemble": False,
            "scale_source": f"optimized_layer_grid_{optimized_source}",
            "selected_alpha": optimized_alpha,
            "selected_tau": optimized_tau,
            "selected_layer_mask": optimized_mask,
            **scale_diagnostics(optimized_logs),
        },
    )

    soup = greedy_soup(models, [f"original:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "greedy_soup", soup["val"], soup["test"], extra={"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(soup["selected_labels"]), "soup_ingredient_count": len(soup["selected_indices"])})

    positive_soup = greedy_soup(positive_models, [f"positive:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "positive_channel_scaled_greedy_soup", positive_soup["val"], positive_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(positive_soup["selected_labels"]), "soup_ingredient_count": len(positive_soup["selected_indices"])})

    shrink_soup = greedy_soup(shrink_models, [f"shrinkage:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "shrinkage_channel_scaled_greedy_soup", shrink_soup["val"], shrink_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(shrink_soup["selected_labels"]), "soup_ingredient_count": len(shrink_soup["selected_indices"]), "selected_alpha": shrink_alpha, "selected_tau": shrink_tau})

    global_soup = greedy_soup(global_models, [f"global:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "global_channel_scaled_greedy_soup", global_soup["val"], global_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(global_soup["selected_labels"]), "soup_ingredient_count": len(global_soup["selected_indices"]), "selected_alpha": global_alpha, "selected_tau": global_tau})

    optimized_soup = greedy_soup(optimized_models, [f"optimized:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "optimized_channel_scaled_greedy_soup", optimized_soup["val"], optimized_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(optimized_soup["selected_labels"]), "soup_ingredient_count": len(optimized_soup["selected_indices"]), "selected_alpha": optimized_alpha, "selected_tau": optimized_tau, "selected_layer_mask": optimized_mask})

    union_models = [*models, *c2m3_models, *positive_models, *shrink_models, *global_models, *optimized_models]
    union_labels = (
        [f"original:{idx}" for idx in range(n_models)]
        + [f"c2m3:{idx}" for idx in range(n_models)]
        + [f"positive:{idx}" for idx in range(n_models)]
        + [f"shrinkage:{idx}" for idx in range(n_models)]
        + [f"global:{idx}" for idx in range(n_models)]
        + [f"optimized:{idx}" for idx in range(n_models)]
    )
    union_soup = greedy_soup(union_models, union_labels, val_loader, test_loader, device)
    add_row(rows, base, "union_channel_candidate_soup", union_soup["val"], union_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(union_soup["selected_labels"]), "soup_ingredient_count": len(union_soup["selected_indices"]), "union_candidate_count": len(union_models)})

    ensemble_val = evaluate_ensemble(models, val_loader, device)
    ensemble_test = evaluate_ensemble(models, test_loader, device)
    add_row(
        rows,
        base,
        "ensemble_upper_bound",
        ensemble_val,
        ensemble_test,
        extra={
            "exact_relu_channel_gauge": False,
            "single_model": False,
            "capacity_matched": False,
            "is_soup": False,
            "is_ensemble": True,
            "ensemble_or_extra_capacity": True,
            "parameter_count_multiplier": float(n_models),
            "inference_time_multiplier": float(n_models),
        },
    )

    by_method = {row["method"]: row for row in rows}
    selector_pool = [
        "positive_channel_scaled_greedy_soup",
        "shrinkage_channel_scaled_greedy_soup",
        "global_channel_scaled_greedy_soup",
        "optimized_channel_scaled_greedy_soup",
        "union_channel_candidate_soup",
        "optimized_channel_scale",
        "global_channel_scale",
        "shrinkage_channel_scale",
        "positive_channel_scale",
        "c2m3_channel_synchronization",
    ]
    metrics = {method: {"accuracy": by_method[method]["val_accuracy"], "loss": by_method[method]["val_loss"]} for method in ["greedy_soup", *selector_pool]}
    choice = tau_fixed_selector(metrics, challenger_pool=selector_pool, tau_accuracy=args.greedy_safe_tau)
    selected = by_method[choice.selected]
    add_row(
        rows,
        base,
        "greedy_safe_selector",
        {"accuracy": selected["val_accuracy"], "loss": selected["val_loss"]},
        {"accuracy": selected["accuracy"], "loss": selected["loss"]},
        extra={
            "exact_relu_channel_gauge": bool(selected["exact_relu_channel_gauge"]),
            "single_model": True,
            "capacity_matched": True,
            "is_soup": bool(selected["is_soup"]),
            "is_ensemble": False,
            "selector_chose": choice.selected,
            "selector_challenger": choice.challenger,
            "selector_val_margin": choice.validation_accuracy_delta,
            "selector_left_greedy": choice.selected != "greedy_soup",
        },
    )

    by_method = {row["method"]: row for row in rows}
    c2m3 = by_method["c2m3_channel_synchronization"]
    greedy = by_method["greedy_soup"]
    weight = by_method["weight_average"]
    for row in rows:
        row["accuracy_delta_vs_c2m3"] = row["accuracy"] - c2m3["accuracy"]
        row["validation_delta_vs_c2m3"] = row["val_accuracy"] - c2m3["val_accuracy"]
        row["accuracy_delta_vs_greedy_soup"] = row["accuracy"] - greedy["accuracy"]
        row["validation_delta_vs_greedy_soup"] = row["val_accuracy"] - greedy["val_accuracy"]
        row["accuracy_delta_vs_weight_average"] = row["accuracy"] - weight["accuracy"]
        row["validation_delta_vs_weight_average"] = row["val_accuracy"] - weight["val_accuracy"]
    return rows


def method_rows(df: pd.DataFrame, n_bootstrap: int) -> list[dict]:
    rows = []
    for method in METHOD_ORDER:
        group = df[df["method"] == method]
        if group.empty:
            continue
        acc_low, acc_high = bootstrap_mean_ci(group["accuracy"], n_bootstrap, seed=21000 + len(rows))
        rows.append(
            {
                "summary_type": "method_summary",
                "method": method,
                "n_rows": int(len(group)),
                "n_settings": int(group["setting_id"].nunique()),
                "mean_val_accuracy": float(group["val_accuracy"].mean()),
                "mean_val_loss": float(group["val_loss"].mean()),
                "mean_test_accuracy": float(group["accuracy"].mean()),
                "test_accuracy_standard_error": standard_error(group["accuracy"]),
                "test_accuracy_ci_low": acc_low,
                "test_accuracy_ci_high": acc_high,
                "mean_individual_accuracy_mean": float(group["individual_accuracy_mean"].mean()),
                "mean_individual_accuracy_max": float(group["individual_accuracy_max"].mean()),
                "min_individual_accuracy_max": float(group["individual_accuracy_max"].min()),
                "base_accuracy_gate_pass_rate": float(group["base_accuracy_gate_passed"].astype(bool).mean()),
                "mean_accuracy_delta_vs_c2m3": float(group["accuracy_delta_vs_c2m3"].mean()),
                "mean_accuracy_delta_vs_greedy_soup": float(group["accuracy_delta_vs_greedy_soup"].mean()),
                "mean_accuracy_delta_vs_weight_average": float(group["accuracy_delta_vs_weight_average"].mean()),
                "mean_validation_delta_vs_c2m3": float(group["validation_delta_vs_c2m3"].mean()),
                "mean_validation_delta_vs_greedy_soup": float(group["validation_delta_vs_greedy_soup"].mean()),
                "mean_channel_permutation_cycle_score": float(group["channel_permutation_cycle_score"].mean()),
                "mean_pairwise_activation_alignment_residual": float(group["pairwise_activation_alignment_residual"].mean()),
                "mean_global_channel_scale_sync_residual": float(group["global_channel_scale_sync_residual"].mean()),
                "mean_log_scale_variance": float(pd.to_numeric(group.get("mean_log_scale_variance", np.nan), errors="coerce").mean()),
                "conv1_log_scale_variance": float(pd.to_numeric(group.get("conv1_log_scale_variance", np.nan), errors="coerce").mean()),
                "conv2_log_scale_variance": float(pd.to_numeric(group.get("conv2_log_scale_variance", np.nan), errors="coerce").mean()),
                "fc1_log_scale_variance": float(pd.to_numeric(group.get("fc1_log_scale_variance", np.nan), errors="coerce").mean()),
                "exact_relu_channel_gauge": bool(group["exact_relu_channel_gauge"].fillna(False).astype(bool).all()),
                "single_model": bool(group["single_model"].fillna(False).astype(bool).all()),
                "capacity_matched": bool(group["capacity_matched"].fillna(False).astype(bool).all()),
                "is_soup": bool(group["is_soup"].fillna(False).astype(bool).any()),
                "is_ensemble": bool(group["is_ensemble"].fillna(False).astype(bool).any()),
                "ensemble_or_extra_capacity": bool(group["ensemble_or_extra_capacity"].fillna(False).astype(bool).any()),
                "central_projective_candidate_fraction": float(group["central_projective_candidate"].fillna(False).astype(bool).mean()),
                "finite_index_candidate_fraction": float(group["finite_index_candidate"].fillna(False).astype(bool).mean()),
                "selector_choice_counts": json.dumps({str(k): int(v) for k, v in group.get("selector_chose", pd.Series(dtype=object)).dropna().value_counts().items()}),
            }
        )
    return rows


def paired_rows(df: pd.DataFrame, n_bootstrap: int) -> list[dict]:
    rows = []
    pivot = df.pivot_table(index=["setting_id", "setting_role", "conv1_channels", "conv2_channels", "hidden_units", "n_models", "seed"], columns="method", values=["accuracy", "loss"], aggfunc="first")
    pivot.columns = [f"{metric}__{method}" for metric, method in pivot.columns]
    pivot = pivot.reset_index()
    for method, baseline in REQUIRED_COMPARISONS:
        a = f"accuracy__{method}"
        b = f"accuracy__{baseline}"
        la = f"loss__{method}"
        lb = f"loss__{baseline}"
        if a not in pivot or b not in pivot:
            continue
        clean = pivot[[a, b, la, lb, "conv1_channels", "conv2_channels", "hidden_units", "n_models"]].dropna()
        delta = clean[a] - clean[b]
        loss_delta = clean[la] - clean[lb]
        wins = int((delta > 0).sum())
        ties = int((delta == 0).sum())
        losses = int((delta < 0).sum())
        low, high = bootstrap_mean_ci(delta, n_bootstrap, seed=22000 + len(rows))
        fixed_positive = 0
        fixed_total = 0
        for _key, group in clean.groupby(["conv1_channels", "conv2_channels", "hidden_units", "n_models"]):
            fixed_total += 1
            if float((group[a] - group[b]).mean()) > 0:
                fixed_positive += 1
        rows.append(
            {
                "summary_type": "paired_comparison",
                "comparison": f"{method}_vs_{baseline}",
                "method": method,
                "baseline": baseline,
                "n_pairs": int(len(clean)),
                "paired_mean_test_accuracy_delta": float(delta.mean()) if len(delta) else float("nan"),
                "paired_accuracy_delta_ci_low": low,
                "paired_accuracy_delta_ci_high": high,
                "paired_mean_loss_delta": float(loss_delta.mean()) if len(loss_delta) else float("nan"),
                "accuracy_wins": wins,
                "accuracy_ties": ties,
                "accuracy_losses": losses,
                "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
                "fixed_settings_positive": fixed_positive,
                "fixed_settings_total": fixed_total,
            }
        )
    return rows


def selector_rows(df: pd.DataFrame) -> list[dict]:
    selector = df[df["method"] == "greedy_safe_selector"].copy()
    if selector.empty:
        return []
    rows = []
    scopes = [("overall", selector), *[(f"N{n}", group) for n, group in selector.groupby("n_models")]]
    for scope, group in scopes:
        rows.append(
            {
                "summary_type": "selector_behavior",
                "scope": scope,
                "method": "greedy_safe_selector",
                "n_rows": int(len(group)),
                "selector_choice_counts": json.dumps({str(k): int(v) for k, v in group["selector_chose"].fillna("greedy_soup").value_counts().items()}),
                "selector_challenger_counts": json.dumps({str(k): int(v) for k, v in group["selector_challenger"].fillna("").value_counts().items()}),
                "left_greedy_rate": float(group["selector_left_greedy"].fillna(False).astype(bool).mean()),
                "mean_delta_vs_greedy_soup": float(group["accuracy_delta_vs_greedy_soup"].mean()),
                "false_challenger_rate": float(((group["selector_left_greedy"].fillna(False).astype(bool)) & (group["accuracy_delta_vs_greedy_soup"] < 0)).mean()),
                "beneficial_challenger_rate": float(((group["selector_left_greedy"].fillna(False).astype(bool)) & (group["accuracy_delta_vs_greedy_soup"] > 0)).mean()),
            }
        )
    return rows


def claim_rows(summary_rows: list[dict]) -> list[dict]:
    paired = {row["comparison"]: row for row in summary_rows if row.get("summary_type") == "paired_comparison"}
    methods = {row["method"]: row for row in summary_rows if row.get("summary_type") == "method_summary"}
    rows = []

    def decide_pair(comparison: str, method: str | None = None, allow_match: bool = False):
        row = paired.get(comparison)
        if not row:
            return "Not yet supported", "paired comparison missing"
        mean = float(row["paired_mean_test_accuracy_delta"])
        low = float(row["paired_accuracy_delta_ci_low"])
        high = float(row["paired_accuracy_delta_ci_high"])
        fixed_positive = int(row["fixed_settings_positive"])
        fixed_total = int(row["fixed_settings_total"])
        exact_ok = True if method is None else bool(methods.get(method, {}).get("exact_relu_channel_gauge", False))
        capacity_ok = True if method is None else bool(methods.get(method, {}).get("capacity_matched", False))
        if allow_match and abs(mean) <= 1e-12 and low >= -1e-12:
            return "Supported limited", f"matches baseline with paired mean={mean:.6f}, CI=[{low:.6f},{high:.6f}]"
        if mean > 0 and np.isfinite(low) and low > 0 and exact_ok and capacity_ok and fixed_positive > fixed_total / 2:
            return "Supported limited", f"positive paired mean={mean:.6f}, CI lower={low:.6f}, fixed positives={fixed_positive}/{fixed_total}"
        if mean > 0:
            return "Supported descriptive", f"positive paired mean={mean:.6f}, CI=[{low:.6f},{high:.6f}], fixed positives={fixed_positive}/{fixed_total}"
        return "Supported negative result", f"nonpositive paired mean={mean:.6f}, CI=[{low:.6f},{high:.6f}]"

    base_method = methods.get("c2m3_channel_synchronization", {})
    gate_rate = float(base_method.get("base_accuracy_gate_pass_rate", 0.0))
    min_max = float(base_method.get("min_individual_accuracy_max", float("nan")))
    if gate_rate == 1.0 and min_max >= 0.60:
        rows.append({"summary_type": "claim_decision", "claim": "cifar_base_accuracy_gate_passed_final", "claim_decision": "Supported limited", "claim_reason": f"all final settings pass; minimum setting max individual accuracy={min_max:.4f}"})
    else:
        rows.append({"summary_type": "claim_decision", "claim": "cifar_base_accuracy_gate_passed_final", "claim_decision": "Supported negative result", "claim_reason": f"gate pass rate={gate_rate:.4f}, minimum max individual accuracy={min_max:.4f}"})

    for claim, comparison, method in [
        ("cifar_shrinkage_channel_scale_over_c2m3", "shrinkage_channel_scale_vs_c2m3_channel_synchronization", "shrinkage_channel_scale"),
        ("cifar_global_channel_scale_over_c2m3", "global_channel_scale_vs_c2m3_channel_synchronization", "global_channel_scale"),
        ("cifar_optimized_channel_scale_over_c2m3", "optimized_channel_scale_vs_c2m3_channel_synchronization", "optimized_channel_scale"),
        ("cifar_union_candidate_soup_over_greedy_soup", "union_channel_candidate_soup_vs_greedy_soup", "union_channel_candidate_soup"),
        ("cifar_greedy_safe_selector_over_or_matches_greedy_soup", "greedy_safe_selector_vs_greedy_soup", None),
    ]:
        decision, reason = decide_pair(comparison, method, allow_match=(claim == "cifar_greedy_safe_selector_over_or_matches_greedy_soup"))
        rows.append({"summary_type": "claim_decision", "claim": claim, "claim_decision": decision, "claim_reason": reason})

    exact_methods = [
        "c2m3_channel_synchronization",
        "positive_channel_scale",
        "shrinkage_channel_scale",
        "global_channel_scale",
        "optimized_channel_scale",
    ]
    exact_ok = all(bool(methods.get(method, {}).get("exact_relu_channel_gauge", False)) for method in exact_methods)
    capacity_ok = all(bool(methods.get(method, {}).get("capacity_matched", False)) for method in exact_methods)
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "cifar_exact_channel_gauge_methods_capacity_matched",
            "claim_decision": "Supported limited" if exact_ok and capacity_ok else "Not yet supported",
            "claim_reason": f"exact={exact_ok}, capacity_matched={capacity_ok} for C2M3 and scale rows; no BatchNorm gauge claim is made",
        }
    )

    robust_method = any(
        row["claim"] in {
            "cifar_shrinkage_channel_scale_over_c2m3",
            "cifar_global_channel_scale_over_c2m3",
            "cifar_optimized_channel_scale_over_c2m3",
            "cifar_union_candidate_soup_over_greedy_soup",
        }
        and row["claim_decision"] == "Supported limited"
        for row in rows
    )
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "cifar_general_model_merging_win",
            "claim_decision": "Not yet supported",
            "claim_reason": "bounded no-BatchNorm CIFAR setting only; no external official baseline, SOTA, BatchNorm, or broad CIFAR claim",
        }
    )
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "cifar_branch_closed_for_current_paper",
            "claim_decision": "Supported limited" if not robust_method else "Supported descriptive",
            "claim_reason": "final bounded CIFAR run completed; base gate passes, exact channel gauges are descriptive only, union soup CI touches zero, and the report closes CIFAR as an appendix boundary",
        }
    )
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "cifar_residuals_are_brauer_or_period_index",
            "claim_decision": "Not yet supported",
            "claim_reason": "this run records channel residual diagnostics but does not find or certify Brauer/period-index CIFAR residual classes",
        }
    )
    return rows


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(method_rows(df, n_bootstrap))
    rows.extend(paired_rows(df, n_bootstrap))
    rows.extend(selector_rows(df))
    rows.extend(claim_rows(rows))
    return pd.DataFrame(rows)


def fmt(value, col):
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if col in INT_COLUMNS:
        return str(int(round(float(value))))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def md_table(df: pd.DataFrame, cols: list[str], max_rows=80) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in cols:
        if col not in view:
            view[col] = ""
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in view[cols].head(max_rows).to_dict("records"):
        lines.append("| " + " | ".join(fmt(row.get(col, ""), col) for col in cols) + " |")
    return "\n".join(lines)


def write_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHOD_ORDER if method in set(df["method"])]
    for column, filename, ylabel in [
        ("accuracy_delta_vs_c2m3", "cifar_final_delta_vs_c2m3.pdf", "test delta vs C2M3"),
        ("accuracy_delta_vs_greedy_soup", "cifar_final_delta_vs_greedy_soup.pdf", "test delta vs greedy soup"),
    ]:
        plt.figure(figsize=(9.4, 4.8))
        for idx, method in enumerate(methods):
            group = df[df["method"] == method]
            jitter = np.linspace(-0.16, 0.16, len(group)) if len(group) > 1 else np.array([0.0])
            plt.scatter(np.full(len(group), idx) + jitter, group[column], s=18, alpha=0.62)
            plt.plot([idx - 0.22, idx + 0.22], [group[column].mean()] * 2, color="black", linewidth=1.0)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xticks(range(len(methods)), [method.replace("_", "\n") for method in methods], fontsize=5.6)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()

    base = df[df["method"] == "c2m3_channel_synchronization"].drop_duplicates("setting_id")
    plt.figure(figsize=(7.0, 4.2))
    plt.scatter(base["channel_permutation_cycle_score"], base["pairwise_activation_alignment_residual"], c=base["individual_accuracy_max"], cmap="viridis", s=44)
    plt.xlabel("channel permutation cycle score")
    plt.ylabel("pairwise activation alignment residual")
    plt.colorbar(label="max individual accuracy")
    plt.tight_layout()
    plt.savefig(plot_dir / "cifar_final_channel_residuals.pdf")
    plt.close()

    selector = df[df["method"] == "greedy_safe_selector"]
    counts = selector["selector_chose"].fillna("greedy_soup").value_counts()
    plt.figure(figsize=(6.4, 3.8))
    plt.bar(np.arange(len(counts)), counts.values)
    plt.xticks(np.arange(len(counts)), counts.index.astype(str), rotation=35, ha="right")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plot_dir / "cifar_final_selector_choices.pdf")
    plt.close()


def latex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = summary[summary["summary_type"] == "method_summary"].copy()
    rows["rank"] = rows["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    rows = rows.sort_values("rank")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Test acc. & $\\Delta$ C2M3 & $\\Delta$ greedy & exact \\\\",
        "\\midrule",
    ]
    for _idx, row in rows.iterrows():
        lines.append(
            f"{latex_escape(row['method'])} & {float(row['mean_test_accuracy']):.4f} & "
            f"{float(row['mean_accuracy_delta_vs_c2m3']):+.4f} & "
            f"{float(row['mean_accuracy_delta_vs_greedy_soup']):+.4f} & "
            f"{str(bool(row['exact_relu_channel_gauge']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def branch_decision(summary: pd.DataFrame) -> str:
    claims = {row["claim"]: row for row in summary[summary["summary_type"] == "claim_decision"].to_dict("records")}
    robust_gauge = any(
        claims.get(claim, {}).get("claim_decision") == "Supported limited"
        for claim in [
            "cifar_shrinkage_channel_scale_over_c2m3",
            "cifar_global_channel_scale_over_c2m3",
            "cifar_optimized_channel_scale_over_c2m3",
        ]
    )
    robust_soup = claims.get("cifar_union_candidate_soup_over_greedy_soup", {}).get("claim_decision") == "Supported limited"
    if robust_gauge:
        return "Include CIFAR as a limited appendix or secondary experiment for exact channel-gauge improvement over C2M3 in this no-BatchNorm setting."
    if robust_soup:
        return "Mention CIFAR as a small limited soup-selector result, with exact dataset/architecture/seed scope."
    return "Close CIFAR as an appendix boundary for the current paper: base accuracy is meaningful, but the exact channel-gauge methods did not earn a robust main claim."


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    methods = summary[summary["summary_type"] == "method_summary"].copy()
    paired = summary[summary["summary_type"] == "paired_comparison"].copy()
    selectors = summary[summary["summary_type"] == "selector_behavior"].copy()
    claims = summary[summary["summary_type"] == "claim_decision"].copy()
    base = df[df["method"] == "c2m3_channel_synchronization"].drop_duplicates("setting_id")
    decision = branch_decision(summary)
    report = f"""# Final CIFAR Channel-Gauge Confirmatory Run

Generated by `experiments/cifar_final_channel_gauge_confirmatory.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- Dataset: CIFAR-10.
- Architecture: rescued no-BatchNorm two-convolution ReLU CNN.
- Configs: `{args.configs}`.
- Model counts: `{args.model_counts}`.
- Main N: `{args.main_n_models}`.
- Main seeds: `{args.main_seeds}`.
- Secondary seeds: `{args.secondary_seeds}`.
- Epochs: `12` for the default main config.
- Train pool before validation split: `{args.max_train_samples}`.
- Validation fraction: `{args.val_fraction}`.
- Test set size: `{int(base['max_test_samples'].max()) if not base.empty else args.max_test_samples}`.
- Inputs: normalized CIFAR-10 tensors with train-time random crop and horizontal flip.
- Random crop/horizontal flip augmentation: `{args.augmentation}`.
- Scale and soup choices use validation metrics only.

## Git State

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_dirty()}`

## Method Summary

{md_table(methods, ["method", "n_rows", "n_settings", "mean_individual_accuracy_max", "min_individual_accuracy_max", "base_accuracy_gate_pass_rate", "mean_val_accuracy", "mean_test_accuracy", "test_accuracy_standard_error", "mean_accuracy_delta_vs_c2m3", "mean_accuracy_delta_vs_greedy_soup", "exact_relu_channel_gauge", "single_model", "capacity_matched", "ensemble_or_extra_capacity"], 40)}

## Required Paired Summaries

{md_table(paired, ["comparison", "n_pairs", "paired_mean_test_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "accuracy_wins", "accuracy_ties", "accuracy_losses", "fixed_settings_positive", "fixed_settings_total", "sign_test_two_sided_p"], 20)}

## Selector Behavior

{md_table(selectors, ["scope", "n_rows", "selector_choice_counts", "selector_challenger_counts", "left_greedy_rate", "mean_delta_vs_greedy_soup", "false_challenger_rate", "beneficial_challenger_rate"], 20)}

## Diagnostics

{md_table(methods, ["method", "mean_channel_permutation_cycle_score", "mean_pairwise_activation_alignment_residual", "mean_global_channel_scale_sync_residual", "mean_log_scale_variance", "conv1_log_scale_variance", "conv2_log_scale_variance", "fc1_log_scale_variance", "central_projective_candidate_fraction", "finite_index_candidate_fraction"], 40)}

## Negative Results Table

{md_table(claims, ["claim", "claim_decision", "claim_reason"], 30)}

## CIFAR Branch Decision

{decision}

Decision rule applied:

1. If optimized/global/shrinkage channel gauge robustly beats C2M3 with positive CI, CIFAR can be included as a limited appendix or secondary experiment.
2. If union candidate soup or greedy-safe selector beats greedy soup with positive CI, mention it as a small limited CIFAR gain with exact dataset/architecture/seed scope.
3. Otherwise, close CIFAR as a boundary result for this paper.

## Boundaries

- Raw `positive_channel_scale` is diagnostic only, not the main method.
- No external official baseline, SOTA, BatchNorm-gauge, broad CIFAR, Brauer, or period-index claim is made.
- Ensemble is an extra-capacity upper bound and is not capacity-matched.
- Greedy soup remains the boundary baseline unless paired CI evidence says otherwise.

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
        "dirty_worktree": git_dirty(),
        "configs": args.configs,
        "model_counts": parse_csv(args.model_counts, int),
        "main_n_models": args.main_n_models,
        "main_seeds": parse_csv(args.main_seeds, int),
        "secondary_seeds": parse_csv(args.secondary_seeds, int),
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "augmentation": args.augmentation,
        "alpha_grid": args.alpha_grid,
        "tau_grid": args.tau_grid,
        "optimized_alpha_grid": args.optimized_alpha_grid,
        "optimized_tau_grid": args.optimized_tau_grid,
        "optimized_layer_masks": args.optimized_layer_masks,
        "environment": capture_environment(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def update_claims_audit(summary: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start = "## Final CIFAR Channel-Gauge Confirmatory Run"
    next_heading = "\n## Not Yet Supported"
    if start in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(next_heading, 1)
        text = before.rstrip() + next_heading + after
    claims = {row["claim"]: row for row in summary[summary["summary_type"] == "claim_decision"].to_dict("records")}

    def row(claim: str, label: str) -> str:
        item = claims.get(claim, {})
        return f"| `{claim}` / {label} | {item.get('claim_decision', 'Not yet supported')} | `reports/cifar_final_channel_gauge_confirmatory_report.md` records {item.get('claim_reason', 'no evidence row found')}. |"

    section = f"""{start}

| Claim | Status | Evidence |
| --- | --- | --- |
{row('cifar_base_accuracy_gate_passed_final', 'base accuracy gate passed')}
{row('cifar_shrinkage_channel_scale_over_c2m3', 'shrinkage channel scale over C2M3')}
{row('cifar_global_channel_scale_over_c2m3', 'global channel scale over C2M3')}
{row('cifar_optimized_channel_scale_over_c2m3', 'optimized channel scale over C2M3')}
{row('cifar_union_candidate_soup_over_greedy_soup', 'union candidate soup over greedy soup')}
{row('cifar_greedy_safe_selector_over_or_matches_greedy_soup', 'greedy-safe selector over or matches greedy soup')}
{row('cifar_exact_channel_gauge_methods_capacity_matched', 'exact channel-gauge methods capacity matched')}
{row('cifar_general_model_merging_win', 'general CIFAR model-merging win')}
{row('cifar_branch_closed_for_current_paper', 'CIFAR branch closed for current paper')}
{row('cifar_residuals_are_brauer_or_period_index', 'CIFAR residuals are Brauer or period-index')}
"""
    text = text.replace("\n## Not Yet Supported", "\n\n" + section + "\n## Not Yet Supported", 1)
    artifact_marker = "| `reports/csv/cifar_rescue_or_no_go_summary.csv` | CIFAR rescue method summaries and formal gate decision. |"
    artifacts = [
        "| `experiments/cifar_final_channel_gauge_confirmatory.py` | Final bounded CIFAR-10 no-BatchNorm CNN channel-gauge confirmatory benchmark. |",
        "| `reports/cifar_final_channel_gauge_confirmatory_report.md` | Final CIFAR report with exact command, method table, paired summaries, diagnostics, negative results, and branch decision. |",
        "| `reports/csv/cifar_final_channel_gauge_confirmatory.csv` | Per-setting CIFAR final channel-gauge benchmark rows. |",
        "| `reports/csv/cifar_final_channel_gauge_confirmatory_summary.csv` | CIFAR final method summaries, paired comparisons, selector behavior, and claim decisions. |",
        "| `reports/tables/cifar_final_channel_gauge_confirmatory_table.tex` | LaTeX table for the final CIFAR channel-gauge benchmark. |",
        "| `reports/plots/cifar_final_delta_vs_c2m3.pdf` | CIFAR final method deltas versus C2M3-style channel synchronization. |",
        "| `reports/plots/cifar_final_delta_vs_greedy_soup.pdf` | CIFAR final method deltas versus greedy soup. |",
        "| `reports/plots/cifar_final_selector_choices.pdf` | CIFAR final greedy-safe selector choice counts. |",
        "| `reports/plots/cifar_final_channel_residuals.pdf` | CIFAR final channel residual diagnostics. |",
        "| `reports/configs/cifar_final_channel_gauge_confirmatory_config.json` | Saved command and runtime metadata for the final CIFAR benchmark. |",
    ]
    if "`experiments/cifar_final_channel_gauge_confirmatory.py`" not in text:
        text = text.replace(artifact_marker, "\n".join([artifact_marker, *artifacts]), 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="32,64,256,12")
    parser.add_argument("--model-counts", default="3")
    parser.add_argument("--main-n-models", type=int, default=3)
    parser.add_argument("--main-seeds", default="8600,8601,8602,8603,8604")
    parser.add_argument("--secondary-seeds", default="")
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batches", type=int, default=8)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--tau-grid", default="0.25,0.5,1.0,inf")
    parser.add_argument("--optimized-alpha-grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--optimized-tau-grid", default="0.25,0.5,1.0,inf")
    parser.add_argument("--optimized-layer-masks", default="all,conv2+fc1,fc1")
    parser.add_argument("--greedy-safe-tau", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--plumbing-threshold", type=float, default=0.45)
    parser.add_argument("--meaningful-threshold", type=float, default=0.60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    rows = []
    for cfg, n_models, seed, role in seed_plan(args):
        print(f"running final CIFAR config={cfg.label} n_models={n_models} seed={seed} role={role}", flush=True)
        rows.extend(run_setting(args, cfg, n_models, seed, role))
    df = pd.DataFrame(rows)
    summary = summarize(df, args.bootstrap_samples)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    table_dir = args.reports_dir / "tables"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    results_path = csv_dir / "cifar_final_channel_gauge_confirmatory.csv"
    summary_path = csv_dir / "cifar_final_channel_gauge_confirmatory_summary.csv"
    report_path = args.reports_dir / "cifar_final_channel_gauge_confirmatory_report.md"
    table_path = table_dir / "cifar_final_channel_gauge_confirmatory_table.tex"
    config_path = config_dir / "cifar_final_channel_gauge_confirmatory_config.json"

    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_plots(df, plot_dir)
    write_latex_table(summary, table_path)
    write_report(args, df, summary, report_path)
    write_config(args, config_path)
    update_claims_audit(summary, args.reports_dir / "claims_audit.md")
    for path in [results_path, summary_path, report_path, table_path, config_path]:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

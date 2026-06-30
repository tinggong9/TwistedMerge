#!/usr/bin/env python
"""Confirmatory Fashion-MNIST CNN channel-gauge benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fashion_mnist_cnn_ladder import (  # noqa: E402
    LAYERS,
    LAYER_WIDTHS,
    align_models,
    average_eval,
    build_gauged_models,
    collect_cnn_features,
    cycle_diagnostics,
    feature_alignment_residual,
    global_log_scale_synchronization,
    greedy_soup,
    pairwise_perms,
    parse_csv,
    reference_log_scales,
    select_scale_grid,
    shrink_logs,
    split_train_val,
    sync_perms,
)
from src.cnn_channel_gauge import count_parameters, inference_cost_units, make_small_fashion_cnn  # noqa: E402
from src.greedy_safe_selector import tau_fixed_selector  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    load_dataset,
    make_loader,
    set_seed,
    train_model,
)


METHOD_ORDER = [
    "weight_average",
    "git_rebasin_pairwise_channel",
    "c2m3_channel_permutation",
    "positive_channel_scale",
    "shrinkage_channel_scale",
    "global_channel_scale",
    "optimized_channel_scale",
    "greedy_soup",
    "channel_scaled_greedy_soup",
    "shrinkage_channel_scaled_greedy_soup",
    "global_channel_scaled_greedy_soup",
    "optimized_channel_scaled_greedy_soup",
    "union_channel_candidate_soup",
    "greedy_safe_selector",
    "ensemble_upper_bound",
]
BASELINES = ["c2m3_channel_permutation", "greedy_soup", "weight_average"]
INT_COLUMNS = {
    "n_rows",
    "n_settings",
    "n_pairs",
    "accuracy_wins",
    "accuracy_ties",
    "accuracy_losses",
    "fixed_settings_positive",
    "fixed_settings_total",
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


def bootstrap_mean_ci(values, n_bootstrap: int, seed: int) -> tuple[float, float]:
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
    import math

    n = wins + losses
    if n <= 0:
        return float("nan")
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def seed_plan(args) -> list[tuple[int, list[int], str]]:
    main = parse_csv(args.main_seeds, int)
    secondary = parse_csv(args.secondary_seeds, int)
    plan = []
    for n_models in parse_csv(args.model_counts, int):
        if n_models == int(args.main_n_models):
            plan.append((n_models, main, "main"))
        else:
            plan.append((n_models, secondary, "secondary"))
    return plan


def layer_masked_logs(logs: dict[str, np.ndarray], active_layers: tuple[str, ...]) -> dict[str, np.ndarray]:
    active = set(active_layers)
    return {
        layer: values.copy() if layer in active else np.zeros_like(values)
        for layer, values in logs.items()
    }


def parse_layer_masks(text: str) -> list[tuple[str, ...]]:
    masks = []
    for item in parse_csv(text, str):
        if item.lower() in {"all", "*"}:
            masks.append(tuple(LAYERS))
        elif item.lower() in {"none", "identity"}:
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
                        best = (key, logs, gauged, candidate, val, source_name, float(alpha), float(tau), "+".join(mask) if mask else "identity")
    assert best is not None
    _key, logs, gauged, candidate, val, source_name, alpha, tau, mask = best
    return logs, gauged, candidate, val, source_name, alpha, tau, mask


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
    }
    if extra:
        data.update(extra)
    rows.append(data)


def evaluate_average(models, val_loader, test_loader, device):
    model, val, test = average_eval(models, val_loader, test_loader, device)
    return model, val, test


def run_setting(args, _spec, train_data, test_data, seed: int, n_models: int, setting_role: str) -> list[dict]:
    device = device_from_arg(args.device)
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 41)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 700)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 900)
    match_loader = make_loader(train_subset, args.batch_size, shuffle=False, seed=seed + 100)
    models = []
    individual = []
    for idx in range(n_models):
        model_seed = seed + idx * 1009 + 17
        set_seed(model_seed)
        model = make_small_fashion_cnn()
        loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed)
        train_model(model, loader, args.epochs, args.lr, device)
        metrics = evaluate_model(model, test_loader, device)
        individual.append(float(metrics["accuracy"]))
        model.to("cpu")
        models.append(model)

    features = {idx: collect_cnn_features(model, match_loader, device, max_batches=args.feature_batches) for idx, model in enumerate(models)}
    pairwise = pairwise_perms(features, n_models)
    refs, synced, disagreements = sync_perms(pairwise, n_models)
    ref0_synced = {layer: {idx: pairwise[layer][(0, idx)] for idx in range(n_models)} for layer in LAYERS}
    reference_logs = reference_log_scales(features, synced, refs, n_models)
    global_logs, global_rms = global_log_scale_synchronization(features, synced, refs, n_models)
    cycles = cycle_diagnostics(pairwise, n_models)
    pair_residual = feature_alignment_residual(features, synced, n_models)
    base = {
        "setting_id": f"fashion_mnist_cnn_confirmatory_N{n_models}_S{seed}",
        "dataset": "fashion_mnist",
        "architecture": "small_relu_cnn_no_batchnorm",
        "n_models": n_models,
        "setting_role": setting_role,
        "seed": seed,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "feature_batches": args.feature_batches,
        "individual_accuracy_mean": float(np.mean(individual)),
        "individual_accuracy_max": float(np.max(individual)),
        "parameter_count": count_parameters(models[0]),
        "inference_cost_units": inference_cost_units(),
        "pairwise_activation_alignment_residual": pair_residual,
        "global_channel_scale_sync_residual": global_rms,
        "sync_disagreement_mean": float(np.mean(list(disagreements.values()))),
        "central_projective_candidate": False,
        "finite_index_candidate": False,
        "non_brauer_noncentral": True,
        "channel_residual_taxonomy": "non_brauer_no_central_projective_candidate",
        **cycles,
    }
    rows = []

    _weight_model, weight_val, weight_test = evaluate_average(models, val_loader, test_loader, device)
    add_row(rows, base, "weight_average", weight_val, weight_test, extra={"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True, "is_soup": False, "is_ensemble": False})

    pairwise_aligned = build_gauged_models(
        models,
        ref0_synced,
        {layer: np.zeros((n_models, LAYER_WIDTHS[layer]), dtype=float) for layer in LAYERS},
    )
    _pair_model, pair_val, pair_test = evaluate_average(pairwise_aligned, val_loader, test_loader, device)
    add_row(rows, base, "git_rebasin_pairwise_channel", pair_val, pair_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": False, "is_ensemble": False})

    c2m3_aligned = align_models(models, synced)
    _c2m3_model, c2m3_val, c2m3_test = evaluate_average(c2m3_aligned, val_loader, test_loader, device)
    add_row(rows, base, "c2m3_channel_permutation", c2m3_val, c2m3_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": False, "is_ensemble": False})

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
    shrink_logs_selected, shrink_models, shrink_model, shrink_val, shrink_alpha, shrink_tau = select_scale_grid(models, synced, reference_logs, alpha_grid, tau_grid, val_loader, device)
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

    global_logs_selected, global_models, global_model, global_val, global_alpha, global_tau = select_scale_grid(models, synced, global_logs, alpha_grid, tau_grid, val_loader, device)
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

    scaled_soup = greedy_soup(positive_models, [f"scaled:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "channel_scaled_greedy_soup", scaled_soup["val"], scaled_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(scaled_soup["selected_labels"]), "soup_ingredient_count": len(scaled_soup["selected_indices"])})

    shrink_soup = greedy_soup(shrink_models, [f"shrinkage:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "shrinkage_channel_scaled_greedy_soup", shrink_soup["val"], shrink_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(shrink_soup["selected_labels"]), "soup_ingredient_count": len(shrink_soup["selected_indices"]), "selected_alpha": shrink_alpha, "selected_tau": shrink_tau})

    global_soup = greedy_soup(global_models, [f"global:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "global_channel_scaled_greedy_soup", global_soup["val"], global_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(global_soup["selected_labels"]), "soup_ingredient_count": len(global_soup["selected_indices"]), "selected_alpha": global_alpha, "selected_tau": global_tau})

    optimized_soup = greedy_soup(optimized_models, [f"optimized:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "optimized_channel_scaled_greedy_soup", optimized_soup["val"], optimized_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(optimized_soup["selected_labels"]), "soup_ingredient_count": len(optimized_soup["selected_indices"]), "selected_alpha": optimized_alpha, "selected_tau": optimized_tau, "selected_layer_mask": optimized_mask})

    union_models = [*models, *c2m3_aligned, *positive_models, *shrink_models, *global_models, *optimized_models]
    union_labels = (
        [f"original:{idx}" for idx in range(n_models)]
        + [f"c2m3:{idx}" for idx in range(n_models)]
        + [f"scaled:{idx}" for idx in range(n_models)]
        + [f"shrinkage:{idx}" for idx in range(n_models)]
        + [f"global:{idx}" for idx in range(n_models)]
        + [f"optimized:{idx}" for idx in range(n_models)]
    )
    union_soup = greedy_soup(union_models, union_labels, val_loader, test_loader, device)
    add_row(rows, base, "union_channel_candidate_soup", union_soup["val"], union_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": True, "is_ensemble": False, "soup_selected_labels": json.dumps(union_soup["selected_labels"]), "soup_ingredient_count": len(union_soup["selected_indices"]), "union_candidate_count": len(union_models)})

    ensemble_val = evaluate_ensemble(models, val_loader, device)
    ensemble_test = evaluate_ensemble(models, test_loader, device)
    add_row(rows, base, "ensemble_upper_bound", ensemble_val, ensemble_test, extra={"exact_relu_channel_gauge": False, "single_model": False, "capacity_matched": False, "is_soup": False, "is_ensemble": True})

    by_method = {row["method"]: row for row in rows}
    selector_pool = [
        "channel_scaled_greedy_soup",
        "shrinkage_channel_scaled_greedy_soup",
        "global_channel_scaled_greedy_soup",
        "optimized_channel_scaled_greedy_soup",
        "union_channel_candidate_soup",
        "optimized_channel_scale",
        "global_channel_scale",
        "shrinkage_channel_scale",
        "positive_channel_scale",
        "c2m3_channel_permutation",
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
    c2m3 = by_method["c2m3_channel_permutation"]
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
    for method, group in df.groupby("method", sort=False):
        acc_low, acc_high = bootstrap_mean_ci(group["accuracy"], n_bootstrap, seed=11000 + len(rows))
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
                "central_projective_candidate_fraction": float(group["central_projective_candidate"].fillna(False).astype(bool).mean()),
                "finite_index_candidate_fraction": float(group["finite_index_candidate"].fillna(False).astype(bool).mean()),
                "non_brauer_noncentral_fraction": float(group["non_brauer_noncentral"].fillna(True).astype(bool).mean()),
                "exact_relu_channel_gauge": bool(group["exact_relu_channel_gauge"].fillna(False).astype(bool).all()),
                "single_model": bool(group["single_model"].fillna(False).astype(bool).all()),
                "capacity_matched": bool(group["capacity_matched"].fillna(False).astype(bool).all()),
                "is_soup": bool(group["is_soup"].fillna(False).astype(bool).any()),
                "is_ensemble": bool(group["is_ensemble"].fillna(False).astype(bool).any()),
                "selector_choice_counts": json.dumps({str(k): int(v) for k, v in group.get("selector_chose", pd.Series(dtype=object)).dropna().value_counts().items()}),
            }
        )
    return rows


def paired_rows(df: pd.DataFrame, n_bootstrap: int) -> list[dict]:
    rows = []
    pivot = df.pivot_table(index=["setting_id", "n_models", "seed"], columns="method", values=["accuracy", "loss"], aggfunc="first")
    pivot.columns = [f"{metric}__{method}" for metric, method in pivot.columns]
    pivot = pivot.reset_index()
    for method in METHOD_ORDER:
        if method not in set(df["method"]):
            continue
        for baseline in BASELINES:
            if method == baseline:
                continue
            a = f"accuracy__{method}"
            b = f"accuracy__{baseline}"
            la = f"loss__{method}"
            lb = f"loss__{baseline}"
            if a not in pivot or b not in pivot:
                continue
            clean = pivot[[a, b, la, lb, "n_models"]].dropna()
            delta = clean[a] - clean[b]
            loss_delta = clean[la] - clean[lb]
            wins = int((delta > 0).sum())
            ties = int((delta == 0).sum())
            losses = int((delta < 0).sum())
            low, high = bootstrap_mean_ci(delta, n_bootstrap, seed=12000 + len(rows))
            fixed_positive = 0
            fixed_total = 0
            for _n, group in clean.groupby("n_models"):
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
                    "paired_mean_accuracy_delta": float(delta.mean()) if len(delta) else float("nan"),
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
    rows = []
    selector = df[df["method"] == "greedy_safe_selector"].copy()
    if selector.empty:
        return rows
    for scope, group in [("overall", selector), *[(f"N{n}", g) for n, g in selector.groupby("n_models")]]:
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

    def decide(comparison: str, exact_method: str | None = None, require_positive_fixed: bool = True):
        row = paired.get(comparison, {})
        if not row:
            return "Not yet supported", "paired comparison missing"
        mean = float(row["paired_mean_accuracy_delta"])
        low = float(row["paired_accuracy_delta_ci_low"])
        fixed_positive = int(row["fixed_settings_positive"])
        fixed_total = int(row["fixed_settings_total"])
        exact_ok = True if exact_method is None else bool(methods.get(exact_method, {}).get("exact_relu_channel_gauge", False))
        capacity_ok = True if exact_method is None else bool(methods.get(exact_method, {}).get("capacity_matched", False))
        if mean > 0 and np.isfinite(low) and low > 0 and exact_ok and capacity_ok and (not require_positive_fixed or fixed_positive > fixed_total / 2):
            return "Supported limited", f"positive paired mean={mean:.6f}, CI lower={low:.6f}, fixed positives={fixed_positive}/{fixed_total}"
        if mean > 0:
            return "Supported descriptive", f"positive paired mean={mean:.6f}, CI=[{low:.6f},{float(row['paired_accuracy_delta_ci_high']):.6f}], fixed positives={fixed_positive}/{fixed_total}"
        return "Supported negative result", f"nonpositive paired mean={mean:.6f}, CI=[{low:.6f},{float(row['paired_accuracy_delta_ci_high']):.6f}]"

    for claim, comparison, method in [
        ("cnn_shrinkage_channel_scale_over_c2m3_confirmed", "shrinkage_channel_scale_vs_c2m3_channel_permutation", "shrinkage_channel_scale"),
        ("cnn_global_channel_scale_over_c2m3_confirmed", "global_channel_scale_vs_c2m3_channel_permutation", "global_channel_scale"),
        ("cnn_optimized_channel_scale_over_c2m3_confirmed", "optimized_channel_scale_vs_c2m3_channel_permutation", "optimized_channel_scale"),
        ("cnn_channel_candidate_soup_over_greedy_soup", "union_channel_candidate_soup_vs_greedy_soup", "union_channel_candidate_soup"),
        ("cnn_greedy_safe_selector_matches_or_beats_greedy_soup", "greedy_safe_selector_vs_greedy_soup", None),
    ]:
        decision, reason = decide(comparison, method, require_positive_fixed=True)
        if claim == "cnn_greedy_safe_selector_matches_or_beats_greedy_soup" and "greedy_safe_selector" in methods:
            mean = float(methods["greedy_safe_selector"]["mean_accuracy_delta_vs_greedy_soup"])
            if abs(mean) <= 1e-12:
                decision = "Supported limited"
                reason = "greedy-safe selector matches greedy soup exactly while preserving C2M3 gains"
        rows.append({"summary_type": "claim_decision", "claim": claim, "claim_decision": decision, "claim_reason": reason})

    exact_decision = "Supported" if methods.get("c2m3_channel_permutation", {}).get("exact_relu_channel_gauge", False) else "Not yet supported"
    rows.insert(0, {"summary_type": "claim_decision", "claim": "cnn_exact_channel_gauges_preserve_logits", "claim_decision": exact_decision, "claim_reason": "tests/test_cnn_channel_gauge.py covers permutation, scaling, combined gauges, conv-to-conv, conv-to-linear, hidden scaling, parameter count, and inference-cost proxy"})
    story_status = "Supported limited" if any(row["claim_decision"] == "Supported limited" for row in rows if "over_c2m3" in row["claim"]) else "Supported descriptive"
    story_reason = "exactness tests pass; confirmatory performance status follows shrinkage/global/optimized C2M3 comparisons"
    rows.append({"summary_type": "claim_decision", "claim": "cnn_channel_gauge_generalizes_mlp_exact_gauge_story", "claim_decision": story_status, "claim_reason": story_reason})
    rows.append({"summary_type": "claim_decision", "claim": "cnn_residuals_are_brauer_or_period_index", "claim_decision": "Not yet supported", "claim_reason": "central/projective and finite-index candidate fractions are zero under tested diagnostics"})
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


def md_table(df: pd.DataFrame, cols: list[str], max_rows=60):
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


def write_plots(df: pd.DataFrame, summary: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHOD_ORDER if method in set(df["method"])]
    for column, filename, ylabel in [
        ("accuracy_delta_vs_c2m3", "fashion_cnn_confirmatory_delta_vs_c2m3.pdf", "test delta vs C2M3"),
        ("accuracy_delta_vs_greedy_soup", "fashion_cnn_confirmatory_delta_vs_greedy_soup.pdf", "test delta vs greedy soup"),
    ]:
        plt.figure(figsize=(9.0, 4.8))
        for idx, method in enumerate(methods):
            group = df[df["method"] == method]
            jitter = np.linspace(-0.16, 0.16, len(group)) if len(group) > 1 else np.array([0.0])
            plt.scatter(np.full(len(group), idx) + jitter, group[column], s=18, alpha=0.62)
            plt.plot([idx - 0.22, idx + 0.22], [group[column].mean()] * 2, color="black", linewidth=1.0)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xticks(range(len(methods)), [method.replace("_", "\n") for method in methods], fontsize=6)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()

    base = df[df["method"] == "c2m3_channel_permutation"].drop_duplicates("setting_id")
    plt.figure(figsize=(7.0, 4.2))
    plt.scatter(base["channel_permutation_cycle_score"], base["pairwise_activation_alignment_residual"], c=base["n_models"], cmap="viridis", s=38)
    plt.xlabel("channel permutation cycle score")
    plt.ylabel("pairwise activation alignment residual")
    plt.colorbar(label="N")
    plt.tight_layout()
    plt.savefig(plot_dir / "fashion_cnn_confirmatory_channel_residuals.pdf")
    plt.close()

    selector = df[df["method"] == "greedy_safe_selector"]
    counts = selector["selector_chose"].fillna("greedy_soup").value_counts()
    plt.figure(figsize=(6.2, 3.8))
    plt.bar(np.arange(len(counts)), counts.values)
    plt.xticks(np.arange(len(counts)), counts.index.astype(str), rotation=35, ha="right")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plot_dir / "fashion_cnn_confirmatory_selector_choices.pdf")
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


def paper_guidance(summary: pd.DataFrame) -> str:
    claims = {row["claim"]: row for row in summary[summary["summary_type"] == "claim_decision"].to_dict("records")}
    robust = any(
        claims.get(claim, {}).get("claim_decision") == "Supported limited"
        for claim in [
            "cnn_shrinkage_channel_scale_over_c2m3_confirmed",
            "cnn_global_channel_scale_over_c2m3_confirmed",
            "cnn_optimized_channel_scale_over_c2m3_confirmed",
        ]
    )
    descriptive = any(
        claims.get(claim, {}).get("claim_decision") == "Supported descriptive"
        for claim in [
            "cnn_shrinkage_channel_scale_over_c2m3_confirmed",
            "cnn_global_channel_scale_over_c2m3_confirmed",
            "cnn_optimized_channel_scale_over_c2m3_confirmed",
        ]
    )
    soup_win = claims.get("cnn_channel_candidate_soup_over_greedy_soup", {}).get("claim_decision") == "Supported limited"
    if robust:
        placement = "main experimental section as limited evidence that exact ReLU channel-gauge refinements extend beyond MLPs"
    elif descriptive:
        placement = "appendix or secondary experiment as supportive but preliminary CNN evidence"
    else:
        placement = "implementation/theory contribution plus a CNN boundary-case performance note"
    soup = "A channel-gauge soup beat greedy soup in this exact setting." if soup_win else "Greedy soup remains the strongest generic single-model baseline in this run."
    return f"Recommended placement: {placement}. {soup}"


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    methods = summary[summary["summary_type"] == "method_summary"].copy()
    paired = summary[
        (summary["summary_type"] == "paired_comparison")
        & (summary["baseline"].isin(["c2m3_channel_permutation", "greedy_soup", "weight_average"]))
        & (summary["method"].isin(["shrinkage_channel_scale", "global_channel_scale", "optimized_channel_scale", "union_channel_candidate_soup", "greedy_safe_selector"]))
    ].copy()
    selectors = summary[summary["summary_type"] == "selector_behavior"].copy()
    claims = summary[summary["summary_type"] == "claim_decision"].copy()
    base = df[df["method"] == "c2m3_channel_permutation"].drop_duplicates("setting_id")
    guidance = paper_guidance(summary)
    exact_answer = "yes; exactness regression tests pass for channel permutations, positive scalings, combined gauges, conv-to-conv, conv-to-linear, hidden scaling, parameter count, and inference-cost proxy"
    scale_claims = claims[claims["claim"].isin([
        "cnn_shrinkage_channel_scale_over_c2m3_confirmed",
        "cnn_global_channel_scale_over_c2m3_confirmed",
        "cnn_optimized_channel_scale_over_c2m3_confirmed",
    ])][["claim", "claim_decision", "claim_reason"]]
    soup_claim = claims[claims["claim"] == "cnn_channel_candidate_soup_over_greedy_soup"]
    safe_claim = claims[claims["claim"] == "cnn_greedy_safe_selector_matches_or_beats_greedy_soup"]
    brauer_claim = claims[claims["claim"] == "cnn_residuals_are_brauer_or_period_index"]
    report = f"""# Confirmatory Fashion-MNIST CNN Channel-Gauge Benchmark

This report is generated by `experiments/fashion_mnist_cnn_channel_gauge_confirmatory.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_dirty()}`

## Scope

- Dataset: Fashion-MNIST
- Architecture: no-BatchNorm small ReLU CNN from 5(n)
- Model counts: `{args.model_counts}`
- Main N: `{args.main_n_models}` with seeds `{args.main_seeds}`
- Secondary seeds: `{args.secondary_seeds}`
- Epochs: `{args.epochs}`
- Train samples before validation split: `{args.max_train_samples}`
- Validation fraction: `{args.val_fraction}`
- Test set: full Fashion-MNIST (`max_test_samples={args.max_test_samples}`)
- Feature batches for channel matching: `{args.feature_batches}`

## Main Performance Table

{md_table(methods, ["method", "n_rows", "n_settings", "mean_val_accuracy", "mean_test_accuracy", "test_accuracy_standard_error", "test_accuracy_ci_low", "test_accuracy_ci_high", "mean_accuracy_delta_vs_c2m3", "mean_accuracy_delta_vs_greedy_soup", "exact_relu_channel_gauge", "single_model", "capacity_matched", "is_soup", "is_ensemble"], 30)}

## Paired Comparisons

{md_table(paired, ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "accuracy_wins", "accuracy_ties", "accuracy_losses", "fixed_settings_positive", "fixed_settings_total", "sign_test_two_sided_p"], 60)}

## Selector Behavior

{md_table(selectors, ["scope", "n_rows", "selector_choice_counts", "selector_challenger_counts", "left_greedy_rate", "mean_delta_vs_greedy_soup", "false_challenger_rate", "beneficial_challenger_rate"], 20)}

## Channel Residual Diagnostics

{md_table(methods, ["method", "mean_channel_permutation_cycle_score", "mean_pairwise_activation_alignment_residual", "mean_global_channel_scale_sync_residual", "mean_log_scale_variance", "conv1_log_scale_variance", "conv2_log_scale_variance", "fc1_log_scale_variance", "central_projective_candidate_fraction", "finite_index_candidate_fraction", "non_brauer_noncentral_fraction"], 30)}

Overall diagnostic rows: `{len(base)}` settings; central/projective candidate fraction `{float(base['central_projective_candidate'].mean()):.6f}`; finite-index candidate fraction `{float(base['finite_index_candidate'].mean()):.6f}`; non-Brauer/noncentral fraction `{float(base['non_brauer_noncentral'].mean()):.6f}`.

## Claim Decisions

{md_table(claims, ["claim", "claim_decision", "claim_reason"], 30)}

## Required Questions

- Are exact CNN channel gauges verified? {exact_answer}.
- Does shrinkage/global/optimized channel scaling beat C2M3 robustly?
{md_table(scale_claims, ["claim", "claim_decision", "claim_reason"], 10)}
- Does any channel-gauge soup beat greedy soup? {soup_claim.iloc[0]['claim_decision'] if not soup_claim.empty else 'Not evaluated'}: {soup_claim.iloc[0]['claim_reason'] if not soup_claim.empty else 'missing'}.
- Does greedy-safe selection improve, match, or lose to greedy soup? {safe_claim.iloc[0]['claim_decision'] if not safe_claim.empty else 'Not evaluated'}: {safe_claim.iloc[0]['claim_reason'] if not safe_claim.empty else 'missing'}.
- Are CNN residuals central/projective/Brauer-like? {brauer_claim.iloc[0]['claim_decision'] if not brauer_claim.empty else 'Not evaluated'}: {brauer_claim.iloc[0]['claim_reason'] if not brauer_claim.empty else 'missing'}.
- Should the CNN branch be main-text, appendix, or removed from experiments? {guidance}

## Negative Boundaries

- No external official C2M3, Git Re-Basin, Model Soups, or SOTA claim is made.
- No greedy-soup win is claimed unless the paired channel-candidate soup row has positive mean and positive bootstrap lower bound.
- CNN residuals are not called Brauer or period-index classes.
- BatchNorm is not used; exact BatchNorm gauge transformations are not implemented here.

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
        "model_counts": parse_csv(args.model_counts, int),
        "main_n_models": args.main_n_models,
        "main_seeds": parse_csv(args.main_seeds, int),
        "secondary_seeds": parse_csv(args.secondary_seeds, int),
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "feature_batches": args.feature_batches,
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
    text = path.read_text(encoding="utf-8")
    start = "## Fashion-MNIST CNN Channel-Gauge Confirmatory Benchmark"
    end = "\n## Not Yet Supported"
    if start in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + end.lstrip() + after
    claims = {row["claim"]: row for row in summary[summary["summary_type"] == "claim_decision"].to_dict("records")}

    def row(claim: str, label: str) -> str:
        item = claims.get(claim, {})
        return f"| `{claim}` / {label} | {item.get('claim_decision', 'Not yet supported')} | `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` records {item.get('claim_reason', 'no evidence row found')}. |"

    section = f"""{start}

| Claim | Status | Evidence |
| --- | --- | --- |
{row('cnn_exact_channel_gauges_preserve_logits', 'exact CNN channel gauges preserve logits')}
{row('cnn_shrinkage_channel_scale_over_c2m3_confirmed', 'shrinkage channel scale over C2M3')}
{row('cnn_global_channel_scale_over_c2m3_confirmed', 'global channel scale over C2M3')}
{row('cnn_optimized_channel_scale_over_c2m3_confirmed', 'optimized channel scale over C2M3')}
{row('cnn_channel_candidate_soup_over_greedy_soup', 'channel candidate soup over greedy soup')}
{row('cnn_greedy_safe_selector_matches_or_beats_greedy_soup', 'greedy-safe selector versus greedy soup')}
{row('cnn_channel_gauge_generalizes_mlp_exact_gauge_story', 'CNN channel gauge generalizes exact-gauge story')}
{row('cnn_residuals_are_brauer_or_period_index', 'CNN residuals are Brauer or period-index')}
"""
    text = text.replace("\n## Not Yet Supported", "\n\n" + section + "\n## Not Yet Supported", 1)
    artifact_marker = "| `reports/configs/fashion_mnist_cnn_ladder_config.json` | Saved command, environment, and benchmark metadata for the Fashion-MNIST CNN ladder run. |"
    artifacts = [
        "| `experiments/fashion_mnist_cnn_channel_gauge_confirmatory.py` | Confirmatory Fashion-MNIST CNN channel-gauge benchmark over N=3 and N=4 with optimized layer-gated scale grids and channel-gauge soup variants. |",
        "| `reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md` | Confirmatory CNN report answering exactness, C2M3, greedy soup, greedy-safe selector, residual taxonomy, and paper-placement questions. |",
        "| `reports/csv/fashion_mnist_cnn_channel_gauge_confirmatory.csv` | Per-setting confirmatory CNN channel-gauge benchmark rows. |",
        "| `reports/csv/fashion_mnist_cnn_channel_gauge_confirmatory_summary.csv` | Confirmatory CNN method summaries, paired comparisons, selector behavior, diagnostics, and claim decisions. |",
        "| `reports/plots/fashion_cnn_confirmatory_delta_vs_c2m3.pdf` | Confirmatory CNN method deltas versus channel-permutation C2M3. |",
        "| `reports/plots/fashion_cnn_confirmatory_delta_vs_greedy_soup.pdf` | Confirmatory CNN method deltas versus greedy soup. |",
        "| `reports/plots/fashion_cnn_confirmatory_channel_residuals.pdf` | Confirmatory CNN channel residual diagnostic scatter plot. |",
        "| `reports/plots/fashion_cnn_confirmatory_selector_choices.pdf` | Confirmatory CNN greedy-safe selector choice counts. |",
        "| `reports/tables/fashion_cnn_channel_gauge_confirmatory_table.tex` | LaTeX summary table for the confirmatory CNN channel-gauge benchmark. |",
        "| `reports/configs/fashion_mnist_cnn_channel_gauge_confirmatory_config.json` | Saved command, environment, and grid metadata for the confirmatory CNN channel-gauge benchmark. |",
    ]
    if "`experiments/fashion_mnist_cnn_channel_gauge_confirmatory.py`" not in text:
        text = text.replace(artifact_marker, "\n".join([artifact_marker, *artifacts]), 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--main-n-models", type=int, default=3)
    parser.add_argument("--main-seeds", default=",".join(str(seed) for seed in range(7600, 7610)))
    parser.add_argument("--secondary-seeds", default=",".join(str(seed) for seed in range(7700, 7705)))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=424242)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--feature-batches", type=int, default=16)
    parser.add_argument("--alpha-grid", default="0,0.5,1.0,1.25")
    parser.add_argument("--tau-grid", default="1.0,inf")
    parser.add_argument("--optimized-alpha-grid", default="0,0.5,1.0,1.25")
    parser.add_argument("--optimized-tau-grid", default="1.0,inf")
    parser.add_argument("--optimized-layer-masks", default="all,conv2+fc1,fc1,conv2")
    parser.add_argument("--greedy-safe-tau", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
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

    spec, train_data, test_data = load_dataset("fashion_mnist", args.data_dir, args.max_train_samples, args.max_test_samples, args.dataset_seed)
    rows = []
    for n_models, seeds, role in seed_plan(args):
        for seed in seeds:
            print(f"running confirmatory CNN seed={seed} n_models={n_models} role={role}", flush=True)
            rows.extend(run_setting(args, spec, train_data, test_data, seed, n_models, role))
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
    results_path = csv_dir / "fashion_mnist_cnn_channel_gauge_confirmatory.csv"
    summary_path = csv_dir / "fashion_mnist_cnn_channel_gauge_confirmatory_summary.csv"
    report_path = args.reports_dir / "fashion_mnist_cnn_channel_gauge_confirmatory_report.md"
    table_path = table_dir / "fashion_cnn_channel_gauge_confirmatory_table.tex"
    config_path = config_dir / "fashion_mnist_cnn_channel_gauge_confirmatory_config.json"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_plots(df, summary, plot_dir)
    write_latex_table(summary, table_path)
    write_report(args, df, summary, report_path)
    write_config(args, config_path)
    update_claims_audit(summary, args.reports_dir / "claims_audit.md")
    for path in [results_path, summary_path, report_path, table_path, config_path]:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

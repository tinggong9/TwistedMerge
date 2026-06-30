#!/usr/bin/env python
"""Expanded rotated/colored-MNIST bridge benchmark for CNN channel gauges."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.cifar_or_colored_mnist_feasibility import (  # noqa: E402
    average_eval,
    bootstrap_mean_ci,
    build_gauged_models,
    collect_features,
    feature_alignment_residual,
    greedy_soup,
    md_table,
    pairwise_perms,
    parse_csv,
    reference_log_scales,
    split_train_val,
    sync_perms,
    zero_logs,
)
from src.cnn_channel_gauge import (  # noqa: E402
    CnnGaugeSpec,
    SmallFashionCNN,
    average_cnn_models,
    count_parameters,
    inference_cost_units,
)
from src.greedy_safe_selector import tau_fixed_selector  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    cycle_score,
    device_from_arg,
    evaluate_model,
    load_dataset,
    make_loader,
    require_torch,
    require_torchvision,
    set_seed,
    train_model,
)


LAYERS = ("conv1", "conv2", "fc1")
BASELINES = ("c2m3_channel_synchronization", "greedy_soup", "weight_average")
METHOD_ORDER = [
    "weight_average",
    "git_rebasin_channel_alignment",
    "c2m3_channel_synchronization",
    "positive_channel_scale",
    "shrinkage_channel_scale",
    "global_channel_scale",
    "optimized_channel_scale",
    "greedy_soup",
    "greedy_safe_selector",
]
INT_COLUMNS = {
    "n_rows",
    "n_settings",
    "n_pairs",
    "n_models",
    "seed",
    "epochs",
    "train_samples",
    "test_samples",
    "accuracy_wins",
    "accuracy_ties",
    "accuracy_losses",
    "fixed_settings_positive",
    "fixed_settings_total",
}


class RotatedDataset(require_torch()[0].utils.data.Dataset):
    def __init__(self, base, angle_degrees: float):
        self.base = base
        self.angle_degrees = float(angle_degrees)
        _torchvision, transforms = require_torchvision()
        self.functional = transforms.functional
        self.interpolation = transforms.InterpolationMode.BILINEAR

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.functional.rotate(x, self.angle_degrees, interpolation=self.interpolation), y


class ColoredMNISTDataset(require_torch()[0].utils.data.Dataset):
    """Deterministic label-palette colored digits on a black background."""

    def __init__(self, base):
        self.base = base
        self.palette = np.asarray(
            [
                (0.90, 0.10, 0.12),
                (0.10, 0.55, 0.90),
                (0.18, 0.72, 0.24),
                (0.95, 0.65, 0.12),
                (0.58, 0.25, 0.82),
                (0.05, 0.78, 0.75),
                (0.92, 0.35, 0.55),
                (0.60, 0.60, 0.18),
                (0.25, 0.35, 0.90),
                (0.65, 0.28, 0.10),
            ],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        torch, _, _ = require_torch()
        x, y = self.base[idx]
        color = torch.tensor(self.palette[int(y) % len(self.palette)], dtype=x.dtype).view(3, 1, 1)
        return x.repeat(3, 1, 1) * color, y


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


def make_model(spec: CnnGaugeSpec) -> SmallFashionCNN:
    return SmallFashionCNN(spec)


def layer_widths(spec: CnnGaugeSpec) -> dict[str, int]:
    return {"conv1": spec.conv1_channels, "conv2": spec.conv2_channels, "fc1": spec.hidden_units}


def bridge_spec(in_channels: int) -> CnnGaugeSpec:
    return CnnGaugeSpec(in_channels=in_channels, spatial_after_pool=7)


def global_log_scale_synchronization(features_by_model, synced, refs, n_models: int, widths: dict[str, int]) -> tuple[dict[str, np.ndarray], float]:
    logs = {}
    residuals = []
    for layer in LAYERS:
        width = widths[layer]
        ref = refs[layer]
        aligned = {idx: features_by_model[idx][layer][:, np.asarray(synced[layer][idx], dtype=int)] for idx in range(n_models)}
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
                source = aligned[i][:, unit]
                target = aligned[j][:, unit]
                denom = max(float(np.dot(source, source)), 1e-12)
                scale = float(np.dot(source, target) / denom)
                if not np.isfinite(scale) or scale <= 0.0:
                    scale = 1.0
                b.append(np.log(np.clip(scale, 1e-3, 1e3)))
            gauge = np.zeros(n_models, dtype=float)
            gauge[ref] = float(max(n_models, 1))
            A.append(gauge)
            b.append(0.0)
            A_arr = np.vstack(A)
            b_arr = np.asarray(b, dtype=float)
            sol, *_ = np.linalg.lstsq(A_arr, b_arr, rcond=None)
            sol = sol - sol[ref]
            layer_logs[:, unit] = sol
            residuals.extend((A_arr[:-1] @ sol - b_arr[:-1]).tolist())
        logs[layer] = layer_logs
    rms = float(np.sqrt(np.mean(np.asarray(residuals) ** 2))) if residuals else 0.0
    return logs, rms


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
        key = item.strip().lower()
        if key in {"all", "*"}:
            masks.append(tuple(LAYERS))
        elif key in {"none", "identity"}:
            masks.append(tuple())
        else:
            layers = tuple(part.strip() for part in item.split("+") if part.strip())
            bad = [layer for layer in layers if layer not in LAYERS]
            if bad:
                raise ValueError(f"unknown layer mask item(s): {bad}")
            masks.append(layers)
    return masks


def select_scale_grid(models, synced, raw_logs, alpha_grid, tau_grid, val_loader, device):
    best = None
    for alpha in alpha_grid:
        for tau in tau_grid:
            logs = shrink_logs(raw_logs, alpha, tau)
            gauged = build_gauged_models(models, synced, logs)
            candidate = average_cnn_models(gauged)
            val = evaluate_model(candidate, val_loader, device)
            key = (float(val["accuracy"]), -float(val["loss"]))
            if best is None or key > best[0]:
                best = (key, logs, gauged, candidate, val, float(alpha), float(tau))
    assert best is not None
    return best[1:]


def choose_optimized_logs(*, models, synced, reference_logs, global_logs, alpha_grid, tau_grid, layer_masks, val_loader, device):
    best = None
    for source_name, source_logs in [("reference", reference_logs), ("global", global_logs)]:
        for alpha in alpha_grid:
            for tau in tau_grid:
                shrunk = shrink_logs(source_logs, alpha, tau)
                for mask in layer_masks:
                    logs = layer_masked_logs(shrunk, mask)
                    gauged = build_gauged_models(models, synced, logs)
                    candidate = average_cnn_models(gauged)
                    val = evaluate_model(candidate, val_loader, device)
                    key = (float(val["accuracy"]), -float(val["loss"]), len(mask), source_name)
                    if best is None or key > best[0]:
                        best = (key, logs, gauged, candidate, val, source_name, float(alpha), float(tau), "+".join(mask) if mask else "identity")
    assert best is not None
    _key, logs, gauged, candidate, val, source_name, alpha, tau, mask = best
    return logs, gauged, candidate, val, source_name, alpha, tau, mask


def scale_diagnostics(logs: dict[str, np.ndarray], prefix: str = "") -> dict[str, float]:
    out = {}
    variances = []
    mean_abs = []
    for layer in LAYERS:
        values = np.asarray(logs[layer], dtype=float)
        out[f"{prefix}{layer}_log_scale_variance"] = float(np.var(values))
        out[f"{prefix}{layer}_mean_abs_log_scale"] = float(np.mean(np.abs(values)))
        variances.append(out[f"{prefix}{layer}_log_scale_variance"])
        mean_abs.append(out[f"{prefix}{layer}_mean_abs_log_scale"])
    out[f"{prefix}mean_log_scale_variance"] = float(np.mean(variances))
    out[f"{prefix}mean_abs_log_scale"] = float(np.mean(mean_abs))
    return out


def cycle_diagnostics(pairwise, n_models: int, widths: dict[str, int]) -> dict[str, float]:
    out = {}
    scores = []
    for layer in LAYERS:
        score, _rows = cycle_score(pairwise[layer], n_models, widths[layer])
        out[f"{layer}_permutation_cycle_score"] = float(score)
        scores.append(float(score))
    out["channel_permutation_cycle_score"] = float(np.mean(scores)) if scores else 0.0
    return out


def add_row(rows, base, method, val, test, *, extra=None):
    row = {
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
        row.update(extra)
    rows.append(row)


def build_dataset_variants(args):
    _spec, base_train, base_test = load_dataset("mnist", args.data_dir, args.max_train_samples, args.max_test_samples, args.dataset_seed)
    variants = []
    for angle in parse_csv(args.angles, float):
        variants.append(
            {
                "dataset": "rotated_mnist",
                "dataset_variant": f"rotated_mnist_{int(angle) if float(angle).is_integer() else angle}deg",
                "angle_degrees": float(angle),
                "coloring": "none",
                "spec": bridge_spec(1),
                "train": RotatedDataset(base_train, angle),
                "test": RotatedDataset(base_test, angle),
            }
        )
    if args.include_colored:
        variants.append(
            {
                "dataset": "colored_mnist",
                "dataset_variant": "colored_mnist_label_palette",
                "angle_degrees": 0.0,
                "coloring": "label_palette",
                "spec": bridge_spec(3),
                "train": ColoredMNISTDataset(base_train),
                "test": ColoredMNISTDataset(base_test),
            }
        )
    return variants


def planned_settings(args, variants):
    main_seeds = parse_csv(args.main_seeds, int)
    secondary_seeds = parse_csv(args.secondary_seeds, int)
    model_counts = parse_csv(args.model_counts, int)
    main_variant = f"rotated_mnist_{int(args.main_angle) if float(args.main_angle).is_integer() else args.main_angle}deg"
    settings = []
    for variant in variants:
        for n_models in model_counts:
            is_main = variant["dataset_variant"] == main_variant and n_models == int(args.main_n_models)
            seeds = main_seeds if is_main else secondary_seeds
            if not args.secondary_colored and variant["dataset"] == "colored_mnist" and not is_main:
                seeds = secondary_seeds[:1]
            for seed in seeds:
                settings.append((variant, int(n_models), int(seed), "main" if is_main else "secondary"))
    return settings


def run_setting(args, variant: dict, n_models: int, seed: int, setting_role: str) -> list[dict]:
    device = device_from_arg(args.device)
    spec = variant["spec"]
    widths = layer_widths(spec)
    train_subset, val_subset = split_train_val(variant["train"], args.val_fraction, seed + 41)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 700)
    test_loader = make_loader(variant["test"], args.batch_size, shuffle=False, seed=seed + 900)
    match_loader = make_loader(train_subset, args.batch_size, shuffle=False, seed=seed + 100)

    models = []
    individual = []
    for idx in range(n_models):
        model_seed = seed + idx * 1009 + 17
        set_seed(model_seed)
        model = make_model(spec)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed)
        train_model(model, train_loader, args.epochs, args.lr, device)
        metrics = evaluate_model(model, test_loader, device)
        individual.append(float(metrics["accuracy"]))
        model.to("cpu")
        models.append(model)

    features = {idx: collect_features(model, match_loader, device, widths, args.feature_batches) for idx, model in enumerate(models)}
    pairwise = pairwise_perms(features, n_models, widths)
    refs, synced, disagreements = sync_perms(pairwise, n_models)
    ref0_synced = {layer: {idx: pairwise[layer][(0, idx)] for idx in range(n_models)} for layer in LAYERS}
    reference_logs = reference_log_scales(features, synced, refs, n_models, widths)
    global_logs, global_rms = global_log_scale_synchronization(features, synced, refs, n_models, widths)
    cycles = cycle_diagnostics(pairwise, n_models, widths)
    pair_residual = feature_alignment_residual(features, synced, n_models)
    eligible = float(np.max(individual)) >= args.bridge_threshold
    base = {
        "setting_id": f"{variant['dataset_variant']}_N{n_models}_S{seed}",
        "dataset": variant["dataset"],
        "dataset_variant": variant["dataset_variant"],
        "setting_role": setting_role,
        "architecture": "small_relu_cnn_no_batchnorm",
        "n_models": n_models,
        "seed": seed,
        "epochs": args.epochs,
        "train_samples": len(variant["train"]),
        "test_samples": len(variant["test"]),
        "val_fraction": args.val_fraction,
        "angle_degrees": variant["angle_degrees"],
        "coloring": variant["coloring"],
        "individual_accuracy_mean": float(np.mean(individual)),
        "individual_accuracy_max": float(np.max(individual)),
        "bridge_accuracy_threshold": args.bridge_threshold,
        "bridge_claims_allowed": bool(eligible),
        "feasibility_status": "bridge_claims_allowed" if eligible else "bridge_below_accuracy_threshold",
        "parameter_count": count_parameters(models[0]),
        "inference_cost_units": inference_cost_units(spec),
        "pairwise_activation_alignment_residual": pair_residual,
        "global_channel_scale_sync_residual": global_rms,
        "sync_disagreement_mean": float(np.mean(list(disagreements.values()))),
        "central_projective_candidate": False,
        "finite_index_candidate": False,
        "non_brauer_noncentral": True,
        "channel_residual_taxonomy": "bridge_non_brauer_no_central_projective_candidate",
        **cycles,
    }
    rows = []

    _weight_model, weight_val, weight_test = average_eval(models, val_loader, test_loader, device)
    add_row(rows, base, "weight_average", weight_val, weight_test, extra={"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True, "is_soup": False})

    pairwise_aligned = build_gauged_models(models, ref0_synced, zero_logs(n_models, widths))
    _pair_model, pair_val, pair_test = average_eval(pairwise_aligned, val_loader, test_loader, device)
    add_row(rows, base, "git_rebasin_channel_alignment", pair_val, pair_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": False})

    c2m3_models = build_gauged_models(models, synced, zero_logs(n_models, widths))
    _c2m3_model, c2m3_val, c2m3_test = average_eval(c2m3_models, val_loader, test_loader, device)
    add_row(rows, base, "c2m3_channel_synchronization", c2m3_val, c2m3_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": False})

    positive_models = build_gauged_models(models, synced, reference_logs)
    _positive_model, positive_val, positive_test = average_eval(positive_models, val_loader, test_loader, device)
    add_row(rows, base, "positive_channel_scale", positive_val, positive_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": False, "scale_source": "reference_raw", "selected_alpha": 1.0, "selected_tau": float("inf"), "selected_layer_mask": "all", **scale_diagnostics(reference_logs)})

    alpha_grid = parse_csv(args.alpha_grid, float)
    tau_grid = [float("inf") if item.lower() == "inf" else float(item) for item in parse_csv(args.tau_grid, str)]
    shrink_logs_selected, _shrink_models, shrink_model, shrink_val, shrink_alpha, shrink_tau = select_scale_grid(models, synced, reference_logs, alpha_grid, tau_grid, val_loader, device)
    shrink_test = evaluate_model(shrink_model, test_loader, device)
    add_row(rows, base, "shrinkage_channel_scale", shrink_val, shrink_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": False, "scale_source": "reference_shrinkage_validation_grid", "selected_alpha": shrink_alpha, "selected_tau": shrink_tau, "selected_layer_mask": "all", **scale_diagnostics(shrink_logs_selected)})

    global_logs_selected, _global_models, global_model, global_val, global_alpha, global_tau = select_scale_grid(models, synced, global_logs, alpha_grid, tau_grid, val_loader, device)
    global_test = evaluate_model(global_model, test_loader, device)
    add_row(rows, base, "global_channel_scale", global_val, global_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": False, "scale_source": "global_log_scale_sync_validation_grid", "selected_alpha": global_alpha, "selected_tau": global_tau, "selected_layer_mask": "all", **scale_diagnostics(global_logs_selected)})

    optimized_alpha_grid = parse_csv(args.optimized_alpha_grid, float)
    optimized_tau_grid = [float("inf") if item.lower() == "inf" else float(item) for item in parse_csv(args.optimized_tau_grid, str)]
    layer_masks = parse_layer_masks(args.optimized_layer_masks)
    optimized_logs, _optimized_models, optimized_model, optimized_val, optimized_source, optimized_alpha, optimized_tau, optimized_mask = choose_optimized_logs(
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
    add_row(rows, base, "optimized_channel_scale", optimized_val, optimized_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "is_soup": False, "scale_source": f"optimized_layer_grid_{optimized_source}", "selected_alpha": optimized_alpha, "selected_tau": optimized_tau, "selected_layer_mask": optimized_mask, **scale_diagnostics(optimized_logs)})

    soup = greedy_soup(models, [f"original:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "greedy_soup", soup["val"], soup["test"], extra={"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True, "is_soup": True, "soup_selected_labels": json.dumps(soup["selected_labels"]), "soup_ingredient_count": len(soup["selected_indices"])})

    by_method = {row["method"]: row for row in rows}
    selector_pool = [
        "optimized_channel_scale",
        "global_channel_scale",
        "shrinkage_channel_scale",
        "positive_channel_scale",
        "c2m3_channel_synchronization",
        "git_rebasin_channel_alignment",
        "weight_average",
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
    git_rebasin = by_method["git_rebasin_channel_alignment"]
    for row in rows:
        row["accuracy_delta_vs_c2m3"] = row["accuracy"] - c2m3["accuracy"]
        row["accuracy_delta_vs_greedy_soup"] = row["accuracy"] - greedy["accuracy"]
        row["accuracy_delta_vs_weight_average"] = row["accuracy"] - weight["accuracy"]
        row["accuracy_delta_vs_git_rebasin"] = row["accuracy"] - git_rebasin["accuracy"]
        row["validation_delta_vs_c2m3"] = row["val_accuracy"] - c2m3["val_accuracy"]
        row["validation_delta_vs_greedy_soup"] = row["val_accuracy"] - greedy["val_accuracy"]
    return rows


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


def method_summary(scope: str, df: pd.DataFrame, n_bootstrap: int, seed_offset: int) -> list[dict]:
    rows = []
    for method in METHOD_ORDER:
        group = df[df["method"] == method]
        if group.empty:
            continue
        low, high = bootstrap_mean_ci(group["accuracy"], n_bootstrap, seed=seed_offset + len(rows))
        rows.append(
            {
                "summary_type": "method_summary",
                "scope": scope,
                "method": method,
                "n_rows": int(len(group)),
                "n_settings": int(group["setting_id"].nunique()),
                "mean_val_accuracy": float(group["val_accuracy"].mean()),
                "mean_test_accuracy": float(group["accuracy"].mean()),
                "test_accuracy_standard_error": standard_error(group["accuracy"]),
                "test_accuracy_ci_low": low,
                "test_accuracy_ci_high": high,
                "mean_individual_accuracy_max": float(group["individual_accuracy_max"].mean()),
                "mean_delta_vs_c2m3": float(group["accuracy_delta_vs_c2m3"].mean()),
                "mean_delta_vs_greedy_soup": float(group["accuracy_delta_vs_greedy_soup"].mean()),
                "mean_delta_vs_weight_average": float(group["accuracy_delta_vs_weight_average"].mean()),
                "mean_delta_vs_git_rebasin": float(group["accuracy_delta_vs_git_rebasin"].mean()),
                "bridge_claims_allowed_fraction": float(group["bridge_claims_allowed"].astype(bool).mean()),
                "exact_relu_channel_gauge": bool(group["exact_relu_channel_gauge"].fillna(False).astype(bool).all()),
                "single_model": bool(group["single_model"].fillna(False).astype(bool).all()),
                "capacity_matched": bool(group["capacity_matched"].fillna(False).astype(bool).all()),
                "is_soup": bool(group["is_soup"].fillna(False).astype(bool).any()),
                "selector_choice_counts": json.dumps({str(k): int(v) for k, v in group.get("selector_chose", pd.Series(dtype=object)).dropna().value_counts().items()}),
            }
        )
    return rows


def paired_summary(scope: str, df: pd.DataFrame, n_bootstrap: int, seed_offset: int) -> list[dict]:
    rows = []
    pivot = df.pivot_table(index=["setting_id", "dataset", "dataset_variant", "n_models", "seed"], columns="method", values=["accuracy", "loss"], aggfunc="first")
    pivot.columns = [f"{metric}__{method}" for metric, method in pivot.columns]
    pivot = pivot.reset_index()
    for method in METHOD_ORDER:
        for baseline in BASELINES:
            if method == baseline:
                continue
            a = f"accuracy__{method}"
            b = f"accuracy__{baseline}"
            la = f"loss__{method}"
            lb = f"loss__{baseline}"
            if a not in pivot or b not in pivot:
                continue
            clean = pivot[[a, b, la, lb, "dataset_variant", "n_models"]].dropna()
            if clean.empty:
                continue
            delta = clean[a] - clean[b]
            loss_delta = clean[la] - clean[lb]
            wins = int((delta > 0).sum())
            ties = int((delta == 0).sum())
            losses = int((delta < 0).sum())
            low, high = bootstrap_mean_ci(delta, n_bootstrap, seed=seed_offset + 100 + len(rows))
            fixed_positive = 0
            fixed_total = 0
            for _setting, group in clean.groupby(["dataset_variant", "n_models"]):
                fixed_total += 1
                if float((group[a] - group[b]).mean()) > 0:
                    fixed_positive += 1
            rows.append(
                {
                    "summary_type": "paired_comparison",
                    "scope": scope,
                    "comparison": f"{method}_vs_{baseline}",
                    "method": method,
                    "baseline": baseline,
                    "n_pairs": int(len(clean)),
                    "paired_mean_accuracy_delta": float(delta.mean()),
                    "paired_accuracy_delta_ci_low": low,
                    "paired_accuracy_delta_ci_high": high,
                    "paired_mean_loss_delta": float(loss_delta.mean()),
                    "accuracy_wins": wins,
                    "accuracy_ties": ties,
                    "accuracy_losses": losses,
                    "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
                    "fixed_settings_positive": fixed_positive,
                    "fixed_settings_total": fixed_total,
                }
            )
    return rows


def claim_rows(summary_rows: list[dict], df: pd.DataFrame) -> list[dict]:
    paired = {(row.get("scope"), row.get("comparison")): row for row in summary_rows if row.get("summary_type") == "paired_comparison"}
    main_boundary = paired.get(("main_rotated25_N3", "greedy_soup_vs_c2m3_channel_synchronization"), {})
    overall_boundary = paired.get(("overall", "greedy_soup_vs_c2m3_channel_synchronization"), {})
    selector_boundary = paired.get(("overall", "greedy_safe_selector_vs_greedy_soup"), {})
    min_base = float(df.groupby("setting_id")["individual_accuracy_max"].first().min())
    rows = [
        {
            "summary_type": "claim_decision",
            "claim": "bridge_accuracy_gate",
            "claim_decision": "Supported limited" if min_base >= 0.80 else "Mixed",
            "claim_reason": f"minimum setting-level individual max accuracy={min_base:.4f}; bridge threshold=0.8000",
        }
    ]
    if main_boundary:
        mean = float(main_boundary["paired_mean_accuracy_delta"])
        low = float(main_boundary["paired_accuracy_delta_ci_low"])
        rows.append(
            {
                "summary_type": "claim_decision",
                "claim": "main_bridge_c2m3_vs_greedy_boundary",
                "claim_decision": "Supported limited" if mean > 0.0 and low > 0.0 else "Descriptive/mixed",
                "claim_reason": f"main rotated-25 N=3 greedy soup delta vs C2M3={mean:.6f}, CI=[{low:.6f},{float(main_boundary['paired_accuracy_delta_ci_high']):.6f}], n_pairs={int(main_boundary['n_pairs'])}",
            }
        )
    if overall_boundary:
        mean = float(overall_boundary["paired_mean_accuracy_delta"])
        low = float(overall_boundary["paired_accuracy_delta_ci_low"])
        rows.append(
            {
                "summary_type": "claim_decision",
                "claim": "overall_bridge_c2m3_vs_greedy_boundary",
                "claim_decision": "Supported descriptive" if mean > 0.0 else "Mixed/not supported",
                "claim_reason": f"overall greedy soup delta vs C2M3={mean:.6f}, CI=[{low:.6f},{float(overall_boundary['paired_accuracy_delta_ci_high']):.6f}], n_pairs={int(overall_boundary['n_pairs'])}; bridge-only, no CIFAR/general-vision implication",
            }
        )
    if selector_boundary:
        rows.append(
            {
                "summary_type": "claim_decision",
                "claim": "greedy_safe_selector_boundary",
                "claim_decision": "Supported descriptive",
                "claim_reason": f"greedy-safe selector mean delta vs greedy soup={float(selector_boundary['paired_mean_accuracy_delta']):.6f}; selector uses validation metrics only",
            }
        )
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "bridge_not_cifar_general_vision",
            "claim_decision": "Boundary recorded",
            "claim_reason": "Rotated/colored MNIST bridge datasets do not imply CIFAR or general vision performance.",
        }
    )
    return rows


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    scopes = [("overall", df), ("main_rotated25_N3", df[df["setting_role"] == "main"]), ("rotated_mnist", df[df["dataset"] == "rotated_mnist"])]
    if (df["dataset"] == "colored_mnist").any():
        scopes.append(("colored_mnist", df[df["dataset"] == "colored_mnist"]))
    for idx, (scope, group) in enumerate(scopes):
        if group.empty:
            continue
        rows.extend(method_summary(scope, group, n_bootstrap, seed_offset=20000 + idx * 1000))
        rows.extend(paired_summary(scope, group, n_bootstrap, seed_offset=30000 + idx * 1000))
    rows.extend(claim_rows(rows, df))
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


def local_md_table(df: pd.DataFrame, cols: list[str], max_rows=80) -> str:
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


def write_tex_table(summary: pd.DataFrame, path: Path):
    methods = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "main_rotated25_N3")].copy()
    methods = methods[methods["method"].isin(["weight_average", "git_rebasin_channel_alignment", "c2m3_channel_synchronization", "positive_channel_scale", "shrinkage_channel_scale", "global_channel_scale", "optimized_channel_scale", "greedy_soup", "greedy_safe_selector"])]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Rows & Test acc. & $\\Delta$ C2M3 & $\\Delta$ greedy \\\\",
        "\\midrule",
    ]
    for row in methods.to_dict("records"):
        label = str(row["method"]).replace("_", "\\_")
        lines.append(
            f"{label} & {int(row['n_rows'])} & {float(row['mean_test_accuracy']):.4f} & {float(row['mean_delta_vs_c2m3']):+.4f} & {float(row['mean_delta_vs_greedy_soup']):+.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path):
    methods = summary[summary["summary_type"] == "method_summary"].copy()
    paired = summary[summary["summary_type"] == "paired_comparison"].copy()
    claims = summary[summary["summary_type"] == "claim_decision"].copy()
    main_methods = methods[methods["scope"] == "main_rotated25_N3"]
    overall_pairs = paired[(paired["scope"] == "overall") & paired["comparison"].isin(["greedy_soup_vs_c2m3_channel_synchronization", "greedy_safe_selector_vs_greedy_soup", "optimized_channel_scale_vs_c2m3_channel_synchronization"])]
    setting_summary = (
        df.groupby(["dataset", "dataset_variant", "n_models", "setting_role"])
        .agg(
            n_settings=("setting_id", "nunique"),
            mean_individual_accuracy_max=("individual_accuracy_max", "mean"),
            mean_c2m3_accuracy=("accuracy", lambda s: float(df.loc[s.index][df.loc[s.index, "method"] == "c2m3_channel_synchronization"]["accuracy"].mean())),
            mean_greedy_accuracy=("accuracy", lambda s: float(df.loc[s.index][df.loc[s.index, "method"] == "greedy_soup"]["accuracy"].mean())),
        )
        .reset_index()
    )
    report = f"""# Bridge Dataset Channel-Gauge Expansion

Generated by `experiments/bridge_dataset_channel_gauge_expansion.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- Bridge datasets only: rotated-MNIST angles `{args.angles}` and colored-MNIST included: `{args.include_colored}`.
- Main bridge setting: rotated-MNIST `{args.main_angle}` degrees, `N={args.main_n_models}`, seeds `{args.main_seeds}`.
- Secondary settings use seeds `{args.secondary_seeds}` and model counts `{args.model_counts}`.
- Test set: `{df["test_samples"].max()}` examples per setting. This is the full MNIST test set when `--max-test-samples 0`.
- Methods: weight average, Git-ReBasin-style pairwise channel alignment, C2M3-style channel synchronization, positive/shrinkage/global/optimized exact channel scaling, greedy soup, and greedy-safe selector.
- This report supports bridge-dataset pattern checks only. It does not imply CIFAR or general vision performance.

## Claim Decisions

{local_md_table(claims, ["claim", "claim_decision", "claim_reason"], 20)}

## Main Setting Method Summary

{local_md_table(main_methods, ["method", "n_rows", "n_settings", "mean_individual_accuracy_max", "mean_val_accuracy", "mean_test_accuracy", "mean_delta_vs_c2m3", "mean_delta_vs_greedy_soup", "test_accuracy_ci_low", "test_accuracy_ci_high"], 30)}

## Overall Boundary Comparisons

{local_md_table(overall_pairs, ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "accuracy_wins", "accuracy_ties", "accuracy_losses"], 20)}

## Setting Coverage

{local_md_table(setting_summary, ["dataset", "dataset_variant", "n_models", "setting_role", "n_settings", "mean_individual_accuracy_max", "mean_c2m3_accuracy", "mean_greedy_accuracy"], 80)}

## Interpretation

- Bridge datasets support checking whether the C2M3-versus-greedy boundary pattern persists across simple MNIST-derived shifts.
- Greedy soup is a single-model soup baseline selected with validation data; greedy-safe selection also uses validation metrics only.
- Channel synchronization and positive channel scaling are exact ReLU reparameterizations for the no-BatchNorm CNN, but that does not guarantee improved accuracy.
- Rotated/colored-MNIST results must not be promoted to CIFAR or general vision claims.

## Environment

```json
{json.dumps({**capture_environment(), "git_commit": git_commit(), "dirty_worktree": git_dirty()}, indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", default="15,25,45")
    parser.add_argument("--include-colored", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--secondary-colored", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--main-angle", type=float, default=25.0)
    parser.add_argument("--main-n-models", type=int, default=3)
    parser.add_argument("--main-seeds", default="8400,8401,8402,8403,8404,8405,8406,8407,8408,8409")
    parser.add_argument("--secondary-seeds", default="8500")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=6000)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--bridge-threshold", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=515151)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--feature-batches", type=int, default=8)
    parser.add_argument("--alpha-grid", default="0,0.5,1")
    parser.add_argument("--tau-grid", default="0.5,1,inf")
    parser.add_argument("--optimized-alpha-grid", default="0,0.5,1")
    parser.add_argument("--optimized-tau-grid", default="0.5,inf")
    parser.add_argument("--optimized-layer-masks", default="all,conv1+conv2,fc1")
    parser.add_argument("--greedy-safe-tau", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    env_prefix = [f"{name}={os.environ[name]}" for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR") if os.environ.get(name)]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    variants = build_dataset_variants(args)
    settings = planned_settings(args, variants)
    rows = []
    for variant, n_models, seed, role in settings:
        print(f"running {variant['dataset_variant']} n_models={n_models} seed={seed} role={role}", flush=True)
        rows.extend(run_setting(args, variant, n_models, seed, role))

    df = pd.DataFrame(rows)
    summary = summarize(df, args.bootstrap_samples)
    csv_dir = args.reports_dir / "csv"
    table_dir = args.reports_dir / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "bridge_dataset_channel_gauge_expansion.csv"
    summary_path = csv_dir / "bridge_dataset_channel_gauge_expansion_summary.csv"
    report_path = args.reports_dir / "bridge_dataset_channel_gauge_expansion.md"
    table_path = table_dir / "bridge_dataset_channel_gauge_expansion.tex"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(args, df, summary, report_path)
    write_tex_table(summary, table_path)
    for path in [results_path, summary_path, report_path, table_path]:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

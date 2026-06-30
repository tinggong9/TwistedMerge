#!/usr/bin/env python
"""Fashion-MNIST small-CNN channel-gauge ladder benchmark."""

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

from src.cnn_channel_gauge import (  # noqa: E402
    apply_inverse_positive_alignment,
    average_cnn_models,
    clone_cnn,
    count_parameters,
    inference_cost_units,
    make_small_fashion_cnn,
)
from src.greedy_safe_selector import tau_fixed_selector  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    activation_permutation,
    cycle_score,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    load_dataset,
    make_loader,
    require_torch,
    set_seed,
    synchronize_permutations,
    train_model,
)


LAYERS = ("conv1", "conv2", "fc1")
LAYER_WIDTHS = {"conv1": 16, "conv2": 32, "fc1": 128}
METHOD_ORDER = [
    "weight_average",
    "git_rebasin_pairwise_channel",
    "c2m3_channel_permutation",
    "positive_channel_scale",
    "shrinkage_channel_scale",
    "global_channel_scale",
    "greedy_soup",
    "channel_scaled_greedy_soup",
    "greedy_safe_selector",
    "ensemble_upper_bound",
]
INT_COLUMNS = {"n_rows", "n_settings", "n_pairs", "accuracy_wins", "accuracy_ties", "accuracy_losses"}


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


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


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * float(val_fraction)))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def collect_cnn_features(model, loader, device, max_batches: int = 8) -> dict[str, np.ndarray]:
    torch, _, _ = require_torch()
    model.to(device)
    model.eval()
    rows = {layer: [] for layer in LAYERS}
    with torch.no_grad():
        for batch_idx, (x, _y) in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            x = x.to(device)
            _logits, features = model(x, return_features=True)
            rows["conv1"].append(features["conv1"].mean(dim=(2, 3)).detach().cpu())
            rows["conv2"].append(features["conv2"].mean(dim=(2, 3)).detach().cpu())
            rows["fc1"].append(features["fc1"].detach().cpu())
    return {layer: torch.cat(parts, dim=0).numpy() for layer, parts in rows.items()}


def pairwise_perms(features_by_model: dict[int, dict[str, np.ndarray]], n_models: int) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    out = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        width = LAYER_WIDTHS[layer]
        for i, j in product(range(n_models), repeat=2):
            out[layer][(i, j)] = np.arange(width) if i == j else activation_permutation(features_by_model[i][layer], features_by_model[j][layer])
    return out


def sync_perms(pairwise: dict[str, dict[tuple[int, int], np.ndarray]], n_models: int):
    refs = {}
    synced = {}
    disagreements = {}
    for layer in LAYERS:
        ref, q, residual = synchronize_permutations(pairwise[layer], n_models)
        refs[layer] = ref
        synced[layer] = q
        disagreements[layer] = residual
    return refs, synced, disagreements


def estimate_scale(source: np.ndarray, target: np.ndarray) -> float:
    denom = max(float(np.dot(source, source)), 1e-12)
    scale = float(np.dot(source, target) / denom)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(np.clip(scale, 1e-3, 1e3))


def reference_log_scales(features_by_model, synced, refs, n_models: int) -> dict[str, np.ndarray]:
    logs = {}
    for layer in LAYERS:
        width = LAYER_WIDTHS[layer]
        ref = refs[layer]
        aligned = {
            idx: features_by_model[idx][layer][:, np.asarray(synced[layer][idx], dtype=int)]
            for idx in range(n_models)
        }
        layer_logs = np.zeros((n_models, width), dtype=float)
        reference = aligned[ref]
        for idx in range(n_models):
            if idx == ref:
                continue
            for unit in range(width):
                layer_logs[idx, unit] = np.log(estimate_scale(reference[:, unit], aligned[idx][:, unit]))
        logs[layer] = layer_logs
    return logs


def global_log_scale_synchronization(features_by_model, synced, refs, n_models: int) -> tuple[dict[str, np.ndarray], float]:
    logs = {}
    residuals = []
    for layer in LAYERS:
        width = LAYER_WIDTHS[layer]
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


def shrink_logs(logs: dict[str, np.ndarray], alpha: float, tau: float) -> dict[str, np.ndarray]:
    out = {}
    for layer, values in logs.items():
        clipped = values if not np.isfinite(tau) else np.clip(values, -float(tau), float(tau))
        out[layer] = float(alpha) * clipped
    return out


def build_gauged_models(models, synced, logs: dict[str, np.ndarray]):
    out = []
    for idx, model in enumerate(models):
        out.append(
            apply_inverse_positive_alignment(
                model,
                conv1_perm=synced["conv1"][idx],
                conv2_perm=synced["conv2"][idx],
                hidden_perm=synced["fc1"][idx],
                conv1_reference_to_model_scales=np.exp(logs["conv1"][idx]),
                conv2_reference_to_model_scales=np.exp(logs["conv2"][idx]),
                hidden_reference_to_model_scales=np.exp(logs["fc1"][idx]),
            )
        )
    return out


def align_models(models, synced):
    return build_gauged_models(models, synced, {layer: np.zeros((len(models), LAYER_WIDTHS[layer])) for layer in LAYERS})


def average_eval(models, val_loader, test_loader, device):
    model = average_cnn_models(models)
    return model, evaluate_model(model, val_loader, device), evaluate_model(model, test_loader, device)


def greedy_soup(models, labels, val_loader, test_loader, device):
    scored = []
    for idx, model in enumerate(models):
        val = evaluate_model(model, val_loader, device)
        scored.append((float(val["accuracy"]), -float(val["loss"]), idx))
    order = [idx for _acc, _neg_loss, idx in sorted(scored, reverse=True)]
    selected = [order[0]]
    soup = clone_cnn(models[order[0]])
    best_val = evaluate_model(soup, val_loader, device)
    for idx in order[1:]:
        candidate_ids = selected + [idx]
        candidate = average_cnn_models([models[item] for item in candidate_ids])
        candidate_val = evaluate_model(candidate, val_loader, device)
        if candidate_val["accuracy"] > best_val["accuracy"] or (
            candidate_val["accuracy"] == best_val["accuracy"] and candidate_val["loss"] <= best_val["loss"]
        ):
            selected = candidate_ids
            soup = candidate
            best_val = candidate_val
    return {
        "model": soup,
        "selected_indices": selected,
        "selected_labels": [labels[idx] for idx in selected],
        "val": best_val,
        "test": evaluate_model(soup, test_loader, device),
    }


def feature_alignment_residual(features_by_model, synced, n_models: int) -> float:
    values = []
    for layer in LAYERS:
        aligned = {
            idx: features_by_model[idx][layer][:, np.asarray(synced[layer][idx], dtype=int)]
            for idx in range(n_models)
        }
        for i in range(n_models):
            for j in range(i + 1, n_models):
                a = aligned[i] - aligned[i].mean(axis=0, keepdims=True)
                b = aligned[j] - aligned[j].mean(axis=0, keepdims=True)
                denom = max(float(np.linalg.norm(a, ord="fro")), float(np.linalg.norm(b, ord="fro")), 1e-12)
                values.append(float(np.linalg.norm(a - b, ord="fro") / denom))
    return float(np.mean(values)) if values else 0.0


def cycle_diagnostics(pairwise, n_models: int) -> dict[str, float]:
    out = {}
    scores = []
    for layer in LAYERS:
        score, _rows = cycle_score(pairwise[layer], n_models, LAYER_WIDTHS[layer])
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
    }
    if extra:
        data.update(extra)
    rows.append(data)


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


def run_setting(args, spec, train_data, test_data, seed: int, n_models: int) -> list[dict]:
    device = device_from_arg(args.device)
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 41)
    train_loaders = []
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
        train_loaders.append(loader)

    features = {idx: collect_cnn_features(model, match_loader, device, max_batches=args.feature_batches) for idx, model in enumerate(models)}
    pairwise = pairwise_perms(features, n_models)
    refs, synced, disagreements = sync_perms(pairwise, n_models)
    ref0_synced = {layer: {idx: pairwise[layer][(0, idx)] for idx in range(n_models)} for layer in LAYERS}
    reference_logs = reference_log_scales(features, synced, refs, n_models)
    global_logs, global_rms = global_log_scale_synchronization(features, synced, refs, n_models)
    cycles = cycle_diagnostics(pairwise, n_models)
    pair_residual = feature_alignment_residual(features, synced, n_models)
    base = {
        "setting_id": f"fashion_mnist_cnn_N{n_models}_S{seed}",
        "dataset": "fashion_mnist",
        "architecture": "small_relu_cnn_no_batchnorm",
        "n_models": n_models,
        "seed": seed,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "individual_accuracy_mean": float(np.mean(individual)),
        "individual_accuracy_max": float(np.max(individual)),
        "parameter_count": count_parameters(models[0]),
        "inference_cost_units": inference_cost_units(),
        "pairwise_alignment_residual": pair_residual,
        "channel_scale_sync_rms_residual": global_rms,
        "sync_disagreement_mean": float(np.mean(list(disagreements.values()))),
        "real_central_projective_candidate": False,
        "finite_index_candidate": False,
        "channel_residual_taxonomy": "channel_gauge_diagnostic_no_brauer_candidate",
        **cycles,
    }
    rows = []

    weight_model, weight_val, weight_test = average_eval(models, val_loader, test_loader, device)
    add_row(rows, base, "weight_average", weight_val, weight_test, extra={"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True})

    pairwise_aligned = [
        apply_inverse_positive_alignment(
            model,
            conv1_perm=ref0_synced["conv1"][idx],
            conv2_perm=ref0_synced["conv2"][idx],
            hidden_perm=ref0_synced["fc1"][idx],
            conv1_reference_to_model_scales=np.ones(16),
            conv2_reference_to_model_scales=np.ones(32),
            hidden_reference_to_model_scales=np.ones(128),
        )
        for idx, model in enumerate(models)
    ]
    _pair_model, pair_val, pair_test = average_eval(pairwise_aligned, val_loader, test_loader, device)
    add_row(rows, base, "git_rebasin_pairwise_channel", pair_val, pair_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True})

    c2m3_aligned = align_models(models, synced)
    _c2m3_model, c2m3_val, c2m3_test = average_eval(c2m3_aligned, val_loader, test_loader, device)
    add_row(rows, base, "c2m3_channel_permutation", c2m3_val, c2m3_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True})

    positive_models = build_gauged_models(models, synced, reference_logs)
    _positive_model, positive_val, positive_test = average_eval(positive_models, val_loader, test_loader, device)
    add_row(rows, base, "positive_channel_scale", positive_val, positive_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "scale_source": "reference_raw", "selected_alpha": 1.0, "selected_tau": float("inf")})

    alpha_grid = parse_csv(args.alpha_grid, float)
    tau_grid = [float("inf") if item.lower() == "inf" else float(item) for item in parse_csv(args.tau_grid, str)]
    shrink_logs_selected, shrink_models, _shrink_model, shrink_val, shrink_alpha, shrink_tau = select_scale_grid(models, synced, reference_logs, alpha_grid, tau_grid, val_loader, device)
    shrink_test = evaluate_model(_shrink_model, test_loader, device)
    add_row(rows, base, "shrinkage_channel_scale", shrink_val, shrink_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "scale_source": "reference_shrinkage_validation_grid", "selected_alpha": shrink_alpha, "selected_tau": shrink_tau})

    global_logs_selected, global_models, _global_model, global_val, global_alpha, global_tau = select_scale_grid(models, synced, global_logs, alpha_grid, tau_grid, val_loader, device)
    global_test = evaluate_model(_global_model, test_loader, device)
    add_row(rows, base, "global_channel_scale", global_val, global_test, extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "scale_source": "global_log_scale_sync_validation_grid", "selected_alpha": global_alpha, "selected_tau": global_tau})

    soup = greedy_soup(models, [f"original:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "greedy_soup", soup["val"], soup["test"], extra={"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True, "soup_selected_labels": json.dumps(soup["selected_labels"]), "soup_ingredient_count": len(soup["selected_indices"])})

    scaled_soup = greedy_soup(positive_models, [f"scaled:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(rows, base, "channel_scaled_greedy_soup", scaled_soup["val"], scaled_soup["test"], extra={"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "soup_selected_labels": json.dumps(scaled_soup["selected_labels"]), "soup_ingredient_count": len(scaled_soup["selected_indices"])})

    ensemble_val = evaluate_ensemble(models, val_loader, device)
    ensemble_test = evaluate_ensemble(models, test_loader, device)
    add_row(rows, base, "ensemble_upper_bound", ensemble_val, ensemble_test, extra={"exact_relu_channel_gauge": False, "single_model": False, "capacity_matched": False})

    by_method = {row["method"]: row for row in rows}
    metrics = {method: {"accuracy": row["val_accuracy"], "loss": row["val_loss"]} for method, row in by_method.items() if method != "ensemble_upper_bound"}
    choice = tau_fixed_selector(
        metrics,
        challenger_pool=["channel_scaled_greedy_soup", "global_channel_scale", "shrinkage_channel_scale", "positive_channel_scale", "c2m3_channel_permutation"],
        tau_accuracy=args.greedy_safe_tau,
    )
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
            "selector_chose": choice.selected,
            "selector_challenger": choice.challenger,
            "selector_val_margin": choice.validation_accuracy_delta,
            "selector_left_greedy": choice.selected != "greedy_soup",
        },
    )

    by_method = {row["method"]: row for row in rows}
    c2m3_acc = by_method["c2m3_channel_permutation"]["accuracy"]
    c2m3_val_acc = by_method["c2m3_channel_permutation"]["val_accuracy"]
    greedy_acc = by_method["greedy_soup"]["accuracy"]
    greedy_val_acc = by_method["greedy_soup"]["val_accuracy"]
    weight_acc = by_method["weight_average"]["accuracy"]
    for row in rows:
        row["accuracy_delta_vs_c2m3"] = row["accuracy"] - c2m3_acc
        row["validation_delta_vs_c2m3"] = row["val_accuracy"] - c2m3_val_acc
        row["accuracy_delta_vs_greedy_soup"] = row["accuracy"] - greedy_acc
        row["validation_delta_vs_greedy_soup"] = row["val_accuracy"] - greedy_val_acc
        row["accuracy_delta_vs_weight_average"] = row["accuracy"] - weight_acc
    return rows


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


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method", sort=False):
        ci_low, ci_high = bootstrap_mean_ci(group["accuracy_delta_vs_c2m3"], n_bootstrap, seed=7500 + len(rows))
        rows.append(
            {
                "summary_type": "method_summary",
                "method": method,
                "n_rows": int(len(group)),
                "n_settings": int(group["setting_id"].nunique()),
                "mean_val_accuracy": float(group["val_accuracy"].mean()),
                "mean_test_accuracy": float(group["accuracy"].mean()),
                "mean_delta_vs_c2m3": float(group["accuracy_delta_vs_c2m3"].mean()),
                "delta_vs_c2m3_ci_low": ci_low,
                "delta_vs_c2m3_ci_high": ci_high,
                "mean_delta_vs_greedy_soup": float(group["accuracy_delta_vs_greedy_soup"].mean()),
                "mean_delta_vs_weight_average": float(group["accuracy_delta_vs_weight_average"].mean()),
                "mean_channel_permutation_cycle_score": float(group["channel_permutation_cycle_score"].mean()),
                "mean_pairwise_alignment_residual": float(group["pairwise_alignment_residual"].mean()),
                "mean_channel_scale_sync_rms_residual": float(group["channel_scale_sync_rms_residual"].mean()),
                "exact_relu_channel_gauge": bool(group["exact_relu_channel_gauge"].fillna(False).astype(bool).all()),
                "single_model": bool(group["single_model"].fillna(False).astype(bool).all()),
                "capacity_matched": bool(group["capacity_matched"].fillna(False).astype(bool).all()),
                "central_projective_candidate_fraction": float(group["real_central_projective_candidate"].fillna(False).astype(bool).mean()),
                "finite_index_candidate_fraction": float(group["finite_index_candidate"].fillna(False).astype(bool).mean()),
            }
        )
    for claim, method, baseline, forbidden in [
        ("cnn_channel_scale_over_c2m3", "positive_channel_scale", "c2m3_channel_permutation", False),
        ("cnn_shrinkage_channel_scale_over_c2m3", "shrinkage_channel_scale", "c2m3_channel_permutation", False),
        ("cnn_global_channel_scale_over_c2m3", "global_channel_scale", "c2m3_channel_permutation", False),
        ("cnn_greedy_safe_over_greedy_soup", "greedy_safe_selector", "greedy_soup", True),
    ]:
        pivot = df[df["method"].isin([method, baseline])].pivot_table(index="setting_id", columns="method", values="accuracy", aggfunc="first").dropna()
        delta = pivot[method] - pivot[baseline] if method in pivot and baseline in pivot else pd.Series(dtype=float)
        low, high = bootstrap_mean_ci(delta, n_bootstrap, seed=8100 + len(rows))
        if len(delta) and float(delta.mean()) > 0 and np.isfinite(low) and low > 0:
            decision = "Supported limited"
        elif len(delta) and float(delta.mean()) > 0 and not forbidden:
            decision = "Supported descriptive"
        else:
            decision = "Not yet supported" if forbidden else "Supported negative result"
        rows.append(
            {
                "summary_type": "claim_decision",
                "claim": claim,
                "claim_decision": decision,
                "claim_reason": f"mean paired delta={float(delta.mean()) if len(delta) else float('nan'):.6f}, CI=[{low:.6f},{high:.6f}]",
            }
        )
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "cnn_exact_channel_gauges_preserve_logits",
            "claim_decision": "Supported",
            "claim_reason": "tests/test_cnn_channel_gauge.py verifies permutation, scaling, combined gauges, parameter count, and inference-cost invariance",
        }
    )
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "cnn_residuals_are_brauer_period_index",
            "claim_decision": "Not yet supported",
            "claim_reason": "CNN benchmark records zero central/projective and finite-index candidates and does not run a Brauer detector",
        }
    )
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


def md_table(df: pd.DataFrame, cols: list[str], max_rows=40):
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


def write_plots(df, summary, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    for column, path_name, ylabel in [
        ("accuracy_delta_vs_c2m3", "fashion_cnn_delta_vs_c2m3.pdf", "test delta vs C2M3 channel permutation"),
        ("accuracy_delta_vs_greedy_soup", "fashion_cnn_delta_vs_greedy_soup.pdf", "test delta vs greedy soup"),
    ]:
        plt.figure(figsize=(7.6, 4.2))
        for idx, method in enumerate(methods):
            group = df[df["method"] == method]
            jitter = np.linspace(-0.14, 0.14, len(group)) if len(group) > 1 else np.array([0.0])
            plt.scatter(np.full(len(group), idx) + jitter, group[column], s=22, alpha=0.7)
            plt.plot([idx - 0.18, idx + 0.18], [group[column].mean()] * 2, color="black", linewidth=1.1)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xticks(range(len(methods)), [m.replace("_", "\n") for m in methods], fontsize=6)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(plot_dir / path_name)
        plt.close()
    tax = df[df["method"] == "c2m3_channel_permutation"].drop_duplicates("setting_id")
    plt.figure(figsize=(5.5, 3.5))
    vals = [
        float(tax["real_central_projective_candidate"].fillna(False).astype(bool).mean()),
        float(tax["finite_index_candidate"].fillna(False).astype(bool).mean()),
        1.0,
    ]
    plt.bar(["central/projective", "finite-index", "diagnostic/no-Brauer"], vals)
    plt.ylim(0.0, 1.05)
    plt.ylabel("fraction")
    plt.tight_layout()
    plt.savefig(plot_dir / "fashion_cnn_channel_residual_taxonomy.pdf")
    plt.close()


def latex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def write_latex_table(summary, path: Path):
    rows = summary[summary["summary_type"] == "method_summary"].copy()
    rows["rank"] = rows["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    rows = rows.sort_values("rank")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Val acc. & Test acc. & $\\Delta$ C2M3 & $\\Delta$ greedy \\\\",
        "\\midrule",
    ]
    for _idx, row in rows.iterrows():
        lines.append(
            f"{latex_escape(row['method'])} & {float(row['mean_val_accuracy']):.4f} & {float(row['mean_test_accuracy']):.4f} & "
            f"{float(row['mean_delta_vs_c2m3']):+.4f} & {float(row['mean_delta_vs_greedy_soup']):+.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(args, df, summary, path: Path):
    methods = summary[summary["summary_type"] == "method_summary"].copy()
    claims = summary[summary["summary_type"] == "claim_decision"].copy()
    report = f"""# Fashion-MNIST CNN Channel-Gauge Ladder Report

This report is generated by `experiments/fashion_mnist_cnn_ladder.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- Dataset: Fashion-MNIST
- Architecture: no-BatchNorm small ReLU CNN with channels 16/32 and hidden width 128
- Model counts: `{args.model_counts}`
- Seeds: `{args.seeds}`
- Epochs: `{args.epochs}`
- Train samples before validation split: `{args.max_train_samples}`
- Test samples: `{args.max_test_samples}` (`0` means full test set)
- N=4 was not run in this initial pass unless included in `--model-counts`; the default is the requested feasible N=3 first run.

## Exactness Status

`tests/test_cnn_channel_gauge.py` verifies channel permutation, positive channel scaling, combined gauges, unchanged parameter count, and unchanged static inference-cost proxy. Therefore channel permutation and positive channel scaling rows are labeled exact ReLU channel gauges.

## Git State

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_dirty()}`

## Main Performance Table

{md_table(methods, ["method", "n_rows", "mean_val_accuracy", "mean_test_accuracy", "mean_delta_vs_c2m3", "delta_vs_c2m3_ci_low", "delta_vs_c2m3_ci_high", "mean_delta_vs_greedy_soup", "exact_relu_channel_gauge", "single_model", "capacity_matched"], 30)}

## Residual And Gauge Diagnostics

{md_table(methods, ["method", "mean_channel_permutation_cycle_score", "mean_pairwise_alignment_residual", "mean_channel_scale_sync_rms_residual", "central_projective_candidate_fraction", "finite_index_candidate_fraction"], 30)}

## Claim Decisions

{md_table(claims, ["claim", "claim_decision", "claim_reason"], 20)}

## Negative Boundaries

- This does not claim a greedy-soup win unless the paired CNN table supports it.
- This does not compare against official external baselines.
- CNN residuals are not claimed to be Brauer or period-index classes.
- General block/channel rotations are not claimed as exact ReLU symmetries; only tested permutations and positive channel scalings are exact here.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_config(args, path: Path):
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "dirty_worktree": git_dirty(),
        "model_counts": parse_csv(args.model_counts, int),
        "seeds": parse_csv(args.seeds, int),
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "environment": capture_environment(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-counts", default="3")
    parser.add_argument("--seeds", default="6400,6401,6402")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=424242)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--feature-batches", type=int, default=8)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--tau-grid", default="0.5,1.0,2.0,inf")
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
    for n_models in parse_csv(args.model_counts, int):
        for seed in parse_csv(args.seeds, int):
            print(f"running CNN seed={seed} n_models={n_models}", flush=True)
            rows.extend(run_setting(args, spec, train_data, test_data, seed, n_models))
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
    results_path = csv_dir / "fashion_mnist_cnn_ladder.csv"
    summary_path = csv_dir / "fashion_mnist_cnn_ladder_summary.csv"
    table_path = table_dir / "fashion_cnn_ladder_table.tex"
    report_path = args.reports_dir / "fashion_mnist_cnn_ladder_report.md"
    config_path = config_dir / "fashion_mnist_cnn_ladder_config.json"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_plots(df, summary, plot_dir)
    write_latex_table(summary, table_path)
    write_report(args, df, summary, report_path)
    write_config(args, config_path)
    for path in [results_path, summary_path, table_path, report_path, config_path]:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

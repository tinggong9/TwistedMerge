#!/usr/bin/env python
"""Bridge-dataset/CIFAR feasibility benchmark for CNN channel-gauge merging."""

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

from src.cnn_channel_gauge import (  # noqa: E402
    CnnGaugeSpec,
    SmallFashionCNN,
    apply_inverse_positive_alignment,
    average_cnn_models,
    clone_cnn,
    count_parameters,
    inference_cost_units,
)
from src.greedy_safe_selector import tau_fixed_selector  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    activation_permutation,
    device_from_arg,
    evaluate_model,
    load_dataset,
    make_loader,
    require_torch,
    require_torchvision,
    set_seed,
    synchronize_permutations,
    train_model,
)


LAYERS = ("conv1", "conv2", "fc1")
METHOD_ORDER = [
    "weight_average",
    "c2m3_channel_permutation",
    "positive_channel_scale",
    "greedy_soup",
    "greedy_safe_selector",
    "individual_probe",
]
INT_COLUMNS = {"n_rows", "n_settings", "n_models", "seed", "epochs", "max_train_samples", "max_test_samples"}


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


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


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


def dataset_spec_for(name: str) -> CnnGaugeSpec:
    if name == "cifar10":
        return CnnGaugeSpec(in_channels=3, spatial_after_pool=8)
    return CnnGaugeSpec(in_channels=1, spatial_after_pool=7)


def layer_widths(spec: CnnGaugeSpec) -> dict[str, int]:
    return {"conv1": spec.conv1_channels, "conv2": spec.conv2_channels, "fc1": spec.hidden_units}


def make_model(spec: CnnGaugeSpec) -> SmallFashionCNN:
    return SmallFashionCNN(spec)


def load_bridge_dataset(args):
    _spec, train_data, test_data = load_dataset(
        "mnist",
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
    )
    return (
        "rotated_mnist",
        dataset_spec_for("rotated_mnist"),
        RotatedDataset(train_data, args.rotation_degrees),
        RotatedDataset(test_data, args.rotation_degrees),
    )


def collect_features(model, loader, device, widths: dict[str, int], max_batches: int) -> dict[str, np.ndarray]:
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
    return {layer: torch.cat(parts, dim=0).numpy()[:, : widths[layer]] for layer, parts in rows.items()}


def pairwise_perms(features_by_model: dict[int, dict[str, np.ndarray]], n_models: int, widths: dict[str, int]):
    out = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        width = widths[layer]
        for i, j in product(range(n_models), repeat=2):
            out[layer][(i, j)] = np.arange(width) if i == j else activation_permutation(features_by_model[i][layer], features_by_model[j][layer])
    return out


def sync_perms(pairwise, n_models: int):
    refs = {}
    synced = {}
    residuals = {}
    for layer in LAYERS:
        ref, q, residual = synchronize_permutations(pairwise[layer], n_models)
        refs[layer] = ref
        synced[layer] = q
        residuals[layer] = residual
    return refs, synced, residuals


def estimate_scale(source: np.ndarray, target: np.ndarray) -> float:
    denom = max(float(np.dot(source, source)), 1e-12)
    scale = float(np.dot(source, target) / denom)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(np.clip(scale, 1e-3, 1e3))


def reference_log_scales(features_by_model, synced, refs, n_models: int, widths: dict[str, int]):
    logs = {}
    for layer in LAYERS:
        width = widths[layer]
        ref = refs[layer]
        aligned = {idx: features_by_model[idx][layer][:, np.asarray(synced[layer][idx], dtype=int)] for idx in range(n_models)}
        reference = aligned[ref]
        layer_logs = np.zeros((n_models, width), dtype=float)
        for idx in range(n_models):
            if idx == ref:
                continue
            for unit in range(width):
                layer_logs[idx, unit] = np.log(estimate_scale(reference[:, unit], aligned[idx][:, unit]))
        logs[layer] = layer_logs
    return logs


def zero_logs(n_models: int, widths: dict[str, int]):
    return {layer: np.zeros((n_models, widths[layer]), dtype=float) for layer in LAYERS}


def build_gauged_models(models, synced, logs):
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
        aligned = {idx: features_by_model[idx][layer][:, np.asarray(synced[layer][idx], dtype=int)] for idx in range(n_models)}
        for i in range(n_models):
            for j in range(i + 1, n_models):
                a = aligned[i] - aligned[i].mean(axis=0, keepdims=True)
                b = aligned[j] - aligned[j].mean(axis=0, keepdims=True)
                denom = max(float(np.linalg.norm(a, ord="fro")), float(np.linalg.norm(b, ord="fro")), 1e-12)
                values.append(float(np.linalg.norm(a - b, ord="fro") / denom))
    return float(np.mean(values)) if values else 0.0


def add_row(rows: list[dict], base: dict, method: str, val: dict, test: dict, extra: dict | None = None):
    row = {
        **base,
        "method": method,
        "val_accuracy": float(val.get("accuracy", np.nan)),
        "val_loss": float(val.get("loss", np.nan)),
        "accuracy": float(test.get("accuracy", np.nan)),
        "loss": float(test.get("loss", np.nan)),
        "selection_used_validation_only": True,
        "evaluation_status": "evaluated",
    }
    if extra:
        row.update(extra)
    rows.append(row)


def merge_claim_status(dataset: str, individual_max: float, args) -> tuple[bool, str]:
    if dataset == "cifar10":
        if individual_max >= args.cifar_meaningful_threshold:
            return True, "meaningful_cifar_claims_allowed"
        if individual_max >= args.cifar_plumbing_threshold:
            return False, "cifar_plumbing_only"
        return False, "cifar_below_plumbing_threshold"
    if individual_max >= args.bridge_threshold:
        return True, "bridge_claims_allowed"
    return False, "bridge_plumbing_only_below_accuracy_threshold"


def run_merge_setting(args, dataset_name: str, spec: CnnGaugeSpec, train_data, test_data, seed: int, n_models: int, epochs: int, max_train_samples: int, max_test_samples: int):
    device = device_from_arg(args.device)
    widths = layer_widths(spec)
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 41)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 700)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 900)
    match_loader = make_loader(train_subset, args.batch_size, shuffle=False, seed=seed + 100)

    models = []
    individual = []
    for idx in range(n_models):
        model_seed = seed + idx * 1009 + 17
        set_seed(model_seed)
        model = make_model(spec)
        loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed)
        train_model(model, loader, epochs, args.lr, device)
        metrics = evaluate_model(model, test_loader, device)
        individual.append(float(metrics["accuracy"]))
        model.to("cpu")
        models.append(model)

    eligible, status = merge_claim_status(dataset_name, float(np.max(individual)), args)
    features = {idx: collect_features(model, match_loader, device, widths, args.feature_batches) for idx, model in enumerate(models)}
    pairwise = pairwise_perms(features, n_models, widths)
    refs, synced, sync_residuals = sync_perms(pairwise, n_models)
    logs = reference_log_scales(features, synced, refs, n_models, widths)
    residual = feature_alignment_residual(features, synced, n_models)
    base = {
        "setting_id": f"{dataset_name}_cnn_N{n_models}_S{seed}",
        "dataset": dataset_name,
        "dataset_role": "bridge" if dataset_name != "cifar10" else "cifar_probe_or_plumbing",
        "architecture": "small_relu_cnn_no_batchnorm",
        "n_models": n_models,
        "seed": seed,
        "epochs": epochs,
        "max_train_samples": max_train_samples,
        "max_test_samples": max_test_samples,
        "individual_accuracy_mean": float(np.mean(individual)),
        "individual_accuracy_max": float(np.max(individual)),
        "bridge_accuracy_threshold": args.bridge_threshold,
        "cifar_plumbing_threshold": args.cifar_plumbing_threshold,
        "cifar_meaningful_threshold": args.cifar_meaningful_threshold,
        "merge_claims_allowed": bool(eligible),
        "feasibility_status": status,
        "parameter_count": count_parameters(models[0]),
        "inference_cost_units": inference_cost_units(spec),
        "pairwise_alignment_residual": residual,
        "sync_disagreement_mean": float(np.mean(list(sync_residuals.values()))),
        "exact_positive_channel_scale_available": True,
    }
    rows = []

    _weight_model, weight_val, weight_test = average_eval(models, val_loader, test_loader, device)
    add_row(rows, base, "weight_average", weight_val, weight_test, {"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True})

    c2m3_models = build_gauged_models(models, synced, zero_logs(n_models, widths))
    _c2m3_model, c2m3_val, c2m3_test = average_eval(c2m3_models, val_loader, test_loader, device)
    add_row(rows, base, "c2m3_channel_permutation", c2m3_val, c2m3_test, {"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True})

    positive_models = build_gauged_models(models, synced, logs)
    _positive_model, positive_val, positive_test = average_eval(positive_models, val_loader, test_loader, device)
    add_row(rows, base, "positive_channel_scale", positive_val, positive_test, {"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True})

    soup = greedy_soup(models, [f"original:{idx}" for idx in range(n_models)], val_loader, test_loader, device)
    add_row(
        rows,
        base,
        "greedy_soup",
        soup["val"],
        soup["test"],
        {"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True, "soup_ingredient_count": len(soup["selected_indices"]), "soup_selected_labels": json.dumps(soup["selected_labels"])},
    )

    by_method = {row["method"]: row for row in rows}
    metrics = {method: {"accuracy": row["val_accuracy"], "loss": row["val_loss"]} for method, row in by_method.items()}
    choice = tau_fixed_selector(
        metrics,
        challenger_pool=["positive_channel_scale", "c2m3_channel_permutation", "weight_average"],
        tau_accuracy=args.greedy_safe_tau,
    )
    selected = by_method[choice.selected]
    add_row(
        rows,
        base,
        "greedy_safe_selector",
        {"accuracy": selected["val_accuracy"], "loss": selected["val_loss"]},
        {"accuracy": selected["accuracy"], "loss": selected["loss"]},
        {
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
    greedy_acc = by_method["greedy_soup"]["accuracy"]
    weight_acc = by_method["weight_average"]["accuracy"]
    for row in rows:
        row["accuracy_delta_vs_c2m3"] = row["accuracy"] - c2m3_acc
        row["accuracy_delta_vs_greedy_soup"] = row["accuracy"] - greedy_acc
        row["accuracy_delta_vs_weight_average"] = row["accuracy"] - weight_acc
    return rows


def run_cifar_probe(args, seed: int):
    try:
        _spec0, train_data, test_data = load_dataset("cifar10", args.data_dir, args.cifar_probe_train_samples, args.cifar_probe_test_samples, args.dataset_seed + 9000)
    except Exception as exc:
        return [
            {
                "setting_id": f"cifar10_probe_S{seed}",
                "dataset": "cifar10",
                "dataset_role": "cifar_probe_or_plumbing",
                "method": "individual_probe",
                "seed": seed,
                "n_models": 1,
                "evaluation_status": "not_run_data_or_dependency_error",
                "feasibility_status": "cifar_probe_failed",
                "failure_reason": repr(exc),
                "merge_claims_allowed": False,
            }
        ]
    device = device_from_arg(args.device)
    spec = dataset_spec_for("cifar10")
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 77)
    train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=seed + 1)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 2)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 3)
    set_seed(seed + 333)
    model = make_model(spec)
    train_model(model, train_loader, args.cifar_probe_epochs, args.lr, device)
    val = evaluate_model(model, val_loader, device)
    test = evaluate_model(model, test_loader, device)
    eligible, status = merge_claim_status("cifar10", float(test["accuracy"]), args)
    row = {
        "setting_id": f"cifar10_probe_S{seed}",
        "dataset": "cifar10",
        "dataset_role": "cifar_probe_or_plumbing",
        "architecture": "small_relu_cnn_no_batchnorm",
        "method": "individual_probe",
        "seed": seed,
        "n_models": 1,
        "epochs": args.cifar_probe_epochs,
        "max_train_samples": args.cifar_probe_train_samples,
        "max_test_samples": args.cifar_probe_test_samples,
        "val_accuracy": float(val["accuracy"]),
        "val_loss": float(val["loss"]),
        "accuracy": float(test["accuracy"]),
        "loss": float(test["loss"]),
        "individual_accuracy_mean": float(test["accuracy"]),
        "individual_accuracy_max": float(test["accuracy"]),
        "bridge_accuracy_threshold": args.bridge_threshold,
        "cifar_plumbing_threshold": args.cifar_plumbing_threshold,
        "cifar_meaningful_threshold": args.cifar_meaningful_threshold,
        "merge_claims_allowed": bool(eligible),
        "feasibility_status": status,
        "evaluation_status": "probe_only",
        "exact_positive_channel_scale_available": True,
        "single_model": True,
        "capacity_matched": True,
    }
    if float(test["accuracy"]) < args.cifar_plumbing_threshold:
        return [row]
    rows = [row]
    rows.extend(
        run_merge_setting(
            args,
            "cifar10",
            spec,
            train_data,
            test_data,
            seed,
            int(args.cifar_model_count),
            args.cifar_merge_epochs,
            args.cifar_probe_train_samples,
            args.cifar_probe_test_samples,
        )
    )
    return rows


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


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in df.groupby(["dataset", "method"], sort=False):
        low, high = bootstrap_mean_ci(group.get("accuracy_delta_vs_c2m3", pd.Series(dtype=float)), n_bootstrap, seed=9100 + len(rows))
        rows.append(
            {
                "summary_type": "method_summary",
                "dataset": dataset,
                "method": method,
                "n_rows": int(len(group)),
                "n_settings": int(group["setting_id"].nunique()),
                "mean_val_accuracy": float(group["val_accuracy"].mean()) if "val_accuracy" in group else float("nan"),
                "mean_test_accuracy": float(group["accuracy"].mean()) if "accuracy" in group else float("nan"),
                "mean_individual_accuracy_max": float(group["individual_accuracy_max"].mean()) if "individual_accuracy_max" in group else float("nan"),
                "mean_delta_vs_c2m3": float(group["accuracy_delta_vs_c2m3"].mean()) if "accuracy_delta_vs_c2m3" in group else float("nan"),
                "delta_vs_c2m3_ci_low": low,
                "delta_vs_c2m3_ci_high": high,
                "mean_delta_vs_greedy_soup": float(group["accuracy_delta_vs_greedy_soup"].mean()) if "accuracy_delta_vs_greedy_soup" in group else float("nan"),
                "mean_delta_vs_weight_average": float(group["accuracy_delta_vs_weight_average"].mean()) if "accuracy_delta_vs_weight_average" in group else float("nan"),
                "merge_claims_allowed_fraction": float(group["merge_claims_allowed"].fillna(False).astype(bool).mean()) if "merge_claims_allowed" in group else 0.0,
                "feasibility_statuses": ",".join(sorted(set(group.get("feasibility_status", pd.Series(dtype=str)).dropna().astype(str)))),
            }
        )
    for dataset, group in df.groupby("dataset", sort=False):
        statuses = set(group.get("feasibility_status", pd.Series(dtype=str)).dropna().astype(str))
        max_base = float(group["individual_accuracy_max"].max()) if "individual_accuracy_max" in group else float("nan")
        if dataset == "rotated_mnist":
            decision = "Supported limited" if max_base >= 0.80 else "Plumbing-only"
            reason = f"max individual accuracy={max_base:.4f}; bridge threshold=0.8000"
        elif dataset == "cifar10":
            if max_base >= 0.60:
                decision = "Meaningful CIFAR claims allowed"
            elif max_base >= 0.45:
                decision = "CIFAR plumbing-only"
            else:
                decision = "CIFAR not run beyond probe"
            reason = f"max individual/probe accuracy={max_base:.4f}; plumbing threshold=0.4500; meaningful threshold=0.6000; statuses={sorted(statuses)}"
        else:
            decision = "Unknown"
            reason = f"statuses={sorted(statuses)}"
        rows.append(
            {
                "summary_type": "claim_decision",
                "dataset": dataset,
                "claim": f"{dataset}_feasibility_status",
                "claim_decision": decision,
                "claim_reason": reason,
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


def md_table(df: pd.DataFrame, cols: list[str], max_rows=60) -> str:
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


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path):
    methods = summary[summary["summary_type"] == "method_summary"].copy()
    claims = summary[summary["summary_type"] == "claim_decision"].copy()
    bridge = df[df["dataset"] == "rotated_mnist"]
    cifar = df[df["dataset"] == "cifar10"]
    report = f"""# CIFAR Or Rotated-MNIST Feasibility Benchmark

Generated by `experiments/cifar_or_colored_mnist_feasibility.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- Bridge dataset tried first: deterministic rotated-MNIST at `{args.rotation_degrees}` degrees.
- Bridge claim gate: individual base accuracy must exceed `0.80`.
- CIFAR-10 gate: probe must exceed `0.45` for plumbing, and `0.60` for meaningful claims.
- Methods on eligible merge settings: weight average, C2M3-style channel/permutation synchronization, greedy soup, exact positive channel scale, and greedy-safe selector.
- The CNN has no BatchNorm, so channel permutations and positive channel scales are exact ReLU reparameterizations for this architecture.

## Method Summary

{md_table(methods, ["dataset", "method", "n_rows", "n_settings", "mean_individual_accuracy_max", "mean_val_accuracy", "mean_test_accuracy", "mean_delta_vs_c2m3", "delta_vs_c2m3_ci_low", "delta_vs_c2m3_ci_high", "mean_delta_vs_greedy_soup", "feasibility_statuses"], 80)}

## Claim Gates

{md_table(claims, ["dataset", "claim", "claim_decision", "claim_reason"], 20)}

## Bridge Rows

{md_table(bridge, ["setting_id", "method", "individual_accuracy_max", "val_accuracy", "accuracy", "accuracy_delta_vs_c2m3", "accuracy_delta_vs_greedy_soup", "feasibility_status"], 40)}

## CIFAR Status

{md_table(cifar, ["setting_id", "method", "individual_accuracy_max", "val_accuracy", "accuracy", "evaluation_status", "feasibility_status"], 20)}

## Interpretation

- Rotated-MNIST is a bridge dataset only; it is not evidence for CIFAR or broad vision generality.
- If CIFAR probe accuracy is below `0.45`, CIFAR is labeled not run beyond probe and all CIFAR statements are plumbing-only.
- If any base threshold is missed, merge rows are retained as diagnostics but not promoted to merge-performance claims.
- Greedy-safe selection uses validation metrics only.

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
        "thresholds": {
            "bridge": args.bridge_threshold,
            "cifar_plumbing": args.cifar_plumbing_threshold,
            "cifar_meaningful": args.cifar_meaningful_threshold,
        },
        "environment": capture_environment(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-counts", default="3")
    parser.add_argument("--seeds", default="7300,7301")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=6000)
    parser.add_argument("--max-test-samples", type=int, default=1200)
    parser.add_argument("--rotation-degrees", type=float, default=25.0)
    parser.add_argument("--bridge-threshold", type=float, default=0.80)
    parser.add_argument("--try-cifar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cifar-probe-epochs", type=int, default=2)
    parser.add_argument("--cifar-merge-epochs", type=int, default=2)
    parser.add_argument("--cifar-probe-train-samples", type=int, default=2500)
    parser.add_argument("--cifar-probe-test-samples", type=int, default=1000)
    parser.add_argument("--cifar-model-count", type=int, default=3)
    parser.add_argument("--cifar-plumbing-threshold", type=float, default=0.45)
    parser.add_argument("--cifar-meaningful-threshold", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=424242)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--feature-batches", type=int, default=8)
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

    dataset_name, spec, train_data, test_data = load_bridge_dataset(args)
    rows = []
    for n_models in parse_csv(args.model_counts, int):
        for seed in parse_csv(args.seeds, int):
            print(f"running {dataset_name} seed={seed} n_models={n_models}", flush=True)
            rows.extend(
                run_merge_setting(
                    args,
                    dataset_name,
                    spec,
                    train_data,
                    test_data,
                    seed,
                    n_models,
                    args.epochs,
                    args.max_train_samples,
                    args.max_test_samples,
                )
            )
    if args.try_cifar:
        print("running CIFAR-10 gated probe", flush=True)
        rows.extend(run_cifar_probe(args, int(parse_csv(args.seeds, int)[0]) + 50000))

    df = pd.DataFrame(rows)
    summary = summarize(df, args.bootstrap_samples)
    csv_dir = args.reports_dir / "csv"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "cifar_or_colored_mnist_feasibility.csv"
    summary_path = csv_dir / "cifar_or_colored_mnist_feasibility_summary.csv"
    report_path = args.reports_dir / "cifar_or_colored_mnist_feasibility.md"
    config_path = config_dir / "cifar_or_colored_mnist_feasibility_config.json"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(args, df, summary, report_path)
    write_config(args, config_path)
    for path in [results_path, summary_path, report_path, config_path]:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

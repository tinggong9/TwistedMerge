#!/usr/bin/env python
"""Small MLP/CNN model-merging benchmark on MNIST and CIFAR-10."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    DomainShiftDataset,
    average_models,
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
    primary_alignment_layer,
    primary_pairwise_permutations,
    permute_model_to_reference,
    rank_lifted_branch_models,
    require_torch,
    save_checkpoint,
    set_seed,
    synchronize_layerwise_permutations,
    synchronize_permutations,
    train_model,
)


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def default_architecture(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset in {"mnist", "fake-mnist", "fashion_mnist", "fashion-mnist", "fashionmnist"}:
        return "mlp2"
    return "small_cnn"


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def train_with_args(model, train_loader, args, device):
    return train_model(
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


def compute_alignment_bundle(models, architecture: str, loader, device, method: str):
    pairwise_by_layer = compute_layerwise_pairwise_permutations(models, architecture, loader, device, method)
    primary = primary_pairwise_permutations(pairwise_by_layer, architecture)
    return pairwise_by_layer, primary


def reference_perms(pairwise_by_layer: dict[str, dict[tuple[int, int], np.ndarray]], ref: int, idx: int) -> dict[str, np.ndarray]:
    return {layer: pairwise[(ref, idx)] for layer, pairwise in pairwise_by_layer.items()}


def synced_perms(synced_by_layer: dict[str, dict[int, np.ndarray]], idx: int) -> dict[str, np.ndarray]:
    return {layer: synced[idx] for layer, synced in synced_by_layer.items()}


def synchronize_alignment_bundle(pairwise_by_layer: dict[str, dict[tuple[int, int], np.ndarray]], n_models: int):
    if len(pairwise_by_layer) == 1:
        layer = next(iter(pairwise_by_layer))
        ref, synced, residual = synchronize_permutations(pairwise_by_layer[layer], n_models)
        return str(ref), {layer: synced}, residual
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
            widths_by_layer[layer],
            swap_fraction,
            seed + 104729 * layer_idx,
        )
        for layer_idx, (layer, pairwise) in enumerate(pairwise_by_layer.items())
    }


def run_setting(args, dataset_name: str, n_models: int, width: int, domain_shift: str) -> tuple[list[dict], list[dict]]:
    torch, _, _ = require_torch()
    device = device_from_arg(args.device)
    architecture = args.architecture if args.architecture != "auto" else default_architecture(dataset_name)
    setting_id = f"{dataset_name}_{architecture}_N{n_models}_W{width}_{domain_shift}"
    spec, train_base, test_base = load_dataset(
        dataset_name,
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.seed,
        augmentation=args.augmentation,
    )
    test_loader = make_loader(test_base, args.batch_size, shuffle=False, seed=args.seed + 999)

    models = []
    per_model_rows = []
    for model_idx in range(n_models):
        set_seed(args.seed + 1000 * model_idx + 17 * width + n_models)
        shifted = DomainShiftDataset(train_base, domain_shift, model_idx, n_models)
        train_subset, val_subset = split_train_val(shifted, args.val_fraction, args.seed + model_idx)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=args.seed + model_idx)
        val_loader_model = make_loader(val_subset, args.batch_size, shuffle=False, seed=args.seed + 100 + model_idx)
        model = make_model(architecture, spec, width)
        train_with_args(model, train_loader, args, device)
        test_metrics = evaluate_model(model, test_loader, device)
        val_metrics = evaluate_model(model, val_loader_model, device)
        model.to("cpu")
        ckpt_path = args.reports_dir / "checkpoints" / setting_id / f"model_{model_idx}.pt"
        save_checkpoint(
            model,
            ckpt_path,
            {
                "dataset": dataset_name,
                "architecture": architecture,
                "model_index": model_idx,
                "n_models": n_models,
                "width": width,
                "domain_shift": domain_shift,
                "epochs": args.epochs,
                "seed": args.seed + 1000 * model_idx,
            },
        )
        models.append(model)
        per_model_rows.append(
            {
                "setting_id": setting_id,
                "dataset": dataset_name,
                "architecture": architecture,
                "n_models": n_models,
                "width": width,
                "domain_shift": domain_shift,
                "model_index": model_idx,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "checkpoint": str(ckpt_path),
            }
        )

    val_loader = make_loader(train_base, args.batch_size, shuffle=False, seed=args.seed + 500)
    match_loader = make_loader(train_base, args.batch_size, shuffle=False, seed=args.seed + 501)
    pairwise_by_layer, pairwise = compute_alignment_bundle(models, architecture, match_loader, device, args.matching)
    primary_layer = primary_alignment_layer(architecture)
    primary_width = model_layer_widths(models[0], architecture)[primary_layer]
    score, cycle_rows = cycle_score(pairwise, n_models, primary_width)
    ref, synced_by_layer, sync_disagreement = synchronize_alignment_bundle(pairwise_by_layer, n_models)
    aligned_to_zero = [
        permute_model_to_reference(model, architecture, spec, width, reference_perms(pairwise_by_layer, 0, idx))
        for idx, model in enumerate(models)
    ]
    aligned_synced = [
        permute_model_to_reference(model, architecture, spec, width, synced_perms(synced_by_layer, idx))
        for idx, model in enumerate(models)
    ]

    baseline_metrics: list[dict] = []
    model_metrics = [row["test_accuracy"] for row in per_model_rows]
    single_best = max(model_metrics)

    def add_baseline(name: str, metrics: dict, extra: dict | None = None) -> None:
        merged_degradation = single_best - metrics["accuracy"]
        row = {
            "setting_id": setting_id,
            "dataset": dataset_name,
            "architecture": architecture,
            "n_models": n_models,
            "width": width,
            "domain_shift": domain_shift,
            "matching": args.matching,
            "baseline": name,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "single_best_accuracy": single_best,
            "merge_degradation": merged_degradation,
            "cycle_score": score,
            "cycle_score_layer": primary_layer,
            "sync_disagreement": sync_disagreement,
            "sync_reference": ref,
        }
        if extra:
            row.update(extra)
        baseline_metrics.append(row)

    weight_avg = average_models(models, architecture, spec, width)
    add_baseline("weight_average", evaluate_model(weight_avg, test_loader, device))

    soup, soup_indices, soup_metrics = greedy_soup(models, val_loader, test_loader, device, architecture, spec, width)
    add_baseline("greedy_soup", soup_metrics, {"soup_indices": json.dumps(soup_indices)})

    git_rebasin = average_models(aligned_to_zero, architecture, spec, width)
    add_baseline("git_rebasin_pairwise", evaluate_model(git_rebasin, test_loader, device))

    c2m3 = average_models(aligned_synced, architecture, spec, width)
    add_baseline("c2m3_cycle_consistent", evaluate_model(c2m3, test_loader, device))

    branches = rank_lifted_branch_models(
        aligned_synced,
        pairwise,
        args.rank_lift_branches,
        architecture,
        spec,
        width,
    )
    add_baseline(
        f"twisted_rank_lift_{len(branches)}",
        evaluate_ensemble(branches, test_loader, device),
        {"rank_lift_branches": len(branches)},
    )

    add_baseline("ensemble_upper_bound", evaluate_ensemble(models, test_loader, device))

    for name, model in [
        ("weight_average", weight_avg),
        ("greedy_soup", soup),
        ("git_rebasin_pairwise", git_rebasin),
        ("c2m3_cycle_consistent", c2m3),
    ]:
        save_checkpoint(
            model.to("cpu"),
            args.reports_dir / "checkpoints" / setting_id / f"{name}.pt",
            {"setting_id": setting_id, "baseline": name, "cycle_score": score},
        )
    for branch_idx, branch in enumerate(branches):
        save_checkpoint(
            branch.to("cpu"),
            args.reports_dir / "checkpoints" / setting_id / f"twisted_branch_{branch_idx}.pt",
            {"setting_id": setting_id, "baseline": "twisted_rank_lift", "branch_index": branch_idx},
        )

    cycle_csv = args.reports_dir / "csv" / "model_merging_cycle_defects.csv"
    cycle_rows = [
        {
            "setting_id": setting_id,
            "dataset": dataset_name,
            "architecture": architecture,
            "n_models": n_models,
            "width": width,
            "domain_shift": domain_shift,
            "cycle_score_layer": primary_layer,
            **row,
        }
        for row in cycle_rows
    ]
    if cycle_rows:
        cycle_df = pd.DataFrame(cycle_rows)
        if cycle_csv.exists():
            old = pd.read_csv(cycle_csv)
            cycle_df = pd.concat([old[old["setting_id"] != setting_id], cycle_df], ignore_index=True)
        cycle_csv.parent.mkdir(parents=True, exist_ok=True)
        cycle_df.to_csv(cycle_csv, index=False)

    return baseline_metrics, per_model_rows


def plot_cycle_scatter(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    part = df[df["baseline"].isin(["weight_average", "git_rebasin_pairwise", "c2m3_cycle_consistent"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for baseline, group in part.groupby("baseline"):
        ax.scatter(group["cycle_score"], group["merge_degradation"], label=baseline, alpha=0.8)
    ax.set_xlabel("cycle obstruction score")
    ax.set_ylabel("single-best accuracy minus merged accuracy")
    ax.set_title("Cycle score versus merge degradation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation(df: pd.DataFrame, x_col: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    part = df[df["baseline"].isin(["weight_average", "c2m3_cycle_consistent", "twisted_rank_lift_2", "ensemble_upper_bound"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for baseline, group in part.groupby("baseline"):
        summary = group.groupby(x_col)["accuracy"].mean().reset_index()
        ax.plot(summary[x_col], summary["accuracy"], marker="o", label=baseline)
    ax.set_xlabel(x_col)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def safe_pearson(x, y) -> float:
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    if xv.size < 2 or np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def safe_spearman(x, y) -> float:
    xv = pd.Series(list(x), dtype=float).rank(method="average").to_numpy()
    yv = pd.Series(list(y), dtype=float).rank(method="average").to_numpy()
    return safe_pearson(xv, yv)


def bootstrap_corr_ci(x, y, method: str, n_boot: int, seed: int) -> tuple[float, float]:
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    if xv.size < 4 or np.std(xv) == 0.0 or np.std(yv) == 0.0 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, xv.size, size=xv.size)
        if np.std(xv[idx]) == 0.0 or np.std(yv[idx]) == 0.0:
            continue
        if method == "spearman":
            vals.append(safe_spearman(xv[idx], yv[idx]))
        else:
            vals.append(safe_pearson(xv[idx], yv[idx]))
    if len(vals) < max(20, n_boot // 10):
        return float("nan"), float("nan")
    low, high = np.quantile(vals, [0.025, 0.975])
    return float(low), float(high)


def delta_status(delta: float) -> str:
    if math.isnan(delta):
        return "not_available"
    if delta > 0.005:
        return "descriptive_positive_mean_delta"
    if delta < -0.005:
        return "negative_mean_delta"
    return "no_mean_gain"


def cycle_prediction_status(row: dict, negative_control: bool) -> tuple[str, str]:
    if row["cycle_score_std"] == 0.0 or row["weight_degradation_std"] == 0.0:
        return "not_testable", "no variation in cycle score or weight-average degradation"
    if row["n_rows"] < 10:
        return "unsupported", "descriptive only: fewer than 10 repeated rows"
    if negative_control:
        return "not_interpretable", "alignment-injection negative control reuses the same trained models"
    ci_low = row["pearson_ci_low"]
    ci_high = row["pearson_ci_high"]
    if not math.isnan(ci_low) and ci_low > 0.0 and row["pearson"] > 0.3 and row["spearman"] > 0.3:
        return "supported_descriptive", "bootstrap CI excludes zero in the positive direction"
    if not math.isnan(ci_high) and ci_high < 0.0:
        return "negative_association", "bootstrap CI excludes zero in the negative direction"
    return "unsupported", "bootstrap CI crosses zero or correlations are weak"


def add_verification_baseline(rows: list[dict], base: dict, name: str, metrics: dict, extra: dict | None = None) -> None:
    single_best = base["single_best_accuracy"]
    row = {
        **base,
        "baseline": name,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "merge_degradation": single_best - metrics["accuracy"],
        "is_single_model": name not in {"ensemble_upper_bound"} and not name.startswith("twisted_rank_lift_"),
        "capacity_matched_to_weight_average": name not in {"ensemble_upper_bound"} and not name.startswith("twisted_rank_lift_"),
        "method_note": "single model",
    }
    if name == "ensemble_upper_bound":
        row["is_single_model"] = False
        row["capacity_matched_to_weight_average"] = False
        row["method_note"] = "ensemble upper bound, extra capacity"
    if name.startswith("twisted_rank_lift_"):
        row["is_single_model"] = False
        row["capacity_matched_to_weight_average"] = False
        row["method_note"] = "rank-lifted branch ensemble, extra capacity"
    if extra:
        row.update(extra)
    rows.append(row)


def run_verification_setting(args, dataset_name: str, n_models: int, width: int, domain_shift: str, seed: int) -> tuple[list[dict], list[dict]]:
    torch, _, _ = require_torch()
    device = device_from_arg(args.device)
    architecture = args.architecture if args.architecture != "auto" else default_architecture(dataset_name)
    setting_id = f"verify_{dataset_name}_{architecture}_N{n_models}_W{width}_{domain_shift}_S{seed}"
    spec, train_base, test_base = load_dataset(
        dataset_name,
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
        augmentation=args.augmentation,
    )
    test_loader = make_loader(test_base, args.batch_size, shuffle=False, seed=args.dataset_seed + 999)

    models = []
    per_model_rows = []
    for model_idx in range(n_models):
        model_seed = seed + 1000 * model_idx + 17 * width + n_models
        set_seed(model_seed)
        shifted = DomainShiftDataset(train_base, domain_shift, model_idx, n_models)
        train_subset, val_subset = split_train_val(shifted, args.val_fraction, args.dataset_seed + model_idx)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed + 11)
        val_loader_model = make_loader(val_subset, args.batch_size, shuffle=False, seed=args.dataset_seed + 100 + model_idx)
        model = make_model(architecture, spec, width)
        train_with_args(model, train_loader, args, device)
        test_metrics = evaluate_model(model, test_loader, device)
        val_metrics = evaluate_model(model, val_loader_model, device)
        model.to("cpu")
        if args.save_verification_checkpoints:
            ckpt_path = args.reports_dir / "checkpoints" / setting_id / f"model_{model_idx}.pt"
            save_checkpoint(
                model,
                ckpt_path,
                {
                    "dataset": dataset_name,
                    "architecture": architecture,
                    "model_index": model_idx,
                    "n_models": n_models,
                    "width": width,
                    "domain_shift": domain_shift,
                    "epochs": args.epochs,
                    "seed": model_seed,
                    "verification_mode": True,
                },
            )
            checkpoint = str(ckpt_path)
        else:
            checkpoint = ""
        models.append(model)
        per_model_rows.append(
            {
                "setting_id": setting_id,
                "dataset": dataset_name,
                "architecture": architecture,
                "n_models": n_models,
                "width": width,
                "domain_shift": domain_shift,
                "seed": seed,
                "model_index": model_idx,
                "model_seed": model_seed,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "checkpoint": checkpoint,
            }
        )

    val_loader = make_loader(train_base, args.batch_size, shuffle=False, seed=args.dataset_seed + 500)
    match_loader = make_loader(train_base, args.batch_size, shuffle=False, seed=args.dataset_seed + 501)
    observed_pairwise_by_layer, observed_pairwise = compute_alignment_bundle(models, architecture, match_loader, device, args.matching)
    primary_layer = primary_alignment_layer(architecture)
    primary_width = model_layer_widths(models[0], architecture)[primary_layer]
    widths_by_layer = model_layer_widths(models[0], architecture)
    model_accuracies = [row["test_accuracy"] for row in per_model_rows]
    single_best = max(model_accuracies)
    mean_individual = float(np.mean(model_accuracies))

    weight_avg = average_models(models, architecture, spec, width)
    weight_metrics = evaluate_model(weight_avg, test_loader, device)
    soup, soup_indices, soup_metrics = greedy_soup(models, val_loader, test_loader, device, architecture, spec, width)
    ensemble_metrics = evaluate_ensemble(models, test_loader, device)

    variants: list[
        tuple[
            str,
            str,
            float,
            dict[str, dict[tuple[int, int], np.ndarray]],
            dict[tuple[int, int], np.ndarray],
            bool,
        ]
    ] = [
        ("observed", "observed", 0.0, observed_pairwise_by_layer, observed_pairwise, True)
    ]
    for noise in parse_csv(args.alignment_noise_levels, float):
        noisy_by_layer = inject_layerwise_permutation_noise(
            observed_pairwise_by_layer,
            n_models,
            widths_by_layer,
            noise,
            seed + 70000 + int(round(10000 * noise)) + 13 * width + n_models,
        )
        variants.append(
            (
                f"injected_{noise:g}",
                "observed_plus_injected_noise",
                noise,
                noisy_by_layer,
                primary_pairwise_permutations(noisy_by_layer, architecture),
                False,
            )
        )

    rows = []
    for variant, alignment_source, noise, pairwise_by_layer, pairwise, independent_draw in variants:
        score, _cycle_rows = cycle_score(pairwise, n_models, primary_width)
        ref, synced_by_layer, sync_disagreement = synchronize_alignment_bundle(pairwise_by_layer, n_models)
        aligned_to_zero = [
            permute_model_to_reference(model, architecture, spec, width, reference_perms(pairwise_by_layer, 0, idx))
            for idx, model in enumerate(models)
        ]
        aligned_synced = [
            permute_model_to_reference(model, architecture, spec, width, synced_perms(synced_by_layer, idx))
            for idx, model in enumerate(models)
        ]
        base = {
            "run_id": f"{setting_id}_{variant}",
            "setting_id": setting_id,
            "dataset": dataset_name,
            "architecture": architecture,
            "n_models": n_models,
            "width": width,
            "domain_shift": domain_shift,
            "seed": seed,
            "matching": args.matching,
            "alignment_variant": variant,
            "alignment_source": alignment_source,
            "alignment_noise": noise,
            "cycle_score_layer": primary_layer,
            "independent_model_draw": independent_draw,
            "single_best_accuracy": single_best,
            "mean_individual_accuracy": mean_individual,
            "cycle_score": score,
            "sync_disagreement": sync_disagreement,
            "sync_reference": ref,
        }
        add_verification_baseline(rows, base, "weight_average", weight_metrics)
        add_verification_baseline(rows, base, "greedy_soup", soup_metrics, {"soup_indices": json.dumps(soup_indices)})
        git_rebasin = average_models(aligned_to_zero, architecture, spec, width)
        add_verification_baseline(rows, base, "git_rebasin_pairwise", evaluate_model(git_rebasin, test_loader, device))
        c2m3 = average_models(aligned_synced, architecture, spec, width)
        add_verification_baseline(rows, base, "c2m3_cycle_consistent", evaluate_model(c2m3, test_loader, device))
        branches = rank_lifted_branch_models(
            aligned_synced,
            pairwise,
            args.rank_lift_branches,
            architecture,
            spec,
            width,
        )
        add_verification_baseline(
            rows,
            base,
            f"twisted_rank_lift_{len(branches)}",
            evaluate_ensemble(branches, test_loader, device),
            {"rank_lift_branches": len(branches)},
        )
        add_verification_baseline(rows, base, "ensemble_upper_bound", ensemble_metrics)

    return rows, per_model_rows


def make_verification_wide(results: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
        "run_id",
        "setting_id",
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "seed",
        "matching",
        "alignment_variant",
        "alignment_source",
        "alignment_noise",
        "cycle_score_layer",
        "independent_model_draw",
        "single_best_accuracy",
        "mean_individual_accuracy",
        "cycle_score",
        "sync_disagreement",
    ]
    wide = results.pivot_table(index=index_cols, columns="baseline", values="accuracy", aggfunc="first").reset_index()
    wide.columns.name = None
    twisted_cols = [col for col in wide.columns if isinstance(col, str) and col.startswith("twisted_rank_lift_")]
    wide["twisted_rank_lift_accuracy"] = wide[twisted_cols[0]] if twisted_cols else np.nan
    wide["weight_merge_degradation"] = wide["single_best_accuracy"] - wide["weight_average"]
    wide["c2m3_delta_vs_weight"] = wide["c2m3_cycle_consistent"] - wide["weight_average"]
    wide["c2m3_delta_vs_greedy"] = wide["c2m3_cycle_consistent"] - wide["greedy_soup"]
    wide["twisted_delta_vs_weight"] = wide["twisted_rank_lift_accuracy"] - wide["weight_average"]
    wide["twisted_delta_vs_greedy"] = wide["twisted_rank_lift_accuracy"] - wide["greedy_soup"]
    wide["greedy_delta_vs_weight"] = wide["greedy_soup"] - wide["weight_average"]
    return wide


def compute_verification_stats(results: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    wide = make_verification_wide(results)
    rows = []

    def emit(scope: str, part: pd.DataFrame, negative_control: bool, labels: dict | None = None) -> None:
        if part.empty:
            return
        labels = labels or {}
        row = {
            "scope": scope,
            "dataset": labels.get("dataset", "ALL"),
            "architecture": labels.get("architecture", "ALL"),
            "n_models": labels.get("n_models", np.nan),
            "width": labels.get("width", np.nan),
            "domain_shift": labels.get("domain_shift", "ALL"),
            "alignment_source": labels.get("alignment_source", "ALL"),
            "n_rows": int(len(part)),
            "n_unique_seeds": int(part["seed"].nunique()),
            "cycle_score_mean": float(part["cycle_score"].mean()),
            "cycle_score_std": float(part["cycle_score"].std(ddof=0)),
            "weight_degradation_mean": float(part["weight_merge_degradation"].mean()),
            "weight_degradation_std": float(part["weight_merge_degradation"].std(ddof=0)),
            "pearson": safe_pearson(part["cycle_score"], part["weight_merge_degradation"]),
            "spearman": safe_spearman(part["cycle_score"], part["weight_merge_degradation"]),
            "c2m3_delta_vs_weight_mean": float(part["c2m3_delta_vs_weight"].mean()),
            "c2m3_delta_vs_greedy_mean": float(part["c2m3_delta_vs_greedy"].mean()),
            "twisted_delta_vs_weight_mean": float(part["twisted_delta_vs_weight"].mean()),
            "twisted_delta_vs_greedy_mean": float(part["twisted_delta_vs_greedy"].mean()),
            "greedy_delta_vs_weight_mean": float(part["greedy_delta_vs_weight"].mean()),
            "rank_lift_result_type": "branch ensemble / extra capacity",
        }
        row["pearson_ci_low"], row["pearson_ci_high"] = bootstrap_corr_ci(
            part["cycle_score"],
            part["weight_merge_degradation"],
            "pearson",
            bootstrap_samples,
            7100 + len(rows),
        )
        row["spearman_ci_low"], row["spearman_ci_high"] = bootstrap_corr_ci(
            part["cycle_score"],
            part["weight_merge_degradation"],
            "spearman",
            bootstrap_samples,
            9100 + len(rows),
        )
        cycle_status, status_note = cycle_prediction_status(row, negative_control)
        row["cycle_score_predicts_degradation"] = cycle_status
        row["c2m3_improves_weight_average"] = delta_status(row["c2m3_delta_vs_weight_mean"])
        row["twisted_improves_weight_average"] = delta_status(row["twisted_delta_vs_weight_mean"])
        row["twisted_improves_greedy"] = delta_status(row["twisted_delta_vs_greedy_mean"])
        row["statistical_status"] = status_note
        rows.append(row)

    observed = wide[wide["alignment_source"] == "observed"]
    injected = wide[wide["alignment_source"] == "observed_plus_injected_noise"]
    emit("overall_observed", observed, False)
    emit("overall_all_alignment_variants", wide, True)
    emit("overall_injected_negative_control", injected, True, {"alignment_source": "observed_plus_injected_noise"})

    for n_models, part in observed.groupby("n_models"):
        emit("fixed_N_observed", part, False, {"n_models": n_models, "alignment_source": "observed"})
    for n_models, part in wide.groupby("n_models"):
        emit("fixed_N_all_alignment_variants", part, True, {"n_models": n_models})
    for (dataset, architecture, width), part in observed.groupby(["dataset", "architecture", "width"]):
        emit(
            "fixed_dataset_arch_width_observed",
            part,
            False,
            {"dataset": dataset, "architecture": architecture, "width": width, "alignment_source": "observed"},
        )
    for (dataset, architecture, width, n_models), part in observed.groupby(["dataset", "architecture", "width", "n_models"]):
        emit(
            "fixed_dataset_arch_width_N_observed",
            part,
            False,
            {
                "dataset": dataset,
                "architecture": architecture,
                "width": width,
                "n_models": n_models,
                "alignment_source": "observed",
            },
        )
    for (dataset, architecture, width, n_models), part in wide.groupby(["dataset", "architecture", "width", "n_models"]):
        emit(
            "fixed_dataset_arch_width_N_all_alignment_variants",
            part,
            True,
            {"dataset": dataset, "architecture": architecture, "width": width, "n_models": n_models},
        )
    for (dataset, architecture, width, n_models), part in injected.groupby(["dataset", "architecture", "width", "n_models"]):
        emit(
            "fixed_dataset_arch_width_N_injected_negative_control",
            part,
            True,
            {
                "dataset": dataset,
                "architecture": architecture,
                "width": width,
                "n_models": n_models,
                "alignment_source": "observed_plus_injected_noise",
            },
        )
    return pd.DataFrame(rows)


def plot_verification_fixed_n(results: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    part = results[results["baseline"] == "weight_average"].copy()
    n_values = sorted(part["n_models"].unique())
    colors = {width: color for width, color in zip(sorted(part["width"].unique()), ["tab:blue", "tab:orange", "tab:green", "tab:red"])}
    markers = {"observed": "o", "observed_plus_injected_noise": "x"}
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(n_values), figsize=(5.2 * len(n_values), 4.0), sharey=True)
    if len(n_values) == 1:
        axes = [axes]
    for ax, n_models in zip(axes, n_values):
        panel = part[part["n_models"] == n_models]
        for (width, source), group in panel.groupby(["width", "alignment_source"]):
            ax.scatter(
                group["cycle_score"],
                group["merge_degradation"],
                label=f"W{width} {source}",
                alpha=0.75,
                marker=markers.get(source, "o"),
                color=colors.get(width, "tab:gray"),
            )
        ax.set_title(f"fixed N={n_models}")
        ax.set_xlabel("cycle obstruction score")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("single-best accuracy minus weight-average accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path)
    plt.close(fig)


def read_cifar_smoke_status(reports_dir: Path) -> str:
    path = reports_dir / "csv" / "model_merging_individual_models.csv"
    if not path.exists():
        return "CIFAR not evaluated in this verification run; no prior smoke CSV was found."
    try:
        old = pd.read_csv(path)
    except Exception as exc:
        return f"CIFAR not evaluated in this verification run; prior smoke CSV could not be read: {exc}."
    cifar = old[old["dataset"].astype(str).str.contains("cifar", case=False, na=False)]
    if cifar.empty:
        return "CIFAR not evaluated in this verification run; prior smoke CSV has no CIFAR rows."
    max_acc = float(cifar["test_accuracy"].max())
    return (
        f"CIFAR was skipped for the verification grid because the prior smoke run has max individual "
        f"CIFAR accuracy {max_acc:.4f}, below the 0.20 threshold requested for non-plumbing claims."
    )


def write_verification_report(args, results: pd.DataFrame, stats: pd.DataFrame, per_model: pd.DataFrame, report_path: Path) -> None:
    key_stats = stats[
        stats["scope"].isin([
            "overall_observed",
            "fixed_N_observed",
            "fixed_dataset_arch_width_observed",
            "fixed_dataset_arch_width_N_observed",
            "overall_injected_negative_control",
        ])
    ].copy()
    stat_cols = [
        "scope",
        "n_models",
        "width",
        "n_rows",
        "pearson",
        "spearman",
        "cycle_score_predicts_degradation",
        "c2m3_improves_weight_average",
        "twisted_improves_weight_average",
        "twisted_improves_greedy",
    ]
    negative = stats[stats["scope"] == "fixed_dataset_arch_width_N_observed"].copy()
    negative_cols = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "n_unique_seeds",
        "cycle_score_predicts_degradation",
        "c2m3_improves_weight_average",
        "twisted_improves_weight_average",
        "twisted_improves_greedy",
        "statistical_status",
    ]
    individual = (
        per_model.groupby(["dataset", "architecture", "n_models", "width"])["test_accuracy"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .to_dict("records")
    )
    capacity_rows = (
        results[results["baseline"].str.startswith("twisted_rank_lift_")]
        [["baseline", "is_single_model", "capacity_matched_to_weight_average", "method_note"]]
        .drop_duplicates()
        .to_dict("records")
    )
    report = f"""# Model Merging Verification Report

This report is generated by `experiments/model_merging_benchmark.py --mode verification`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- Dataset grid: `{args.datasets}`.
- Architecture setting: `{args.architecture}` (`auto` resolves to `mlp2` for MNIST/Fashion-MNIST after the training-quality sweep).
- Fixed-N repeated-seed settings: `N in {args.model_counts}`, widths `{args.widths}`, seeds `{args.seeds}`.
- Training uses `{args.max_train_samples}` MNIST training samples, `{args.max_test_samples}` test samples, `{args.epochs}` epochs, optimizer `{args.optimizer}`, scheduler `{args.scheduler}`, weight decay `{args.weight_decay}`, and augmentation `{args.augmentation}`.
- Alignment variants include observed activation matching plus controlled injected pairwise permutation noise levels `{args.alignment_noise_levels}`.
- Multi-layer architectures use layerwise permutation alignment and report cycle score on the primary downstream layer.
- Injected alignment rows are a negative/control intervention: they vary cycle score while reusing the same trained models, so they should not be interpreted as independent evidence that cycle score predicts weight-average degradation.

{read_cifar_smoke_status(args.reports_dir)}

## Outputs

- Verification CSV: `reports/csv/model_merging_verification.csv`
- Statistics CSV: `reports/csv/model_merging_stats.csv`
- Fixed-N plot: `reports/plots/model_merging_cycle_score_fixed_N.pdf`
- This report: `reports/model_merging_verification_report.md`

## Individual Model Accuracy

{format_markdown_table(individual, ["dataset", "architecture", "n_models", "width", "mean", "min", "max"])}

## Correlation And Delta Summary

{format_markdown_table(key_stats.to_dict("records"), stat_cols)}

## Negative Results By Fixed Setting

{format_markdown_table(negative.to_dict("records"), negative_cols)}

## Rank-Lift Capacity Label

{format_markdown_table(capacity_rows, ["baseline", "is_single_model", "capacity_matched_to_weight_average", "method_note"])}

## Interpretation

- The fixed-N observed rows are the relevant repeated-seed check for whether cycle score predicts ordinary weight-average degradation beyond the trivial `N=2` versus `N=3` confound.
- The injected-alignment rows test score sensitivity under controlled pairwise inconsistency, but weight averaging itself does not use alignments. Correlation there is a negative-control diagnostic, not a validated large-scale result.
- C2M3-style and Git-Re-Basin-style rows are single-model, capacity-matched merged models in this implementation.
- The TwistedMerge/rank-lift row is a branch ensemble with extra capacity. It must not be described as beating single merged models unless a future capacity-matched comparison is added.
- Claims are marked descriptive unless the corresponding fixed-setting repeated-seed statistics provide enough rows and a bootstrap interval that does not cross zero.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def run_verification_mode(args) -> None:
    all_results = []
    all_models = []
    for dataset_name in parse_csv(args.datasets, str):
        for n_models in parse_csv(args.model_counts, int):
            for width in parse_csv(args.widths, int):
                for domain_shift in parse_csv(args.domain_shifts, str):
                    for seed in parse_csv(args.seeds, int):
                        results, model_rows = run_verification_setting(args, dataset_name, n_models, width, domain_shift, seed)
                        all_results.extend(results)
                        all_models.extend(model_rows)

    results_df = pd.DataFrame(all_results)
    models_df = pd.DataFrame(all_models)
    stats_df = compute_verification_stats(results_df, args.bootstrap_samples)
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    verification_path = csv_dir / "model_merging_verification.csv"
    stats_path = csv_dir / "model_merging_stats.csv"
    results_df.to_csv(verification_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    plot_verification_fixed_n(results_df, plot_dir / "model_merging_cycle_score_fixed_N.pdf")
    save_json(
        args.reports_dir / "configs" / "model_merging_verification_config.json",
        {
            "argv": sys.argv,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "command_string"
            },
            "environment": capture_environment(),
        },
    )
    write_verification_report(args, results_df, stats_df, models_df, args.reports_dir / "model_merging_verification_report.md")
    print(f"wrote {verification_path}")
    print(f"wrote {stats_path}")
    print(f"wrote {plot_dir / 'model_merging_cycle_score_fixed_N.pdf'}")
    print(f"wrote {args.reports_dir / 'model_merging_verification_report.md'}")


def write_report(args, results: pd.DataFrame, per_model: pd.DataFrame, report_path: Path) -> None:
    columns = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "domain_shift",
        "baseline",
        "accuracy",
        "loss",
        "cycle_score",
        "merge_degradation",
    ]
    rows = results[columns].sort_values(["dataset", "architecture", "domain_shift", "width", "n_models", "baseline"]).to_dict("records")
    best_rows = (
        results.sort_values("accuracy", ascending=False)
        .groupby("setting_id")
        .head(1)[["setting_id", "baseline", "accuracy", "cycle_score"]]
        .to_dict("records")
    )
    corr = float(results[results["baseline"] == "weight_average"][["cycle_score", "merge_degradation"]].corr().iloc[0, 1]) if len(results[results["baseline"] == "weight_average"]) > 1 else float("nan")
    pivot = results.pivot_table(index="setting_id", columns="baseline", values="accuracy", aggfunc="mean")
    c2m3_delta = float((pivot["c2m3_cycle_consistent"] - pivot["weight_average"]).mean()) if {"c2m3_cycle_consistent", "weight_average"}.issubset(pivot.columns) else float("nan")
    twisted_delta = float((pivot.filter(like="twisted_rank_lift").iloc[:, 0] - pivot["weight_average"]).mean()) if "weight_average" in pivot.columns and any(col.startswith("twisted_rank_lift") for col in pivot.columns) else float("nan")
    single_summary = (
        per_model.groupby(["dataset", "architecture"])["test_accuracy"]
        .agg(["mean", "max"])
        .reset_index()
        .to_dict("records")
    )
    report = f"""# Model Merging Benchmark Report

This report is generated by `experiments/model_merging_benchmark.py`.

## Commands

```bash
{args.command_string}
```

## Scope

This is a small controlled benchmark for one-hidden-layer MLPs and one-block
CNNs.  It compares ordinary weight averaging, greedy model soup, pairwise
Git-Re-Basin-style permutation alignment, a C2M3-style cycle-consistent
synchronization, an ensemble upper bound, and a cycle-aware rank-lifted branch
ensemble.  The rank-lifted branch result is not a single merged model; it is
reported separately to avoid hiding that extra capacity.

## Outputs

- Baseline CSV: `reports/csv/model_merging_benchmark.csv`
- Per-model CSV: `reports/csv/model_merging_individual_models.csv`
- Cycle defects CSV: `reports/csv/model_merging_cycle_defects.csv`
- Scatter plot: `reports/plots/model_merging_cycle_score_vs_degradation.png`
- N ablation plot: `reports/plots/model_merging_ablation_n_models.png`
- Width ablation plot: `reports/plots/model_merging_ablation_width.png`
- Domain-shift table: `reports/csv/model_merging_domain_shift_summary.csv`
- Checkpoints: `reports/checkpoints/`

## Main Table

{format_markdown_table(rows, columns)}

## Best Baseline Per Setting

{format_markdown_table(best_rows, ["setting_id", "baseline", "accuracy", "cycle_score"])}

## Individual Model Accuracy

{format_markdown_table(single_summary, ["dataset", "architecture", "mean", "max"])}

## Diagnostic Claim Status

- Cycle score versus weight-average degradation correlation in this run:
  `{corr:.4f}`. Treat this as descriptive only for small smoke runs.
- C2M3-style synchronization average accuracy delta versus weight averaging:
  `{c2m3_delta:.4f}`.
- Rank-lifted branch ensemble average accuracy delta versus weight averaging:
  `{twisted_delta:.4f}`.
- The models are intentionally undertrained (`{args.epochs}` epoch, at most
  `{args.max_train_samples}` train samples per setting). CIFAR-10 accuracies are
  near chance in this smoke run, so CIFAR rows mainly test the plumbing rather
  than a validated large-scale image-model result.
- A positive result here means the score co-varies with merge degradation in
  controlled small settings. It is not yet a claim that the method beats all
  external baselines at scale.
- This run supports a weak diagnostic claim for weight averaging, but it does
  not support a general claim that cycle-consistent or rank-lifted merging
  improves accuracy across all settings.
- Negative results should be preserved in the CSV; the report is regenerated
  from all rows produced by the command above.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="benchmark", choices=["benchmark", "verification"])
    parser.add_argument("--datasets", default="mnist")
    parser.add_argument("--architecture", default="auto", choices=["auto", "mlp", "mlp2", "cnn", "small_cnn"])
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="128")
    parser.add_argument("--domain-shifts", default="none")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--augmentation", default="none", choices=["none", "light"])
    parser.add_argument("--matching", default="activation", choices=["activation", "weight"])
    parser.add_argument("--rank-lift-branches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--seeds", default="1000,1001,1002,1003,1004")
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--alignment-noise-levels", default="0.05,0.15,0.30")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--save-verification-checkpoints", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    if args.mode == "verification":
        run_verification_mode(args)
        return

    all_results = []
    all_models = []
    for dataset_name in parse_csv(args.datasets, str):
        for n_models in parse_csv(args.model_counts, int):
            for width in parse_csv(args.widths, int):
                for domain_shift in parse_csv(args.domain_shifts, str):
                    results, model_rows = run_setting(args, dataset_name, n_models, width, domain_shift)
                    all_results.extend(results)
                    all_models.extend(model_rows)

    results_df = pd.DataFrame(all_results)
    models_df = pd.DataFrame(all_models)
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "model_merging_benchmark.csv"
    models_path = csv_dir / "model_merging_individual_models.csv"
    domain_path = csv_dir / "model_merging_domain_shift_summary.csv"
    results_df.to_csv(results_path, index=False)
    models_df.to_csv(models_path, index=False)
    domain_summary = (
        results_df.groupby(["dataset", "architecture", "domain_shift", "baseline"])[["accuracy", "merge_degradation", "cycle_score"]]
        .mean()
        .reset_index()
    )
    domain_summary.to_csv(domain_path, index=False)
    plot_cycle_scatter(results_df, plot_dir / "model_merging_cycle_score_vs_degradation.png")
    plot_ablation(results_df, "n_models", plot_dir / "model_merging_ablation_n_models.png", "Ablation over number of models")
    plot_ablation(results_df, "width", plot_dir / "model_merging_ablation_width.png", "Ablation over width")
    save_json(
        args.reports_dir / "configs" / "model_merging_benchmark_config.json",
        {
            "argv": sys.argv,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "command_string"
            },
            "environment": capture_environment(),
        },
    )
    write_report(args, results_df, models_df, args.reports_dir / "model_merging_report.md")
    print(f"wrote {results_path}")
    print(f"wrote {models_path}")
    print(f"wrote {domain_path}")
    print(f"wrote {args.reports_dir / 'model_merging_report.md'}")


if __name__ == "__main__":
    main()

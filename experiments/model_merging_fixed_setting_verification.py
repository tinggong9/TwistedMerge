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
    compute_pairwise_permutations,
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
    permutation_disagreement,
    permute_model_to_reference,
    rank_lifted_branch_models,
    require_torch,
    save_checkpoint,
    set_seed,
    synchronize_permutations,
    train_model,
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
BRANCH_CAPACITY_BASELINES = (
    "random_branch_ensemble",
    "validation_branch_ensemble",
    "c2m3_cluster_branch_ensemble",
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


def summarize_seed_list(seeds: list[int]) -> str:
    if not seeds:
        return ""
    if len(seeds) > 4 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}:{seeds[-1]}"
    return ",".join(str(seed) for seed in seeds)


def permutation_json(pairwise_perms: dict[tuple[int, int], np.ndarray]) -> str:
    payload = {f"{i}->{j}": pairwise_perms[(i, j)].astype(int).tolist() for i, j in sorted(pairwise_perms)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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


def pairwise_alignment_residuals(models: list, pairwise_perms: dict[tuple[int, int], np.ndarray], loader, device, max_batches: int) -> dict:
    features = [collect_features(model, loader, device, max_batches=max_batches) for model in models]
    rows = []
    for i, j in combinations(range(len(models)), 2):
        perm = pairwise_perms[(i, j)]
        fi = features[i] - features[i].mean(axis=0, keepdims=True)
        fj = features[j][:, perm] - features[j][:, perm].mean(axis=0, keepdims=True)
        denom = float(np.linalg.norm(fi) + np.linalg.norm(fj) + 1e-12)
        residual = float(np.linalg.norm(fi - fj) / denom)
        rows.append({"pair": f"{i}-{j}", "residual": residual})
    return {
        "pairwise_alignment_residual_mean": safe_mean([row["residual"] for row in rows]),
        "pairwise_alignment_residual_max": max([row["residual"] for row in rows], default=float("nan")),
        "pairwise_alignment_residual_json": json.dumps(rows, sort_keys=True, separators=(",", ":")),
    }


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
    val_loader,
    test_loader,
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

    pairwise_aligned = [
        models[0] if idx == 0 else permute_model_to_reference(models[idx], architecture, spec, width, pairwise_perms[(0, idx)])
        for idx in range(len(models))
    ]
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

    sync_ref, synced_perms, sync_disagreement = synchronize_permutations(pairwise_perms, len(models))
    c2m3_aligned = [
        models[idx] if idx == sync_ref else permute_model_to_reference(models[idx], architecture, spec, width, synced_perms[idx])
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
    c2m3 = lookup.get("c2m3_synchronized", {})
    for row in rows:
        row["single_best_merge_degradation"] = single_best_accuracy - float(row["test_accuracy"])
        row["mean_individual_merge_degradation"] = mean_individual_accuracy - float(row["test_accuracy"])
        row["weight_merge_degradation"] = (
            single_best_accuracy - float(weight.get("test_accuracy", float("nan")))
            if row["method"] == "weight_average"
            else float("nan")
        )
        row["delta_vs_weight_average"] = float(row["test_accuracy"]) - float(weight.get("test_accuracy", float("nan")))
        row["delta_vs_greedy_soup"] = float(row["test_accuracy"]) - float(greedy.get("test_accuracy", float("nan")))
        row["delta_vs_c2m3_synchronized"] = float(row["test_accuracy"]) - float(c2m3.get("test_accuracy", float("nan")))


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
        train_model(model, train_loader, args.epochs, args.lr, device)
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

    pairwise = compute_pairwise_permutations(models, architecture, match_loader, device, matching)
    residuals = pairwise_alignment_residuals(models, pairwise, match_loader, device, args.feature_batches)
    observed_sync_ref, _observed_synced, observed_sync_disagreement = synchronize_permutations(pairwise, n_models)
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
        "dataset_seed": args.dataset_seed,
        "val_fraction": args.val_fraction,
        "mean_individual_accuracy": mean_individual_accuracy,
        "single_best_accuracy": single_best_accuracy,
        "single_worst_accuracy": single_worst_accuracy,
        "individual_accuracy_spread": single_best_accuracy - single_worst_accuracy,
        "pairwise_alignment_permutations_json": permutation_json(pairwise),
        **residuals,
    }

    run_rows = []
    triangle_out = []
    observed_cycle_score, observed_triangles = triangle_rows(shared_base, pairwise, n_models, width, "observed", 0.0)
    triangle_out.extend(observed_triangles)
    observed_base = {
        **shared_base,
        "alignment_source": "observed",
        "alignment_noise_fraction": 0.0,
        "is_injected_alignment_control": False,
        "cycle_score": observed_cycle_score,
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
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        base=observed_base,
    )
    add_paired_deltas(rows, single_best_accuracy, mean_individual_accuracy)
    run_rows.extend(rows)

    for noise_fraction in parse_float_csv(args.alignment_noise_levels):
        if noise_fraction <= 0:
            continue
        noisy = inject_pairwise_permutation_noise(pairwise, n_models, width, noise_fraction, seed + int(round(10000 * noise_fraction)))
        noisy_residuals = pairwise_alignment_residuals(models, noisy, match_loader, device, args.feature_batches)
        sync_ref, _synced, sync_disagreement = synchronize_permutations(noisy, n_models)
        noisy_cycle_score, noisy_triangles = triangle_rows(shared_base, noisy, n_models, width, "injected_noise", noise_fraction)
        triangle_out.extend(noisy_triangles)
        noisy_base = {
            **shared_base,
            **noisy_residuals,
            "pairwise_alignment_permutations_json": permutation_json(noisy),
            "alignment_source": "injected_noise",
            "alignment_noise_fraction": noise_fraction,
            "is_injected_alignment_control": True,
            "cycle_score": noisy_cycle_score,
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
            val_loader=val_loader,
            test_loader=test_loader,
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


def write_report(
    args,
    runs: pd.DataFrame,
    stats: pd.DataFrame,
    individuals: pd.DataFrame,
    triangles: pd.DataFrame,
    paired_deltas: pd.DataFrame,
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
        "--datasets mnist,fashion_mnist --architecture mlp --model-counts 3,4 --widths 32,64 "
        "--domain-shifts none,input_noise,brightness --seeds 1000:1029 --epochs 5 "
        "--max-train-samples 5000 --max-test-samples 2000 --batch-size 128 "
        "--device auto --matching activation,weight --bootstrap-samples 1000 "
        "--alignment-noise-levels 0.15"
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
    )
    if artifact_marker in text and "fixed_setting_verification_report.md" not in text:
        text = text.replace(artifact_marker, artifact_new)
    elif "real_obstruction_paired_deltas.csv" not in text:
        text += (
            "\n| `reports/real_obstruction_degradation_report.md` | Paper-facing real obstruction-degradation report with capacity-matched rank-lift branch controls. |"
            f"\n| `reports/csv/{REAL_OBSTRUCTION_RUNS_CSV}` | Paper-facing alias for per-method real obstruction-degradation rows, including capacity metadata and branch controls. |"
            f"\n| `reports/csv/{REAL_OBSTRUCTION_PAIRED_DELTAS_CSV}` | Paired rank-lift minus branch-capacity matched baseline deltas with bootstrap confidence intervals. |"
        )
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--architecture", default="mlp", choices=["mlp"])
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="32,64")
    parser.add_argument("--domain-shifts", default="none,input_noise,brightness")
    parser.add_argument("--seeds", default="1000:1029")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--matching", default="activation,weight")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--alignment-noise-levels", default="0.15")
    parser.add_argument("--rank-lift-branches", type=int, default=2)
    parser.add_argument("--feature-batches", type=int, default=8)
    parser.add_argument("--dataset-seed", type=int, default=314159)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    datasets = parse_csv(args.datasets, str)
    model_counts = parse_csv(args.model_counts, int)
    widths = parse_csv(args.widths, int)
    domain_shifts = parse_csv(args.domain_shifts, str)
    matchings = parse_csv(args.matching, str)
    seeds = parse_seeds(args.seeds)
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
    stats = compute_stats(runs, args.bootstrap_samples)
    paired_deltas = compute_branch_paired_deltas(runs, args.bootstrap_samples)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    runs_path = csv_dir / RUNS_CSV
    stats_path = csv_dir / STATS_CSV
    triangles_path = csv_dir / TRIANGLES_CSV
    individuals_path = csv_dir / INDIVIDUALS_CSV
    paired_deltas_path = csv_dir / REAL_OBSTRUCTION_PAIRED_DELTAS_CSV
    runs.to_csv(runs_path, index=False, lineterminator="\n")
    stats.to_csv(stats_path, index=False, lineterminator="\n")
    triangles.to_csv(triangles_path, index=False, lineterminator="\n")
    individuals.to_csv(individuals_path, index=False, lineterminator="\n")
    paired_deltas.to_csv(paired_deltas_path, index=False, lineterminator="\n")
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
        args.reports_dir / "fixed_setting_verification_report.md",
    )
    write_report(
        args,
        runs,
        stats,
        individuals,
        triangles,
        paired_deltas,
        args.reports_dir / "real_obstruction_degradation_report.md",
        title="Real Obstruction Degradation Verification",
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
    update_claims_audit(args.reports_dir / "claims_audit.md")
    print(f"wrote {runs_path}")
    print(f"wrote {stats_path}")
    print(f"wrote {triangles_path}")
    print(f"wrote {individuals_path}")
    print(f"wrote {paired_deltas_path}")
    print(f"wrote {args.reports_dir / 'fixed_setting_verification_report.md'}")
    print(f"wrote {args.reports_dir / 'real_obstruction_degradation_report.md'}")


if __name__ == "__main__":
    main()
